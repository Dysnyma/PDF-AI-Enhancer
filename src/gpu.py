"""Torch utilities: device detection, tiled inference, OOM-safe execution."""
import logging

import torch

log = logging.getLogger("pdfenhance")


def detect_device(preferred: str = "auto") -> torch.device:
    """Detect best device. Checks CUDA availability and free VRAM."""
    if preferred == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        free, total = torch.cuda.mem_get_info(0)
        log.info("GPU: %s | free VRAM %.1f GB / %.1f GB",
                 torch.cuda.get_device_name(0), free / 1024**3, total / 1024**3)
        return dev
    log.info("CUDA unavailable, using CPU")
    return torch.device("cpu")


def vram_headroom_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    free, _ = torch.cuda.mem_get_info(0)
    return free / 1024**3


def tiled_forward(model, x, tile=512, overlap=32, on_oom_shrink=True, scale=1):
    """Run model.forward on a large image using overlapping tiles.

    Tiles are feather-blended to avoid seams. On CUDA OOM the tile size is
    reduced automatically (512 -> 384 -> 256 -> 192) instead of crashing.

    scale: output/input ratio (SR models upsample). Feather window and
    accumulation buffer are sized to the *output* resolution.
    """
    b, c, h, w = x.shape
    device = x.device

    if h <= tile and w <= tile:
        return model(x)

    out_h, out_w = h * scale, w * scale
    out_ch = model_out_channels(model, c)

    stride = tile - overlap
    out = torch.zeros((b, out_ch, out_h, out_w), device=device)
    weight = torch.zeros((1, 1, out_h, out_w), device=device)

    # 2D linear feather window (output resolution) over the overlap band
    ft = tile * scale
    fo = overlap * scale
    feather = torch.ones((ft, ft), device=device)
    if fo > 0:
        ramp = torch.linspace(0.0, 1.0, fo + 1, device=device)[1:]
        feather[:fo, :] *= ramp.unsqueeze(1)
        feather[-fo:, :] *= ramp.flip(0).unsqueeze(1)
        feather[:, :fo] *= ramp.unsqueeze(0)
        feather[:, -fo:] *= ramp.flip(0).unsqueeze(0)

    tiles_y = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    tiles_x = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    if tiles_y[-1] + tile < h:
        tiles_y.append(h - tile)
    if tiles_x[-1] + tile < w:
        tiles_x.append(w - tile)

    for ty in tiles_y:
        for tx in tiles_x:
            patch = x[:, :, ty:ty + tile, tx:tx + tile]
            try:
                with torch.no_grad():
                    pred = model(patch)
            except torch.cuda.OutOfMemoryError:
                if on_oom_shrink and tile > 192:
                    torch.cuda.empty_cache()
                    smaller = {512: 384, 384: 256, 256: 192}.get(tile, 192)
                    log.warning("OOM at tile=%d, retrying whole image with tile=%d", tile, smaller)
                    return tiled_forward(model, x, tile=smaller, overlap=overlap, scale=scale)
                raise
            ph, pw = pred.shape[-2], pred.shape[-1]
            oy, ox = ty * scale, tx * scale
            f = feather[:ph, :pw]
            out[:, :, oy:oy + ph, ox:ox + pw] += pred * f
            weight[:, :, oy:oy + ph, ox:ox + pw] += f

    out = out / weight.clamp(min=1e-8)
    return out


def model_out_channels(model, in_channels):
    """Best-effort output channel count for output buffer allocation."""
    # Both our backends output 3 channels
    return 3


def load_checkpoint(path, weights_only=True):
    """torch.load wrapper handling torch>=2.6 default and DataParallel keys."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=weights_only)
    except Exception:
        # Older pickles may need weights_only=False (file comes from a known
        # upstream source we downloaded ourselves).
        log.warning("weights_only load failed for %s, retrying permissive load", path)
        state = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "params" in state:
        state = state["params"]

    # strip DataParallel "module." prefix if present
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state
