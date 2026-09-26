"""Super-resolution backends.

Backends implement:  process(img_bgr: np.ndarray) -> np.ndarray (BGR)
Registered in SR_BACKENDS by name; config selects via
super_resolution.backend.
"""
import logging

import cv2
import numpy as np
import torch

from third_party.srvgg import SRVGGNetCompact
from src.gpu import load_checkpoint, tiled_forward, vram_headroom_gb
from src.backends.base import SRBackend, register_sr

log = logging.getLogger("pdfenhance")


@register_sr("realesrgan-general")
class RealESRGANGeneral(SRBackend):
    """realesr-general-x4v3 (SRVGGNetCompact, 5MB) — fast, doc-friendly SR.

    Native upscale is x4. For --scale 2 we run x4 and Lanczos-downsample to
    x2, which yields cleaner anti-aliased text than plain x2 models.

    Note: SRVGG overflows in fp16 (produces NaN) — this backend always runs
    fp32. The model is tiny so fp32 is fast and VRAM-safe.
    """

    default_weights = "realesr-general-x4v3.pth"
    default_params = {"scale": 2, "tile": 384}

    def __init__(self, weights_path: str, device: torch.device,
                 fp16: bool = True, scale: int = 2, tile: int = 384):
        assert scale in (2, 4), "scale must be 2 or 4"
        self.scale = scale
        self.device = device
        self.fp16 = False  # fp32 forced: fp16 NaNs on SRVGG (verified)
        if fp16 and device.type == "cuda":
            log.info("SRVGG runs fp32 (fp16 produces NaN on this arch)")

        # Pick tile by available VRAM
        if device.type == "cuda":
            headroom = vram_headroom_gb(device)
            if headroom < 3.0 and tile > 256:
                tile = 256
                log.info("low VRAM (%.1f GB free), tile reduced to %d", headroom, tile)
        self.tile = tile

        self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                     num_conv=32, upscale=4, act_type="prelu")
        state = load_checkpoint(weights_path)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(device)
        if self.fp16:
            self.model.half()
        log.info("realesr-general-x4v3 loaded (scale=%d, tile=%d, fp16=%s)",
                 scale, tile, self.fp16)

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(img_bgr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        t = t.float() / 255.0  # SRVGG expects [0,1] input (official convention)
        with torch.no_grad():
            pred = tiled_forward(self.model, t, tile=self.tile, overlap=32, scale=4)
        out = pred[0].permute(1, 2, 0).float().cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

        if self.scale == 2:
            h, w = img_bgr.shape[:2]
            out = cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        return out


@register_sr("none")
class NoopSR(SRBackend):
    needs_weights = False

    def __init__(self, *args, **kwargs):
        pass

    def process(self, img_bgr):
        return img_bgr
