"""SwinIR super-resolution backend (real-world SR, SwinIR-M).

This is the *example* backend that proves the pluggable architecture: it has
a totally different network (Swin Transformer vs. SRVGG conv net), different
weights (67MB vs. 5MB), different memory profile, and a different param set —
yet the pipeline instantiates it with zero changes.

SwinIR is a heavy transformer: on an 8GB card we infer in small tiles (the
default tile is small enough for the attention maps) and use fp16 to keep
VRAM within budget. Quality is noticeably higher than SRVGG on natural
images, at ~10-20x the compute.
"""
import logging

import cv2
import numpy as np
import torch

from third_party.swinir import build_swinir
from src.gpu import load_checkpoint, tiled_forward, vram_headroom_gb
from src.backends.base import SRBackend, register_sr

log = logging.getLogger("pdfenhance")


@register_sr("swinir")
class SwinIRBackend(SRBackend):
    """SwinIR-M real-world x4 SR (003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN).

    Native upscale is x4. ``scale`` 2 runs x4 and Lanczos-downsamples to x2.
    """

    default_weights = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    default_params = {"scale": 2, "tile": 128}

    def __init__(self, weights_path: str, device: torch.device,
                 fp16: bool = True, scale: int = 2, tile: int = 128):
        assert scale in (2, 4), "scale must be 2 or 4"
        self.scale = scale
        self.device = device
        # SwinIR (Swin Transformer) underflows to all-zero output in fp16 —
        # verified on this arch. Force fp32 like SRVGG. VRAM is modest: even
        # tile=128 peaks ~0.3GB in fp32.
        self.fp16 = False
        if fp16 and device.type == "cuda":
            log.info("SwinIR runs fp32 (fp16 produces all-zero output on this arch)")

        # SwinIR tile must be a multiple of window_size (8).
        tile = max(64, (tile // 8) * 8)
        if device.type == "cuda":
            headroom = vram_headroom_gb(device)
            if headroom < 3.0 and tile > 128:
                tile = 128
                log.info("low VRAM (%.1f GB free), tile reduced to %d", headroom, tile)
            elif headroom < 1.5:
                tile = 64
                log.info("very low VRAM (%.1f GB free), tile reduced to %d", headroom, tile)
        self.tile = tile

        self.model = build_swinir("m")
        state = load_checkpoint(weights_path)
        # official SwinIR checkpoints store weights under "params_ema"
        state = state.get("params_ema", state)
        # The checkpoint also stores pre-computed cyclic-shift attention masks
        # (buffers) for a fixed img_size; we rebuild those on the fly for any
        # tile size, so drop them here.
        state = {k: v for k, v in state.items() if "attn_mask" not in k}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.to(device)
        if self.fp16:
            self.model.half()
        log.info("SwinIR-M loaded (scale=%d, tile=%d, fp16=%s, weights=%s)",
                 scale, tile, self.fp16, weights_path)

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(img_bgr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        t = t.float() / 255.0  # SwinIR expects [0,1] (img_range=1.0)
        if self.fp16:
            t = t.half()
        with torch.no_grad():
            pred = tiled_forward(self.model, t, tile=self.tile, overlap=32, scale=4)
        out = pred[0].permute(1, 2, 0).float().cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

        if self.scale == 2:
            h, w = img_bgr.shape[:2]
            out = cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        return out
