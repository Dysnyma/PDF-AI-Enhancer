"""Document restoration backends.

Backends implement:  process(img_bgr: np.ndarray) -> np.ndarray (BGR)
Registered in RESTORE_BACKENDS by name; config selects via
document_restoration.backend.
"""
import logging

import cv2
import numpy as np
import torch

from third_party.docres_restormer import Restormer
from src.gpu import load_checkpoint
from src.backends.base import RestoreBackend, register_restore

log = logging.getLogger("pdfenhance")


# --------------------------------------------------------------------------
# DTSPrompt builders — adapted from the official DocRes inference.py
# (ZZHANG-jx/DocRes, CVPR 2024). Pure OpenCV, no model involved.
# --------------------------------------------------------------------------

def appearance_prompt(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    img = cv2.resize(img, (1024, 1024))
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    result_norm = cv2.resize(result_norm, (w, h))
    return result_norm


def deshadow_prompt(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    img = cv2.resize(img, (1024, 1024))
    rgb_planes = cv2.split(img)
    bg_imgs = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        bg_imgs.append(bg_img)
    bg_imgs = cv2.merge(bg_imgs)
    bg_imgs = cv2.resize(bg_imgs, (w, h))
    return bg_imgs


def deblur_prompt(img: np.ndarray) -> np.ndarray:
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    high_frequency = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    high_frequency = cv2.cvtColor(high_frequency, cv2.COLOR_BGR2GRAY)
    high_frequency = cv2.cvtColor(high_frequency, cv2.COLOR_GRAY2BGR)
    return high_frequency


def stride_integral(img: np.ndarray, stride: int = 8):
    """Pad image top/left so dimensions are multiples of stride (official)."""
    h, w = img.shape[:2]
    padding_h = padding_w = 0
    if (h % stride) != 0:
        padding_h = stride - (h % stride)
        img = cv2.copyMakeBorder(img, padding_h, 0, 0, 0, borderType=cv2.BORDER_REPLICATE)
    if (w % stride) != 0:
        padding_w = stride - (w % stride)
        img = cv2.copyMakeBorder(img, 0, 0, padding_w, 0, borderType=cv2.BORDER_REPLICATE)
    return img, padding_h, padding_w


@register_restore("docres")
class DocResBackend(RestoreBackend):
    """DocRes generalist restoration model (CVPR 2024).

    Tasks: appearance | deshadowing | deblurring  (binarization in phase 4;
    dewarping needs the separate 713MB MBD model and is intentionally not
    bundled).

    For large pages the official code infers at 1600x1600 and divides the
    illumination map back out at full resolution. Restormer at 1600x1600
    peaks ~10.9GB VRAM (official target was 24GB cards) — we default to
    1024x1024 (the illumination map is low-frequency, quality impact is
    negligible) and shrink further automatically on OOM.
    """

    default_weights = "docres.pkl"
    default_params = {"task": "appearance", "max_size": 1024}

    def __init__(self, weights_path: str, device: torch.device,
                 fp16: bool = True, task: str = "appearance", max_size: int = 1024):
        assert task in ("appearance", "deshadowing", "deblurring"), \
            f"unsupported DocRes task: {task}"
        self.task = task
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        self.max_size = max_size

        self.model = Restormer(
            inp_channels=6, out_channels=3, dim=48,
            num_blocks=[2, 3, 3, 4],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=True,
        )
        state = load_checkpoint(weights_path)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(device)
        if self.fp16:
            self.model.half()
        log.info("DocRes loaded (task=%s, fp16=%s, weights=%s)",
                 task, self.fp16, weights_path)

    # ---- official inference paths ---------------------------------------

    def _prompt(self, img):
        if self.task == "appearance":
            return appearance_prompt(img)
        if self.task == "deshadowing":
            return deshadow_prompt(img)
        return deblur_prompt(img)

    def _forward_direct(self, in_im: np.ndarray):
        """Full-resolution forward (input already 6-channel, padded to /8).

        Normalizes to [0,1] as in the official inference code; returns uint8.
        """
        t = torch.from_numpy(in_im.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        t = t.float() / 255.0
        if self.fp16:
            t = t.half()
        with torch.no_grad():
            pred = self.model(t)
        pred = torch.clamp(pred, 0, 1)
        pred = pred[0].permute(1, 2, 0).float().cpu().numpy()
        return (pred * 255).astype(np.uint8)

    def _forward_ratio(self, img_bgr: np.ndarray, prompt: np.ndarray,
                       max_size: int) -> np.ndarray:
        """Large-image path: infer illumination at max_size, divide at full res."""
        h, w = img_bgr.shape[:2]
        in_im = np.concatenate((img_bgr, prompt), -1)
        small = cv2.resize(in_im, (max_size, max_size))
        pred = self._forward_direct(small)
        pred[pred == 0] = 1
        shadow_map = cv2.resize(img_bgr, (max_size, max_size)).astype(np.float64) \
            / pred.astype(np.float64)
        shadow_map = cv2.resize(shadow_map, (w, h))
        shadow_map[shadow_map == 0] = 0.00001
        return np.clip(img_bgr.astype(np.float64) / shadow_map, 0, 255).astype(np.uint8)

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        prompt = self._prompt(img_bgr)

        if self.task == "deblurring":
            in_im, padding_h, padding_w = stride_integral(
                np.concatenate((img_bgr, prompt), -1), 8)
            try:
                out_im = self._forward_direct(in_im)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.warning("DocRes deblurring OOM at %dx%d, falling back to "
                            "appearance-style ratio path", w, h)
                prompt = appearance_prompt(img_bgr)
                return self._forward_ratio(img_bgr, prompt, self.max_size)
            return out_im[padding_h:, padding_w:]

        if max(w, h) < self.max_size:
            in_im, padding_h, padding_w = stride_integral(
                np.concatenate((img_bgr, prompt), -1), 8)
            out_im = self._forward_direct(in_im)
            return out_im[padding_h:, padding_w:]

        # Large image (typical 300 DPI scan): ratio path with OOM shrink retry
        for size in (self.max_size, 768, 512, 384):
            try:
                return self._forward_ratio(img_bgr, prompt, size)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.warning("DocRes OOM at infer size %d, shrinking", size)
        raise RuntimeError("DocRes OOM even at 384x384 — reduce render DPI")


@register_restore("none")
class NoopRestore(RestoreBackend):
    needs_weights = False

    def __init__(self, *args, **kwargs):
        pass

    def process(self, img_bgr):
        return img_bgr
