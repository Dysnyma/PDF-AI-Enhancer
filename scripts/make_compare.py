"""Generate side-by-side 400% zoom comparison crops for the benchmark report.

For each pipeline config, renders pages at 150 DPI (keeps SwinIR fast enough),
crops a text region (page 1) and a color-figure region (page 2), upsamples 4x
with nearest-neighbor (to show the true pixels), and stacks them into a labeled
comparison image.

Crop rectangles are defined in *source PDF point* space so they stay aligned
across configs with different output resolutions (SR x2 changes pixel size,
not the underlying page geometry).

Usage:  env\\Scripts\\python.exe scripts\\make_compare.py [output_dir]
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.backends import available_backends
from src.backends.base import resolve_weights
from src.gpu import detect_device
from src.render import render_pdf_pages

MODELS = os.path.join(ROOT, "models")

DPI = 150  # lower DPI keeps SwinIR (transformer) fast enough for the report

# configs in display order (left -> right)
CONFIGS = [
    ("baseline", "原始扫描", "none", "none"),
    ("docres", "DocRes 修复", "docres", "none"),
    ("realesrgan", "DocRes+RealESRGAN", "docres", "realesrgan-general"),
    ("swinir", "DocRes+SwinIR", "docres", "swinir"),
]

# crop regions in page point coordinates (A4 = 595 x 842 pt), independent of
# render DPI. These are converted to pixels using the per-config scale.
CROP_TEXT_PT = (60, 150, 340, 150)    # Chinese body text block on page 1
CROP_FIG_PT = (60, 330, 400, 190)     # color bar chart on page 2


def build(regs, device, restore_name, sr_name):
    def make(registry, name):
        if name == "none":
            return registry["none"]()
        cls = registry[name]
        w = resolve_weights(ROOT, cls.default_weights)
        return cls.create(device, w, dict(cls.default_params), fp16=True)
    return make(regs["restore"], restore_name), make(regs["sr"], sr_name)


def pt_to_px(rect_pt, scale):
    """Convert a (x, y, w, h) rect in points to pixel coords for a given scale
    (px per point)."""
    x, y, w, h = rect_pt
    return (int(x * scale), int(y * scale), int(w * scale), int(h * scale))


def zoom_crop(img, rect_px, factor=4):
    x, y, w, h = rect_px
    crop = img[y:y + h, x:x + w]
    return cv2.resize(crop, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_NEAREST)


def stack(imgs, labels, title, out_path):
    """Horizontal stack of crops, each with a label banner on top.

    All crops are resized to the same height (of the first) to tolerate +/-1px
    rounding differences in the zoom factor.
    """
    h = imgs[0].shape[0]
    imgs = [img if img.shape[0] == h else cv2.resize(img, (img.shape[1], h))
            for img in imgs]
    labeled = []
    for img, lab in zip(imgs, labels):
        banner = np.full((40, img.shape[1], 3), 240, np.uint8)
        cv2.putText(banner, lab, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)
        labeled.append(np.vstack([banner, img]))
    grid = np.hstack(labeled)
    cv2.imwrite(out_path, grid)
    print(f"wrote {out_path}  ({grid.shape[1]}x{grid.shape[0]})")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "output", "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(ROOT, "input", "test_scan.pdf")
    device = detect_device("auto")
    regs = available_backends()

    pages = render_pdf_pages(pdf, DPI)
    base_scale = DPI / 72.0  # px per point at the render DPI

    # process every config, keep the enhanced page images + their scale
    enhanced = {}
    for key, label, rn, sn in CONFIGS:
        restore, sr = build(regs, device, rn, sn)
        imgs = []
        for p in pages:
            img = p["image"]
            img = restore.process(img)
            img = sr.process(img)
            imgs.append(img)
        # scale factor of the enhanced image vs the base render (SR x2 -> 2.0)
        scale = imgs[0].shape[1] / pages[0]["image"].shape[1]
        enhanced[key] = (imgs, scale)
        print(f"processed config {key} (scale={scale:.1f}x)", flush=True)

    # --- text region comparison (page 1) ---
    text_imgs = []
    for key, label, *_ in CONFIGS:
        imgs, scale = enhanced[key]
        px = pt_to_px(CROP_TEXT_PT, base_scale * scale)
        # zoom factor compensates for SR: SR x2 already has 2x pixels, so we
        # zoom by 4/scale to land every config at the same final display size
        # (base DPI x 4), making the crops directly comparable.
        text_imgs.append(zoom_crop(imgs[0], px, factor=4.0 / scale))
    stack(text_imgs, [l for _, l, *_ in CONFIGS], "text",
          os.path.join(out_dir, "compare_text.png"))

    # --- color figure comparison (page 2) ---
    fig_imgs = []
    for key, label, *_ in CONFIGS:
        imgs, scale = enhanced[key]
        px = pt_to_px(CROP_FIG_PT, base_scale * scale)
        fig_imgs.append(zoom_crop(imgs[1], px, factor=4.0 / scale))
    stack(fig_imgs, [l for _, l, *_ in CONFIGS], "figure",
          os.path.join(out_dir, "compare_figure.png"))


if __name__ == "__main__":
    main()
