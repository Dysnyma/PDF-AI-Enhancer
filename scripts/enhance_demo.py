# -*- coding: utf-8 -*-
"""One-page enhancement demo for the textbook (page 129, 0-based 128).

The book embeds each page as a native 1439x2005 image placed at a nominal
72 DPI, so we render at zoom=1 (zero interpolation, native pixels), run
DocRes + RealESRGAN x4, rebuild a single-page PDF, and produce a
side-by-side "reader zoom vs enhanced" comparison strip.
"""
import os
import sys
import time

ROOT = r"E:\04_Projects\PDF_AI_Enhancer"
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import fitz  # noqa: E402
import numpy as np  # noqa: E402

from src.backends import available_backends  # noqa: E402
from src.classify import classify_page  # noqa: E402
from src.gpu import detect_device  # noqa: E402
from src.pipeline import _build_stage_backend  # noqa: E402
from src.render import rebuild_pdf  # noqa: E402

BOOK = (r"E:\04_Projects\PDF_AI_Enhancer\input"
        r"\全国高等学校计算机教育研究会十四五规划教材  大数据与人工智能技术丛书  云计算导论  第3版 ---1.pdf")
PNO = 128  # 0-based -> physical page 129
DIAG = os.path.join(ROOT, "temp", "diag")
OUT_PDF = os.path.join(ROOT, "output", "云计算导论_页129_增强演示.pdf")
OUT_CMP = os.path.join(ROOT, "output", "compare_zoom_page129.png")

# crop box in native page pixels (matches temp/diag/orig_native_crop.png)
CX0, CY0, CX1, CY1 = 144, 882, 1425, 1182


def build_backends():
    device = detect_device("auto")
    regs = available_backends()
    restore = _build_stage_backend(
        regs["restore"], "docres", ROOT,
        {"task": "appearance", "max_size": 1024}, device, fp16=True)
    sr = _build_stage_backend(
        regs["sr"], "realesrgan-general", ROOT,
        {"scale": 4, "tile": 384}, device, fp16=True)
    return device, restore, sr


def main():
    doc = fitz.open(BOOK)
    page = doc[PNO]
    print(f"page {PNO + 1}: rect={page.rect}")

    # native-pixel render: page is a 1439x2005 image placed at 72 DPI,
    # so zoom=1 returns the scan's own pixels with zero interpolation
    t0 = time.time()
    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csRGB,
                          alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(f"native render: {img.shape[1]}x{img.shape[0]} in {time.time()-t0:.1f}s")

    device, restore, sr = build_backends()
    print(f"device: {device}")

    t0 = time.time()
    img = restore.process(img)
    print(f"docres done in {time.time()-t0:.1f}s")

    t0 = time.time()
    enh = sr.process(img)
    print(f"SR x4 done in {time.time()-t0:.1f}s -> {enh.shape[1]}x{enh.shape[0]}")

    info = classify_page(enh)
    ptype = info["type"]
    print(f"classified: {ptype} (color={info['color_ratio']:.3f} "
          f"text={info['text_ratio']:.3f})")

    cv2.imwrite(os.path.join(DIAG, "enhanced_page129.png"), enh)

    rebuild_pdf(
        [{"index": 0, "image": enh, "width_pt": float(page.rect.width),
          "height_pt": float(page.rect.height), "ptype": ptype}],
        OUT_PDF, jpeg_quality=95, monochrome="1bit")
    print(f"single-page PDF -> {OUT_PDF}")

    # ---- comparison strip ----
    native = cv2.imread(os.path.join(DIAG, "orig_native_crop.png"))
    k = 4  # SR scale
    enh_crop = enh[CY0 * k:CY1 * k, CX0 * k:CX1 * k]
    orig_zoom = cv2.resize(native, (enh_crop.shape[1], enh_crop.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(os.path.join(ROOT, "tools", "fonts", "simsun.ttc"),
                              34)
    W = 2400

    def label_row(text, fill=(245, 245, 245)):
        im = Image.new("RGB", (W, 52), fill)
        d = ImageDraw.Draw(im)
        d.text((14, 6), text, font=font, fill=(20, 20, 20))
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    def fitw(im):
        s = W / im.shape[1]
        return cv2.resize(im, (W, max(1, int(im.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)

    parts = [
        label_row("原版放大 4x（阅读器插值 —— 你看到的\"糊\"）", (255, 228, 225)),
        fitw(orig_zoom),
        label_row("AI 增强：DocRes 修复 + SR x4（真像素 5756x8020，非插值）",
                  (220, 240, 255)),
        fitw(enh_crop),
    ]
    strip = np.vstack(parts)
    cv2.imwrite(OUT_CMP, strip)
    print(f"comparison -> {OUT_CMP} ({strip.shape[1]}x{strip.shape[0]})")


if __name__ == "__main__":
    main()
