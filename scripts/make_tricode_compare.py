# -*- coding: utf-8 -*-
"""Three-way encoding comparison for page 129, same field of view as the
user's first screenshot: original (interpolated zoom) vs AI-enhanced q95
JPEG (grayscale-preserving) vs AI-enhanced 1-bit binarized."""
import os

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = r"E:\04_Projects\PDF_AI_Enhancer"
OUT = os.path.join(ROOT, "output", "compare_encoding_p129.png")

BOOK = os.path.join(ROOT, "input",
                    "全国高等学校计算机教育研究会十四五规划教材  大数据与人工智能技术丛书  云计算导论  第3版 ---1.pdf")
V1 = os.path.join(ROOT, "output", "云计算导论_页129_增强演示.pdf")    # q95 mixed
V2 = os.path.join(ROOT, "output", "云计算导论_页129_增强演示_v2.pdf")  # 1-bit bw-text

CLIP = fitz.Rect(184, 967, 584, 1177)  # "2. 天文计算领域" + 2 body lines (pt)
ZOOM = 6  # reader-side zoom factor


def crop_of(path, pno):
    doc = fitz.open(path)
    pix = doc[pno].get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM),
                              colorspace=fitz.csRGB, alpha=False, clip=CLIP)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(
        pix.height, pix.width, 3)
    doc.close()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    rows = [
        ("原版（~197dpi 源扫描，阅读器放大即糊）", (255, 228, 225),
         crop_of(BOOK, 128)),
        ("AI 增强 → q95 JPEG（灰度忠实、边缘柔和、5.1MB/页）", (255, 240, 200),
         crop_of(V1, 0)),
        ("AI 增强 → 1-bit 二值化（黑实硬边、0.5MB/页、省 10 倍空间）", (220, 240, 255),
         crop_of(V2, 0)),
    ]

    font = ImageFont.truetype(os.path.join(ROOT, "tools", "fonts", "simsun.ttc"),
                              34)
    W = 2400

    def label_row(text, fill):
        im = Image.new("RGB", (W, 52), fill)
        ImageDraw.Draw(im).text((14, 6), text, font=font, fill=(20, 20, 20))
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    def fitw(im):
        s = W / im.shape[1]
        return cv2.resize(im, (W, max(1, int(im.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)

    parts = []
    for text, fill, im in rows:
        parts.append(label_row(text, fill))
        parts.append(fitw(im))
    strip = np.vstack(parts)
    cv2.imwrite(OUT, strip)
    print(f"-> {OUT} ({strip.shape[1]}x{strip.shape[0]})")


if __name__ == "__main__":
    main()
