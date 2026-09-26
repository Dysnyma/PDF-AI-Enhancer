"""Synthesize a small low-quality scanned textbook PDF for E2E testing.

Creates 3 pages (A4) at ~150 DPI effective resolution with realistic scan
degradations: Gaussian blur, JPEG q30 artifacts, uneven illumination
gradient, mild noise. Page 1: Chinese text; page 2: English text + color
figure; page 3: mixed.

Usage:  env\Scripts\python.exe scripts\make_test_pdf.py [output.pdf]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

A4_PT = (595.28, 841.88)
SCAN_W, SCAN_H = 1240, 1754  # ~150 DPI A4
FONT_SIMSUN = "C:/Windows/Fonts/simsun.ttc"
FONT_TIMES = "C:/Windows/Fonts/times.ttf"

CH_TEXT = [
    "第一章 绪论",
    "",
    "扫描版文档的图像质量直接影响阅读体验与光学字符识别",
    "（OCR）的准确率。常见的扫描退化包括：分辨率不足、",
    "模糊、JPEG 压缩伪影、光照不均、纸张泛黄与阴影。",
    "",
    "本文档用于测试文档图像增强流水线。处理目标是在保留",
    "原始版面与彩色信息的前提下，尽可能提高文字区域的",
    "清晰度。测试文本包含中文、English、数字 1234567890",
    "以及标点符号，用于综合评估。",
    "",
    "1.1 研究背景与意义",
    "",
    "文档图像超分辨率（Document Image Super Resolution）",
    "与文档图像恢复（Document Image Restoration）是文档",
    "智能化的基础环节。与自然图像不同，文档图像具有强",
    "结构先验：笔画、字形与版面布局。",
]

EN_TEXT = [
    "Chapter 1  Introduction",
    "",
    "The quality of scanned documents directly affects both",
    "human readability and optical character recognition (OCR)",
    "accuracy. Common scanning degradations include insufficient",
    "resolution, blur, JPEG compression artifacts, uneven",
    "illumination, yellowed paper, and shadows.",
    "",
    "This document is used to test the document enhancement",
    "pipeline. The goal is to maximize text clarity while",
    "preserving layout and color information. Test text includes",
    "Chinese, English, digits 1234567890, and punctuation,",
    "for comprehensive evaluation.",
]


def draw_page(mode: int) -> np.ndarray:
    img = Image.new("RGB", (SCAN_W, SCAN_H), (248, 246, 240))
    d = ImageDraw.Draw(img)
    margin = 110

    if mode in (0, 2):
        font_title = ImageFont.truetype(FONT_SIMSUN, 40)
        font_body = ImageFont.truetype(FONT_SIMSUN, 26)
        y = 120
        for line in CH_TEXT:
            f = font_title if line.startswith("第") or line.startswith("1.") else font_body
            d.text((margin, y), line, fill=(30, 30, 30), font=f)
            y += 22 + (18 if f is font_title else 0)
    if mode in (1, 2):
        y0 = 120 if mode == 1 else SCAN_H - 620
        font_title = ImageFont.truetype(FONT_TIMES, 38)
        font_body = ImageFont.truetype(FONT_TIMES, 25)
        y = y0
        for line in EN_TEXT:
            f = font_title if line.startswith("Chapter") else font_body
            d.text((margin, y), line, fill=(30, 30, 30), font=f)
            y += 20 + (16 if f is font_title else 0)

    # color figure (a fake chart) on pages 1 & 2
    if mode in (1, 2):
        x0, y0, x1, y1 = margin, (700 if mode == 1 else SCAN_H - 320), SCAN_W - margin, \
                         (1000 if mode == 1 else SCAN_H - 120)
        d.rectangle([x0, y0, x1, y1], outline=(60, 60, 60), width=2)
        bar_colors = [(200, 60, 60), (60, 120, 200), (70, 150, 80), (200, 160, 50)]
        n_bars = 6
        bw = (x1 - x0 - 40) // n_bars
        rng = np.random.RandomState(42)
        for i in range(n_bars):
            h = int((y1 - y0 - 40) * (0.25 + 0.7 * rng.rand()))
            color = bar_colors[i % len(bar_colors)]
            d.rectangle([x0 + 20 + i * bw, y1 - 20 - h, x0 + 20 + (i + 1) * bw - 6, y1 - 20],
                        fill=color)

    arr = np.array(img)[:, :, ::-1].copy()  # RGB -> BGR

    # --- degradations: uneven illumination + blur + JPEG q30 + noise ---
    illum = np.ones((SCAN_H, SCAN_W), np.float32)
    xx, yy = np.meshgrid(np.linspace(-1, 1, SCAN_W), np.linspace(-1, 1, SCAN_H))
    illum *= 1.0 - 0.28 * (xx * 0.6 + yy * 0.8) ** 2  # darker toward a corner
    illum = np.clip(illum, 0.6, 1.0)
    arr = (arr.astype(np.float32) * illum[..., None]).astype(np.uint8)

    arr = cv2.GaussianBlur(arr, (3, 3), 1.2)
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 30])
    arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    noise = np.random.normal(0, 5, arr.shape).astype(np.float32)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return arr


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "input", "test_scan.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    import pymupdf as fitz
    doc = fitz.open()
    for mode in (0, 1, 2):
        img = draw_page(mode)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
        page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(out)
    doc.close()
    print(f"test PDF written: {out}")


if __name__ == "__main__":
    main()
