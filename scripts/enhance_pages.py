# -*- coding: utf-8 -*-
"""Partial-book enhancement: pages [--from, --to] (0-based, inclusive).

Same pipeline as the full run (auto DPI render -> DocRes -> by-type SR x4 ->
classify -> q95 encode + hidden OCR text layer), just restricted to a page
range so a sample PDF can be produced quickly.
"""
import argparse
import os
import sys
import time

ROOT = r"E:\04_Projects\PDF_AI_Enhancer"
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import fitz  # noqa: E402
import numpy as np  # noqa: E402

from src.backends import available_backends  # noqa: E402
from src.classify import classify_page, coarse_type  # noqa: E402
from src.gpu import detect_device  # noqa: E402
from src.ocr import (RapidOCRBackend, boxes_to_pdf_text,  # noqa: E402
                     render_page_for_ocr)
from src.pipeline import _build_stage_backend  # noqa: E402
from src.render import _page_nominal_dpi, rebuild_pdf  # noqa: E402

BOOK = (r"E:\04_Projects\PDF_AI_Enhancer\input"
        r"\全国高等学校计算机教育研究会十四五规划教材  大数据与人工智能技术丛书  云计算导论  第3版 ---1.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=BOOK, help="source PDF path")
    ap.add_argument("--from", dest="p0", type=int, required=True)
    ap.add_argument("--to", dest="p1", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = detect_device("auto")
    regs = available_backends()
    restore = _build_stage_backend(
        regs["restore"], "docres", ROOT,
        {"task": "appearance", "max_size": 1024}, device, fp16=True)
    sr_text = _build_stage_backend(
        regs["sr"], "realesrgan-general", ROOT,
        {"scale": 4, "tile": 384}, device, fp16=True)
    sr_image = _build_stage_backend(
        regs["sr"], "swinir", ROOT,
        {"scale": 4, "tile": 128}, device, fp16=True)
    ocr = RapidOCRBackend(os.path.join(ROOT, "models", "rapidocr"),
                          use_cuda=False)
    font = os.path.join(ROOT, "tools", "fonts", "simsun.ttc")

    doc = fitz.open(args.input)
    idxs = list(range(args.p0, args.p1 + 1))

    # --- OCR pass (hidden text layer, 200 DPI, maps to original page pt) ---
    text_layers = {}
    for i in idxs:
        img, zoom = render_page_for_ocr(args.input, i, 200)
        boxes = ocr.recognize(img)
        text_layers[i] = boxes_to_pdf_text(boxes, zoom)
        print(f"page {i + 1} OCR: {len(boxes)} spans", flush=True)

    # --- enhance pass ---
    out_pages = []
    for i in idxs:
        page = doc[i]
        nominal = _page_nominal_dpi(page)
        dpi = 300 if nominal <= 0 else int(min(max(nominal, 72.0), 450.0))
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                              colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = restore.process(img)
        ctype = coarse_type(img)
        t0 = time.time()
        img = (sr_image if ctype == "image" else sr_text).process(img)
        sr_name = "swinir" if ctype == "image" else "realesrgan"
        info = classify_page(img)
        out_pages.append({
            "index": i, "image": img,
            "width_pt": page.rect.width, "height_pt": page.rect.height,
            "ptype": info["type"],
        })
        print(f"page {i + 1}: {info['type']} <- {sr_name} "
              f"(sr {time.time() - t0:.1f}s)", flush=True)

    rebuild_pdf(out_pages, args.out, jpeg_quality=95,
                text_layers=text_layers, font_file=font, monochrome="1bit",
                jbig2_bin=os.path.join(ROOT, "tools", "jbig2", "jbig2.exe"))
    size = os.path.getsize(args.out) / 1024 / 1024
    print(f"done: {args.out} ({len(out_pages)} pages, {size:.1f} MB)")


if __name__ == "__main__":
    main()
