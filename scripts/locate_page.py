# -*- coding: utf-8 -*-
"""Locate the '天文计算' page via OCR (the book has no text layer), then
inspect the embedded scan's native resolution / JPEG subsampling and export
crops for the blur comparison."""
import io
import os
import sys

ROOT = r"E:\04_Projects\PDF_AI_Enhancer"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2  # noqa: E402
import fitz  # noqa: E402
from PIL import Image  # noqa: E402

from diag_blur import jpeg_sof_info, BOOK  # noqa: E402
from src.ocr import RapidOCRBackend, render_page_for_ocr  # noqa: E402

OUT = os.path.join(ROOT, "temp", "diag")
os.makedirs(OUT, exist_ok=True)
MODEL_DIR = os.path.join(ROOT, "models", "rapidocr")


def inspect_page_images(doc, pno: int, crop_pt: fitz.Rect):
    page = doc[pno]
    print(f"\n=== page {pno + 1} embedded images ===")
    print(f"page rect: {page.rect}  crop_pt: {crop_pt}")
    for img in page.get_images(full=True):
        xref, _sm, w, h, bpc, cs, _acs, name, filt = img[:9]
        raw = doc.xref_stream_raw(xref)
        print(f"xref={xref} {w}x{h}px bpc={bpc} cs={cs} filter={filt} "
              f"{len(raw) / 1024:.0f} KB ({len(raw) * 8 / (w * h):.2f} bit/px)")
        sof = jpeg_sof_info(raw)
        if sof:
            print(f"  SOF: {sof[0]}x{sof[1]} comps={sof[2]} subsampling={sof[3]}")
        for r in page.get_image_rects(xref):
            if r.is_empty or r.width <= 0:
                continue
            dpi_x, dpi_y = w / r.width * 72, h / r.height * 72
            print(f"  placed {r} -> effective DPI {dpi_x:.0f}x{dpi_y:.0f}")
            if not crop_pt.intersects(r):
                continue
            sx, sy = w / r.width, h / r.height
            box = (max(0, int((crop_pt.x0 - r.x0) * sx)),
                   max(0, int((crop_pt.y0 - r.y0) * sy)),
                   min(w, int((crop_pt.x1 - r.x0) * sx)),
                   min(h, int((crop_pt.y1 - r.y0) * sy)))
            with open(os.path.join(OUT, "orig.stream"), "wb") as f:
                f.write(raw)
            pix = fitz.Pixmap(doc, xref)  # decode any filter (Flate/DCT/...)
            if pix.alpha:
                pix = fitz.Pixmap(pix, 0)
            if pix.colorspace and pix.colorspace.n not in (1, 3):
                pix = fitz.Pixmap(fitz.csRGB, pix)
            ch = pix.n
            import numpy as np
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, ch)
            if ch == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            crop = arr[box[1]:box[3], box[0]:box[2]]
            cv2.imwrite(os.path.join(OUT, "orig_native_crop.png"), crop)
            print(f"  native crop saved {crop.shape[1]}x{crop.shape[0]} "
                  f"(this is ALL the real detail the file has)")
    # render crop @600dpi: what a reader shows when zoomed to ~600dpi
    zoom = 600 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False,
                          clip=crop_pt)
    pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    pil.save(os.path.join(OUT, "orig_render600_crop.png"))
    print(f"render600 crop saved {pil.size}")


def main():
    doc = fitz.open(BOOK)

    # --page N (0-based): skip OCR search, inspect that page directly with a
    # mid-page text crop (image params are uniform across the book).
    if "--page" in sys.argv:
        pno = int(sys.argv[sys.argv.index("--page") + 1])
        page = doc[pno]
        pw, ph = page.rect.width, page.rect.height
        crop_pt = fitz.Rect(pw * 0.10, ph * 0.44, pw * 0.99, ph * 0.44 + 300)
        inspect_page_images(doc, pno, crop_pt)
        return

    ocr = RapidOCRBackend(MODEL_DIR, use_cuda=False)

    found = None
    for i in range(31, 38):  # 0-based: pages 32..38 (toc: 2.8 at p33)
        img, zoom = render_page_for_ocr(BOOK, i, 200)
        for box, text, score in ocr.recognize(img):
            if "天文" in text or "望远镜阵列" in text:
                xs, ys = box[:, 0], box[:, 1]
                r = fitz.Rect(xs.min() / zoom, ys.min() / zoom,
                              xs.max() / zoom, ys.max() / zoom)
                print(f"page {i + 1}: {text!r} score={score:.2f} rect={r}")
                if "天文" in text and found is None:
                    found = (i, r)

    if found is None:
        print("NOT FOUND in candidate pages")
        sys.exit(1)

    pno, hit_r = found
    page = doc[pno]
    pw = page.rect.width
    crop_pt = fitz.Rect(pw * 0.10, hit_r.y0 - 40, pw * 0.99, hit_r.y1 + 260)
    inspect_page_images(doc, pno, crop_pt)


if __name__ == "__main__":
    main()
