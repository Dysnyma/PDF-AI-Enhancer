# -*- coding: utf-8 -*-
"""MRC verification on physical page 35 (baidu-cloud screenshot + body text).

Enhance once, rebuild twice (whole-page q95 JPEG vs MRC layered), then emit
a three-row comparison strip (original / q95 / MRC) and size numbers.
"""
import os
import sys
import time

ROOT = r"E:\04_Projects\PDF_AI_Enhancer"
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import fitz  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.backends import available_backends  # noqa: E402
from src.classify import classify_page  # noqa: E402
from src.gpu import detect_device  # noqa: E402
from src.pipeline import _build_stage_backend  # noqa: E402
from src.render import _page_nominal_dpi, rebuild_pdf  # noqa: E402

BOOK = os.path.join(ROOT, "input",
                    "全国高等学校计算机教育研究会十四五规划教材  大数据与人工智能技术丛书  云计算导论  第3版 ---1.pdf")
Q95 = os.path.join(ROOT, "output", "verify_p35_q95.pdf")
MRC = os.path.join(ROOT, "output", "verify_p35_mrc.pdf")
CMP = os.path.join(ROOT, "output", "compare_mrc_p35.png")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, default=34, help="0-based page index")
    args = ap.parse_args()
    pno = args.page
    tag = f"p{pno + 1}"
    q95 = os.path.join(ROOT, "output", f"verify_{tag}_q95.pdf")
    mrc = os.path.join(ROOT, "output", f"verify_{tag}_mrc.pdf")
    cmp_png = os.path.join(ROOT, "output", f"compare_mrc_{tag}.png")

    doc = fitz.open(BOOK)
    page = doc[pno]
    nominal = _page_nominal_dpi(page)
    dpi = 300 if nominal <= 0 else int(min(max(nominal, 72.0), 450.0))
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                          colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(
        pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(f"page {pno + 1} native render: {img.shape[1]}x{img.shape[0]} @ {dpi}dpi")

    device = detect_device("auto")
    regs = available_backends()
    restore = _build_stage_backend(regs["restore"], "docres", ROOT,
                                   {"task": "appearance", "max_size": 1024},
                                   device, fp16=True)
    sr = _build_stage_backend(regs["sr"], "realesrgan-general", ROOT,
                              {"scale": 4, "tile": 384}, device, fp16=True)
    t0 = time.time()
    img = restore.process(img)
    print(f"docres {time.time() - t0:.1f}s")
    t0 = time.time()
    enh = sr.process(img)
    print(f"SR x4 {time.time() - t0:.1f}s -> {enh.shape[1]}x{enh.shape[0]}")
    info = classify_page(enh)
    print(f"classified: {info}")

    page_kw = dict(index=0, image=enh, width_pt=float(page.rect.width),
                   height_pt=float(page.rect.height))

    rebuild_pdf([dict(page_kw, ptype="color-image")], q95, jpeg_quality=95)
    rebuild_pdf([dict(page_kw, ptype=info["type"])], mrc, jpeg_quality=95)
    s1 = os.path.getsize(q95) / 1024 / 1024
    s2 = os.path.getsize(mrc) / 1024 / 1024
    print(f"q95 whole-page: {s1:.2f} MB | {info['type']}: {s2:.2f} MB "
          f"({s1 / max(s2, 0.01):.1f}x)")

    # ---- comparison strip: body-text strip AND figure strip ----
    pw, ph = float(page.rect.width), float(page.rect.height)
    clips = [
        fitz.Rect(pw * 0.10, ph * 0.47, pw * 0.50, ph * 0.47 + 150),  # body text
        fitz.Rect(pw * 0.10, ph * 0.60, pw * 0.50, ph * 0.60 + 150),  # figure area
    ]
    Z = 6
    W = 2400

    def crop(path, pno, clip):
        d = fitz.open(path)
        px = d[pno].get_pixmap(matrix=fitz.Matrix(Z, Z), colorspace=fitz.csRGB,
                               alpha=False, clip=clip)
        a = np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width, 3)
        d.close()
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)

    font = ImageFont.truetype(os.path.join(ROOT, "tools", "fonts", "simsun.ttc"), 30)

    def label(text, fill):
        im = Image.new("RGB", (W, 44), fill)
        ImageDraw.Draw(im).text((12, 4), text, font=font, fill=(20, 20, 20))
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    def fitw(im):
        s = W / im.shape[1]
        return cv2.resize(im, (W, max(1, int(im.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)

    parts = []
    for ci, clip in enumerate(clips):
        parts.append(label(f"区域{ci + 1}  原版", (255, 228, 225)))
        parts.append(fitw(crop(BOOK, pno, clip)))
        parts.append(label(f"区域{ci + 1}  整页 q95 JPEG（{s1:.1f}MB/页）", (255, 240, 200)))
        parts.append(fitw(crop(q95, 0, clip)))
        parts.append(label(f"区域{ci + 1}  新分类={info['type']}（{s2:.1f}MB/页）", (220, 240, 255)))
        parts.append(fitw(crop(mrc, 0, clip)))
    strip = np.vstack(parts)
    cv2.imwrite(cmp_png, strip)
    print(f"-> {cmp_png}")


if __name__ == "__main__":
    main()
