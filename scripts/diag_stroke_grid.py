"""Fair stroke comparison: both versions on the SAME 1439px grid.

Source page renders 1:1 natively at zoom 1; the new PDF is rendered at the
same size so MuPDF itself does the downsampling. Then both are binarised and
diffed, so "missing strokes" is measured on equal terms.
"""
import os
import sys

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(ROOT, "tools", "fonts", "simsun.ttc")

SRC = os.path.join(ROOT, "input",
                   "全国高等学校计算机教育研究会十四五规划教材  大数据与人工智能技术丛书  云计算导论  第3版 ---1.pdf")
OLD = os.path.join(ROOT, "output",
                   "全国高等学校计算机教育研究会十四五规划教材  大数据与人工智能技术丛书  云计算导论  第3版 ---1_enhanced.pdf")
NEW = os.path.join(ROOT, "output", "diag_graytext", "mrc_test_scale2.pdf")
PAGE_OLD, PAGE_NEW = 44, 2


def page_gray(doc, idx, zoom):
    pix = doc[idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                              colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)


def ink(g):
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b > 0


def main():
    d_src, d_old, d_new = fitz.open(SRC), fitz.open(OLD), fitz.open(NEW)
    g_src = page_gray(d_src, PAGE_OLD, 1.0)
    g_old = page_gray(d_old, PAGE_OLD, 1.0)
    g_new = page_gray(d_new, PAGE_NEW, 1.0)
    print("grid", g_src.shape, g_old.shape, g_new.shape)

    i_src, i_old, i_new = ink(g_src), ink(g_old), ink(g_new)
    print("ink coverage on the 1439px grid:")
    for nm, i in (("source", i_src), ("old-JPEG", i_old), ("new-MRC", i_new)):
        print(f"   {nm:<10} {i.mean() * 100:6.2f}%")
    for nm, i in (("old-JPEG", i_old), ("new-MRC", i_new)):
        print(f"vs source {nm:<10} missing={(i_src & ~i).mean() * 100:5.2f}%  "
              f"added={(i & ~i_src).mean() * 100:5.2f}%")

    def panel(g, rect, zoom=3):
        x0, y0, x1, y1 = rect
        c = g[y0:y1, x0:x1]
        return cv2.cvtColor(cv2.resize(c, None, fx=zoom, fy=zoom,
                                       interpolation=cv2.INTER_NEAREST),
                            cv2.COLOR_GRAY2BGR)

    def dpanel(a, b, rect, zoom=3):
        x0, y0, x1, y1 = rect
        A, B = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
        vis = np.full((A.shape[0], A.shape[1], 3), 255, np.uint8)
        vis[A & B] = (70, 70, 70)
        vis[A & ~B] = (40, 40, 235)
        vis[B & ~A] = (235, 130, 30)
        return cv2.resize(vis, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)

    # text body region on the 1439 grid
    RECT = (252, 600, 700, 700)
    panels = [
        ("1 source (native 1439)", panel(g_src, RECT)),
        ("2 old: whole-page JPEG", panel(g_old, RECT)),
        ("3 new: MRC", panel(g_new, RECT)),
        ("4 diff OLD  red=lost", dpanel(i_src, i_old, RECT)),
        ("5 diff NEW  red=lost", dpanel(i_src, i_new, RECT)),
    ]

    PAD, GAP, BAR = 20, 12, 32
    W, H = panels[0][1].shape[1], panels[0][1].shape[0]
    canvas_w = PAD * 2 + W * len(panels) + GAP * (len(panels) - 1)
    canvas_h = PAD + 50 + BAR + H + PAD
    cv = Image.new("RGB", (canvas_w, canvas_h), "white")
    dr = ImageDraw.Draw(cv)
    dr.text((PAD, PAD - 4), "stroke audit on the SAME 1439px grid  (rendered 1:1)",
            font=ImageFont.truetype(FONT, 24), fill=(17, 24, 39))
    y = PAD + 50
    for i, (nm, im) in enumerate(panels):
        x = PAD + i * (W + GAP)
        dr.rectangle([x, y, x + W, y + BAR - 8], fill=(90, 90, 95))
        dr.text((x + 8, y + 1), nm, font=ImageFont.truetype(FONT, 15), fill="white")
        cv.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (x, y + BAR))
    out = os.path.join(ROOT, "output", "diag_graytext", "stroke_audit_grid.png")
    cv.save(out)
    print("saved", out, cv.size)


if __name__ == "__main__":
    main()
