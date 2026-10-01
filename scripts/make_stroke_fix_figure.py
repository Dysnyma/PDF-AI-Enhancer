"""Final before/after figure for the missing-stroke fix.

Everything is rendered on the shared 1439 px grid (source native) so the
comparison is on equal terms; panels are magnified with nearest-neighbour
only, never downscaled.
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
BEFORE = os.path.join(ROOT, "output", "diag_graytext", "mrc_test_scale2.pdf")
AFTER = os.path.join(ROOT, "output", "diag_graytext", "mrc_test_fix3.pdf")


def shot(doc, idx):
    pix = doc[idx].get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)


def ink(g):
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b > 0


def main():
    d_src, d_old, d_bef, d_aft = (fitz.open(SRC), fitz.open(OLD),
                                  fitz.open(BEFORE), fitz.open(AFTER))
    g_src, g_old = shot(d_src, 44), shot(d_old, 44)
    g_bef, g_aft = shot(d_bef, 2), shot(d_aft, 2)
    i_src = ink(g_src)

    stats = []
    for nm, g in (("old JPEG", g_old), ("before fix", g_bef), ("after fix", g_aft)):
        i = ink(g)
        stats.append((nm, i.mean() * 100, (i_src & ~i).mean() * 100, (i & ~i_src).mean() * 100))
    print("source ink %.2f%%" % (i_src.mean() * 100))
    for nm, cov, mis, add in stats:
        print(f"  {nm:<12} ink={cov:5.2f}%  missing={mis:5.2f}%  added={add:5.2f}%")

    # ---- row A: glyph, 7x ----
    boxA = (slice(590, 650), slice(250, 350))
    rowA = [("original scan", g_src, (107, 114, 128)),
            ("old: whole-page JPEG", g_old, (180, 83, 9)),
            ("before fix  (open 3x3 + cap 130)", g_bef, (185, 28, 28)),
            ("after fix  (no open, cap 140)", g_aft, (4, 120, 87))]
    # ---- row B: paragraph, 3x ----
    boxB = (slice(596, 676), slice(252, 552))
    rowB = [("original scan", g_src, (107, 114, 128)),
            ("after fix", g_aft, (4, 120, 87))]

    def make(items, box, zoom, width_cm):
        out = []
        for nm, g, col in items:
            c = g[box]
            im = cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
            out.append((nm, cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), col))
        return out

    A = make(rowA, boxA, 7, None)
    B = make(rowB, boxB, 3, None)

    PAD, GAP, BAR0, TITLE, BAR1 = 22, 12, 30, 78, 40
    wa = A[0][1].shape[1]
    ha = A[0][1].shape[0]
    wb = B[0][1].shape[1]
    hb = B[0][1].shape[0]
    wmax = max(wa * 4 + GAP * 3, wb * 2 + GAP)
    canvas = Image.new("RGB", (wmax + PAD * 2, PAD + TITLE + BAR1 + ha + 26 + BAR1 + hb + PAD),
                       (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    f_t = ImageFont.truetype(FONT, 27)
    f_h = ImageFont.truetype(FONT, 16)
    f_s = ImageFont.truetype(FONT, 15)
    f_l = ImageFont.truetype(FONT, 19)

    dr.text((PAD, PAD - 6), "笔画丢失修复  —  《云计算导论》第 45 页", font=f_t, fill=(17, 24, 39))
    dr.text((PAD, PAD + 34),
            "源墨迹 6.39%  ·  旧版 JPEG 丢失 0.04%  ·  修复前丢失 1.02%  ·  修复后丢失 0.09%",
            font=f_s, fill=(90, 96, 105))

    y = PAD + TITLE
    dr.text((PAD, y + 8), "单字 7×", font=f_l, fill=(17, 24, 39))
    y += BAR1
    for i, (nm, im, col) in enumerate(A):
        x = PAD + i * (im.shape[1] + GAP)
        dr.rectangle([x, y, x + im.shape[1], y + BAR0 - 8], fill=col)
        dr.text((x + 8, y + 1), nm, font=f_h, fill=(255, 255, 255))
        canvas.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (x, y + BAR0))
        dr.rectangle([x - 1, y + BAR0 - 1, x + im.shape[1], y + BAR0 + im.shape[0]],
                     outline=(190, 190, 195), width=1)
    y += BAR0 + ha + 26
    dr.text((PAD, y + 8), "正文段落 3×", font=f_l, fill=(17, 24, 39))
    y += BAR1
    for i, (nm, im, col) in enumerate(B):
        x = PAD + i * (im.shape[1] + GAP)
        dr.rectangle([x, y, x + im.shape[1], y + BAR0 - 8], fill=col)
        dr.text((x + 8, y + 1), nm + "   (原始扫描 / 修复后成品)", font=f_h, fill=(255, 255, 255))
        canvas.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (x, y + BAR0))
        dr.rectangle([x - 1, y + BAR0 - 1, x + im.shape[1], y + BAR0 + im.shape[0]],
                     outline=(190, 190, 195), width=1)

    out = os.path.join(ROOT, "output", "diag_graytext", "stroke_fix_final.png")
    canvas.save(out)
    print("saved", out, canvas.size)


if __name__ == "__main__":
    main()
