"""Are strokes actually missing in the MRC output?

Renders the same region from source / old / new at one zoom, binarises each,
and reports ink coverage + a red/blue diff map so missing strokes are visible.
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
ZOOM = 8.0
X0, Y0, W_PT, H_PT = 252.0, 602.0, 90.0, 26.0


def shot(doc, idx):
    r = fitz.Rect(X0, Y0, X0 + W_PT, Y0 + H_PT)
    pix = doc[idx].get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), clip=r,
                              colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)


def ink(gray):
    """Otsu binarisation -> True where there is ink."""
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b > 0


def main():
    d_src, d_old, d_new = fitz.open(SRC), fitz.open(OLD), fitz.open(NEW)
    g_src, g_old, g_new = shot(d_src, PAGE_OLD), shot(d_old, PAGE_OLD), shot(d_new, PAGE_NEW)
    i_src, i_old, i_new = ink(g_src), ink(g_old), ink(g_new)

    print("ink coverage (% of pixels):")
    for nm, i in (("source", i_src), ("old-JPEG", i_old), ("new-MRC", i_new)):
        print(f"   {nm:<10} {i.mean() * 100:6.2f}%")

    def diff(a, b):
        """a = reference (source), b = candidate."""
        missing = a & ~b          # in source but gone in candidate
        added = b & ~a
        return missing.mean() * 100, added.mean() * 100

    for nm, i in (("old-JPEG", i_old), ("new-MRC", i_new)):
        m, a = diff(i_src, i)
        print(f"vs source  {nm:<10} missing={m:5.2f}%  added={a:5.2f}%")

    # ---- visual: grayscale crops + red/blue diff (source ink missing => red) ----
    panel_zoom = 2.6
    gs = [cv2.cvtColor(cv2.resize(g, None, fx=panel_zoom, fy=panel_zoom,
                                  interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
          for g in (g_src, g_old, g_new)]
    H, W = gs[0].shape[:2]

    def diffmap(ref, cand):
        vis = np.full((ref.shape[0], ref.shape[1], 3), 255, np.uint8)
        vis[ref & cand] = (60, 60, 60)          # both -> dark grey
        vis[ref & ~cand] = (40, 40, 230)        # source only -> RED (missing!)
        vis[cand & ~ref] = (230, 120, 40)       # candidate only -> BLUE (extra)
        return cv2.resize(vis, None, fx=panel_zoom, fy=panel_zoom,
                          interpolation=cv2.INTER_NEAREST)

    dm_old = diffmap(i_src, i_old)
    dm_new = diffmap(i_src, i_new)

    labels = ["original scan", "old: whole-page JPEG", "new: MRC",
              "diff vs original (red = lost, blue = added)  OLD",
              "diff vs original (red = lost, blue = added)  NEW"]
    imgs = [gs[0], gs[1], gs[2], dm_old, dm_new]

    PAD, GAP, BAR = 20, 14, 34
    cols = [imgs[0], imgs[1], imgs[2], dm_old, dm_new]
    canvas_w = PAD * 2 + W * 5 + GAP * 4
    canvas_h = PAD + 52 + BAR + H + PAD
    cv = Image.new("RGB", (canvas_w, canvas_h), "white")
    dr = ImageDraw.Draw(cv)
    dr.text((PAD, PAD - 4), f"stroke audit  {ZOOM:.0f}x  stroke width check",
            font=ImageFont.truetype(FONT, 24), fill=(17, 24, 39))
    y = PAD + 52
    for i, im in enumerate(cols):
        x = PAD + i * (W + GAP)
        dr.rectangle([x, y, x + W, y + BAR - 8], fill=(90, 90, 95))
        dr.text((x + 8, y + 1), f"{i + 1}", font=ImageFont.truetype(FONT, 16), fill="white")
        cv.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (x, y + BAR))
    out = os.path.join(ROOT, "output", "diag_graytext", "stroke_audit.png")
    cv.save(out)
    print("saved", out, cv.size)
    for i, l in enumerate(labels):
        print(f"  [{i + 1}] {l}")


if __name__ == "__main__":
    main()
