"""Locate exactly which strokes the MRC output loses.

On the shared 1439px grid: missing = source_ink & ~new_ink. Find its blobs,
rank by size, and dump zoomed crops of the worst spots so the loss is visible.
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


def page_gray(doc, idx, zoom=1.0):
    pix = doc[idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                              colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)


def ink(g):
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b > 0


def main():
    d_src, d_old, d_new = fitz.open(SRC), fitz.open(OLD), fitz.open(NEW)
    g = {k: page_gray(d, i) for k, d, i in (("src", d_src, PAGE_OLD),
                                            ("old", d_old, PAGE_OLD),
                                            ("new", d_new, PAGE_NEW))}
    i = {k: ink(v) for k, v in g.items()}

    for name in ("old", "new"):
        miss = i["src"] & ~i[name]
        core = cv2.erode(miss.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        n, lab, st, cen = cv2.connectedComponentsWithStats(miss.astype(np.uint8), 8)
        areas = st[1:, cv2.CC_STAT_AREA]
        order = np.argsort(-areas)[:6]
        print(f"--- {name}: missing ink {miss.mean() * 100:.2f}% of page, "
              f"{n - 1} blobs; core(missing after 3x3 erode) {core.mean() * 100:.2f}%")
        for k in order:
            j = k + 1
            x, y, w, h, a = st[j]
            print(f"     blob area={a:5d}  bbox=({x},{y},{w}x{h})")
        # how many source components are largely lost?
        ns, ls, ss, _ = cv2.connectedComponentsWithStats(i["src"].astype(np.uint8), 8)
        big = [j for j in range(1, ns) if ss[j, cv2.CC_STAT_AREA] >= 30]
        lost = 0
        for j in big:
            m = (ls == j)
            keep = (m & i[name]).sum() / m.sum()
            if keep < 0.5:
                lost += 1
        print(f"     source components >=30px: {len(big)}, "
              f"of which <50% kept in {name}: {lost} ({lost / max(len(big), 1) * 100:.0f}%)")

    # visual: pick the worst zone in 'new' and show src / new / diff
    miss = i["src"] & ~i["new"]
    n, lab, st, _ = cv2.connectedComponentsWithStats(miss.astype(np.uint8), 8)
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = st[j]
    cx, cy = x + w // 2, y + h // 2
    R = 110
    x0, y0 = max(cx - R, 0), max(cy - R, 0)
    x1, y1 = x0 + 2 * R, y0 + 2 * R
    Z = 4

    def pan(img, rect, zoom=Z, color=True):
        a, b, c, d = rect
        cr = img[b:d, a:c]
        cr = cv2.resize(cr, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(cr, cv2.COLOR_GRAY2BGR) if color else cr

    def dpan(rect):
        a, b, c, d = rect
        A, B = i["src"][b:d, a:c], i["new"][b:d, a:c]
        vis = np.full((A.shape[0], A.shape[1], 3), 255, np.uint8)
        vis[A & B] = (70, 70, 70)
        vis[A & ~B] = (40, 40, 235)
        vis[B & ~A] = (235, 130, 30)
        return cv2.resize(vis, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)

    rect = (x0, y0, x1, y1)
    panels = [("source", pan(g["src"], rect)), ("new MRC", pan(g["new"], rect)),
              ("diff: red=lost", dpan(rect)),
              ("source 8x", pan(g["src"], rect, 8)), ("new 8x", pan(g["new"], rect, 8))]

    PAD, GAP, BAR = 18, 12, 30
    W, H = panels[0][1].shape[1], panels[0][1].shape[0]
    Hmax = max(p[1].shape[0] for p in panels)
    Wmax = 700
    tot = PAD * 2 + sum(min(p[1].shape[1], Wmax) for p in panels) + GAP * (len(panels) - 1)
    cv = Image.new("RGB", (tot, PAD + 46 + BAR + Hmax + PAD), "white")
    dr = ImageDraw.Draw(cv)
    dr.text((PAD, PAD - 4), f"worst missing blob  bbox=({x},{y},{w}x{h})",
            font=ImageFont.truetype(FONT, 22), fill=(17, 24, 39))
    xx = PAD
    yy = PAD + 46
    for nm, im in panels:
        if im.shape[1] > Wmax:
            im = im[:, :Wmax]
        dr.rectangle([xx, yy, xx + im.shape[1], yy + BAR - 8], fill=(90, 90, 95))
        dr.text((xx + 8, yy + 1), nm, font=ImageFont.truetype(FONT, 15), fill="white")
        cv.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (xx, yy + BAR))
        xx += im.shape[1] + GAP
    out = os.path.join(ROOT, "output", "diag_graytext", "stroke_loss.png")
    cv.save(out)
    print("saved", out, cv.size)


if __name__ == "__main__":
    main()
