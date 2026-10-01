"""Showcase: zoom ladder, source vs old output vs new output.

Every panel is a real render at the stated zoom with a crop window chosen so
panels come out PANEL px wide; nothing is downscaled afterwards, so each row
is a true 1x / 3x / 6x view of the page.
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

PAGE_OLD = 44
PAGE_NEW = 2

PANEL = 500                       # px width of every panel
PANEL_H = 170                     # px height of every panel
ROWS = [(1.0, "1x"), (2.0, "2x"), (3.0, "3x"), (6.0, "6x")]
X0, Y0 = 252.0, 602.0             # top-left of the text column, page pt


def shot(doc, idx, zoom):
    r = fitz.Rect(X0, Y0, X0 + PANEL / zoom, Y0 + PANEL_H / zoom)
    pix = doc[idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=r,
                              colorspace=fitz.csRGB, alpha=False)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def page_bytes(doc, idx):
    return sum(len(doc.xref_stream_raw(im[0])) for im in doc[idx].get_images(full=True)) / 1024 / 1024


def img_res(doc, idx):
    ws = [im[2] for im in doc[idx].get_images(full=True)]
    return max(ws) if ws else 0


def main():
    d_src, d_old, d_new = fitz.open(SRC), fitz.open(OLD), fitz.open(NEW)
    cols = [
        ("原始扫描件   {:.2f} MB/页  {} px".format(page_bytes(d_src, PAGE_OLD),
                                                   img_res(d_src, PAGE_OLD)),
         d_src, PAGE_OLD, (107, 114, 128)),
        ("旧版成品 整页 JPEG  {:.2f} MB/页  {} px".format(page_bytes(d_old, PAGE_OLD),
                                                          img_res(d_old, PAGE_OLD)),
         d_old, PAGE_OLD, (180, 83, 9)),
        ("新版成品 MRC 分层  {:.2f} MB/页  {} px".format(page_bytes(d_new, PAGE_NEW),
                                                         img_res(d_new, PAGE_NEW)),
         d_new, PAGE_NEW, (4, 120, 87)),
    ]

    # pre-render every row
    rows = []
    for zoom, lab in ROWS:
        row = [shot(doc, idx, zoom) for _, doc, idx, _ in cols]
        rows.append((lab, row))

    PAD, GAP, BAR, TITLE, GUT = 24, 16, 40, 84, 62
    cw = PANEL
    gx = PAD + GUT
    col_x = [gx + i * (cw + GAP) for i in range(len(cols))]
    canvas_w = col_x[-1] + cw + PAD
    total_rows_h = sum(BAR + r[1][0].shape[0] + 20 for r in rows)
    canvas_h = PAD + TITLE + total_rows_h + PAD

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    f_t = ImageFont.truetype(FONT, 27)
    f_h = ImageFont.truetype(FONT, 17)
    f_l = ImageFont.truetype(FONT, 23)
    f_r = ImageFont.truetype(FONT, 16)

    dr.text((PAD, PAD - 6), "《云计算导论》第 45 页   增强效果对比",
            font=f_t, fill=(17, 24, 39))
    dr.text((PAD, PAD + 34), "每一格都是按标注倍率真实渲染的画面，未做任何缩放后处理",
            font=f_r, fill=(107, 114, 128))

    y = PAD + TITLE
    for lab, row in rows:
        for i, arr in enumerate(row):
            x = col_x[i]
            dr.rectangle([x, y, x + cw, y + BAR - 12], fill=cols[i][3])
            dr.text((x + 11, y + 1), cols[i][0], font=f_h, fill=(255, 255, 255))
            canvas.paste(Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)), (x, y + BAR))
            dr.rectangle([x - 1, y + BAR - 1, x + cw, y + BAR + arr.shape[0]],
                         outline=(185, 185, 190), width=1)
        h = row[0].shape[0]
        dr.text((PAD + 8, y + BAR + h // 2 - 14), lab, font=f_l, fill=(17, 24, 39))
        dr.line([PAD + GUT - 12, y + BAR, PAD + GUT - 12, y + BAR + h],
                fill=(209, 213, 219), width=1)
        y += BAR + h + 20

    out = os.path.join(ROOT, "output", "diag_graytext", "showcase_ladder.png")
    canvas.save(out)
    print("saved", out, canvas.size)
    for d in (d_src, d_old, d_new):
        d.close()


if __name__ == "__main__":
    sys.exit(main())
