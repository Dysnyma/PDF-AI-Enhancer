"""Does /Interpolate true on the 1-bit stencil smooth the jaggies?"""
import os
import sys
import tempfile

import cv2
import numpy as np
import pymupdf as fitz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml  # noqa: E402

from src import mrc  # noqa: E402
from src.jbig2 import _paint_jbig2  # noqa: E402
from src.pipeline import EnhancePipeline  # noqa: E402
from src.render import render_pdf_pages  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "input",
                   "全国高等学校计算机教育研究会十四五规划教材  大数据与人工智能技术丛书  云计算导论  第3版 ---1.pdf")
PNO = 44


def build(img, bg_jpg, layers, w, h, interpolate):
    doc = fitz.open()
    page = doc.new_page(width=1439, height=2005)
    page.insert_image(page.rect, stream=bg_jpg)
    for packed, (b, g, r) in layers:
        xref = doc.get_new_xref()
        extra = " /Interpolate true" if interpolate else ""
        doc.update_object(
            xref,
            f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
            f"/ImageMask true /BitsPerComponent 1 /Decode [0 1]{extra} >>")
        doc.update_stream(xref, packed, compress=1)
        _paint_jbig2(page, xref, page.rect,
                     fill=f"{r / 255:.4f} {g / 255:.4f} {b / 255:.4f} rg ")
    return doc


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml"), encoding="utf-8"))
    pipe = EnhancePipeline(cfg, ROOT)
    wd = tempfile.mkdtemp()
    jb = os.path.join(ROOT, "tools", "jbig2", "jbig2.exe")

    p = render_pdf_pages(PDF, cfg["render"]["dpi"], cfg["render"]["max_dpi"],
                         page_range=(PNO, PNO))[0]
    img = pipe.sr.process(pipe.restore.process(p["image"]))
    bg, layers = mrc.split_mrc(img, jb, wd)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(bg, 5)
    h0, w0 = bg.shape[:2]
    bg = cv2.resize(bg, (w0 // 2, h0 // 2), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", bg, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    packed_layers = [(np.packbits((~s).astype(np.uint8), axis=1).tobytes(), f)
                     for s, f in layers]
    mh, mw = layers[0][0].shape

    panels = []
    docs = {}
    for name, interp in (("stencil: no /Interpolate", False),
                         ("stencil: /Interpolate true", True)):
        d = build(img, jpg.tobytes(), packed_layers, mw, mh, interp)
        docs[name] = d
        pg = d[0]
        z = 6.0
        pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csRGB, alpha=False)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        H, W = a.shape[:2]
        crop = a[int(0.305 * H):int(0.345 * H), int(0.185 * W):int(0.275 * W)]
        panels.append((name, crop))

    # also the source page for reference
    ds = fitz.open(PDF)
    pg = ds[PNO]
    pix = pg.get_pixmap(matrix=fitz.Matrix(6.0, 6.0), colorspace=fitz.csRGB, alpha=False)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    H, W = a.shape[:2]
    panels.insert(0, ("原始扫描件", a[int(0.305 * H):int(0.345 * H),
                                     int(0.185 * W):int(0.275 * W)]))

    from PIL import Image, ImageDraw, ImageFont
    F = os.path.join(ROOT, "tools", "fonts", "simsun.ttc")
    cw = panels[0][1].shape[1]
    PAD, GAP, BAR = 20, 14, 36
    W = PAD * 2 + cw * len(panels) + GAP * (len(panels) - 1)
    Hc = PAD + 58 + BAR + panels[0][1].shape[0] + PAD
    cv = Image.new("RGB", (W, Hc), "white")
    dr = ImageDraw.Draw(cv)
    dr.text((PAD, PAD - 6), "6x zoom  /Interpolate test", font=ImageFont.truetype(F, 24),
            fill=(17, 24, 39))
    y = PAD + 58
    for i, (nm, c) in enumerate(panels):
        x = PAD + i * (cw + GAP)
        dr.rectangle([x, y, x + cw, y + BAR - 8], fill=(90, 90, 95))
        dr.text((x + 10, y + 1), nm, font=ImageFont.truetype(F, 17), fill="white")
        cv.paste(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)), (x, y + BAR))
    out = os.path.join(ROOT, "output", "diag_graytext", "interp_test.png")
    cv.save(out)
    print("saved", out, cv.size)


if __name__ == "__main__":
    main()
