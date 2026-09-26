# -*- coding: utf-8 -*-
"""Diagnose perceived blur: locate the '天文计算' page in the textbook,
inspect the embedded scan's native resolution / JPEG chroma subsampling,
and export a native-pixel crop around the region for later comparison."""
import os
import sys

import pymupdf

BOOK = (r"E:\04_Projects\PDF_AI_Enhancer\input"
        r"\全国高等学校计算机教育研究会十四五规划教材  大数据与人工智能技术丛书  云计算导论  第3版 ---1.pdf")
OUT = r"E:\04_Projects\PDF_AI_Enhancer\temp\diag"
os.makedirs(OUT, exist_ok=True)

NEEDLES = ["天文计算", "望远镜阵列"]


def jpeg_sof_info(data: bytes):
    """Parse SOF marker: returns (width, height, ncomp, sampling_factors)."""
    if not data.startswith(b"\xff\xd8"):
        return None
    i, n = 2, len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            nc = data[i + 9]
            sampling = []
            for c in range(nc):
                samp = data[i + 11 + 3 * c]
                sampling.append((samp >> 4, samp & 0xF))
            return w, h, nc, sampling
        if m == 0xD8 or m == 0x01 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m in (0xD9,):  # EOI
            break
        seglen = (data[i + 2] << 8) | data[i + 3]
        i += 2 + seglen
    return None


def main():
    doc = pymupdf.open(BOOK)
    print("pages:", doc.page_count)

    hits = []
    pages_with_text = 0
    for i in range(doc.page_count):
        t = doc[i].get_text()
        if t.strip():
            pages_with_text += 1
        for needle in NEEDLES:
            if needle in t:
                rects = doc[i].search_for(needle)
                hits.append((i, needle, rects))
                print(f"hit: page {i + 1} needle={needle!r} "
                      f"rects={[tuple(round(v, 1) for v in r) for r in rects[:3]]}")
                break
    print(f"pages with text layer: {pages_with_text}/{doc.page_count}")

    if not hits:
        toc = doc.get_toc()
        print("no text-layer hits; toc entries:", len(toc))
        for lvl, title, pg in toc[:50]:
            print(" ", lvl, title, pg)
        sys.exit(0)

    pno, needle, rects = hits[0]
    page = doc[pno]
    print(f"\n=== page {pno + 1} embedded images ===")
    bbox = rects[0] if rects else None

    for img in page.get_images(full=True):
        xref, _smask, w, h, bpc, cs, _acs, name, filt = img[:9]
        rects_img = page.get_image_rects(xref)
        raw = doc.xref_stream_raw(xref)
        print(f"xref={xref} {w}x{h}px bpc={bpc} cs={cs} filter={filt} "
              f"bytes={len(raw)} ({len(raw) / 1024:.0f} KB)")
        sof = jpeg_sof_info(raw)
        if sof:
            print(f"  SOF: {sof[0]}x{sof[1]} comps={sof[2]} "
                  f"subsampling={sof[3]}")
        for r in rects_img:
            dpi_x = w / r.width * 72 if r.width else 0
            dpi_y = h / r.height * 72 if r.height else 0
            print(f"  placed at {r} -> effective DPI {dpi_x:.0f}x{dpi_y:.0f}")
            if bbox is not None and bbox.intersects(r):
                sx = w / r.width
                sy = h / r.height
                px0 = max(0, int((bbox.x0 - r.x0) * sx) - 60)
                py0 = max(0, int((bbox.y0 - r.y0) * sy) - 260)
                px1 = min(w, int((bbox.x1 - r.x0) * sx) + 900)
                py1 = min(h, int((bbox.y1 - r.y0) * sy) + 900)
                print(f"  crop px box: ({px0},{py0})-({px1},{py1})")
                with open(os.path.join(OUT, "orig.jpeg"), "wb") as f:
                    f.write(raw)
                from PIL import Image
                import io
                im = Image.open(io.BytesIO(raw))
                crop = im.crop((px0, py0, px1, py1))
                crop.save(os.path.join(OUT, "orig_native_crop.png"))
                print(f"  saved orig_native_crop.png {crop.size}")
    print("\nlast modified:", doc.metadata.get("producer"), "|", doc.metadata.get("creator"))


if __name__ == "__main__":
    main()
