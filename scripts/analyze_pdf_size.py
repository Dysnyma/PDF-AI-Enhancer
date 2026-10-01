"""Anatomy of an enhanced PDF: where do the bytes actually go?

For each page, list every image XObject with its compressed stream size and
its encoding (bits-per-component, colorspace, filter), then aggregate:
  * total bytes by encoding category
  * the heaviest pages
  * pages whose type guess (from encoding) is gray/text/mrc

Usage:
    env/Scripts/python.exe scripts/analyze_pdf_size.py output/x_enhanced.pdf
"""
import argparse
import collections
import os
import sys

import pymupdf as fitz

CS = {1: "Gray", 3: "RGB", 4: "CMYK"}
FILT = {"/DCTDecode": "JPEG", "/FlateDecode": "Flate", "/CCITTFaxDecode": "CCITT",
        "/JBIG2Decode": "JBIG2", "/JPXDecode": "JP2"}


def classify_encoding(infos):
    """Guess the page's encoding path from its image objects."""
    if not infos:
        return "vector/none"
    kinds = set()
    for im in infos:
        bpc, ncs, filt = im["bpc"], im["ncs"], im["filter"]
        if filt == "/DCTDecode":
            # any 8-bit JPEG plane (RGB or ICCBased gray) is an image layer
            kinds.add("color-jpeg" if ncs == 3 else "gray-jpeg")
        elif bpc == 1:
            kinds.add("1bit")
        else:
            kinds.add("other")
    if "1bit" in kinds and ("color-jpeg" in kinds or "gray-jpeg" in kinds):
        return "MRC(1bit+jpeg)"
    if "1bit" in kinds:
        return "bw-text(1bit)"
    if "color-jpeg" in kinds:
        return "color/mixed(jpeg)"
    if "gray-jpeg" in kinds:
        return "gray(jpeg)"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    cat_bytes = collections.Counter()
    cat_pages = collections.Counter()
    page_rows = []
    total = 0

    for pno in range(doc.page_count):
        page = doc[pno]
        infos = []
        for im in page.get_images(full=True):
            xref = im[0]
            try:
                raw = doc.xref_stream_raw(xref)
                size = len(raw)
            except Exception:
                size = 0
            obj = doc.xref_object(xref)
            filt = next((f for f in FILT if f in obj), "?")
            infos.append({"xref": xref, "w": im[2], "h": im[3], "bpc": im[4],
                          "ncs": im[5], "filter": filt, "size": size})
        psize = sum(i["size"] for i in infos)
        total += psize
        kind = classify_encoding(infos)
        cat_bytes[kind] += psize
        cat_pages[kind] += 1
        page_rows.append((psize, pno + 1, kind, infos))

    print(f"file: {args.pdf}")
    print(f"pages: {doc.page_count}   total image bytes: {total/1024**2:.1f} MB   "
          f"file size: {os.path.getsize(args.pdf)/1024**2:.1f} MB")
    print("\n=== bytes by encoding path ===")
    for k, v in cat_bytes.most_common():
        print(f"  {k:<20} {v/1024**2:8.1f} MB  ({v/total*100:5.1f}%)  on {cat_pages[k]} pages")

    print(f"\n=== {args.top} heaviest pages ===")
    for psize, pno, kind, infos in sorted(page_rows, reverse=True)[:args.top]:
        detail = "  ".join(f"{i['w']}x{i['h']}bpc{i['bpc']}cs{i['ncs']}"
                           f"{FILT.get(i['filter'], i['filter'])[:4]}:{i['size']/1024:.0f}KB"
                           for i in infos)
        print(f"  p{pno:<4} {psize/1024**2:6.2f} MB  {kind:<16} {detail}")

    avg = total / max(1, doc.page_count) / 1024**2
    print(f"\nmean {avg:.2f} MB/page")
    doc.close()


if __name__ == "__main__":
    main()
