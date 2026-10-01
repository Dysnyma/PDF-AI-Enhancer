"""Monochrome encoding for black-and-white text pages.

Two strategies, selectable via config `compression.monochrome`:

  - "1bit"  (default): adaptive binarization -> 1-bit image, embedded by
    PyMuPDF as FlateDecode/1bpc. Lossless w.r.t. the binarized result, fully
    compatible, and renderable by PyMuPDF itself (so it can be verified and
    keeps the hidden OCR text layer working end-to-end).

  - "jbig2": adaptive binarization -> agl/jbig2enc (tools/jbig2/jbig2.exe)
    symbol mode. Better compression but (a) the symbol merger is lossy and can
    substitute similar glyphs (the "hallucinated character" risk the project
    explicitly wants to avoid), and (b) MuPDF is compiled without JBIG2Decode
    so PyMuPDF cannot render/verify the result. It remains available for
    users who prioritise size over glyph fidelity.

NOTE: binarization itself is lossy (gray -> black/white). It is only ever
applied to pages classified as pure BW text; color/gray/mixed pages never go
through this path (their color is preserved).
"""
import logging
import os
import re
import struct
import subprocess

import cv2
import numpy as np
import pymupdf as fitz

from src.imgio import imwrite_unicode

log = logging.getLogger("pdfenhance")

JBIG2_HEADER_LEN = 27


def binarize(img_gray: np.ndarray) -> np.ndarray:
    """Adaptive-threshold binarization robust to uneven scan illumination."""
    return cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 15)


def encode_1bit(img_gray: np.ndarray) -> bytes:
    """Return a 1-bit PNG stream of the binarized page (PyMuPDF embeds it)."""
    bw = binarize(img_gray)
    ok, png = cv2.imencode(".png", bw, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    if not ok:
        raise RuntimeError("1-bit PNG encode failed")
    return png.tobytes()


def encode_jbig2(img_gray: np.ndarray, jbig2_bin: str,
                 workdir: str, threshold: float = 0.92):
    """Encode one grayscale page to JBIG2 via agl/jbig2enc.

    Returns (sym_bytes, jb2_bytes, width, height). jb2_bytes is the raw JBIG2
    page stream with the jbig2enc custom header stripped.
    """
    bw = binarize(img_gray)
    png_path = os.path.abspath(os.path.join(workdir, "page_bw.png"))
    base = os.path.abspath(os.path.join(workdir, "page"))
    # Unicode-safe write: the workdir derives from the book name, and
    # cv2.imwrite cannot open non-ASCII paths on Windows.
    imwrite_unicode(png_path, bw)

    # -s symbol mode (text); no -r refinement (crashes Acrobat).
    cmd = [os.path.abspath(jbig2_bin), "-s", "-p", "-t", str(threshold),
           "-b", base, png_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, cwd=workdir,
                       timeout=120)
    except subprocess.CalledProcessError as e:
        log.error("jbig2 failed: %s", e.stderr.decode(errors="replace")[-500:])
        raise

    sym_path, page_path = base + ".sym", base + ".0000"
    if not (os.path.isfile(sym_path) and os.path.isfile(page_path)):
        raise RuntimeError("jbig2 produced no output")

    sym_bytes = open(sym_path, "rb").read()
    page_data = open(page_path, "rb").read()
    jb2_bytes = page_data[JBIG2_HEADER_LEN:]
    width, height, _, _ = struct.unpack(
        ">IIII", page_data[11:JBIG2_HEADER_LEN])
    return sym_bytes, jb2_bytes, width, height


def _paint_jbig2(page, img_xref: int, rect, fill: str = ""):
    """Append a content operator that paints an image XObject into `rect`.

    fill: optional PDF paint operator prefix (e.g. "0 0 0 rg " for stencil
    masks, which are painted with the current fill color).
    """
    x0, y0, x1, y1 = rect
    w_pt, h_pt = x1 - x0, y1 - y0
    op = (f"q {fill}{w_pt:.4f} 0 0 {h_pt:.4f} {x0:.4f} {y0:.4f} cm "
          f"/Im{img_xref} Do Q\n")
    doc = page.parent

    contents = page.get_contents()
    if contents:
        # merge into the first existing content stream
        c_xref = contents[0]
        old = page.read_contents().decode("latin1", errors="replace")
        doc.update_stream(c_xref, (old + op).encode("latin1"))
    else:
        # fresh page without a content stream: create one and reference it
        c_xref = doc.get_new_xref()
        doc.update_object(c_xref, f"<< /Length {len(op)} >>")
        doc.update_stream(c_xref, op.encode("latin1"))
        page_obj = doc.xref_object(page.xref)
        page_obj = page_obj.replace(
            "/Resources", f"/Contents {c_xref} 0 R /Resources", 1)
        doc.update_object(page.xref, page_obj)
    _ensure_resource(page, img_xref)


def _ensure_resource(page, img_xref: int):
    """Register /Im<ref> in the page's /Resources /XObject dict."""
    doc = page.parent
    key = f"Im{img_xref}"
    page_obj = doc.xref_object(page.xref)
    m = re.search(r"/Resources\s+(\d+)\s+\d+\s+R", page_obj)
    if not m:
        log.warning("page has no /Resources; cannot register JBIG2 image")
        return
    r_xref = int(m.group(1))
    robj = doc.xref_object(r_xref)
    if key in robj:
        return
    if "/XObject" in robj:
        # resources already has an XObject entry -> add ours inside its dict
        robj = robj.replace("/XObject <<", f"/XObject << /{key} {img_xref} 0 R", 1)
        if f"/{key} {img_xref} 0 R" not in robj:
            # /XObject existed but not as an open dict; wrap ours after it
            robj = robj.replace("/XObject", f"/XObject << /{key} {img_xref} 0 R >>", 1)
    else:
        # append an XObject dict before the final closing >>
        robj = robj.rstrip()
        if robj.endswith(">>"):
            robj = robj[:-2].rstrip() + f" /XObject << /{key} {img_xref} 0 R >> >>"
        else:
            robj += f" /XObject << /{key} {img_xref} 0 R >>"
    doc.update_object(r_xref, robj)


def embed_jbig2(page, sym_bytes: bytes, jb2_bytes: bytes,
                width: int, height: int, rect):
    """Embed a JBIG2 image (globals + page stream) into a page."""
    doc = page.parent

    # globals stream (pure JBIG2 data, no image dict fields)
    g_xref = doc.get_new_xref()
    doc.update_object(g_xref, "<< >>")
    doc.update_stream(g_xref, sym_bytes, compress=0)

    # image XObject
    img_xref = doc.get_new_xref()
    doc.update_object(
        img_xref,
        f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
        f"/ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /JBIG2Decode "
        f"/DecodeParms << /JBIG2Globals {g_xref} 0 R >> >>")
    doc.update_stream(img_xref, jb2_bytes, compress=0)

    _paint_jbig2(page, img_xref, rect)
