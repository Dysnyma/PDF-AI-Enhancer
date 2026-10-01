"""PDF rendering and rebuilding via PyMuPDF."""
import logging
import os
import zlib

import cv2
import pymupdf as fitz  # PyMuPDF
import numpy as np

log = logging.getLogger("pdfenhance")


def _page_nominal_dpi(page) -> float:
    """Nominal DPI of the largest embedded raster on the page (image px per
    placed inch). Returns 0.0 for vector pages without raster images.

    Example: a 1439x2005 scan placed on a 1439x2005 pt page -> 72 DPI, so
    rendering at 72 gives one output pixel per source pixel (zero interp).
    """
    best = 0.0
    try:
        imgs = page.get_images(full=True)
    except Exception:
        return 0.0
    for img in imgs:
        w = img[2]
        if not w:
            continue
        try:
            rects = page.get_image_rects(img[0])
        except Exception:
            continue
        for r in rects:
            if r.width > 0 and not r.is_empty:
                best = max(best, w * 72.0 / r.width)
    return best


def render_pdf_pages(pdf_path: str, dpi=300, max_dpi: int = 450,
                     page_range: tuple[int, int] | None = None):
    """Render pages of a PDF to BGR numpy arrays.

    dpi: int, or "auto" — per page, render at the embedded scan's own nominal
    DPI (zero interpolation: one output pixel per scan pixel), clamped to
    [72, max_dpi]. Vector pages without raster images fall back to 300.

    page_range: optional (start, end) inclusive 0-based page indexes; when
    given, only those pages are rendered (end clamped to the last page).

    Returns list of dicts: {index, image (BGR ndarray), width_pt, height_pt, dpi}
    """
    doc = fitz.open(pdf_path)
    auto = isinstance(dpi, str) and dpi.lower() == "auto"
    pages = []
    dpi_stats: dict[int, int] = {}
    if page_range is not None:
        start, end = page_range
        start = max(0, int(start))
        end = min(doc.page_count - 1, int(end))
        idxs = range(start, end + 1)
    else:
        idxs = range(doc.page_count)
    for i in idxs:
        page = doc[i]
        if auto:
            nominal = _page_nominal_dpi(page)
            if nominal <= 0:
                page_dpi = 300  # vector page: keep the classic default
            else:
                page_dpi = int(min(max(nominal, 72.0), float(max_dpi)))
        else:
            page_dpi = int(dpi)
        zoom = page_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pages.append({
            "index": i,
            "image": img,
            "width_pt": page.rect.width,
            "height_pt": page.rect.height,
            "dpi": page_dpi,
        })
        dpi_stats[page_dpi] = dpi_stats.get(page_dpi, 0) + 1
    doc.close()
    if auto:
        log.info("rendered %d pages @ auto DPI %s from %s",
                 len(pages), dict(sorted(dpi_stats.items())), pdf_path)
    else:
        log.info("rendered %d pages @ %d DPI from %s", len(pages), dpi, pdf_path)
    return pages


def encode_page_image(img_bgr: np.ndarray, page_type: str, jpeg_quality: int) -> bytes:
    """Encode a page image to JPEG bytes appropriate for its type.

    color/color-text/color-image/mixed -> 3-channel JPEG
    gray-*                              -> 1-channel JPEG (1/3 the BGR size)
    bw-text is NOT handled here (see rebuild_pdf -> binarized 1-bit/JBIG2).
    """
    if page_type.startswith("gray"):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = img_bgr
    ok, buf = cv2.imencode(".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def rebuild_pdf(pages, out_path: str, jpeg_quality: int = 85,
                text_layers: dict[int, list[dict]] | None = None,
                font_file: str | None = None,
                monochrome: str = "1bit",
                jbig2_bin: str | None = None,
                mrc_bg_scale="auto", mrc_bg_denoise="auto"):
    """Build a new PDF from processed page images.

    pages: list of dicts with keys image (BGR), width_pt, height_pt, ptype.
    monochrome: "1bit" (default) or "jbig2" — encoding for bw-text pages.
      jbig2 is EXPERIMENTAL: PyMuPDF strips the /JBIG2Decode filter on save
      (MuPDF has no JBIG2 support), which corrupts the page (renders all
      black). A post-save validation reverts such output to the 1-bit path.
    mrc_bg_scale: MRC background-plane downsampling factor — "auto" | int.
      "auto" picks 2 for every text page: the render pipeline runs at the
      source's effective DPI, so a factor of 2 puts the background back at
      exactly the resolution of the original scan — everything beyond that
      is SR interpolation and carries no real information. 1 disables
      downsampling.
    mrc_bg_denoise: odd median-kernel size for the background — "auto" | int.
      "auto" applies 5 on gray-text pages: the residual background is mostly
      per-pixel noise injected by the SR, which costs JPEG bytes and buys
      nothing; median filtering it is a ~2.5x background saving and a
      visibly cleaner paper tone. Figure-bearing pages (mixed/color-text)
      keep 0 so the figure is untouched.
    text_layers: optional {page_index: [spans]} of OCR text (invisible).
    Physical page size is preserved, so effective DPI rises with SR.
    """
    from src.ocr import insert_hidden_text_layer
    from src.jbig2 import encode_1bit, encode_jbig2, embed_jbig2

    doc = fitz.open()
    workdir = os.path.dirname(out_path)
    for p in pages:
        ptype = p.get("ptype", "color")
        page = doc.new_page(width=p["width_pt"], height=p["height_pt"])

        if ptype == "bw-text":
            gray = cv2.cvtColor(p["image"], cv2.COLOR_BGR2GRAY)
            if monochrome == "jbig2" and jbig2_bin and os.path.isfile(jbig2_bin):
                try:
                    sym, jb2, w, h = encode_jbig2(gray, jbig2_bin, workdir)
                    embed_jbig2(page, sym, jb2, w, h, page.rect)
                except Exception as e:
                    log.warning("jbig2 encode failed (%s); using 1-bit", e)
                    png = encode_1bit(gray)
                    page.insert_image(page.rect, stream=png)
            else:
                png = encode_1bit(gray)
                page.insert_image(page.rect, stream=png)
        elif ptype in ("mixed", "color-text", "gray-text"):
            # Dual-encode and keep the smaller: MRC layered (JPEG background
            # with text whitened + per-color 1-bit stencils over glyph
            # strokes) wins on text-heavy pages; whole-page JPEG wins when
            # the figure dominates (background saving < stencil cost) or
            # when the page carries no meaningful text at all. Stencils are
            # compared at their Flate-compressed size (raw packing is
            # ~5.6 MB/page each; zlib ~0.3 MB).
            from src.mrc import encode_mrc, embed_mrc
            scale = mrc_bg_scale
            if scale == "auto":
                scale = 2
            denoise = mrc_bg_denoise
            if denoise == "auto":
                denoise = 5 if ptype == "gray-text" else 0
            mrc = encode_mrc(p["image"], jpeg_quality,
                             bg_gray=(ptype == "gray-text"),
                             jbig2_bin=jbig2_bin, workdir=workdir,
                             bg_scale=int(scale), bg_denoise=int(denoise))
            jpg = encode_page_image(p["image"], ptype, jpeg_quality)
            if mrc[0] is not None and mrc[1]:
                z_est = sum(len(zlib.compress(pk, 6)) for pk, _ in mrc[1])
                if len(mrc[0]) + z_est < len(jpg):
                    bg_jpg, layers, mh, mw = mrc
                    embed_mrc(page, bg_jpg, layers, mh, mw)
                else:
                    page.insert_image(page.rect, stream=jpg)
            else:
                page.insert_image(page.rect, stream=jpg)
        else:
            jpg = encode_page_image(p["image"], ptype, jpeg_quality)
            page.insert_image(page.rect, stream=jpg)

        if text_layers and p["index"] in text_layers:
            insert_hidden_text_layer(page, text_layers[p["index"]], font_file)

    doc.save(out_path, deflate=True, garbage=3)
    doc.close()
    from src.mrc import cleanup_threshold_files
    cleanup_threshold_files(workdir)

    # Sanity check for the experimental jbig2 path: PyMuPDF strips
    # /JBIG2Decode on save, so the raw JBIG2 stream is misread as Flate data
    # and the page renders all black. Detect that and rebuild with 1-bit.
    if monochrome == "jbig2" and any(p.get("ptype") == "bw-text" for p in pages):
        try:
            d2 = fitz.open(out_path)
            corrupted = False
            for p in d2:
                for im in p.get_images(full=True):
                    if im[4] != 1 or "JBIG2" in str(im[8]):
                        continue
                    # render a small probe: an all-black page = broken decode
                    pix = p.get_pixmap(matrix=fitz.Matrix(0.1, 0.1))
                    arr = np.frombuffer(pix.samples, np.uint8)
                    ink = float((arr < 128).mean())
                    if ink > 0.95:
                        corrupted = True
            d2.close()
        except Exception:
            corrupted = True
        if corrupted:
            log.warning(
                "jbig2 output corrupted by PyMuPDF save (filter stripped); "
                "falling back to 1-bit for the whole document")
            return rebuild_pdf(pages, out_path, jpeg_quality=jpeg_quality,
                               text_layers=text_layers, font_file=font_file,
                               monochrome="1bit")

    # Subset embedded CJK fonts: PyMuPDF embeds the *whole* font file (the
    # 18 MB simsun.ttc) for the OCR text layer; subsetting keeps only the
    # glyphs actually used, shrinking it to a few KB.
    if text_layers:
        try:
            d2 = fitz.open(out_path)
            d2.subset_fonts()
            d2.save(out_path + ".subset.pdf", deflate=True, garbage=3)
            d2.close()
            os.replace(out_path + ".subset.pdf", out_path)
        except Exception as e:
            log.warning("font subsetting failed (%s); keeping full font", e)

    log.info("wrote %s (%d pages)", out_path, len(pages))
