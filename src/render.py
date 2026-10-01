"""PDF rendering and rebuilding via PyMuPDF."""
import logging
import os
import time
import zlib

import cv2
import pymupdf as fitz  # PyMuPDF
import numpy as np

from src.imgio import imread_unicode

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


def _page_dpi(page, dpi, max_dpi: int) -> int:
    """""auto" -> the embedded scan's own nominal DPI, clamped to [72, max_dpi].

    Vector pages (no raster at all) keep the classic 300 DPI default.
    """
    if isinstance(dpi, str) and dpi.lower() == "auto":
        nominal = _page_nominal_dpi(page)
        if nominal <= 0:
            return 300
        return int(min(max(nominal, 72.0), float(max_dpi)))
    return int(dpi)


def _render_one(doc, i: int, dpi, max_dpi: int) -> dict:
    """Render a single page to BGR. Returns {index, image, width_pt, height_pt, dpi}."""
    page = doc[i]
    page_dpi = _page_dpi(page, dpi, max_dpi)
    zoom = page_dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                          colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return {
        "index": i,
        "image": img,
        "width_pt": page.rect.width,
        "height_pt": page.rect.height,
        "dpi": page_dpi,
    }


def _page_indexes(doc, page_range):
    if page_range is None:
        return range(doc.page_count)
    start, end = page_range
    start = max(0, int(start))
    end = min(doc.page_count - 1, int(end))
    return range(start, end + 1)


class PdfPages:
    """Lazy, one-page-at-a-time access to a PDF's rendered pages.

    Rendering a whole book up front costs ~9 MB per page in raw BGR even at
    72 DPI (340 pages ≈ 3 GB) and the *enhanced* pages are roughly four times
    that again after a 2x super-resolution pass — holding both lists means a
    full book peaks around 15 GB, and a 4x run cannot fit in memory at all.
    Callers that process pages independently (the pipeline, which checkpoints
    each page to disk anyway) should render on demand instead:

        with PdfPages(pdf, dpi, max_dpi, page_range) as pages:
            for i in pages.indexes:
                p = pages.render(i)      # one page in memory at a time

    ``geometry(i)`` returns the page's point size without rendering, so a
    resumed run can rebuild a cached page without paying for the render.
    """

    def __init__(self, pdf_path: str, dpi=300, max_dpi: int = 450,
                 page_range: tuple[int, int] | None = None):
        self.pdf_path = pdf_path
        self._doc = fitz.open(pdf_path)
        self._dpi = dpi
        self._max_dpi = int(max_dpi)
        self.indexes = list(_page_indexes(self._doc, page_range))
        self._dpi_stats: dict[int, int] = {}
        self._rendered = 0

    def geometry(self, i: int) -> tuple[float, float]:
        r = self._doc[i].rect
        return r.width, r.height

    def render(self, i: int) -> dict:
        p = _render_one(self._doc, i, self._dpi, self._max_dpi)
        self._dpi_stats[p["dpi"]] = self._dpi_stats.get(p["dpi"], 0) + 1
        self._rendered += 1
        return p

    def log_stats(self):
        if not self._rendered:
            return
        if isinstance(self._dpi, str) and self._dpi.lower() == "auto":
            log.info("rendered %d/%d pages @ auto DPI %s from %s",
                     self._rendered, len(self.indexes),
                     dict(sorted(self._dpi_stats.items())), self.pdf_path)
        else:
            log.info("rendered %d/%d pages @ %d DPI from %s",
                     self._rendered, len(self.indexes), self._dpi, self.pdf_path)

    def close(self):
        try:
            self._doc.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def render_pdf_pages(pdf_path: str, dpi=300, max_dpi: int = 450,
                     page_range: tuple[int, int] | None = None):
    """Render ALL pages of a PDF to BGR arrays at once (list of page dicts).

    Convenience wrapper kept for the diagnostic/benchmark scripts, which only
    ever work on a handful of pages. The pipeline uses :class:`PdfPages`
    instead so it never holds a whole book in memory.

    Returns list of dicts: {index, image (BGR ndarray), width_pt, height_pt, dpi}
    """
    with PdfPages(pdf_path, dpi, max_dpi, page_range) as pages:
        out = [pages.render(i) for i in pages.indexes]
        pages.log_stats()
    return out


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


def _replace_with_retry(src: str, dst: str, attempts: int = 6) -> bool:
    """os.replace with backoff, because Windows hands out transient locks.

    The freshly written PDF is ~10-300 MB and real-time antivirus scanning
    opens it immediately; os.replace then fails with WinError 5 ("access
    denied") for a moment even though nothing of ours holds the file. This
    used to silently abort font subsetting, so every searchable PDF shipped
    with the whole 18 MB SimSun embedded (a 2-page test came out at 9.9 MB
    instead of 177 KB).
    """
    delay = 0.15
    for k in range(attempts):
        try:
            os.replace(src, dst)
            return True
        except OSError as e:
            if k == attempts - 1:
                log.warning("could not replace %s -> %s (%s)", src, dst, e)
                return False
            time.sleep(delay)
            delay = min(delay * 2, 1.5)
    return False


def _mrc_plane_params(ptype: str, mrc_bg_scale, mrc_bg_denoise):
    """Resolve the "auto" MRC background settings for one page type."""
    scale = 2 if mrc_bg_scale == "auto" else int(mrc_bg_scale)
    if mrc_bg_denoise == "auto":
        denoise = 5 if ptype == "gray-text" else 0
    else:
        denoise = int(mrc_bg_denoise)
    return scale, denoise


def encode_page_payload(img, ptype: str, *, jpeg_quality: int,
                        monochrome: str, mrc_bg_scale, mrc_bg_denoise,
                        jbig2_bin, workdir: str):
    """Encode one enhanced page into a small, picklable payload.

    This is the slow half of rebuilding: MRC segmentation alone is ~8 s per
    page, which made the rebuild of a 340-page book 45 minutes of
    single-threaded CPU — 44% of the whole run, more than the GPU stages put
    together. It is a pure function of (page image, encoding settings), so
    returning a payload instead of embedding straight into a MuPDF page lets
    the caller run it in a process pool while the MuPDF side stays serial
    (a fitz Document is not safe to touch from several processes).

    Payloads:
      ("1bit",  png_bytes)
      ("jbig2", sym_bytes, jb2_bytes, w, h)
      ("jpeg",  jpg_bytes)
      ("mrc",   bg_jpg, [(packed, fill), ...], h, w)
    """
    if ptype == "bw-text":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if monochrome == "jbig2" and jbig2_bin and os.path.isfile(jbig2_bin):
            try:
                from src.jbig2 import encode_jbig2
                sym, jb2, w, h = encode_jbig2(gray, jbig2_bin, workdir)
                return ("jbig2", sym, jb2, w, h)
            except Exception as e:
                log.warning("jbig2 encode failed (%s); using 1-bit", e)
        from src.jbig2 import encode_1bit
        return ("1bit", encode_1bit(gray))

    if ptype in ("mixed", "color-text", "gray-text"):
        # Dual-encode and keep the smaller: MRC layered (JPEG background with
        # text whitened + per-colour 1-bit stencils over glyph strokes) wins
        # on text-heavy pages; whole-page JPEG wins when the figure dominates
        # or the page carries no meaningful text. Stencils are compared at
        # their Flate-compressed size (raw packing is ~5.6 MB/page each,
        # zlib ~0.3 MB).
        from src.mrc import encode_mrc
        scale, denoise = _mrc_plane_params(ptype, mrc_bg_scale, mrc_bg_denoise)
        mrc = encode_mrc(img, jpeg_quality, bg_gray=(ptype == "gray-text"),
                         jbig2_bin=jbig2_bin, workdir=workdir,
                         bg_scale=scale, bg_denoise=denoise)
        jpg = encode_page_image(img, ptype, jpeg_quality)
        if mrc[0] is not None and mrc[1]:
            z_est = sum(len(zlib.compress(pk, 6)) for pk, _ in mrc[1])
            if len(mrc[0]) + z_est < len(jpg):
                bg_jpg, layers, mh, mw = mrc
                return ("mrc", bg_jpg, layers, mh, mw)
        return ("jpeg", jpg)

    return ("jpeg", encode_page_image(img, ptype, jpeg_quality))


def embed_page_payload(page, payload):
    """Install an encoded payload on a MuPDF page. Serial and cheap."""
    kind = payload[0]
    if kind == "1bit":
        page.insert_image(page.rect, stream=payload[1])
    elif kind == "jbig2":
        from src.jbig2 import embed_jbig2
        embed_jbig2(page, payload[1], payload[2], payload[3], payload[4],
                    page.rect)
    elif kind == "jpeg":
        page.insert_image(page.rect, stream=payload[1])
    else:  # mrc
        from src.mrc import embed_mrc
        embed_mrc(page, payload[1], payload[2], payload[3], payload[4])


_ENC_OPTS: dict | None = None


def _enc_init(opts: dict):
    """Process-pool initializer: stash the (picklable) encoding settings.

    Also pins OpenCV to a single thread per worker. Measured effect on the
    real book: none (0.81 -> 0.80 s/page at 8 workers), so this is hygiene
    rather than an optimisation — it just keeps N workers from each starting
    their own thread pool.
    """
    global _ENC_OPTS
    _ENC_OPTS = opts
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _enc_worker(spec):
    """Encode one page inside a worker process. `spec` = (image_path, ptype)."""
    image_path, ptype = spec
    img = imread_unicode(image_path)
    if img is None:
        raise RuntimeError(f"cannot read page image {image_path}")
    try:
        return encode_page_payload(img, ptype, **_ENC_OPTS)
    finally:
        del img


def resolve_encode_workers(encode_workers) -> int:
    """ "auto" -> half the logical CPUs (the encoder is a mix of a jbig2enc
    subprocess, OpenCV and numpy, so leaving headroom helps)."""
    if encode_workers in (None, "auto"):
        return max(1, (os.cpu_count() or 4) // 2)
    return max(1, int(encode_workers))


def rebuild_pdf(pages, out_path: str, jpeg_quality: int = 85,
                text_layers: dict[int, list[dict]] | None = None,
                font_file: str | None = None,
                monochrome: str = "1bit",
                jbig2_bin: str | None = None,
                mrc_bg_scale="auto", mrc_bg_denoise="auto",
                encode_workers="auto"):
    """Build a new PDF from processed page images.

    pages: list of dicts with keys width_pt, height_pt, ptype and EITHER
      image (BGR ndarray) OR image_path (a file to load it from). The path
      form lets the pipeline rebuild a checkpointed book one page at a time
      instead of holding every enhanced page in memory at once.
    encode_workers: number of processes used to encode pages, or "auto".
      Encoding is per-page independent and CPU-bound, so this is where a
      long book spends most of its rebuild time. Pages given as in-memory
      arrays are always encoded in-process (they would have to be pickled).
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

    workdir = os.path.dirname(out_path)
    pages = list(pages)
    opts = dict(jpeg_quality=jpeg_quality, monochrome=monochrome,
                mrc_bg_scale=mrc_bg_scale, mrc_bg_denoise=mrc_bg_denoise,
                jbig2_bin=jbig2_bin, workdir=workdir)

    # --- encode (parallel when every page lives on disk) ---
    workers = resolve_encode_workers(encode_workers)
    file_backed = all(p.get("image") is None and p.get("image_path")
                      for p in pages)
    payloads = None
    if file_backed and len(pages) > 1 and workers > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            specs = [(p["image_path"], p.get("ptype", "color")) for p in pages]
            t_enc = time.time()
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_enc_init,
                                     initargs=(opts,)) as ex:
                payloads = list(ex.map(_enc_worker, specs, chunksize=2))
            log.info("encoded %d pages on %d workers in %.1fs",
                     len(pages), workers, time.time() - t_enc)
        except Exception as e:
            log.warning("parallel encode unavailable (%s); encoding serially", e)
            payloads = None
    if payloads is None:
        payloads = []
        for p in pages:
            img = p.get("image")
            if img is None:
                img = imread_unicode(p["image_path"])
                if img is None:
                    raise RuntimeError(f"cannot read page image {p['image_path']}")
            payloads.append(encode_page_payload(
                img, p.get("ptype", "color"), **opts))
            del img

    # --- embed (MuPDF is single-threaded here) ---
    doc = fitz.open()
    for p, payload in zip(pages, payloads):
        page = doc.new_page(width=p["width_pt"], height=p["height_pt"])
        embed_page_payload(page, payload)
        if text_layers and p["index"] in text_layers:
            insert_hidden_text_layer(page, text_layers[p["index"]], font_file)
        del payload

    doc.save(out_path, deflate=True, garbage=3)
    doc.close()
    before = os.path.getsize(out_path)
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
            if _replace_with_retry(out_path + ".subset.pdf", out_path):
                log.info("font subset applied: %d bytes released",
                         max(0, before - os.path.getsize(out_path)))
            else:
                log.warning(
                    "font subsetting could not be installed (file locked); "
                    "the PDF keeps the full embedded font — re-run "
                    "--rebuild-only, which reuses the page checkpoint")
        except Exception as e:
            log.warning("font subsetting failed (%s); keeping full font", e)

    log.info("wrote %s (%d pages)", out_path, len(pages))
