# -*- coding: utf-8 -*-
"""MRC (Mixed Raster Content) layered encoding for text+image pages, v3.

Segmentation: agl/jbig2enc's adaptive thresholder (tools/jbig2/jbig2.exe,
Apache-2.0, leptonica 1.87) — the same component Internet Archive runs over
millions of scanned pages — instead of a hand-rolled cv2 threshold.

Guards that keep document reality from ruining the stencil:

  - LARGE colored regions (screenshots, figure fills) never enter the
    foreground: a stencil paints solid fill, so a dark-blue UI panel would
    become a flat silhouette. Large = connected color component covering
    >= BLOB_MIN_AREA of the page.
  - large dark connected blobs (grayscale photos, dark figure fills) are
    excluded for the same reason; text never forms such large components

Colored TEXT (blue keywords, colored captions) is supported since v3:
the foreground is split into per-color groups. Each group becomes its own
1-bit /ImageMask painted with the group's measured fill color (a PDF
ImageMask is painted with the current fill color, so "固件" renders blue
while body text stays black). Up to MAX_COLOR_LAYERS color groups are
kept; rarer colors fall back to the background JPEG untouched.

Encoding: classic MRC stencil structure —
  background: JPEG (ALL stenciled text whitened out, figures untouched)
  foreground: N x [1-bit /ImageMask (JBIG2, Flate fallback) + per-group fill]
  Paint polarity verified empirically: ink bits = 0 + /Decode [0 1].

The stencils are the bulk of the finished file — 54.7% of a 340-page book
(30.6 MB of 56.0 MB) — and they used to be Flate, which is the wrong codec
for text bitmaps. jbig2enc's generic region coder is ~2.6x smaller on them
(measured: 1.56 MB -> 0.62 MB over 20 real pages, rasters bit-identical), so
JBIG2 is now the default and Flate the fallback.

This is encoding only: the *thresholding* that decides the mask was already
jbig2enc's (see _threshold_mask), so the mask itself is unchanged.
"""
import glob
import logging
import os
import struct
import subprocess
import zlib

import cv2
import numpy as np

from src.classify import _color_mask
from src.imgio import imread_unicode, imwrite_unicode

log = logging.getLogger("pdfenhance")

# A connected dark blob >= this fraction of the page area is figure/photo
# content, not text. A bold CJK glyph at 5756x8020 is ~0.05% of the page, a
# photo region is typically >1%. 0.4% sits safely between them.
BLOB_MIN_AREA = 0.004
# Connected components smaller than this (pixels) are SR noise specks.
SPECK_MIN_AREA = 24
# Below this ink coverage MRC has nothing to offer — caller should just use
# whole-page JPEG.
MIN_COVERAGE = 0.0005
# Median stroke saturation <= this counts as black ink (no color layer).
BLACK_SAT_MAX = 45
# OpenCV hue units per color group (6 buckets over 0..179).
HUE_BUCKET = 30
# Color layers kept besides the black one; rarer colors stay in background.
MAX_COLOR_LAYERS = 3
# Paint stencil keeps pixels up to this far above the page's own median ink
# level. jbig2enc's Sauvola threshold alone reads ~35% fatter than the
# grayscale stroke it came from (it swallows the SR soft shoulder); this
# global band trims the light shoulder back to natural stroke weight while
# auto-relaxing on faded scans (median ink high -> cap high -> no trim).
PAINT_CORE_BAND = 70
# Floor of the paint cap. It must sit high enough to keep a stroke whose ink
# only ever reaches mid-grey: CJK horizontals at 1439 px native are a light
# shade, and a cap of 130 deleted those strokes outright (measured: 1.10% of
# the source's ink pixels went missing vs 0.10% at 140). 140 keeps the glyph
# intact while still trimming the anti-aliasing shoulder that made strokes
# read as fake-bold.
PAINT_CAP_MIN, PAINT_CAP_MAX = 140, 190


def _threshold_mask(gray: np.ndarray, jbig2_bin: str | None,
                    workdir: str | None) -> np.ndarray:
    """Adaptive-threshold the page into an ink mask (True = dark ink).

    Primary: jbig2enc -O dump (leptonica local adaptive thresholding).
    Fallback: cv2 adaptive threshold (project's binarize()).
    """
    if jbig2_bin and os.path.isfile(jbig2_bin) and workdir:
        src = os.path.join(workdir, f"_mrc_in_{os.getpid()}.png")
        dst = os.path.join(workdir, f"_mrc_thr_{os.getpid()}.png")
        try:
            imwrite_unicode(src, gray)
            # -O dumps the thresholded image; jbig2enc also prints the raw
            # JBIG2 generic stream to stdout, so silence it.
            subprocess.run(
                [os.path.abspath(jbig2_bin), "-O", os.path.abspath(dst),
                 os.path.abspath(src)],
                check=True, capture_output=True, timeout=180)
            thr = imread_unicode(dst, cv2.IMREAD_GRAYSCALE)
            if thr is not None and thr.shape == gray.shape:
                return thr < 128
            log.warning("jbig2enc threshold dump missing/mismatched; "
                        "using cv2 fallback")
        except Exception as e:
            log.warning("jbig2enc threshold failed (%s); using cv2 "
                        "fallback", e)
        # NOTE: no cleanup here. The two filenames are pid-stable, so each
        # page simply overwrites them; deleting here meant 2 file deletions
        # per page and a 340-page book issued ~680 of them. rebuild_pdf
        # clears the pair once when it is done.
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 10) > 0


def cleanup_threshold_files(workdir: str | None) -> None:
    """Drop the scratch PNGs written by _threshold_mask / encode_jbig2.

    Called once per document from rebuild_pdf. The names are pid-tagged so
    concurrent worker processes do not collide, which means the set to clean
    is only known by globbing — only these exact prefixes are touched, and
    never a recursive delete.
    """
    if not workdir:
        return
    patterns = ("_mrc_in_*.png", "_mrc_thr_*.png", "_jbig2_bw_*.png",
                "_jbig2_*.sym", "_jbig2_*.0000", "_stencil_*.png")
    removed = 0
    for pat in patterns:
        for p in glob.glob(os.path.join(glob.escape(workdir), pat)):
            try:
                os.remove(p)
                removed += 1
            except OSError:
                pass
    if removed:
        log.debug("cleaned %d encoder scratch file(s) in %s", removed, workdir)


def cleanup_own_scratch(workdir: str | None) -> int:
    """Remove only THIS process's scratch files. Safe to call from a worker.

    cleanup_threshold_files() globs every pid, so a worker calling it at exit
    would delete files its siblings are still using. Here the names are
    derived from os.getpid(), so each worker reclaims exactly its own pair.
    Each worker holds one pair for a whole run (the names are reused page
    after page), so this is two deletes per worker rather than two per page.
    """
    if not workdir:
        return 0
    removed = 0
    for name in (f"_mrc_in_{os.getpid()}.png", f"_mrc_thr_{os.getpid()}.png"):
        try:
            os.remove(os.path.join(workdir, name))
            removed += 1
        except OSError:
            pass
    return removed


def _large_dark_blobs(gray: np.ndarray) -> np.ndarray:
    """Mask of big connected dark regions (photos, dark figure fills).

    Text glyphs are small isolated components even at SR resolution; anything
    this large is image content that must stay in the background JPEG.
    """
    dark = (gray < 170).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    blob = np.zeros(gray.shape, bool)
    min_area = BLOB_MIN_AREA * gray.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            blob[labels == i] = True
    if blob.any():
        # safety margin so photo fringes never leak into the stencil
        blob = cv2.dilate(blob.astype(np.uint8),
                          cv2.getStructuringElement(
                              cv2.MORPH_ELLIPSE, (9, 9))) > 0
    return blob


def _large_color_blobs(img_bgr: np.ndarray) -> np.ndarray:
    """Mask of big connected COLORED regions (screenshot panels, figure
    fills). Small colored components — colored text strokes, captions —
    are deliberately NOT here: they belong to the (color-aware) foreground.
    """
    color = _color_mask(img_bgr).astype(np.uint8)
    if not color.any():
        return np.zeros(color.shape, bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(color, connectivity=8)
    blob = np.zeros(color.shape, bool)
    min_area = BLOB_MIN_AREA * color.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            blob[labels == i] = True
    if blob.any():
        blob = cv2.dilate(blob.astype(np.uint8),
                          cv2.getStructuringElement(
                              cv2.MORPH_ELLIPSE, (9, 9))) > 0
    return blob


def _drop_specks(fg: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < SPECK_MIN_AREA:
            fg[labels == i] = False
    return fg


def _core_colors(img_bgr: np.ndarray, labels: np.ndarray,
                 n: int) -> np.ndarray:
    """Vectorized per-component ink color for ALL connected components at once.

    Returns a (n, 3) int array where row `i` is the mean BGR of the *dark
    core* of component label `i` (rows for unused labels are 0).

    This replaces a per-component Python loop (`labels == i` + percentile +
    median) that was O(components x pixels) and took ~50 s/page on 5756x8020
    scans. The vectorized form uses np.bincount (the standard StackOverflow
    technique for per-label statistics) and runs in single-digit ms.

    "Dark core" is implemented as a single global pre-filter: only pixels
    darker than the page's 30th-percentile foreground luminance contribute.
    This reproduces the old median-of-darkest-30% semantics closely enough
    for the black-vs-color decision (which only needs the ink hue), at a
    tiny fraction of the cost.
    """
    fg = labels > 0
    lum = img_bgr.astype(np.int32).sum(axis=2)  # (H, W) per-pixel luminance
    core_mask = fg & (lum <= np.percentile(lum[fg], 30)) \
        if fg.any() else fg
    lab = labels[core_mask]
    if lab.size == 0:
        return np.zeros((n, 3), np.int32)
    # np.bincount per channel: bins = component label, weights = pixel value
    sums = np.stack([
        np.bincount(lab, img_bgr[core_mask, c], minlength=n).astype(np.int64)
        for c in range(3)
    ], axis=1)  # (n, 3)
    cnt = np.bincount(lab, minlength=n).astype(np.int64)  # (n,)
    means = np.zeros((n, 3), np.int32)
    valid = cnt > 0
    means[valid] = (sums[valid] / cnt[valid, None]).astype(np.int32)

    # A component with NO pixels in the global dark-core band (a faint or
    # anti-aliased-only stroke) keeps a (0,0,0) row. Still it IS real ink, so
    # fall back to the mean over all of its pixels — otherwise it would be
    # indistinguishable from black ink (and from an unused label).
    missing = ~valid
    missing[0] = False  # label 0 is the background, never a component
    if missing.any():
        m_all = labels > 0
        all_lab = labels[m_all]
        asums = np.stack([
            np.bincount(all_lab, img_bgr[m_all, c], minlength=n).astype(np.int64)
            for c in range(3)
        ], axis=1)
        acnt = np.bincount(all_lab, minlength=n).astype(np.int64)
        mc = missing & (acnt > 0)
        means[mc] = (asums[mc] / acnt[mc, None]).astype(np.int32)
    return means


def split_mrc(img_bgr: np.ndarray, jbig2_bin: str | None = None,
              workdir: str | None = None):
    """Split a page into (background_bgr, layers).

    layers: list of (stencil_mask_bool, fill_bgr) — one entry per color
    group. Empty list / None means "no meaningful text — don't use MRC".
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    thr = _threshold_mask(gray, jbig2_bin, workdir)
    if not thr.any():
        return img_bgr.copy(), None

    # Only LARGE figure-scale regions are protected from the stencil;
    # small colored strokes (colored text) pass through and get their own
    # color layer below.
    figures = _large_color_blobs(img_bgr) | _large_dark_blobs(gray)

    fg = thr & ~figures
    # NO morphological opening here. A 3x3 open erodes then dilates, so it
    # deletes every structure thinner than 3 px — and at a 2878 px page width
    # the thin horizontal strokes of a CJK glyph are exactly 2 px. Measured on
    # a real book page: the open alone wiped 20% of the ink inside a glyph
    # crop (15.20% -> 12.13%), and after rendering it cost 1.10% of the
    # source's ink pixels, versus 0.10% without it. Salt noise is already
    # handled by the area-based _drop_specks below.
    fg = _drop_specks(fg.copy())

    if float(fg.mean()) < MIN_COVERAGE:
        return img_bgr.copy(), None

    # ---- per-component ink color -> color groups -------------------------
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=8)
    core = _core_colors(img_bgr, labels, n)  # (n, 3) mean BGR of dark core
    groups: dict = {"black": []}
    for i in range(1, n):
        # NOTE: labels 1..n-1 from connectedComponentsWithStats are *all* real
        # components — there are no "unused" labels. Never use a zero core
        # color as a skip sentinel: a pure-black stroke legitimately has the
        # core color (0,0,0), and skipping those silently emptied the
        # foreground on pages whose text is pure black (MRC then always lost
        # the size comparison and every page fell back to whole-page JPEG).
        c = core[i]
        if int(max(c)) - int(min(c)) <= BLACK_SAT_MAX:
            groups["black"].append(i)
        else:
            hsv = cv2.cvtColor(np.array([[c]], np.uint8), cv2.COLOR_BGR2HSV)
            hue_bucket = int(hsv[0, 0, 0]) // HUE_BUCKET
            groups.setdefault(hue_bucket, []).append(i)

    # black group + the MAX_COLOR_LAYERS largest color buckets; components
    # of dropped buckets stay in the background JPEG (no whiten, no stencil)
    buckets = [(k, v) for k, v in groups.items() if k != "black"]
    buckets.sort(
        key=lambda kv: -sum(stats[i, cv2.CC_STAT_AREA] for i in kv[1]))
    kept = ([("black", groups["black"])] if groups["black"] else [])
    kept += buckets[:MAX_COLOR_LAYERS]

    layers = []
    paint_all = np.zeros(fg.shape, bool)
    # adaptive paint cap: page's own median ink level + a fixed band
    ink_med = float(np.median(gray[fg]))
    paint_cap = int(np.clip(ink_med + PAINT_CORE_BAND,
                            PAINT_CAP_MIN, PAINT_CAP_MAX))
    log.debug("MRC paint cap %d (ink median %.0f)", paint_cap, ink_med)
    for _, comps in kept:
        if not comps:
            continue
        mask = np.isin(labels, comps) & (gray <= paint_cap)
        if not mask.any():
            continue
        # area-weighted core color of this layer's components (vectorized,
        # reuses the per-component core colors already computed above)
        areas = stats[comps, cv2.CC_STAT_AREA].astype(np.float64)
        w = areas / max(areas.sum(), 1.0)
        fill = tuple(int(v) for v in np.round(
            (core[comps].astype(np.float64) * w[:, None]).sum(axis=0)))
        if max(fill) - min(fill) <= BLACK_SAT_MAX:
            fill = (0, 0, 0)  # near-black bucket: keep the pure-black look
        # PAINT the trimmed raw mask: dilating the paint stencil makes every
        # stroke read as fake bold (verified on stage strips — the SR
        # grayscale output looks natural, a dilated stencil does not).
        layers.append((mask, fill))
        paint_all |= mask

    if not layers:
        return img_bgr.copy(), None

    # whiten with a safety ring around the ink so no gray stroke shoulder
    # leaks into the background JPEG (wider than the paint mask on purpose)
    whiten = cv2.dilate(paint_all.astype(np.uint8),
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (5, 5))) > 0
    bg = img_bgr.copy()
    bg[whiten] = 255
    return bg, layers


def _png_1bit(packed_rows: bytes, width: int, height: int) -> bytes:
    """Wrap row-packed 1-bit samples in a minimal 1-bpp grayscale PNG.

    jbig2enc only skips its own binarizer when the input is already 1 bpp.
    Hand it an 8-bit PNG and it re-thresholds the mask (adaptively by
    default), i.e. it edits the very bitmap we just decided on.
    """
    stride = (width + 7) // 8
    raw = b"".join(b"\x00" + packed_rows[y * stride:(y + 1) * stride]
                   for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def encode_stencil_jbig2(packed_rows: bytes, width: int, height: int,
                         jbig2_bin: str, workdir: str) -> bytes | None:
    """Encode one 1-bit stencil as JBIG2, or None if the encoder fails.

    Generic region coder (`-p`, produce PDF-ready data). Deliberately NOT
    symbol mode (`-s`): symbol mode merges look-alike glyphs, which is exactly
    the "hallucinated character" risk this project refuses to take. Generic
    mode is lossless, so the raster is bit-identical to the Flate one — it is
    only the container that changes.
    """
    tag = os.getpid()
    png_path = os.path.abspath(os.path.join(workdir, f"_stencil_{tag}.png"))
    try:
        with open(png_path, "wb") as fh:
            fh.write(_png_1bit(packed_rows, width, height))
        res = subprocess.run([os.path.abspath(jbig2_bin), "-p", png_path],
                             capture_output=True, cwd=workdir, timeout=180)
        if res.returncode != 0 or not res.stdout:
            log.warning("jbig2 stencil encode failed: %s",
                        res.stderr.decode(errors="replace")[-300:])
            return None
        return res.stdout
    except Exception as e:  # noqa: BLE001 - any failure must fall back to Flate
        log.warning("jbig2 stencil encode raised (%s); using Flate", e)
        return None
    finally:
        try:
            os.remove(png_path)
        except OSError:
            pass


def encode_mrc(img_bgr: np.ndarray, jpeg_quality: int,
               bg_gray: bool = False, jbig2_bin: str | None = None,
               workdir: str | None = None, bg_scale: int = 1,
               bg_denoise: int = 0, stencil_codec: str = "jbig2"):
    """Encode a page as MRC layers.

    Returns (background_jpeg, layers, height, width) where layers is a list
    of (packed_1bit_bytes, fill_bgr); or (None, [], 0, 0) when the page
    carries no meaningful text (caller falls back to whole-page JPEG).

    bg_scale: integer downsampling factor applied to the background plane
    before JPEG encoding (>1 = smaller). The background carries only
    low-frequency content (paper tone, gradients, illumination, and any large
    figure excluded from the stencil), while every text glyph is preserved at
    full resolution by the 1-bit stencil. Storing the background at 1/N
    resolution and letting the PDF viewer scale it back therefore costs ~N^2
    fewer pixels with no visible loss on text pages — the classic DjVu
    background-plane trick. Measured on a real 340-page book: 1627 KB/page
    (whole-page JPEG) -> 204 KB/page at bg_scale=4 with an identical-looking
    result. Use a milder factor on figure-heavy pages so the figure (which
    lives in the background) is not softened.

    bg_denoise: odd median-kernel size applied to the background before
    encoding (0 = off). The background is what SR left behind after the text
    was lifted out, and it is dominated by per-pixel noise the SR injected —
    noise that costs a lot of JPEG bytes but carries zero information.
    Measured on the same book: a median-5 pass shrinks the full-resolution
    background JPEG by ~2.5x (p130: 869 -> 384 KB) while the remaining
    gradients/figures are untouched at this resolution (7 px is 0.24% of a
    2878 px page width). It composes with bg_scale.
    """
    bg, layers = split_mrc(img_bgr, jbig2_bin, workdir)
    if not layers:
        return None, [], 0, 0

    if bg_gray:
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    if bg_denoise and int(bg_denoise) >= 3:
        k = int(bg_denoise) | 1
        bg = cv2.medianBlur(bg, k)
    if bg_scale and int(bg_scale) > 1:
        f = int(bg_scale)
        h0, w0 = bg.shape[:2]
        bg = cv2.resize(bg, (max(1, w0 // f), max(1, h0 // f)),
                        interpolation=cv2.INTER_AREA)
    bg_quality = min(int(jpeg_quality), 85)
    ok, bg_jpg = cv2.imencode(
        ".jpg", bg, [int(cv2.IMWRITE_JPEG_QUALITY), bg_quality])
    if not ok:
        raise RuntimeError("MRC background JPEG encode failed")

    # ImageMask stencil, 1-bit row-packed, ink = 0 bit (verified polarity:
    # /Decode [0 1] paints zero samples).
    #
    # Each layer carries (mode, data, fill, stored_size). stored_size is what
    # the layer will actually cost in the file, which is what the caller's
    # dual-encode decision compares against a whole-page JPEG: for JBIG2 the
    # bytes are already final, for Flate it is the zlib estimate (PyMuPDF
    # deflates at embed time, and its output lands within a few percent).
    h, w = layers[0][0].shape
    enc_layers = []
    for stencil, fill in layers:
        packed = np.packbits((~stencil).astype(np.uint8), axis=1).tobytes()
        if (stencil_codec == "jbig2" and jbig2_bin and workdir
                and os.path.isfile(jbig2_bin)):
            jb2 = encode_stencil_jbig2(packed, w, h, jbig2_bin, workdir)
            if jb2:
                enc_layers.append(("jbig2", jb2, fill, len(jb2)))
                continue
        enc_layers.append(("flate", packed, fill, len(zlib.compress(packed, 6))))
    return bg_jpg.tobytes(), enc_layers, h, w


def embed_mrc(page, bg_jpg: bytes, layers, height: int, width: int):
    """Insert the background JPEG plus the per-color stencil masks on `page`.

    layers: list of (mode, data, fill_bgr, stored_size) from encode_mrc.

    mode "jbig2": data is a JBIG2 generic-region stream, written verbatim with
      /JBIG2Decode and compress=0 (it is already entropy coded).
      The /Filter key MUST be re-set with xref_set_key AFTER update_stream:
      update_stream() drops every /Filter key, and save(deflate=True) then
      deflates the payload and relabels it /FlateDecode — which decodes to
      garbage and renders the page black. Verified: same call order without
      the re-set writes 294 bytes out of a 1032-byte payload.

    mode "flate": data is the RAW 1-bit row-packed stencil (ink = 0 bit),
      stored with compress=1 so PyMuPDF applies its own FlateDecode — the
      only variant MuPDF renders correctly (a hand-zlib'd stream with a
      manual /Filter entry renders the page all black).
    """
    from src.jbig2 import _paint_jbig2

    page.insert_image(page.rect, stream=bg_jpg)

    doc = page.parent
    for mode, data, (b, g, r), _cost in layers:
        xref = doc.get_new_xref()
        common = (f"<< /Type /XObject /Subtype /Image /Width {width} "
                  f"/Height {height} /ImageMask true /BitsPerComponent 1 "
                  f"/Decode [0 1] >>")
        doc.update_object(xref, common)
        doc.update_stream(xref, data, compress=0 if mode == "jbig2" else 1)
        if mode == "jbig2":
            doc.xref_set_key(xref, "Filter", "/JBIG2Decode")
        _paint_jbig2(page, xref, page.rect,
                     fill=f"{r / 255.0:.4f} {g / 255.0:.4f} "
                          f"{b / 255.0:.4f} rg ")
