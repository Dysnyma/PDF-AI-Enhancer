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
  foreground: N x [1-bit /ImageMask (PyMuPDF Flate) + per-group fill]
  Paint polarity verified empirically: ink bits = 0 + /Decode [0 1].
"""
import logging
import os
import subprocess

import cv2
import numpy as np

from src.classify import _color_mask

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
PAINT_CAP_MIN, PAINT_CAP_MAX = 130, 190


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
            cv2.imwrite(src, gray)
            # -O dumps the thresholded image; jbig2enc also prints the raw
            # JBIG2 generic stream to stdout, so silence it.
            subprocess.run(
                [os.path.abspath(jbig2_bin), "-O", os.path.abspath(dst),
                 os.path.abspath(src)],
                check=True, capture_output=True, timeout=180)
            thr = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)
            if thr is not None and thr.shape == gray.shape:
                return thr < 128
            log.warning("jbig2enc threshold dump missing/mismatched; "
                        "using cv2 fallback")
        except Exception as e:
            log.warning("jbig2enc threshold failed (%s); using cv2 "
                        "fallback", e)
        finally:
            for f in (src, dst):
                try:
                    if f and os.path.isfile(f):
                        os.remove(f)
                except OSError:
                    pass
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 10) > 0


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


def _core_color(img_bgr: np.ndarray, mask: np.ndarray) -> tuple | None:
    """Median BGR of the darkest 30% pixels under `mask`.

    Antialiasing fringes are lighter than the stroke; the dark core carries
    the true ink color (e.g. saturated blue instead of washed-out blue).
    """
    vals = img_bgr[mask]
    if len(vals) == 0:
        return None
    lum = vals.astype(np.int32).sum(axis=1)
    cut = np.percentile(lum, 30)
    core = vals[lum <= cut]
    med = np.median(core, axis=0)
    return tuple(int(v) for v in med)


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
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((3, 3), np.uint8))
    fg = _drop_specks(fg.astype(bool))

    if float(fg.mean()) < MIN_COVERAGE:
        return img_bgr.copy(), None

    # ---- per-component ink color -> color groups -------------------------
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=8)
    groups: dict = {"black": []}
    for i in range(1, n):
        c = _core_color(img_bgr, labels == i)
        if c is None:
            continue
        if max(c) - min(c) <= BLACK_SAT_MAX:
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
        fill = _core_color(img_bgr, mask) or (0, 0, 0)
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


def encode_mrc(img_bgr: np.ndarray, jpeg_quality: int,
               bg_gray: bool = False, jbig2_bin: str | None = None,
               workdir: str | None = None):
    """Encode a page as MRC layers.

    Returns (background_jpeg, layers, height, width) where layers is a list
    of (packed_1bit_bytes, fill_bgr); or (None, [], 0, 0) when the page
    carries no meaningful text (caller falls back to whole-page JPEG).
    """
    bg, layers = split_mrc(img_bgr, jbig2_bin, workdir)
    if not layers:
        return None, [], 0, 0

    if bg_gray:
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    bg_quality = min(int(jpeg_quality), 85)
    ok, bg_jpg = cv2.imencode(
        ".jpg", bg, [int(cv2.IMWRITE_JPEG_QUALITY), bg_quality])
    if not ok:
        raise RuntimeError("MRC background JPEG encode failed")

    # ImageMask stencil, 1-bit row-packed, ink = 0 bit (verified polarity:
    # /Decode [0 1] paints zero samples). Returned RAW: it MUST be stored
    # via doc.update_stream(xref, packed, compress=1) — letting PyMuPDF do
    # its own deflate. A hand-zlib'd stream with a manual /Filter entry
    # triggers a MuPDF render bug (page comes out all black).
    packed_layers = [(np.packbits((~stencil).astype(np.uint8),
                                  axis=1).tobytes(), fill)
                     for stencil, fill in layers]
    h, w = layers[0][0].shape
    return bg_jpg.tobytes(), packed_layers, h, w


def embed_mrc(page, bg_jpg: bytes, layers, height: int, width: int):
    """Insert the background JPEG plus the per-color stencil masks on `page`.

    layers: list of (packed, fill_bgr) from encode_mrc. packed is the RAW
    1-bit row-packed stencil (ink = 0 bit), stored with compress=1 so
    PyMuPDF applies its own FlateDecode — the only variant MuPDF renders
    correctly (hand-compressed streams with a manual /Filter render black).
    """
    from src.jbig2 import _paint_jbig2

    page.insert_image(page.rect, stream=bg_jpg)

    doc = page.parent
    for packed, (b, g, r) in layers:
        xref = doc.get_new_xref()
        doc.update_object(
            xref,
            f"<< /Type /XObject /Subtype /Image /Width {width} "
            f"/Height {height} /ImageMask true /BitsPerComponent 1 "
            f"/Decode [0 1] >>")
        doc.update_stream(xref, packed, compress=1)
        _paint_jbig2(page, xref, page.rect,
                     fill=f"{r / 255.0:.4f} {g / 255.0:.4f} "
                          f"{b / 255.0:.4f} rg ")
