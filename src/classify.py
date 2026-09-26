"""Page type classification.

Full taxonomy (phase 4):

    bw-text      pure black-and-white text page (no gray, no color)
    gray-text    grayscale text page (halftone/anti-aliased text, no color)
    color-text   colored text page (little image area)
    color-image  color image / photo page (little text)
    mixed        text + image mixed page (both significant)
    formula      formula-dense page (heuristic, optional)

The classifier returns a dict with the page type plus the features used to
decide it, so downstream stages (compression, restoration) can branch on the
raw signals rather than the coarse label.
"""
import cv2
import numpy as np


def _color_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    mask = (sat > 45).astype(np.uint8)
    # Real color content (figures, screenshots, covers) forms *connected*
    # regions; RGB scan noise / chroma artifacts amplified by SR are isolated
    # specks. An opening removes the specks so near-BW text pages are not
    # misread as color pages (which would force 3-channel JPEG and ~4x size).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask > 0


def _text_mask(img_gray: np.ndarray) -> np.ndarray:
    """Estimate where text-like (high-gradient) pixels are.

    Text is characterized by strong local contrast. We use a morphological
    gradient: dilation - erosion, thresholded.
    """
    k = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, k)
    # blur to merge nearby strokes into coherent text blocks
    grad = cv2.GaussianBlur(grad, (5, 5), 0)
    return grad > 40


def _max_blob_ratio(mask: np.ndarray) -> float:
    """Area of the largest connected component, as a page fraction.

    Content regions (photos, figure fills, screenshot UIs) form large blobs;
    text antialiasing fringes and scan noise are specks. This is far more
    robust than a page-wide mean, which dilutes a local figure into nothing.
    """
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / mask.size


def classify_page(img_bgr: np.ndarray) -> dict:
    """Classify a BGR page image.

    Returns {"type": str, "color_ratio": float, "text_ratio": float,
             "gray_ratio": float}
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) color signal
    color = _color_mask(img_bgr)
    color_ratio = float(color.mean())
    color_blob = _max_blob_ratio(color)

    # 2) "real grayscale" signal: pixels whose channels are nearly equal
    #    (i.e. NOT colored), measuring whether the page has mid-gray tones.
    b, g, r = img_bgr[:, :, 0].astype(int), img_bgr[:, :, 1].astype(int), img_bgr[:, :, 2].astype(int)
    ch_max = np.maximum(np.maximum(b, g), r)
    ch_min = np.minimum(np.minimum(b, g), r)
    achromatic = (ch_max - ch_min) <= 12  # channels within ~5% -> gray
    # a pixel is "mid-gray" if achromatic and not near-black/white
    mid_gray = achromatic & (gray > 40) & (gray < 220)
    gray_ratio = float(mid_gray.mean())

    # Mid-gray that hugs black strokes is antialiasing fringe (safe to
    # binarize); only mid-gray AWAY from any stroke is figure/photo content
    # (binarizing would destroy it). A whole text line's fringe forms one
    # big blob, so plain blob-size alone cannot tell the two apart.
    ink = gray < 100
    near_ink = cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    content_gray = mid_gray & ~near_ink
    gray_blob = _max_blob_ratio(content_gray)

    # 3) text signal
    text = _text_mask(gray)
    text_ratio = float(text.mean())

    # A connected blob >= 0.08% of the page (~190x190 px at 5756x8020) is
    # real content; anything smaller is antialiasing fringe or noise.
    BLOB_MIN = 0.0008
    has_color = color_blob > BLOB_MIN
    has_gray_fig = gray_blob > BLOB_MIN

    # Decision
    if has_color:
        # meaningful color region: distinguish text vs image vs mixed
        if text_ratio > 0.01:
            ptype = "mixed" if text_ratio > 0.03 else "color-text"
        else:
            ptype = "color-image"
    elif has_gray_fig:
        # grayscale figure/photo region: binarizing would destroy it
        ptype = "gray-text" if text_ratio > 0.005 else "gray-image"
    else:
        # no content blobs: pure text page, safe to binarize (BW).
        # The small gray_ratio is just antialiasing on glyph edges.
        ptype = "bw-text"

    return {
        "type": ptype,
        "color_ratio": color_ratio,
        "text_ratio": text_ratio,
        "gray_ratio": gray_ratio,
    }


def is_binarizable(info: dict) -> bool:
    """Whether a page is safe to binarize (pure BW text)."""
    return info["type"] == "bw-text" and info["color_ratio"] < 0.005


def coarse_type(img_bgr: np.ndarray) -> str:
    """Coarse page-type signal for mixed SR routing (before SR, after restore).

    The full six-way classifier runs *after* SR (restoration/SR change the
    illumination & sharpness that the classifier relies on), but SR model
    selection must happen *before* SR. This chicken-and-egg is resolved by a
    light pre-SR signal that is already reliable after DocRes restoration:

      - "image" -> photo / illustration / figure page (little text) -> SwinIR
        (better on natural-image content, but drifts color on white text pages)
      - "text"  -> text-bearing page -> RealESRGAN (document-trained, no drift)

    The discriminator is the *text stroke* ratio: photo/figure pages have very
    few high-gradient text-like strokes, text pages have many. This mirrors the
    `color-image`/`gray-image` thresholds in `classify_page` (text_ratio < 0.005
    means "little text") so the two stay consistent.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    text_ratio = float(_text_mask(gray).mean())
    return "image" if text_ratio < 0.005 else "text"
