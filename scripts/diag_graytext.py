"""Diagnose gray-text misclassification.

Hypothesis: scanned paper background (a large, light, low-variance, page-edge-
touching mid-gray region) is being counted as a "grayscale content blob" by
classify_page, so text pages fall through to gray-text and get encoded as a
whole-page gray JPEG (~14x the size of a 1-bit BW page).

This script replays the *real* pipeline up to classification
    render -> DocRes restore -> SR -> classify
for a set of sampled pages, then reports for each page:
  * the verdict from the current classifier
  * the top content-gray blobs: area%, mean gray, std, whether it touches the
    page border
  * the estimated paper level and the blob's darkness relative to it
  * a "strict" verdict that additionally requires a blob to be meaningfully
    darker than the paper and to carry internal structure

The gap between current and strict verdicts estimates how much volume is
being wasted.

Usage:
    env/Scripts/python.exe scripts/diag_graytext.py --pdf input/x.pdf \
        --pages 20,45,70,100,130,160,190,220,250,280,310,335
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.classify import _color_mask, _text_mask, classify_page  # noqa: E402
from src.pipeline import EnhancePipeline  # noqa: E402
from src.render import render_pdf_pages  # noqa: E402

BLOB_MIN = 0.0008


def analyze(img_bgr: np.ndarray) -> dict:
    """Recompute the classifier's internals and profile content-gray blobs."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    color = _color_mask(img_bgr)
    color_ratio = float(color.mean())
    color_blob = 0.0
    n, _l, stats, _ = cv2.connectedComponentsWithStats(color.astype(np.uint8), 8)
    if n > 1:
        color_blob = float(stats[1:, cv2.CC_STAT_AREA].max()) / color.size

    b, g, r = (img_bgr[:, :, k].astype(int) for k in range(3))
    ch_max = np.maximum(np.maximum(b, g), r)
    ch_min = np.minimum(np.minimum(b, g), r)
    achromatic = (ch_max - ch_min) <= 12
    mid_gray = achromatic & (gray > 40) & (gray < 220)
    gray_ratio = float(mid_gray.mean())

    ink = gray < 100
    near_ink = cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    content_gray = mid_gray & ~near_ink

    text_ratio = float(_text_mask(gray).mean())

    # paper level: the bright plateau (background of the scanned page)
    paper = float(np.percentile(gray, 90))
    ink_level = float(np.median(gray[ink])) if ink.any() else 0.0

    # profile the largest content-gray blobs
    n2, labels, cstats, _ = cv2.connectedComponentsWithStats(
        content_gray.astype(np.uint8), 8)
    blobs = []
    if n2 > 1:
        order = np.argsort(-cstats[1:, cv2.CC_STAT_AREA])[:3] + 1
        for li in order:
            area = int(cstats[li, cv2.CC_STAT_AREA])
            if area == 0:
                continue
            x0 = int(cstats[li, cv2.CC_STAT_LEFT])
            y0 = int(cstats[li, cv2.CC_STAT_TOP])
            bw = int(cstats[li, cv2.CC_STAT_WIDTH])
            bh = int(cstats[li, cv2.CC_STAT_HEIGHT])
            m = labels == li
            vals = gray[m]
            margin = max(4, int(0.01 * min(h, w)))
            touches = (x0 <= margin or y0 <= margin or
                       x0 + bw >= w - margin or y0 + bh >= h - margin)
            blobs.append({
                "area_pct": area / gray.size * 100.0,
                "bbox": (x0, y0, bw, bh),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "darkness": paper - float(vals.mean()),
                "touches_border": bool(touches),
            })

    # current classifier verdict
    info = classify_page(img_bgr)
    ptype = info["type"]

    # strict verdict: only treat content-gray as real figure if the blob is
    # clearly darker than paper AND has internal structure (not a flat wash).
    strict_fig = any(bd["darkness"] > 35 and bd["std"] > 12 for bd in blobs)
    has_color = color_blob > BLOB_MIN
    if has_color:
        strict = "mixed" if text_ratio > 0.03 else ("color-text" if text_ratio > 0.01 else "color-image")
    elif strict_fig:
        strict = "gray-text" if text_ratio > 0.005 else "gray-image"
    else:
        strict = "bw-text"

    return {
        "ptype": ptype, "strict": strict,
        "size": (w, h),
        "color_ratio": color_ratio, "color_blob": color_blob,
        "gray_ratio": gray_ratio, "text_ratio": text_ratio,
        "paper": paper, "ink": ink_level,
        "blobs": blobs,
        "content_gray": content_gray,
        "gray": gray,
    }


def overlay(r: dict, max_w=460) -> np.ndarray:
    """Build a small side-by-side: gray + content-gray highlighted in red."""
    gray = r["gray"]
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    m = r["content_gray"]
    vis[m] = (0.4 * vis[m] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    for bd in r["blobs"]:
        x0, y0, bw, bh = bd["bbox"]
        cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + bh), (0, 200, 0), 8)
    s = max_w / vis.shape[1]
    return cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", required=True,
                    help="1-based, comma separated, e.g. 20,45,70")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "config.yaml"))
    ap.add_argument("--outdir", default=os.path.join(PROJECT_ROOT, "output", "diag_graytext"))
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pages = [int(x) - 1 for x in args.pages.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[diag] loading pipeline (config: SR {cfg['super_resolution'].get('backend')} "
          f"x{cfg['super_resolution'].get('scale')}, DocRes {cfg['document_restoration'].get('task')})...")
    t0 = time.time()
    pipe = EnhancePipeline(cfg, PROJECT_ROOT)
    print(f"[diag] models ready in {time.time() - t0:.1f}s")

    rows = []
    strips = []
    for i in pages:
        t = time.time()
        p = render_pdf_pages(args.pdf, cfg["render"].get("dpi", 300),
                             int(cfg["render"].get("max_dpi", 450)),
                             page_range=(i, i))[0]
        img = p["image"]
        pre = img.shape[:2]
        img = pipe.restore.process(img)
        if pipe.routing == "by-type":
            from src.classify import coarse_type
            ct = coarse_type(img)
            img = (pipe.sr_image if ct == "image" else pipe.sr).process(img)
            routed = "swinir" if ct == "image" else "realesrgan"
        else:
            img = pipe.sr.process(img)
            routed = cfg["super_resolution"].get("backend", "-")
        r = analyze(img)
        r["index"] = i + 1
        r["routed"] = routed
        r["pre"] = pre
        rows.append(r)

        strip = overlay(r)
        label = np.zeros((34, strip.shape[1], 3), np.uint8)
        txt = (f"p{r['index']} {r['ptype']} -> {r['strict']}  "
               f"sr={r['pre'][1]}x{r['pre'][0]}->{r['size'][1]}x{r['size'][0]}  "
               f"routed={routed}")
        cv2.putText(label, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        strips.append(np.vstack([label, strip]))
        print(f"[diag] p{r['index']:<4} {r['ptype']:<11} -> strict={r['strict']:<11} "
              f"text={r['text_ratio']:.4f} color={r['color_ratio']:.4f} "
              f"paper={r['paper']:.0f} ink={r['ink']:.0f} blobs={len(r['blobs'])} "
              f"({time.time() - t:.1f}s)")

    # contact sheet
    if strips:
        hmax = max(s.shape[0] for s in strips)
        wmax = max(s.shape[1] for s in strips)
        cols = 4
        rowsn = (len(strips) + cols - 1) // cols
        sheet = np.full((rowsn * (hmax + 8), cols * (wmax + 8), 3), 24, np.uint8)
        for k, s in enumerate(strips):
            rr, cc = divmod(k, cols)
            y = rr * (hmax + 8)
            x = cc * (wmax + 8)
            sheet[y:y + s.shape[0], x:x + s.shape[1]] = s
        out = os.path.join(args.outdir, "graytext_sheet.png")
        cv2.imwrite(out, sheet)
        print(f"[diag] contact sheet -> {out}")

    # summary
    cur_gray = sum(1 for r in rows if r["ptype"] in ("gray-text", "gray-image"))
    strict_gray = sum(1 for r in rows if r["strict"] in ("gray-text", "gray-image"))
    cur_bw = sum(1 for r in rows if r["ptype"] == "bw-text")
    strict_bw = sum(1 for r in rows if r["strict"] == "bw-text")
    print("\n=== summary (sampled %d pages) ===" % len(rows))
    print(f"  gray*: current {cur_gray}  ->  strict {strict_gray}")
    print(f"  bw-text: current {cur_bw}  ->  strict {strict_bw}")
    print("  blob details (area% / mean / std / darkness=paper-mean / border):")
    for r in rows:
        for bd in r["blobs"]:
            print(f"    p{r['index']:<4} {bd['area_pct']:6.2f}%  mean={bd['mean']:6.1f} "
                  f"std={bd['std']:5.1f}  dark={bd['darkness']:6.1f}  "
                  f"border={bd['touches_border']}")


if __name__ == "__main__":
    main()
