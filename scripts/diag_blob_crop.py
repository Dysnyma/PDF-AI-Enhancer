"""Crop and dump the largest content-gray blobs at native resolution.

The contact sheet's red overlay clashes with the book's own red design, so this
dumps each top blob region as a clean native-resolution crop so we can see
exactly what the classifier is reacting to (real figure vs. page furniture vs.
scan artifact introduced by SR).

Usage:
    env/Scripts/python.exe scripts/diag_blob_crop.py --pdf input/x.pdf --pages 45,100,130,335
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import EnhancePipeline  # noqa: E402
from src.render import render_pdf_pages  # noqa: E402


def content_gray_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = (img_bgr[:, :, k].astype(int) for k in range(3))
    ch_max = np.maximum(np.maximum(b, g), r)
    ch_min = np.minimum(np.minimum(b, g), r)
    achromatic = (ch_max - ch_min) <= 12
    mid_gray = achromatic & (gray > 40) & (gray < 220)
    ink = gray < 100
    near_ink = cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return gray, mid_gray & ~near_ink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "config.yaml"))
    ap.add_argument("--outdir", default=os.path.join(PROJECT_ROOT, "output", "diag_graytext"))
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.outdir, exist_ok=True)
    pipe = EnhancePipeline(cfg, PROJECT_ROOT)

    for pno in [int(x) for x in args.pages.split(",") if x.strip()]:
        i = pno - 1
        p = render_pdf_pages(args.pdf, cfg["render"].get("dpi", 300),
                             int(cfg["render"].get("max_dpi", 450)),
                             page_range=(i, i))[0]
        restored = pipe.restore.process(p["image"])
        img = pipe.sr.process(restored)

        # side-by-side: restored (pre-SR) vs enhanced, same region
        gray_e, mask_e = content_gray_mask(img)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_e.astype(np.uint8), 8)
        if n <= 1:
            print(f"p{pno}: no blobs")
            continue
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:args.top] + 1
        gray_r = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        scale = img.shape[1] / restored.shape[1]

        tiles = []
        for rank, li in enumerate(order):
            x0 = int(stats[li, cv2.CC_STAT_LEFT]); y0 = int(stats[li, cv2.CC_STAT_TOP])
            bw = int(stats[li, cv2.CC_STAT_WIDTH]); bh = int(stats[li, cv2.CC_STAT_HEIGHT])
            area = int(stats[li, cv2.CC_STAT_AREA])
            pad = 24
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(img.shape[1], x0 + bw + pad), min(img.shape[0], y0 + bh + pad)
            # crop must be >= 200px for a legible zoom
            if cx1 - cx0 < 200:
                cx0 = max(0, (cx0 + cx1) // 2 - 100); cx1 = cx0 + 200
            if cy1 - cy0 < 120:
                cy0 = max(0, (cy0 + cy1) // 2 - 60); cy1 = cy0 + 120

            enc = img[cy0:cy1, cx0:cx1].copy()
            encm = mask_e[cy0:cy1, cx0:cx1]
            enc[encm] = (0.35 * enc[encm] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
            cv2.rectangle(enc, (x0 - cx0, y0 - cy0), (x0 - cx0 + bw, y0 - cy0 + bh), (0, 220, 0), 3)

            rx0, ry0 = int(cx0 / scale), int(cy0 / scale)
            rx1, ry1 = int(cx1 / scale), int(cy1 / scale)
            pre = cv2.cvtColor(gray_r[ry0:ry1, rx0:rx1], cv2.COLOR_GRAY2BGR)

            h = max(enc.shape[0], pre.shape[0])
            w = enc.shape[1]
            row = np.full((h + 26, w * 2 + 8, 3), 30, np.uint8)
            row[26:26 + enc.shape[0], :w] = enc
            row[26:26 + pre.shape[0], w + 8:w + 8 + pre.shape[1]] = pre
            v = gray_e[labels == li]
            meta = (f"p{pno} rank{rank+1} area={area/gray_e.size*100:.2f}% "
                    f"mean={v.mean():.0f} std={v.std():.0f} box={bw}x{bh}@({x0},{y0})")
            cv2.putText(row, meta, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(row, "ENHANCED(mask=magenta)", (w + 14, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1, cv2.LINE_AA)
            tiles.append(row)

        wmax = max(t.shape[1] for t in tiles)
        out = np.full((sum(t.shape[0] + 6 for t in tiles), wmax, 3), 30, np.uint8)
        y = 0
        for t in tiles:
            out[y:y + t.shape[0], :t.shape[1]] = t
            y += t.shape[0] + 6
        path = os.path.join(args.outdir, f"blob_p{pno}.png")
        cv2.imwrite(path, out)
        print(f"p{pno}: {len(tiles)} blobs -> {path}")


if __name__ == "__main__":
    main()
