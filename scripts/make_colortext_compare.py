"""Build a fair before/after comparison for the color-text test page.

Lesson (2026-09-26): never render with a hardcoded matrix. The enhanced
PDF's page size varies with the source (this one is 262pt wide, vs 1439pt
for the scanned textbook). Compute zoom from the actual page rect so both
rows show the same paper region at the same display width.
"""
import argparse
import os

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# relative crop of the blue-keyword paragraph
Y0, Y1, X0, X1 = 0.435, 0.545, 0.04, 0.80
TARGET_W = 1500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"input/计算机组成原理_第17页_彩色测试_扫描降质.png")
    ap.add_argument("--pdf", default=r"output/计算机组成原理_p17_彩色增强.pdf")
    ap.add_argument("--out", default=r"output/compare_colortext_p17.png")
    args = ap.parse_args()

    src = np.array(Image.open(args.src).convert("RGB"))
    H, W = src.shape[:2]
    crop_src = src[int(Y0 * H):int(Y1 * H), int(X0 * W):int(X1 * W)]

    d = fitz.open(args.pdf)
    page = d[0]
    pw, ph = page.rect.width, page.rect.height
    # render exactly the crop region at TARGET_W pixels wide
    clip = fitz.Rect(X0 * pw, Y0 * ph, X1 * pw, Y1 * ph)
    zoom = TARGET_W / clip.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip,
                          colorspace=fitz.csRGB, alpha=False)
    rend = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    d.close()

    top = cv2.resize(crop_src[:, :, ::-1],
                     (TARGET_W, int(crop_src.shape[0] * TARGET_W / crop_src.shape[1])),
                     interpolation=cv2.INTER_CUBIC)
    bot = rend[:, :, ::-1]
    if bot.shape[1] != TARGET_W:
        bot = cv2.resize(bot, (TARGET_W, int(bot.shape[0] * TARGET_W / bot.shape[1])),
                         interpolation=cv2.INTER_AREA)

    gap = np.full((14, TARGET_W, 3), 255, np.uint8)
    strip = np.vstack([top, gap, bot])
    cv2.imwrite(args.out, strip)
    print(f"source crop {crop_src.shape[1]}px -> {TARGET_W}px (cubic up, reader-like)")
    print(f"pdf crop rendered at zoom {zoom:.2f} ({pix.width}px native, no fake upscale)")
    print(f"-> {args.out} {strip.shape} {os.path.getsize(args.out) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
