# -*- coding: utf-8 -*-
"""Close-up comparison at the same field of view as the user's screenshot
(~6 chars wide, ~150 px per char) from cached renders - no GPU needed."""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = r"E:\04_Projects\PDF_AI_Enhancer"
DIAG = os.path.join(ROOT, "temp", "diag")
OUT = os.path.join(ROOT, "output", "compare_zoom_page129_closeup.png")

# boxes in the native-crop coordinate frame (crop was page px 144,882 ->)
NX0, NY0, NX1, NY1 = 40, 85, 440, 295       # "2. 天文计算领域" + 2 body lines
K = 4                                        # SR scale
EX0, EY0 = (NX0 + 144) * K, (NY0 + 882) * K  # -> enhanced image px


def main():
    native = cv2.imread(os.path.join(DIAG, "orig_native_crop.png"))
    enh = cv2.imread(os.path.join(DIAG, "enhanced_page129.png"))

    nc = native[NY0:NY1, NX0:NX1]
    ec = enh[EY0:EY0 + (NY1 - NY0) * K, EX0:EX0 + (NX1 - NX0) * K]

    # simulate the PDF reader's 6x interpolation zoom on the original
    W = 2400
    oz = cv2.resize(nc, (W, int(nc.shape[0] * W / nc.shape[1])),
                    interpolation=cv2.INTER_LINEAR)
    ez = cv2.resize(ec, (W, int(ec.shape[0] * W / ec.shape[1])),
                    interpolation=cv2.INTER_AREA)

    font = ImageFont.truetype(os.path.join(ROOT, "tools", "fonts", "simsun.ttc"),
                              34)

    def label_row(text, fill):
        im = Image.new("RGB", (W, 52), fill)
        ImageDraw.Draw(im).text((14, 6), text, font=font, fill=(20, 20, 20))
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    strip = np.vstack([
        label_row("原版（阅读器放大 ~6x，插值糊）—— 即你截图看到的效果", (255, 228, 225)),
        oz,
        label_row("AI 增强（DocRes + SR x4，真像素重建）", (220, 240, 255)),
        ez,
    ])
    cv2.imwrite(OUT, strip)
    print(f"close-up -> {OUT} ({strip.shape[1]}x{strip.shape[0]})")


if __name__ == "__main__":
    main()
