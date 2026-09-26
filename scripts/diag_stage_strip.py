"""Stage-by-stage visual diagnosis: where does the 'weird' look come from?

Stages: degraded source -> DocRes -> SR x4 (grayscale) -> final MRC PDF.
Same paper region in every row, normalized to the same display width.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from src.backends import available_backends
from src.pipeline import _build_stage_backend
from src.gpu import detect_device

SRC = r"input/计算机组成原理_第17页_彩色测试_扫描降质.png"
PDF = r"output/计算机组成原理_p17_彩色增强.pdf"
OUT = r"output/diag_stage_strip_p17.png"
Y0, Y1, X0, X1 = 0.445, 0.545, 0.04, 0.62  # the 运算器/控制器/存储器 line
TW = 1400

FONT = ImageFont.truetype(r"tools/fonts/simsun.ttc", 34)


def label(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    d.text((10, 6), text, fill=(200, 30, 30), font=FONT)
    return np.array(pil)[:, :, ::-1]


def to_w(img: np.ndarray, w: int) -> np.ndarray:
    h = img.shape[0] * w // img.shape[1]
    interp = cv2.INTER_AREA if img.shape[1] >= w else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def crop_rel(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(Y0 * h):int(Y1 * h), int(X0 * w):int(X1 * w)]


src = np.array(Image.open(SRC).convert("RGB"))[:, :, ::-1]
device = detect_device("auto")
regs = available_backends()
restore = _build_stage_backend(regs["restore"], "docres", ".",
                              {"task": "appearance", "max_size": 1024},
                              device, fp16=True)
sr = _build_stage_backend(regs["sr"], "realesrgan-general", ".",
                          {"scale": 4, "tile": 384}, device, fp16=True)

docres = restore.process(src.copy())
sr_out = sr.process(docres.copy())

rows = [
    (label(to_w(crop_rel(src), TW), "1. 源（150dpi 合成降质）"), 3),
    (label(to_w(crop_rel(docres), TW), "2. DocRes 后（同分辨率）"), 3),
    (label(to_w(crop_rel(sr_out), TW), "3. SR x4 后（原始灰度，未二值化）"), 3),
]
# final PDF render, native pixels
d = fitz.open(PDF)
page = d[0]
pw, ph = page.rect.width, page.rect.height
clip = fitz.Rect(X0 * pw, Y0 * ph, X1 * pw, Y1 * ph)
zoom = TW / clip.width
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip,
                      colorspace=fitz.csRGB, alpha=False)
rend = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
d.close()
if rend.shape[1] != TW:
    rend = rend[:, :TW] if rend.shape[1] > TW else cv2.copyMakeBorder(
        rend, 0, 0, 0, TW - rend.shape[1], cv2.BORDER_REPLICATE)
rows.append((label(rend[:, :, ::-1], "4. 最终 PDF（MRC 二值 stencil）"), 3))

gap = np.full((8, TW, 3), 255, np.uint8)
strip = np.vstack([r for r, _ in rows])
cv2.imwrite(OUT, strip)
print("saved", OUT, strip.shape)
