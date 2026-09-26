"""CER evaluation on real scanned books (no ground truth).

Real scans have no "correct text" to compare against, so an absolute CER is
impossible without manual transcription. Instead we measure what enhancement
*does to OCR quality* via two objective, engine-independent signals on the
same pages, all OCR'd at a fixed input size (long side ~2000px):

  - char_count : how many characters OCR recovers (fuzzy glyphs are dropped)
  - conf_mean  : mean recognition confidence over all detected boxes (clearer
                 glyphs -> higher confidence)

Higher = better on both. This is the honest proxy for "did enhancement make
the page more OCR-readable" when no ground truth exists.

Usage:  env\\Scripts\\python.exe scripts\\cer_real.py <pdf> [page_index ...]
        (default pages: 3, 11, 21 — 1-based)
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.backends import available_backends
from src.backends.base import resolve_weights
from src.gpu import detect_device
from src.ocr import RapidOCRBackend
from src.render import render_pdf_pages

MODELS = os.path.join(ROOT, "models")
OCR_LONG_SIDE = 2000

CONFIGS = {
    "baseline":   ("none", "none"),
    "docres":     ("docres", "none"),
    "realesrgan": ("docres", "realesrgan-general"),
}


def build(regs, device, restore_name, sr_name):
    def make(registry, name):
        if name == "none":
            return registry["none"]()
        cls = registry[name]
        w = resolve_weights(ROOT, cls.default_weights)
        return cls.create(device, w, dict(cls.default_params), fp16=True)
    return make(regs["restore"], restore_name), make(regs["sr"], sr_name)


def ocr_stats(engine, img_bgr):
    h, w = img_bgr.shape[:2]
    if max(h, w) > OCR_LONG_SIDE:
        s = OCR_LONG_SIDE / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
    boxes = engine.recognize(img_bgr)
    chars = sum(len(t) for _, t, _ in boxes)
    scores = [float(b[2]) for b in boxes]
    conf = float(np.mean(scores)) if scores else 0.0
    return len(boxes), chars, conf


def main():
    pdf = sys.argv[1]
    idxs = [int(x) - 1 for x in sys.argv[2:]] if len(sys.argv) > 2 else [2, 10, 20]

    device = detect_device("auto")
    regs = available_backends()
    ocr = RapidOCRBackend(os.path.join(MODELS, "rapidocr"), use_cuda=False)

    pages = render_pdf_pages(pdf, 200)
    print(f"pages: {len(pages)}  sampled: {[i + 1 for i in idxs]}")

    # load backends once
    backends = {k: build(regs, device, r, s) for k, (r, s) in CONFIGS.items()}

    print(f"\n{'config':<12} {'page':>4} {'boxes':>6} {'chars':>7} {'conf':>7}")
    print("-" * 40)
    agg = {k: {"chars": [], "conf": []} for k in CONFIGS}
    for k, (restore, sr) in backends.items():
        for i in idxs:
            img = pages[i]["image"].copy()
            img = restore.process(img)
            img = sr.process(img)
            n, chars, conf = ocr_stats(ocr, img)
            agg[k]["chars"].append(chars)
            agg[k]["conf"].append(conf)
            print(f"{k:<12} {i + 1:>4} {n:>6} {chars:>7} {conf:>7.3f}")

    print("-" * 40)
    print("\n=== 汇总（均值）===")
    for k in CONFIGS:
        mc = np.mean(agg[k]["chars"])
        mf = np.mean(agg[k]["conf"])
        print(f"{k:<12} 平均字符数={mc:7.1f}  平均置信度={mf:.3f}")


if __name__ == "__main__":
    main()
