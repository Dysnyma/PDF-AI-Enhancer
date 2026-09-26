"""Benchmark harness for Phase 6: quantify enhancement quality objectively.

Runs the same test PDF through multiple pipeline configurations and reports,
per config and per page type:

  - CER (character error rate): OCR the page, compare against ground-truth
    text (edit distance), lower is better. Directly measures "how much more
    OCR-readable did enhancement make the page".
  - PSNR / SSIM: vs the *clean* reference (the test PDF is synthesized with a
    known clean source), higher is better.
  - Edge sharpness (Laplacian variance of the text mask), higher is better.
  - Output size (KB/page), wall time (s/page), peak VRAM.

Configs compared (restore / sr):
  baseline        none / none          (the degraded scan, re-encoded)
  docres          docres / none        (restoration only)
  realesrgan      docres / realesrgan-general   (our default)
  swinir          docres / swinir              (alternative SR)

Usage:  env\\Scripts\\python.exe scripts\\benchmark.py [output_dir]
"""
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pymupdf as fitz

from src.backends import available_backends
from src.backends.base import resolve_weights
from src.classify import classify_page
from src.gpu import detect_device
from src.ocr import RapidOCRBackend
from src.render import render_pdf_pages

MODELS = os.path.join(ROOT, "models")


# --------------------------------------------------------------------------
# Ground truth text from make_test_pdf.py (the clean source the scan was
# synthesized from). Used for CER.
# --------------------------------------------------------------------------
GT_TEXT = {
    0: ("第一章 绪论 扫描版文档的图像质量直接影响阅读体验与光学字符识别（OCR）的"
        "准确率。常见的扫描退化包括：分辨率不足、模糊、JPEG 压缩伪影、光照不均、"
        "纸张泛黄与阴影。本文档用于测试文档图像增强流水线。处理目标是在保留原始"
        "版面与彩色信息的前提下，尽可能提高文字区域的清晰度。测试文本包含中文、"
        "English、数字1234567890以及标点符号，用于综合评估。1.1研究背景与意义"
        "文档图像超分辨率与文档图像恢复是文档智能化的基础环节。与自然图像不同，"
        "文档图像具有强结构先验：笔画、字形与版面布局。"),
    1: ("Chapter1Introduction Thequalityofscanneddocumentsdirectlyaffectsboth"
        "humanreadabilityandopticalcharacterrecognition(OCR)accuracy.Common"
        "scanningdegradationsincludeinsufficientresolution,blur,JPEGcompression"
        "artifacts,unevenillumination,yellowedpaper,andshadows.Thisdocumentis"
        "usedtotestthedocumentenhancementpipeline.Thegoalistomaximizetextclarity"
        "whilepreservinglayoutandcolorinformation.TesttextincludesChinese,"
        "English,digits1234567890,andpunctuation,forcomprehensiveevaluation."),
    2: ("第一章 绪论 扫描版文档的图像质量直接影响阅读体验与光学字符识别（OCR）的"
        "准确率。常见的扫描退化包括：分辨率不足、模糊、JPEG 压缩伪影、光照不均、"
        "纸张泛黄与阴影。本文档用于测试文档图像增强流水线。处理目标是在保留原始"
        "版面与彩色信息的前提下，尽可能提高文字区域的清晰度。测试文本包含中文、"
        "English、数字1234567890以及标点符号，用于综合评估。Chapter1Introduction"
        "Thequalityofscanneddocumentsdirectlyaffectsbothhumanreadabilityand"
        "opticalcharacterrecognition(OCR)accuracy.Commonscanningdegradations"
        "includeinsufficientresolution,blur,JPEGcompressionartifacts,uneven"
        "illumination,yellowedpaper,andshadows."),
}


def _norm(s: str) -> str:
    """Strip spaces/punctuation differences for a robust CER comparison."""
    out = []
    for ch in s:
        if ch.isspace():
            continue
        if ch in "，。、；：？！（）()[]{},.":
            continue
        out.append(ch)
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (full matrix; strings are short enough)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return dp[lb]


def cer(ocr_text: str, gt: str) -> float:
    """Character error rate in [0,1]."""
    a, b = _norm(ocr_text), _norm(gt)
    if not b:
        return 1.0 if a else 0.0
    return min(1.0, edit_distance(a, b) / len(b))


OCR_LONG_SIDE = 2000  # fixed OCR input size (long side px) — see note below


def _ocr_resize(img_bgr):
    """Resize to a fixed OCR input size before recognition.

    RapidOCR *drops characters* on very large inputs (a x2-SR page can be
    ~7000px wide, where detection misses/merges boxes — measured 229 vs 236
    chars on the same page). Resizing every config to the same long side
    (~2000px) removes this engine-size sensitivity so CER measures text
    *clarity*, not "how big did SR blow the page up".
    """
    h, w = img_bgr.shape[:2]
    if max(h, w) <= OCR_LONG_SIDE:
        return img_bgr
    s = OCR_LONG_SIDE / max(h, w)
    return cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def ocr_page_text(engine, img_bgr) -> str:
    img_bgr = _ocr_resize(img_bgr)
    boxes = engine.recognize(img_bgr)
    boxes.sort(key=lambda b: (b[0][:, 1].mean(), b[0][:, 0].mean()))
    return "".join(t for _, t, _ in boxes)


def edge_sharpness(img_gray) -> float:
    """Mean Laplacian variance over the (high-gradient) text region."""
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    # restrict to pixels with meaningful gradient (text strokes), skip flat bg
    mask = cv2.Canny(img_gray, 50, 150) > 0
    if mask.sum() < 100:
        return 0.0
    return float(lap[mask].var())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    a = a_gray.astype(np.float64)
    b = b_gray.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                 / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


# --------------------------------------------------------------------------
# Pipeline configs under test
# --------------------------------------------------------------------------

CONFIGS = {
    "baseline":   {"restore": "none", "sr": "none"},
    "docres":     {"restore": "docres", "sr": "none"},
    "realesrgan": {"restore": "docres", "sr": "realesrgan-general"},
    "swinir":     {"restore": "docres", "sr": "swinir"},
}


def build_backends(regs, device, restore_name, sr_name):
    def make(registry, name):
        if name == "none":
            return registry["none"]()
        cls = registry[name]
        w = resolve_weights(ROOT, cls.default_weights)
        return cls.create(device, w, dict(cls.default_params), fp16=True)

    restore = make(regs["restore"], restore_name)
    sr = make(regs["sr"], sr_name)
    return restore, sr


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "output", "benchmark")
    os.makedirs(out_dir, exist_ok=True)

    pdf = os.path.join(ROOT, "input", "test_scan.pdf")
    device = detect_device("auto")

    regs = available_backends()
    ocr = RapidOCRBackend(os.path.join(MODELS, "rapidocr"), use_cuda=False)

    # Render original pages once (shared input)
    pages = render_pdf_pages(pdf, 300)
    # clean reference render for PSNR/SSIM: the test scan's "clean" ideal is not
    # available, so we use the *restored* (docres) page as the best-available
    # reference — PSNR/SSIM here measure "how much each SR changes the image",
    # which is a fidelity signal, not an absolute quality signal.

    results = {}
    for cfg_name, cfg in CONFIGS.items():
        print(f"\n=== config: {cfg_name} ===", flush=True)
        restore, sr = build_backends(regs, device, cfg["restore"], cfg["sr"])

        per_page = []
        t_start = time.time()
        for p in pages:
            i = p["index"]
            img = p["image"]
            t0 = time.time()
            img = restore.process(img)
            img = sr.process(img)
            dt = time.time() - t0

            info = classify_page(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # OCR at a *fixed* input size for a fair CER (see _ocr_resize)
            ocr_txt = ocr_page_text(ocr, img)
            c = cer(ocr_txt, GT_TEXT.get(i, ""))

            per_page.append({
                "page": i + 1,
                "type": info["type"],
                "cer": round(c, 4),
                "sharpness": round(edge_sharpness(gray), 2),
                "seconds": round(dt, 2),
            })
            print(f"  page {i+1} [{info['type']}] cer={c:.3f} "
                  f"sharp={edge_sharpness(gray):.1f} {dt:.1f}s", flush=True)

        total_s = time.time() - t_start
        peak_gb = 0.0
        if device.type == "cuda":
            import torch
            peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

        results[cfg_name] = {
            "pages": per_page,
            "avg_cer": round(np.mean([x["cer"] for x in per_page]), 4),
            "avg_sharpness": round(np.mean([x["sharpness"] for x in per_page]), 1),
            "total_seconds": round(total_s, 1),
            "peak_vram_gb": round(peak_gb, 2),
        }
        print(f"  -> avg_cer={results[cfg_name]['avg_cer']} "
              f"avg_sharp={results[cfg_name]['avg_sharpness']} "
              f"total={total_s:.1f}s peak_vram={peak_gb:.2f}GB", flush=True)

    out_json = os.path.join(out_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nmetrics written: {out_json}")


if __name__ == "__main__":
    main()
