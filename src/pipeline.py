"""Pipeline orchestration: render -> classify -> restore -> SR -> rebuild."""
import logging
import os
import time

import cv2

from src.backends import available_backends
from src.backends.base import resolve_weights
from src.classify import classify_page, coarse_type
from src.render import render_pdf_pages, rebuild_pdf
from src.ocr import (RapidOCRBackend, boxes_to_pdf_text,
                     render_page_for_ocr)

log = logging.getLogger("pdfenhance")


def _build_stage_backend(registry, name, project_root, stage_cfg, device, fp16):
    """Instantiate a backend by name using its own defaults.

    The pipeline only merges the stage's config sub-dict over the backend's
    declared ``default_params`` and resolves the weights file via
    ``default_weights`` (or an explicit ``weights`` override). It never
    hardcodes a backend-specific constructor signature.
    """
    if name == "none":
        return registry["none"]()

    cls = registry.get(name)
    if cls is None:
        raise KeyError(
            f"unknown backend '{name}' for {cls.stage if cls else '?'} "
            f"(available: {sorted(registry)})")

    params = dict(cls.default_params)
    for k, v in stage_cfg.items():
        if k in ("backend", "enabled", "weights"):
            continue
        params[k] = v  # config may override defaults or add backend-specific keys

    weights_name = stage_cfg.get("weights") or cls.default_weights
    weights = resolve_weights(project_root, weights_name) if cls.needs_weights else None
    if cls.needs_weights and (weights is None or not os.path.isfile(weights)):
        raise FileNotFoundError(
            f"weights missing for backend '{name}': {weights_name} "
            f"(expected under models/)")

    return cls.create(device, weights, params, fp16=fp16)


class EnhancePipeline:
    def __init__(self, cfg, project_root: str):
        self.cfg = cfg
        self.project_root = project_root

        import torch
        from src.gpu import detect_device
        self.torch = torch
        self.device = detect_device(cfg["gpu"].get("device", "auto"))
        self._report_vram()

        registries = available_backends()

        # --- restore stage ---
        dr = cfg.get("document_restoration", {})
        dr_name = dr.get("backend", "none") if dr.get("enabled", False) else "none"
        self.restore = _build_stage_backend(
            registries["restore"], dr_name, project_root, dr,
            self.device, cfg["gpu"].get("fp16", True))

        # --- super-resolution stage ---
        sr = cfg.get("super_resolution", {})
        sr_name = sr.get("backend", "none") if sr.get("enabled", False) else "none"

        # Mixed routing: pick SR backend per page type. Text pages -> the
        # document SR (realesrgan-general); image/photo pages -> SwinIR.
        self.routing = sr.get("routing", "off")
        self.sr = None
        self.sr_image = None  # backend for image/photo pages (routing: by-type)
        if self.routing == "by-type":
            self.sr = _build_stage_backend(
                registries["sr"], "realesrgan-general", project_root,
                {"scale": sr.get("scale", 2), "tile": sr.get("tile", 384)},
                self.device, cfg["gpu"].get("fp16", True))
            self.sr_image = _build_stage_backend(
                registries["sr"], "swinir", project_root,
                {"scale": sr.get("scale", 2), "tile": sr.get("tile", 128)},
                self.device, cfg["gpu"].get("fp16", True))
        else:
            self.sr = _build_stage_backend(
                registries["sr"], sr_name, project_root, sr,
                self.device, cfg["gpu"].get("fp16", True))

        # OCR backend (optional; independent of visual pipeline)
        ocr = cfg.get("ocr", {})
        if ocr.get("enabled", False) and ocr.get("backend", "rapidocr") != "none":
            model_dir = os.path.join(project_root, "models", "rapidocr")
            self.ocr = RapidOCRBackend(
                model_dir, use_cuda=bool(ocr.get("use_cuda", False)))
            self.ocr_dpi = int(ocr.get("dpi", 200))
        else:
            self.ocr = None
            self.ocr_dpi = 200

        # CJK font for the hidden text layer (embedded, project-local)
        self.ocr_font = os.path.join(project_root, "tools", "fonts", "simsun.ttc")
        if not os.path.isfile(self.ocr_font):
            self.ocr_font = None

    def _report_vram(self):
        if self.device.type == "cuda" and self.torch is not None:
            self.torch.cuda.reset_peak_memory_stats()
            free, total = self.torch.cuda.mem_get_info(0)
            log.info("VRAM before run: %.1f GB free / %.1f GB", free / 1024**3, total / 1024**3)

    def process_pdf(self, pdf_path: str, out_path: str,
                    save_images_dir: str | None = None):
        cfg = self.cfg
        t0 = time.time()
        pages = render_pdf_pages(pdf_path, cfg["render"].get("dpi", 300),
                                 int(cfg["render"].get("max_dpi", 450)))

        temp_dir = None
        if cfg.get("debug", {}).get("keep_temp", False):
            temp_dir = os.path.join(self.project_root, "temp",
                                    os.path.splitext(os.path.basename(pdf_path))[0])
            os.makedirs(temp_dir, exist_ok=True)

        out_pages = []
        text_layers: dict[int, list[dict]] = {}
        n_pages = len(pages)

        # --- OCR pass (hidden text layer, on original pages @ ocr_dpi) ---
        if self.ocr is not None:
            log.info("OCR pass @ %d DPI (%d pages)", self.ocr_dpi, n_pages)
            t_ocr = time.time()
            for i in range(n_pages):
                img, zoom = render_page_for_ocr(pdf_path, i, self.ocr_dpi)
                boxes = self.ocr.recognize(img)
                text_layers[i] = boxes_to_pdf_text(boxes, zoom)
                log.info("page %d OCR: %d spans", i + 1, len(boxes))
            log.info("OCR done in %.1fs", time.time() - t_ocr)

        for p in pages:
            i = p["index"]
            img = p["image"]
            t_page = time.time()

            img = self.restore.process(img)

            # Mixed SR routing: pick the SR backend by coarse page type
            # (photo/figure -> SwinIR, text -> RealESRGAN). The coarse signal
            # is computed on the *restored* (pre-SR) image, where DocRes has
            # already removed the illumination gradient that would otherwise
            # corrupt the text-vs-image decision.
            routed = None
            if self.routing == "by-type":
                ctype = coarse_type(img)
                sr = self.sr_image if ctype == "image" else self.sr
                routed = f"{'swinir' if ctype == 'image' else 'realesrgan'}"
                img = sr.process(img)
            else:
                img = self.sr.process(img)

            # Classify on the *enhanced* image: restoration removes the
            # illumination gradient that would otherwise make a BW text page
            # look gray, so the binarization decision is only reliable here.
            info = classify_page(img)
            ptype = info["type"]

            if temp_dir:
                cv2.imwrite(os.path.join(temp_dir, f"page_{i:04d}_{ptype}.png"), img)

            out_pages.append({
                "index": i,
                "image": img,
                "width_pt": p["width_pt"],
                "height_pt": p["height_pt"],
                "ptype": ptype,
            })
            log.info("page %d [%s%s] done in %.1fs (color=%.3f text=%.3f)",
                     i + 1, ptype,
                     f" <- {routed}" if routed else "",
                     time.time() - t_page,
                     info["color_ratio"], info["text_ratio"])

        q = int(cfg["compression"].get("jpeg_quality", 85))
        mono = cfg["compression"].get("monochrome", "1bit")
        jbig2_bin = os.path.join(self.project_root, "tools", "jbig2", "jbig2.exe")
        rebuild_pdf(out_pages, out_path, jpeg_quality=q,
                    text_layers=text_layers if self.ocr is not None else None,
                    font_file=self.ocr_font if self.ocr is not None else None,
                    monochrome=mono,
                    jbig2_bin=jbig2_bin if mono == "jbig2" else None)

        if save_images_dir:
            os.makedirs(save_images_dir, exist_ok=True)
            for p in out_pages:
                cv2.imwrite(os.path.join(save_images_dir, f"page_{p['index']:04d}.png"),
                            p["image"])

        if self.device.type == "cuda":
            peak = self.torch.cuda.max_memory_allocated() / 1024**3
            log.info("peak VRAM allocated: %.2f GB", peak)

        log.info("total: %.1fs for %d pages -> %s",
                 time.time() - t0, len(pages), out_path)
