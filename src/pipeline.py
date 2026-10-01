"""Pipeline orchestration: render -> classify -> restore -> SR -> rebuild.

Pages are processed one at a time and each finished page is written to a
checkpoint (see :mod:`src.checkpoint`), so an interrupted run resumes
instead of starting over, and the peak memory footprint stays at a single
page rather than a whole book.
"""
import logging
import os
import shutil
import time

from src.backends import available_backends
from src.backends.base import resolve_weights
from src.checkpoint import (OcrStore, PageStore, SpansView, clear_checkpoints,
                            human_size)
from src.classify import classify_page, coarse_type
from src.ocr import (RapidOCRBackend, boxes_to_pdf_text,
                     render_page_for_ocr)
from src.render import PdfPages, rebuild_pdf

log = logging.getLogger("pdfenhance")


class Cancelled(Exception):
    """Raised when a caller-requested cancel is detected mid-pipeline.

    The caller passes ``should_stop`` (a zero-arg callable) to ``process_pdf``;
    the pipeline checks it between pages and aborts early so a long全本 run can
    be stopped from the GUI without killing the process. Every page finished
    so far is already in the checkpoint, so the next run continues from there.
    """


def _check_stop(should_stop):
    if should_stop is not None and should_stop():
        raise Cancelled()


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
        if k in ("backend", "enabled", "weights", "routing"):
            continue  # pipeline-level keys, not backend constructor params
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

    # ------------------------------------------------------------------
    def process_pdf(self, pdf_path: str, out_path: str,
                    save_images_dir: str | None = None,
                    page_range: tuple[int, int] | None = None,
                    should_stop=None,
                    resume: bool = True,
                    rebuild_only: bool = False,
                    clear_checkpoint: bool = False):
        """Enhance one PDF.

        resume: reuse pages already finished by an earlier (interrupted) run
          instead of reprocessing them. Finished pages are checkpointed to
          ``temp/<book>/ckpt/`` as they complete.
        rebuild_only: skip enhancement entirely and rebuild the PDF from the
          checkpoint. Fails loudly if any page is missing, so it can never
          silently emit a partial document. This turns "re-tune the MRC
          encoder and look at the result" from ~104 minutes into minutes.
        clear_checkpoint: wipe every checkpoint of this book first.
        """
        cfg = self.cfg
        t0 = time.time()
        book = os.path.splitext(os.path.basename(pdf_path))[0]
        ckpt_root = os.path.join(self.project_root, "temp", book, "ckpt")

        if clear_checkpoint:
            clear_checkpoints(ckpt_root)
            log.info("checkpoint cleared for %s", book)

        store = PageStore(ckpt_root, pdf_path, cfg, page_range, use_cache=resume)
        log.info("checkpoint: %s", store.dir)

        ocr_store = None
        if self.ocr is not None and not rebuild_only:
            ocr_store = OcrStore(ckpt_root, pdf_path, cfg, page_range,
                                 self.ocr_dpi, use_cache=resume)
        elif self.ocr is not None and rebuild_only:
            # A rebuild still wants the hidden text layer: attach to the
            # existing OCR checkpoint instead of creating a new one.
            ocr_store = OcrStore(ckpt_root, pdf_path, cfg, page_range,
                                 self.ocr_dpi, use_cache=True)

        cached = store.count() if resume else 0
        if cached:
            log.info("resume: %d page(s) already enhanced (%s on disk)",
                     cached, human_size(store.dir_size()))

        temp_dir = None
        if cfg.get("debug", {}).get("keep_temp", False):
            temp_dir = os.path.join(self.project_root, "temp", book)
            os.makedirs(temp_dir, exist_ok=True)

        with PdfPages(pdf_path, cfg["render"].get("dpi", 300),
                      int(cfg["render"].get("max_dpi", 450)),
                      page_range=page_range) as pages:
            total = len(pages.indexes)
            log.info("plan: %d page(s), %d cached, %d to enhance",
                     total, cached, total - cached)

            # --- OCR pass (hidden text layer, on original pages @ ocr_dpi) ---
            if ocr_store is not None and not rebuild_only:
                todo = [i for i in pages.indexes if not ocr_store.has(i)]
                if todo:
                    log.info("OCR pass @ %d DPI (%d page(s), %d cached)",
                             self.ocr_dpi, len(todo), total - len(todo))
                    t_ocr = time.time()
                    for i in todo:
                        _check_stop(should_stop)
                        img, zoom = render_page_for_ocr(pdf_path, i, self.ocr_dpi)
                        boxes = self.ocr.recognize(img)
                        del img
                        ocr_store.save(i, boxes_to_pdf_text(boxes, zoom))
                        log.info("page %d OCR: %d spans", i + 1, len(boxes))
                    log.info("OCR done in %.1fs", time.time() - t_ocr)
                else:
                    log.info("OCR pass: all %d page(s) cached", total)

            # --- enhancement pass (one page in memory at a time) ---
            out_pages = []
            missing = []
            for i in pages.indexes:
                _check_stop(should_stop)
                t_page = time.time()

                if store.has(i):
                    out_pages.append(self._cached_entry(store, i))
                    if not rebuild_only:
                        log.info("page %d [%s] cached, skipped",
                                 i + 1, store.meta(i)["ptype"])
                    continue

                if rebuild_only:
                    missing.append(i + 1)
                    continue

                p = pages.render(i)
                img = p["image"]
                img = self.restore.process(img)

                # Mixed SR routing picks the backend by coarse page type
                # (photo/figure -> SwinIR, text -> RealESRGAN). The coarse
                # signal is computed on the *restored* (pre-SR) image, where
                # DocRes has already removed the illumination gradient that
                # would otherwise corrupt the text-vs-image decision.
                routed = None
                if self.routing == "by-type":
                    ctype = coarse_type(img)
                    sr = self.sr_image if ctype == "image" else self.sr
                    routed = "swinir" if ctype == "image" else "realesrgan"
                    img = sr.process(img)
                else:
                    img = self.sr.process(img)

                # Classify on the *enhanced* image: restoration removes the
                # illumination gradient that would otherwise make a BW text
                # page look gray, so the binarization decision is only
                # reliable here.
                info = classify_page(img)
                ptype = info["type"]

                store.save(i, img, ptype, p["width_pt"], p["height_pt"], p["dpi"])

                if temp_dir:
                    self._link_or_copy(store.image_path(i),
                                       os.path.join(temp_dir,
                                                    f"page_{i:04d}_{ptype}.png"))

                out_pages.append({
                    "index": i,
                    "image_path": store.image_path(i),
                    "ptype": ptype,
                    "width_pt": p["width_pt"],
                    "height_pt": p["height_pt"],
                })
                log.info("page %d [%s%s] done in %.1fs (color=%.3f text=%.3f)",
                         i + 1, ptype,
                         f" <- {routed}" if routed else "",
                         time.time() - t_page,
                         info["color_ratio"], info["text_ratio"])
                del img, p  # keep exactly one page in memory

            pages.log_stats()

            if rebuild_only and missing:
                head = ", ".join(str(n) for n in missing[:10])
                more = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10})"
                raise RuntimeError(
                    f"rebuild-only: {len(missing)} page(s) not in the checkpoint "
                    f"(pages {head}{more}). Run the full pipeline once "
                    f"(without --rebuild-only) to produce them.")

        # --- rebuild ---
        q = int(cfg["compression"].get("jpeg_quality", 85))
        mono = cfg["compression"].get("monochrome", "1bit")
        bg_scale = cfg["compression"].get("mrc_bg_scale", "auto")
        bg_denoise = cfg["compression"].get("mrc_bg_denoise", "auto")
        enc_workers = cfg["compression"].get("encode_workers", "auto")
        enc_priority = cfg["compression"].get("encode_priority", "below-normal")
        jbig2_bin = os.path.join(self.project_root, "tools", "jbig2", "jbig2.exe")
        text_layers = SpansView(ocr_store) if ocr_store is not None else None
        log.info("rebuild: encoding %d pages [quality=%d monochrome=%s bg_scale=%s "
                 "bg_denoise=%s workers=%s priority=%s] -> %s",
                 len(out_pages), q, mono, bg_scale, bg_denoise, enc_workers,
                 enc_priority, os.path.basename(out_path))
        # jbig2_bin is handed to rebuild_pdf unconditionally: MRC's text
        # segmentation (leptonica adaptive threshold) needs it regardless of
        # how bw-text pages are encoded. rebuild_pdf only uses it for the
        # *encoding* of bw-text pages when monochrome == "jbig2".
        rebuild_pdf(out_pages, out_path, jpeg_quality=q,
                    text_layers=text_layers,
                    font_file=self.ocr_font if text_layers else None,
                    monochrome=mono,
                    mrc_bg_scale=bg_scale,
                    mrc_bg_denoise=bg_denoise,
                    jbig2_bin=jbig2_bin,
                    encode_workers=enc_workers,
                    encode_priority=enc_priority)

        if save_images_dir:
            os.makedirs(save_images_dir, exist_ok=True)
            for p in out_pages:
                shutil.copyfile(p["image_path"], os.path.join(
                    save_images_dir, f"page_{p['index']:04d}.png"))

        if self.device.type == "cuda":
            peak = self.torch.cuda.max_memory_allocated() / 1024**3
            log.info("peak VRAM allocated: %.2f GB", peak)

        keep = cfg.get("debug", {}).get("keep_checkpoint", True)
        if keep:
            log.info("checkpoint kept: %s (%s) — re-tuning compression can "
                     "rebuild from it with --rebuild-only",
                     store.dir, human_size(store.dir_size()))
        else:
            clear_checkpoints(ckpt_root)
            log.info("checkpoint removed (debug.keep_checkpoint=false)")

        log.info("total: %.1fs for %d pages -> %s",
                 time.time() - t0, len(out_pages), out_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _cached_entry(store: PageStore, i: int) -> dict:
        m = store.meta(i)
        return {
            "index": i,
            "image_path": store.image_path(i),
            "ptype": m["ptype"],
            "width_pt": m["width_pt"],
            "height_pt": m["height_pt"],
        }

    @staticmethod
    def _link_or_copy(src: str, dst: str):
        """Hardlink when possible so keep_temp does not duplicate the
        checkpoint on disk; fall back to a copy."""
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)
        except OSError:
            try:
                shutil.copyfile(src, dst)
            except OSError as e:
                log.warning("could not export %s: %s", dst, e)
