"""CLI entry point: run.bat -> env python -m src.main [args]"""
import argparse
import logging
import os
import sys
import glob

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Scanned PDF AI enhancer")
    parser.add_argument("pdf", nargs="?", default=None,
                        help="PDF file (default: process everything in input/)")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "config.yaml"))
    parser.add_argument("--scale", type=int, choices=[2, 4], default=None)
    parser.add_argument("--model", default=None,
                        help="SR backend: realesrgan-general | swinir | none")
    parser.add_argument("--routing", default=None, choices=["off", "by-type"],
                        help="mixed SR routing: off (single backend) | by-type "
                             "(text->realesrgan, photo->swinir)")
    parser.add_argument("--task", default=None,
                        help="DocRes task: appearance | deshadowing | deblurring")
    parser.add_argument("--quality", type=int, default=None, help="JPEG quality")
    parser.add_argument("--device", default=None, help="auto | cuda | cpu")
    parser.add_argument("--ocr", action="store_true", default=None)
    parser.add_argument("--no-ocr", dest="ocr", action="store_false")
    parser.add_argument("--no-restore", action="store_true", help="disable DocRes")
    parser.add_argument("--no-sr", action="store_true", help="disable super resolution")
    parser.add_argument("--save-images", action="store_true",
                        help="also export page PNGs to output/<name>_images/")
    parser.add_argument("--dpi", default=None,
                        help="render DPI (int, or 'auto' = per-page native scan DPI)")
    parser.add_argument("--pages", default=None,
                        help="page range to process: '50' (first 50) or '10-59' "
                             "(1-based inclusive)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        default=True,
                        help="ignore existing checkpoints and redo every page "
                             "(default: resume, reusing pages finished by an "
                             "interrupted earlier run)")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="skip enhancement and rebuild the PDF from the "
                             "checkpoint (re-tune compression in minutes); "
                             "fails if any page is missing")
    parser.add_argument("--clear-checkpoint", action="store_true",
                        help="delete this book's checkpoint before processing")
    parser.add_argument("--list-backends", action="store_true",
                        help="list available restore/SR backends and exit")
    args = parser.parse_args()

    if args.list_backends:
        from src.backends import available_backends
        regs = available_backends()
        print("restore backends:")
        for name, cls in sorted(regs["restore"].items()):
            print(f"  {name:<20} weights={cls.default_weights or '-'}")
        print("super-resolution backends:")
        for name, cls in sorted(regs["sr"].items()):
            print(f"  {name:<20} weights={cls.default_weights or '-'}")
        sys.exit(0)

    cfg = load_config(args.config)

    # CLI overrides config
    if args.scale:
        cfg["super_resolution"]["scale"] = args.scale
    if args.model:
        cfg["super_resolution"]["backend"] = args.model
    if args.routing:
        cfg["super_resolution"]["routing"] = args.routing
    if args.task:
        cfg["document_restoration"]["task"] = args.task
    if args.quality:
        cfg["compression"]["jpeg_quality"] = args.quality
    if args.device:
        cfg["gpu"]["device"] = args.device
    if args.dpi:
        cfg["render"]["dpi"] = (args.dpi if str(args.dpi).lower() == "auto"
                                else int(args.dpi))
    if args.no_restore:
        cfg["document_restoration"]["enabled"] = False
    if args.no_sr:
        cfg["super_resolution"]["enabled"] = False
    if args.ocr is True:
        cfg["ocr"]["enabled"] = True
    elif args.ocr is False:
        cfg["ocr"]["enabled"] = False

    # collect input PDFs
    if args.pdf:
        pdfs = [args.pdf]
    else:
        pdfs = sorted(glob.glob(os.path.join(PROJECT_ROOT, "input", "*.pdf")))
    if not pdfs:
        print("no input PDF found (put files in input/ or pass a path)")
        sys.exit(1)

    # parse page range (1-based inclusive, or a bare count = first N pages)
    page_range = None
    if args.pages:
        spec = str(args.pages).strip()
        if "-" in spec:
            a, b = spec.split("-", 1)
            page_range = (int(a) - 1, int(b) - 1)
        else:
            page_range = (0, int(spec) - 1)

    os.makedirs(os.path.join(PROJECT_ROOT, "output"), exist_ok=True)

    from src.pipeline import EnhancePipeline
    pipe = EnhancePipeline(cfg, PROJECT_ROOT)

    for pdf in pdfs:
        pdf = os.path.abspath(pdf)
        name = os.path.splitext(os.path.basename(pdf))[0]
        log_path = os.path.join(PROJECT_ROOT, "logs", f"{name}.log")
        setup_logging(log_path)
        log = logging.getLogger("pdfenhance")
        log.info("=== processing %s ===", pdf)

        out_pdf = os.path.join(PROJECT_ROOT, "output", f"{name}_enhanced.pdf")
        img_dir = os.path.join(PROJECT_ROOT, "output", f"{name}_images") \
            if args.save_images else None
        pipe.process_pdf(pdf, out_pdf, save_images_dir=img_dir,
                         page_range=page_range,
                         resume=args.resume,
                         rebuild_only=args.rebuild_only,
                         clear_checkpoint=args.clear_checkpoint)

    print("done.")


if __name__ == "__main__":
    main()
