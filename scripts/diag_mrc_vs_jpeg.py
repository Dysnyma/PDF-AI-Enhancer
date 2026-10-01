"""Head-to-head: whole-page JPEG vs MRC on real enhanced pages.

Answers why the dual-encode in rebuild_pdf never picks MRC: measure the actual
byte sizes of both encodings for the same page image.

Usage:
    env/Scripts/python.exe scripts/diag_mrc_vs_jpeg.py --pdf input/x.pdf --pages 130,45,220
"""
import argparse
import os
import sys
import tempfile
import zlib

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.classify import classify_page  # noqa: E402
from src.pipeline import EnhancePipeline  # noqa: E402
from src.render import encode_page_image, render_pdf_pages  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "config.yaml"))
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pipe = EnhancePipeline(cfg, PROJECT_ROOT)
    jbig2 = os.path.join(PROJECT_ROOT, "tools", "jbig2", "jbig2.exe")
    jbig2 = jbig2 if os.path.isfile(jbig2) else None
    workdir = tempfile.mkdtemp(prefix="mrc_diag_")

    from src.mrc import encode_mrc

    for pno in [int(x) for x in args.pages.split(",") if x.strip()]:
        i = pno - 1
        p = render_pdf_pages(args.pdf, cfg["render"].get("dpi", 300),
                             int(cfg["render"].get("max_dpi", 450)),
                             page_range=(i, i))[0]
        img = pipe.sr.process(pipe.restore.process(p["image"]))
        info = classify_page(img)
        ptype = info["type"]

        jpg = encode_page_image(img, ptype, args.quality)
        try:
            mrc = encode_mrc(img, args.quality, bg_gray=(ptype == "gray-text"),
                             jbig2_bin=jbig2, workdir=workdir)
            bg, layers, mh, mw = mrc
            zst = sum(len(zlib.compress(pk, 6)) for pk, _ in layers) if layers else 0
            mrc_total = (len(bg) + zst) if bg is not None else None
        except Exception as e:
            import traceback
            traceback.print_exc()
            bg, layers, mrc_total = None, [], None

        print(f"\np{pno}  type={ptype}  {img.shape[1]}x{img.shape[0]}")
        print(f"  whole-page JPEG : {len(jpg)/1024:8.1f} KB")
        if mrc_total is not None:
            print(f"  MRC background  : {len(bg)/1024:8.1f} KB")
            print(f"  MRC stencils    : {zst/1024:8.1f} KB   ({len(layers)} layers)")
            print(f"  MRC total       : {mrc_total/1024:8.1f} KB")
            win = "MRC" if mrc_total < len(jpg) else "JPEG"
            print(f"  -> winner: {win}  "
                  f"(MRC/JPEG = {mrc_total/len(jpg):.2f})")
        else:
            print("  MRC: FAILED / returned nothing")


if __name__ == "__main__":
    main()
