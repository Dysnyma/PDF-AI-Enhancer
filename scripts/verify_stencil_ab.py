"""Bit-exact A/B acceptance for the MRC stencil codec.

Why this exists
---------------
``compression.mrc_stencil`` switches the 1-bit foreground templates between
JBIG2 and Flate. The templates are the bulk of the file (54.7% of the
340-page book), so this switch is exactly where a silent, catastrophic
regression would hide: an earlier bug wrote the JBIG2 stream with a wrong
``/Filter`` and the only visible symptom was *fully black pages* -- something
a four-page spot check found by luck rather than by design.

So instead of re-rendering and eyeballing thumbnails, decode both files and
compare the layer bitmaps exactly. Rendering at any reduced DPI resamples a
2878x4010 1-bit template and can hide single-pixel differences; comparing the
decoded bitmaps cannot.

What is compared
----------------
Per page, the positional list of image layers must match on
(width, height, bits-per-component, /Filter), and:

* **1-bit layers (the templates)** -- decoded to a ``(h, w)`` bit array and
  compared bit for bit. Padding-agnostic: only the ``w`` valid bits of each
  row take part, so a codec that pads rows differently is not falsely flagged.
* **photo layers (the JPEG background)** -- compared as compressed bytes
  verbatim, which is stricter than comparing pixels.

Also reported: the min/max ink ratio over every template, which flags
all-black or all-white pages -- the historical failure signature.

A note on a false alarm
-----------------------
A naive sanity check that a ``/JBIG2Decode`` stream starts with the JBIG2
file header ``0x97 'JB2'`` reports *every* stream as corrupt. It is wrong:
PDF embeds JBIG2 in its *segment* format, which carries no file header, so
such a stream legitimately starts with a segment header (typically
``00 00 00 00 30 00 01 00``). Do not "fix" a file because of that check.

Usage
-----
    python -m scripts.verify_stencil_ab A.pdf B.pdf [--json report.json]

Exit code is 0 only when every page matches.
"""

import argparse
import json
import sys
from collections import Counter

import numpy as np
import pymupdf


def _layers(doc, pno):
    """Layers of one page, sorted by identity so the two files are aligned positionally.

    Geometry comes from the ``get_images`` tuple rather than ``extract_image``:
    the latter fully decodes the image, which would decode every JBIG2 template
    twice for no reason. ``img`` is (xref, smask, width, height, bpc, ...); an
    ImageMask reports bpc 0 and is 1-bit by definition.

    The identity key deliberately **excludes /Filter**: the two files are meant
    to differ in exactly that, so folding it in would report every page as a
    mismatch. Filters are counted separately for the census.
    """
    items = []
    for img in doc[pno].get_images(full=True):
        xref = img[0]
        w, h, bpc = int(img[2] or 0), int(img[3] or 0), int(img[4] or 0)
        if bpc == 0:
            bpc = 1
        filt = (doc.xref_get_key(xref, "Filter")[1] or "?")
        # Polarity. Both layer kinds this project emits ink on sample 0:
        #   * ImageMask + /Decode [0 1] -> 0 paints, so a text page is ~99% set
        #     samples and only ~1% inked;
        #   * 1-bit DeviceGray (the bw-text pages) -> 0 is black, so a text page
        #     is ~97% white samples.
        # Reading the sample mean as "ink" in either case reports ~97-99% ink on
        # perfectly normal text pages. Only an explicit /Decode [1 0] inverts it.
        dec = (doc.xref_get_key(xref, "Decode")[1] or "").replace(" ", "")
        paint_on_zero = not dec.startswith("[10]")
        key = (w, h, bpc)
        try:
            payload = doc.xref_stream(xref)
        except Exception as e:
            payload = b""
            key = key + (f"stream-error:{e}",)
        items.append((key, filt, paint_on_zero, payload))
    items.sort(key=lambda t: (t[0], len(t[3])))
    return items


def _to_bits(payload, w, h):
    """Decode packed 1-bit data into a (h, w) boolean array, ignoring row padding.

    JBIG2 and Flate may pad each row to a different boundary; only the w valid
    bits per row are content, so mask the padding off instead of comparing it.
    """
    row_bytes = (w + 7) // 8
    if len(payload) < row_bytes * h:
        return None
    arr = np.frombuffer(payload[:row_bytes * h], dtype=np.uint8).reshape(h, row_bytes)
    return np.unpackbits(arr, axis=1)[:, :w].astype(bool)


def _cmp(key, pay_a, pay_b):
    w, h, bpc = key[0], key[1], key[2]
    if bpc == 1 and w and h:
        ba, bb = _to_bits(pay_a, w, h), _to_bits(pay_b, w, h)
        if ba is None or bb is None:
            return False, f"short stream {len(pay_a)} / {len(pay_b)}", None
        n = int(np.count_nonzero(ba != bb))
        return (n == 0), (f"{n} of {ba.size} bits differ" if n else ""), ba
    if pay_a != pay_b:
        return False, f"bytes differ ({len(pay_a)} vs {len(pay_b)})", None
    return True, "", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--json", default=None, help="write a machine-readable report")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    da, db = pymupdf.open(args.file_a), pymupdf.open(args.file_b)
    if da.page_count != db.page_count:
        print(f"FAIL page count {da.page_count} vs {db.page_count}")
        return 2

    bad, n_bit, n_photo = [], 0, 0
    ink_min, ink_max, ink_min_page = 1.0, 0.0, 0
    filters = Counter()

    for pno in range(da.page_count):
        la, lb = _layers(da, pno), _layers(db, pno)
        if [k for k, _, _, _ in la] != [k for k, _, _, _ in lb]:
            bad.append({"page": pno + 1,
                        "why": f"layer identities differ: "
                               f"{[k for k, _, _, _ in la]} vs {[k for k, _, _, _ in lb]}"})
            continue
        for (key, filt_a, on_zero, pay_a), (_, filt_b, _, pay_b) in zip(la, lb):
            filters[f"{filt_a} -> {filt_b}"] += 1
            ok, why, bits = _cmp(key, pay_a, pay_b)
            if key[2] == 1:
                n_bit += 1
                if bits is not None:
                    painted = (1.0 - float(np.count_nonzero(bits)) / bits.size
                               if on_zero else
                               float(np.count_nonzero(bits)) / bits.size)
                    if painted < ink_min:
                        ink_min, ink_min_page = painted, pno + 1
                    ink_max = max(ink_max, painted)
                    if painted > 0.90:
                        bad.append({"page": pno + 1, "key": str(key),
                                    "why": f"template is {painted:.1%} inked "
                                           f"(all-black page?)"})
            else:
                n_photo += 1
            if not ok:
                bad.append({"page": pno + 1, "key": str(key), "why": why})
        if not args.quiet and (pno + 1) % 50 == 0:
            print(f"  ... {pno + 1}/{da.page_count} pages checked")

    report = {
        "file_a": args.file_a, "file_b": args.file_b,
        "pages": da.page_count,
        "templates_compared": n_bit,
        "photo_layers_compared": n_photo,
        "filters": dict(filters),
        "template_ink_ratio": {"min": round(ink_min, 6), "min_page": ink_min_page,
                               "max": round(ink_max, 6)},        "mismatches": bad,
        "verdict": "IDENTICAL" if not bad else "MISMATCH",
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\npages                : {report['pages']}")
    print(f"1-bit templates      : {n_bit} compared")
    print(f"photo layers         : {n_photo} compared")
    print(f"filter census        : {dict(filters)}")
    print(f"template ink ratio   : min {ink_min:.6f} (p{ink_min_page})  max {ink_max:.6f}")
    if bad:
        print(f"\n{len(bad)} MISMATCH/ANOMALY item(s):")
        for b in bad[:20]:
            print("  page", b["page"], "--", b["why"])
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")
    print(f"\nVERDICT: {report['verdict']}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
