"""Per-page checkpointing so an interrupted long run can resume.

A 340-page book takes ~104 minutes; this project has already lost two such
runs to an interruption, throwing away every finished page. Pages are
independent of each other, so each finished page is written to disk as a
lossless PNG plus a small JSON of its metadata, and a restart picks up from
the last finished page instead of starting over.

Layout (inside the project, so deleting the project still uninstalls
everything)::

    temp/<book>/ckpt/enhance_<sig12>/p00007.png   enhanced page, lossless
    temp/<book>/ckpt/enhance_<sig12>/p00007.json  metadata (the "done" marker)
    temp/<book>/ckpt/enhance_<sig12>/manifest.json
    temp/<book>/ckpt/ocr_<sig12>/p00007.json      OCR spans for one page

Two independent signatures are used:

* **enhance** — render DPI, DocRes backend/task, SR backend/scale/tile,
  routing, device and fp16: everything that changes the *pixels* of an
  enhanced page.
* **ocr** — OCR backend, DPI and language: unrelated to the enhancement
  chain and much cheaper to redo, so changing SR settings must not throw
  away a finished OCR pass.

Compression settings (JPEG quality, the MRC background parameters, the
monochrome encoding) are deliberately **not** part of either signature:
they only affect the rebuild step, which reads the cached pages back from
disk. That is what makes ``--rebuild-only`` possible — re-tuning the MRC
encoder costs a rebuild (minutes) instead of a full enhancement pass (hours).

Crash safety: the image is written to ``*.part`` and renamed, and the JSON
metadata is written *after* it. A page counts as done only when its JSON
exists, so a crash between the two writes just means that page is redone.
"""

import hashlib
import json
import logging
import os

import cv2
import numpy as np

from src.imgio import imread_unicode, imwrite_unicode

log = logging.getLogger("pdfenhance")

CHECKPOINT_VERSION = 2


def _source_id(pdf_path: str) -> dict:
    try:
        size = os.path.getsize(pdf_path)
    except OSError:
        size = -1
    return {"name": os.path.basename(pdf_path), "size": size}


def _sig(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def enhance_signature(cfg: dict, pdf_path: str, page_range) -> str:
    """Hash of everything that determines the enhanced pixels of a page."""
    sr = dict(cfg.get("super_resolution", {}) or {})
    dr = dict(cfg.get("document_restoration", {}) or {})
    gpu = dict(cfg.get("gpu", {}) or {})
    return _sig({
        "v": CHECKPOINT_VERSION,
        "kind": "enhance",
        "source": _source_id(pdf_path),
        "pages": list(page_range) if page_range else None,
        "render": cfg.get("render", {}),
        "restore": dr,
        "sr": sr,
        "device": gpu.get("device", "auto"),
        "fp16": gpu.get("fp16", True),
    })


def ocr_signature(cfg: dict, pdf_path: str, page_range, ocr_dpi: int) -> str:
    """Hash of everything that determines a page's OCR spans."""
    ocr = dict(cfg.get("ocr", {}) or {})
    return _sig({
        "v": CHECKPOINT_VERSION,
        "kind": "ocr",
        "source": _source_id(pdf_path),
        "pages": list(page_range) if page_range else None,
        "backend": ocr.get("backend", "rapidocr"),
        "lang": ocr.get("lang", "ch"),
        "use_cuda": ocr.get("use_cuda", False),
        "dpi": int(ocr_dpi),
    })


def _atomic_write_bytes(path: str, data: bytes):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _atomic_imwrite(path: str, img: np.ndarray):
    """Write a PNG atomically, Unicode-safely.

    Two Windows traps are handled here: the temp name keeps the .png
    extension (cv2.imwrite picks its encoder from the extension and refuses
    to write "page.png.part"), and the write goes through
    :mod:`src.imgio` because cv2.imwrite cannot handle the Chinese book names
    this project actually uses.

    Compression level 1: the checkpoint is a transient intermediate, and on
    a 2878x4010 page the difference between level 1 and level 6 is seconds
    of CPU per page for a few percent of disk.
    """
    tmp = path + ".part.png"
    if not imwrite_unicode(tmp, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1]):
        raise RuntimeError(f"failed to write checkpoint image {tmp}")
    os.replace(tmp, path)


def _rm_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


class PageStore:
    """Cache of finished enhanced pages: one lossless PNG + JSON per page.

    ``use_cache=False`` still writes (the rebuild stage reads the pages back
    from here, so the files are the intermediate representation regardless),
    it only stops the store from *reusing* pages left by an earlier run.
    """

    def __init__(self, ckpt_root: str, pdf_path: str, cfg: dict, page_range,
                 use_cache: bool = True):
        self.use_cache = bool(use_cache)
        self.sig = enhance_signature(cfg, pdf_path, page_range)
        self.dir = os.path.join(ckpt_root, f"enhance_{self.sig}")
        self._meta_cache: dict[int, dict] = {}
        try:
            os.makedirs(self.dir, exist_ok=True)
            manifest = os.path.join(self.dir, "manifest.json")
            if not os.path.isfile(manifest):
                _atomic_write_bytes(manifest, json.dumps({
                    "version": CHECKPOINT_VERSION,
                    "kind": "enhance",
                    "signature": self.sig,
                    "source": os.path.basename(pdf_path),
                    "pages": list(page_range) if page_range else None,
                }, indent=2, ensure_ascii=False).encode("utf-8"))
        except OSError as e:  # read-only volume etc. -> degrade to no caching
            log.warning("checkpoint disabled (%s)", e)
            self.use_cache = False

    # ---- paths ----
    def image_path(self, i: int) -> str:
        return os.path.join(self.dir, f"p{i:05d}.png")

    def _meta_path(self, i: int) -> str:
        return os.path.join(self.dir, f"p{i:05d}.json")

    # ---- read side ----
    def has(self, i: int) -> bool:
        return self.use_cache and os.path.isfile(self._meta_path(i))

    def meta(self, i: int) -> dict:
        """Metadata of a finished page (cached; one JSON read per new page)."""
        m = self._meta_cache.get(i)
        if m is None:
            with open(self._meta_path(i), "r", encoding="utf-8") as f:
                m = json.load(f)
            self._meta_cache[i] = m
        return m

    def load_image(self, i: int) -> np.ndarray:
        img = imread_unicode(self.image_path(i))
        if img is None:  # truncated / deleted behind our back
            raise RuntimeError(f"checkpoint image unreadable: {self.image_path(i)}")
        return img

    def count(self) -> int:
        try:
            return sum(1 for n in os.listdir(self.dir)
                       if n.endswith(".json") and n.startswith("p"))
        except OSError:
            return 0

    # ---- write side ----
    def save(self, i: int, image: np.ndarray, ptype: str,
             width_pt: float, height_pt: float, dpi: int):
        _atomic_imwrite(self.image_path(i), image)
        h, w = image.shape[:2]
        meta = {
            "index": i, "ptype": ptype,
            "width_pt": width_pt, "height_pt": height_pt,
            "dpi": dpi, "px_w": w, "px_h": h,
            "version": CHECKPOINT_VERSION,
        }
        _atomic_write_bytes(self._meta_path(i),
                            json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        self._meta_cache[i] = meta

    # ---- housekeeping ----
    def dir_size(self) -> int:
        total = 0
        try:
            for n in os.listdir(self.dir):
                if n.startswith("p") or n.startswith("manifest"):
                    try:
                        total += os.path.getsize(os.path.join(self.dir, n))
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def clear(self):
        """Delete the cached pages. Removes files we created, then the dir.

        Deliberately not ``shutil.rmtree``: only ``p*.png`` / ``p*.json`` /
        ``manifest.json`` inside a directory this class created are removed,
        so a mis-set path can never take out anything else.
        """
        try:
            names = os.listdir(self.dir)
        except OSError:
            names = []
        removed = 0
        for n in names:
            if n.endswith((".png", ".json", ".part")):
                _rm_file(os.path.join(self.dir, n))
                removed += 1
        try:
            os.rmdir(self.dir)
        except OSError:
            pass
        self._meta_cache.clear()
        if removed:
            log.info("checkpoint cleared: %d files in %s", removed, self.dir)


class OcrStore:
    """Cache of per-page OCR spans (JSON only — the spans are tiny)."""

    def __init__(self, ckpt_root: str, pdf_path: str, cfg: dict, page_range,
                 ocr_dpi: int, use_cache: bool = True):
        self.use_cache = bool(use_cache)
        self.sig = ocr_signature(cfg, pdf_path, page_range, ocr_dpi)
        self.dir = os.path.join(ckpt_root, f"ocr_{self.sig}")
        try:
            os.makedirs(self.dir, exist_ok=True)
            manifest = os.path.join(self.dir, "manifest.json")
            if not os.path.isfile(manifest):
                _atomic_write_bytes(manifest, json.dumps({
                    "version": CHECKPOINT_VERSION,
                    "kind": "ocr",
                    "signature": self.sig,
                    "source": os.path.basename(pdf_path),
                    "dpi": int(ocr_dpi),
                }, indent=2, ensure_ascii=False).encode("utf-8"))
        except OSError as e:
            log.warning("OCR checkpoint disabled (%s)", e)
            self.use_cache = False

    def _path(self, i: int) -> str:
        return os.path.join(self.dir, f"p{i:05d}.json")

    def has(self, i: int) -> bool:
        return self.use_cache and os.path.isfile(self._path(i))

    def load(self, i: int) -> list[dict]:
        with open(self._path(i), "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, i: int, spans: list[dict]):
        _atomic_write_bytes(self._path(i),
                            json.dumps(spans, ensure_ascii=False).encode("utf-8"))

    def count(self) -> int:
        try:
            return sum(1 for n in os.listdir(self.dir)
                       if n.endswith(".json") and n.startswith("p"))
        except OSError:
            return 0

    def clear(self):
        try:
            names = os.listdir(self.dir)
        except OSError:
            names = []
        for n in names:
            if n.endswith((".json", ".part")):
                _rm_file(os.path.join(self.dir, n))
        try:
            os.rmdir(self.dir)
        except OSError:
            pass


def clear_checkpoints(book_ckpt_root: str) -> str:
    """Remove every checkpoint of one book (all signatures). Returns the path.

    Only directories matching this module's own naming are touched.
    """
    if not os.path.isdir(book_ckpt_root):
        return book_ckpt_root
    for name in os.listdir(book_ckpt_root):
        if not (name.startswith("enhance_") or name.startswith("ocr_")):
            continue
        sub = os.path.join(book_ckpt_root, name)
        if not os.path.isdir(sub):
            continue
        try:
            for n in os.listdir(sub):
                if n.endswith((".png", ".json", ".part")):
                    _rm_file(os.path.join(sub, n))
            os.rmdir(sub)
        except OSError:
            pass
    try:
        os.rmdir(book_ckpt_root)
    except OSError:
        pass
    return book_ckpt_root


class SpansView:
    """Read-only, lazily-loaded ``{page_index: spans}`` mapping.

    ``rebuild_pdf`` only tests membership and then indexes individual pages,
    so loading all 340 JSON files into a dict first would be wasted memory
    and I/O. This presents the OCR store as a mapping instead.
    """

    def __init__(self, store: "OcrStore"):
        self._store = store

    def __contains__(self, i) -> bool:
        return self._store.has(i)

    def __getitem__(self, i) -> list[dict]:
        return self._store.load(i)

    def __len__(self) -> int:
        return self._store.count()

    def __bool__(self) -> bool:
        return len(self) > 0


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
