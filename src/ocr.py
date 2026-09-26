"""RapidOCR integration: recognize text and return it as a hidden text layer.

The OCR runs on a *separate* render at a modest DPI (200) so that:
  - recognition is fast (the huge x2/x4 SR images are overkill for OCR),
  - the returned box coordinates map directly onto the original PDF page in
    points, independent of any super-resolution / restoration applied to the
    visual layer.
"""
import logging
import os

import cv2
import numpy as np
import pymupdf as fitz

log = logging.getLogger("pdfenhance")


class RapidOCRBackend:
    """Thin wrapper around rapidocr_onnxruntime.RapidOCR.

    The ONNX models ship inside the package but we copy them into
    models/rapidocr/ so the project stays fully self-contained (delete the
    project folder = nothing left behind).
    """

    def __init__(self, model_dir: str, use_cuda: bool = False):
        from rapidocr_onnxruntime import RapidOCR
        self.model_dir = model_dir
        self._engine = None
        self._use_cuda = use_cuda
        self._RapidOCR = RapidOCR

    def _ensure_engine(self):
        if self._engine is None:
            kwargs = {}
            if self._use_cuda:
                kwargs["det_use_cuda"] = True
                kwargs["rec_use_cuda"] = True
                kwargs["cls_use_cuda"] = True
            # Point RapidOCR at the project-local models.
            if self.model_dir and os.path.isdir(self.model_dir):
                kwargs["det_model_path"] = os.path.join(
                    self.model_dir, "ch_PP-OCRv3_det_infer.onnx")
                kwargs["rec_model_path"] = os.path.join(
                    self.model_dir, "ch_PP-OCRv3_rec_infer.onnx")
                kwargs["cls_model_path"] = os.path.join(
                    self.model_dir, "ch_ppocr_mobile_v2.0_cls_infer.onnx")
            self._engine = self._RapidOCR(**kwargs)
        return self._engine

    def recognize(self, img_bgr: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        """Return list of (box[4x2], text, score)."""
        engine = self._ensure_engine()
        res, _ = engine(img_bgr)
        if not res:
            return []
        out = []
        for item in res:
            box = np.array(item[0], dtype=np.float32)  # 4 corners
            text = str(item[1])
            score = float(item[2])
            out.append((box, text, score))
        return out


def render_page_for_ocr(pdf_path: str, page_index: int, dpi: int) -> tuple[np.ndarray, float]:
    """Render one page at `dpi` and return (BGR image, zoom factor px->pt)."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img, zoom


def boxes_to_pdf_text(boxes, zoom: float) -> list[dict]:
    """Convert OCR boxes (in render pixels) to PDF text spans (in points).

    Returns list of dicts: {bbox: (x0,y0,x1,y1) in points, text, score}.
    """
    spans = []
    for box, text, score in boxes:
        pts = box / zoom  # px -> pt
        x0 = float(pts[:, 0].min())
        y0 = float(pts[:, 1].min())
        x1 = float(pts[:, 0].max())
        y1 = float(pts[:, 1].max())
        spans.append({"bbox": (x0, y0, x1, y1), "text": text, "score": score})
    return spans


def insert_hidden_text_layer(page, spans: list[dict], font_file: str | None = None):
    """Write OCR text as an invisible layer on a PyMuPDF page.

    Text is inserted with render_mode=3 (invisible) so it never affects the
    visual appearance, but stays selectable and searchable (Ctrl+F).
    font_file should point to a CJK-capable font (e.g. simsun.ttc); without it
    non-Latin text degrades to '?' because the builtin helv font is Latin-only.
    """
    for s in spans:
        x0, y0, x1, y1 = s["bbox"]
        rect = fitz.Rect(x0, y0, x1, y1)
        # Small fixed-ish size so text reliably fits the (possibly narrow)
        # OCR box; render_mode=3 hides it anyway, so exact size is cosmetic.
        # A too-large size makes insert_textbox silently drop the text.
        fontsize = max(2.0, (y1 - y0) * 0.55)
        kwargs = dict(
            fontsize=fontsize,
            render_mode=3,     # invisible
            align=fitz.TEXT_ALIGN_LEFT,
        )
        if font_file and os.path.isfile(font_file):
            kwargs["fontfile"] = font_file
            kwargs["fontname"] = "ocr"
        else:
            kwargs["fontname"] = "helv"
        try:
            # NOTE: 2nd positional arg is `buffer` (the text), not `text`.
            # insert_textbox returns the unused height (negative if text
            # overflows); retry with a shrinking size so nothing is dropped.
            ret = page.insert_textbox(rect, s["text"], **kwargs)
            attempt = 0
            while ret < 0 and attempt < 4:
                attempt += 1
                kwargs["fontsize"] = kwargs["fontsize"] * 0.6
                ret = page.insert_textbox(rect, s["text"], **kwargs)
        except Exception as e:  # box too small / odd geometry -> skip
            log.debug("skip textbox %r: %s", s["text"], e)
