"""Unicode-safe image file I/O.

OpenCV's ``imread``/``imwrite`` open the file through the ANSI code page on
Windows, so a path containing non-ASCII characters — and this project stores
scanned books in Chinese, e.g. ``temp/云计算导论/ckpt/...`` — makes
``imwrite`` return False and ``imread`` return None, with no readable error.
The checkpoint writer hit exactly that: the first real book failed to
checkpoint while an ASCII-named test file passed.

These helpers encode/decode in memory and use Python's own file objects
(which are Unicode-correct on every platform) for the actual disk I/O.
"""

import os

import cv2
import numpy as np


def _ext(path: str, default: str = ".png") -> str:
    return os.path.splitext(path)[1] or default


def imwrite_unicode(path: str, img: np.ndarray, params=None) -> bool:
    """cv2.imwrite that survives non-ASCII paths. Returns success."""
    ok, buf = cv2.imencode(_ext(path), img, params if params is not None else [])
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def imread_unicode(path: str, flags=cv2.IMREAD_COLOR) -> np.ndarray | None:
    """cv2.imread that survives non-ASCII paths. None if unreadable."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, flags)
