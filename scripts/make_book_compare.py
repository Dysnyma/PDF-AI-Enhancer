# -*- coding: utf-8 -*-
"""原书 vs 增强成品 单页放大对比图。

成品是 SR×2 的 1-bit/JPEG 图，原书是低分辨率扫描。两者页面几何相同
（都是 A4 595×842pt 左右），直接在各自 PDF 里渲染同一页到同一像素
尺寸，裁同一段文字区域，最近邻放大 400% 左右拼成一张图，直观看
清晰度差异。

用法: env\\Scripts\\python.exe scripts\\make_book_compare.py \
          <原书.pdf> <成品.pdf> <页号(1-based)> [输出png]
"""
import os
import sys

import cv2
import numpy as np
import pymupdf as fitz


def render_page(pdf_path, page_idx, target_w):
    """渲染指定页，缩放到 target_w 宽（保持宽高比），返回 BGR ndarray。"""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    zoom = target_w / page.rect.width
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def main():
    orig_pdf = sys.argv[1]
    enh_pdf = sys.argv[2]
    page_no = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    out_png = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output", "compare_book_page%d.png" % page_no)
    page_idx = page_no - 1

    # 渲染到同一宽度，便于对齐裁剪
    TARGET_W = 1800
    orig = render_page(orig_pdf, page_idx, TARGET_W)
    enh = render_page(enh_pdf, page_idx, TARGET_W)
    print("原书页尺寸:", orig.shape, " 成品页尺寸:", enh.shape)

    h = min(orig.shape[0], enh.shape[0])
    orig, enh = orig[:h], enh[:h]

    # 裁一个文字密集的区域（默认页面上 1/3 处，宽 55%，高 22%）
    cx = int(orig.shape[1] * 0.22)
    cy = int(orig.shape[0] * 0.28)
    cw = int(orig.shape[1] * 0.56)
    ch = int(orig.shape[0] * 0.22)
    c_orig = orig[cy:cy + ch, cx:cx + cw]
    c_enh = enh[cy:cy + ch, cx:cx + cw]

    # 最近邻放大 4x（真实像素，不放平滑）
    z = 4
    c_orig = cv2.resize(c_orig, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    c_enh = cv2.resize(c_enh, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)

    # 顶部标签
    def label(img, text):
        banner = np.full((50, img.shape[1], 3), 245, np.uint8)
        cv2.putText(banner, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 2)
        return np.vstack([banner, img])

    left = label(c_orig, "原书扫描 (第%d页)" % page_no)
    right = label(c_enh, "增强成品 (SRx2 + 1bit)")
    grid = np.hstack([left, right])

    cv2.imwrite(out_png, grid)
    print("wrote", out_png, grid.shape[1], "x", grid.shape[0])


if __name__ == "__main__":
    main()
