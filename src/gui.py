# -*- coding: utf-8 -*-
"""本地 Web GUI 后端：零依赖（仅标准库），封装 EnhancePipeline。

架构
  - ThreadingHTTPServer 处理并发请求（前端轮询 + 文件上传）
  - JobManager 维护任务队列；**单个后台 worker 串行执行**（GPU 独占，避免并发抢显存）
  - 用 logging.Handler 挂到 "pdfenhance" logger，实时捕获逐页日志并解析「阶段 + 进度」
  - REST 接口（JSON）：
      GET  /                       前端页面
      GET  /api/config             配置 + 可用后端清单
      GET  /api/jobs               任务列表
      GET  /api/jobs/<id>          单任务状态
      GET  /api/jobs/<id>/log?after=N   增量日志
      POST /api/jobs               创建任务 {pdf_path, ...params, start?}
      POST /api/jobs/<id>/start    开始单个任务
      POST /api/jobs/start-all     开始全部（排队）
      POST /api/jobs/<id>/stop     停止任务
      POST /api/jobs/<id>/remove   移除任务
      POST /api/jobs/clear-finished 清空已完成/失败
      POST /api/upload             浏览器上传 PDF
      POST /api/pick               本地原生文件选择框（tkinter，可选）
      GET  /api/compare?job=&page=&width=   原书/增强页图（base64）
      GET  /api/open?path=&mode=   在资源管理器中打开/定位
"""
import base64
import json
import logging
import os
import subprocess
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(PROJECT_ROOT, "src", "web")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
VERSION = "0.2"

# 任务的默认参数（前端「高级设置」会覆盖这些默认值）
DEFAULTS = {
    "pages": None,                  # [start, end] 0-based 或 null=全部
    "scale": 2,                     # 2 | 4
    "quality": 85,                  # JPEG 质量
    "ocr": False,                   # 内置 OCR 隐藏层（默认关，交给 FineReader）
    "restore_enabled": True,
    "restore_backend": "docres",    # docres | none
    "sr_enabled": True,
    "sr_backend": "realesrgan-general",  # realesrgan-general | swinir | none
    "routing": "off",               # off | by-type
    "dpi": "auto",                  # auto | 数字
    "device": "auto",               # auto | cuda | cpu
    "monochrome": "1bit",           # 1bit | jbig2
    "fp16": True,
}


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config", "config.yaml"),
              "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def available_backend_names():
    try:
        from src.backends import available_backends
        regs = available_backends()
        return {k: sorted(v.keys()) for k, v in regs.items()}
    except Exception as e:  # pragma: no cover
        return {"restore": ["none"], "sr": ["none"], "error": str(e)}


# --------------------------------------------------------------------------
# 任务
# --------------------------------------------------------------------------
class Job:
    def __init__(self, pdf_path, params):
        self.id = f"job{int(time.time() * 1000)}{os.getpid() % 1000}"
        self.pdf_path = pdf_path
        self.name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.params = dict(DEFAULTS)
        self.params.update({k: v for k, v in (params or {}).items() if v is not None})
        self.status = "draft"        # draft | queued | running | done | error | stopped
        self.stage = "init"          # init | render | ocr | enhance | rebuild | done | stopped
        self.progress = 0.0          # 0..1
        self.current_page = 0
        self.total_pages = 0
        self.page_start = 0          # 处理范围相对原文档的 0-based 起始页
        self.started_at = None
        self.log_lines = []
        self.result = None
        self.error = None
        self._log_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def append_log(self, line):
        with self._log_lock:
            self.log_lines.append(line)
            if len(self.log_lines) > 4000:
                self.log_lines = self.log_lines[-4000:]

    def to_dict(self):
        d = {
            "id": self.id,
            "name": self.name,
            "pdf_path": self.pdf_path,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "current_page": self.current_page,
            "total_pages": self.total_pages,
            "page_start": self.page_start,
            "result": self.result,
            "error": self.error,
        }
        d.update(self.params)
        return d


class JobManager:
    """线程安全的任务队列 + 串行 worker。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}
        self._order = []
        self._wake = threading.Event()
        self._worker = None
        self._worker_lock = threading.Lock()

    # ---- CRUD ----
    def create(self, pdf_path, params):
        job = Job(pdf_path, params)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        return job

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def list(self):
        with self._lock:
            return [self._jobs[i] for i in self._order if i in self._jobs]

    def remove(self, job_id):
        with self._lock:
            self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)

    def clear_finished(self):
        n = 0
        with self._lock:
            for jid in list(self._order):
                j = self._jobs.get(jid)
                if j and j.status in ("done", "error", "stopped"):
                    del self._jobs[jid]
                    self._order.remove(jid)
                    n += 1
        return n

    # ---- scheduling ----
    def enqueue(self, job):
        job.status = "queued"
        self.ensure_worker()
        self._wake.set()

    def enqueue_all(self):
        with self._lock:
            for jid in self._order:
                j = self._jobs.get(jid)
                if j and j.status == "draft":
                    j.status = "queued"
        self.ensure_worker()
        self._wake.set()

    def ensure_worker(self):
        with self._worker_lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._loop, daemon=True)
                self._worker.start()

    def _next(self):
        with self._lock:
            for jid in self._order:
                j = self._jobs.get(jid)
                if j and j.status == "queued":
                    return j
        return None

    def _loop(self):
        while True:
            job = self._next()
            if job is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            try:
                _run_job(job)
            except Exception:  # 兜底，绝不让 worker 挂掉
                traceback.print_exc()


MANAGER = JobManager()


class _JobLogHandler(logging.Handler):
    """把 pipeline 日志转发到任务，并解析阶段与进度。"""

    def __init__(self, job):
        super().__init__()
        self.job = job

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        job = self.job
        job.append_log(msg)
        import re
        try:
            if msg.startswith("render:"):
                job.stage = "render"
            if "OCR pass" in msg:
                job.stage = "ocr"
            m = re.search(r"page (\d+) OCR", msg)
            if m and job.total_pages:
                job.stage = "ocr"
                frac = min(1.0, int(m.group(1)) / job.total_pages)
                job.progress = max(job.progress, 0.03 + 0.22 * frac)
            m = re.search(r"page (\d+) \[", msg)
            if m:
                job.stage = "enhance"
                if job.total_pages:
                    job.current_page = max(1, int(m.group(1)) - job.page_start)
                    frac = min(1.0, job.current_page / job.total_pages)
                    job.progress = max(job.progress, 0.30 + 0.55 * frac)
            if msg.startswith("rebuild:"):
                job.stage = "rebuild"
                job.progress = max(job.progress, 0.90)
            if msg.startswith("total:"):
                job.stage = "done"
                job.progress = 1.0
        except Exception:
            pass


def _run_job(job: Job):
    """worker：真正跑一个任务（阻塞）。"""
    from src.pipeline import EnhancePipeline, Cancelled

    plog = logging.getLogger("pdfenhance")
    plog.setLevel(logging.INFO)
    handler = _JobLogHandler(job)
    handler.setFormatter(logging.Formatter("%(message)s"))
    plog.addHandler(handler)

    try:
        job.status = "running"
        job.stage = "init"
        job.started_at = time.time()
        job.append_log(f"===== 开始处理: {job.name} =====")

        # 依 GUI 参数构造 cfg
        cfg = load_config()
        p = job.params
        cfg["render"]["dpi"] = _norm_dpi(p.get("dpi", "auto"))
        cfg["document_restoration"]["enabled"] = bool(p.get("restore_enabled", True))
        cfg["document_restoration"]["backend"] = p.get("restore_backend", "docres")
        cfg["super_resolution"]["enabled"] = bool(p.get("sr_enabled", True))
        cfg["super_resolution"]["backend"] = p.get("sr_backend", "realesrgan-general")
        cfg["super_resolution"]["scale"] = int(p.get("scale", 2))
        if p.get("routing"):
            cfg["super_resolution"]["routing"] = p["routing"]
        cfg["ocr"]["enabled"] = bool(p.get("ocr", False))
        cfg["compression"]["jpeg_quality"] = int(p.get("quality", 85))
        cfg["compression"]["monochrome"] = p.get("monochrome", "1bit")
        cfg["gpu"]["device"] = p.get("device", "auto")
        cfg["gpu"]["fp16"] = bool(p.get("fp16", True))
        cfg["debug"]["keep_temp"] = False

        # 统计页数 / 计算范围
        import pymupdf as fitz
        doc = fitz.open(job.pdf_path)
        total = doc.page_count
        doc.close()

        pages = p.get("pages")
        if pages and total:
            start = max(0, min(int(pages[0]), total - 1))
            end_raw = pages[1]
            end = (total - 1) if end_raw is None else max(
                start, min(int(end_raw), total - 1))
            page_range = (start, end)
            job.page_start = start
            job.total_pages = end - start + 1
        else:
            page_range = None
            job.page_start = 0
            job.total_pages = total

        in_size = os.path.getsize(job.pdf_path)
        out_name = f"{job.name}_enhanced.pdf"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        pipe = EnhancePipeline(cfg, PROJECT_ROOT)

        job.append_log(f"参数: SR×{job.params['scale']} · JPEG q{job.params['quality']} "
                       f"· 修复={job.params['restore_backend']} "
                       f"· SR={job.params['sr_backend']} · 路由={job.params['routing']} "
                       f"· 设备={job.params['device']} · 页码={page_range or '全部'}")

        t0 = time.time()
        pipe.process_pdf(job.pdf_path, out_path, save_images_dir=None,
                         page_range=page_range, should_stop=job._stop.is_set)
        elapsed = time.time() - t0

        if job._stop.is_set():
            job.status = "stopped"
            job.stage = "stopped"
            job.append_log("已手动停止")
            return

        out_size = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
        job.result = {
            "out_path": out_path,
            "out_name": out_name,
            "in_size": in_size,
            "out_size": out_size,
            "elapsed": elapsed,
            "ratio": (out_size / in_size) if in_size else 0,
        }
        job.status = "done"
        job.stage = "done"
        job.progress = 1.0
        job.append_log(f"完成: {out_name} "
                       f"({in_size/1024/1024:.1f}MB -> {out_size/1024/1024:.1f}MB, "
                       f"{elapsed:.0f}s)")
    except Cancelled:
        job.status = "stopped"
        job.stage = "stopped"
        job.append_log("已手动停止")
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.append_log(f"错误: {e}")
        job.append_log(traceback.format_exc())
    finally:
        try:
            plog.removeHandler(handler)
        except Exception:
            pass


def _norm_dpi(v):
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
        return s or "auto"
    return v


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = f"PDFEnhancerGUI/{VERSION}"

    def log_message(self, *args):
        pass  # 静默访问日志

    # ---- helpers ----
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        if not os.path.isfile(path):
            self._send_json({"error": "not found"}, 404)
            return
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    # ---- GET ----
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._send_file(os.path.join(WEB_DIR, "index.html"),
                            "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            fn = os.path.basename(path)
            ext = os.path.splitext(fn)[1][1:].lower()
            mime = {"css": "text/css", "js": "application/javascript",
                    "png": "image/png", "svg": "image/svg+xml",
                    "ico": "image/x-icon"}.get(ext, "text/plain")
            self._send_file(os.path.join(WEB_DIR, "static", fn), mime)
        elif path == "/api/config":
            self._send_json({
                "config": load_config(),
                "backends": available_backend_names(),
                "defaults": DEFAULTS,
                "version": VERSION,
                "root": PROJECT_ROOT,
            })
        elif path == "/api/jobs":
            self._send_json([j.to_dict() for j in MANAGER.list()])
        elif path.startswith("/api/jobs/") and path.endswith("/log"):
            job = MANAGER.get(path.split("/")[3])
            if not job:
                self._send_json({"error": "no such job"}, 404)
                return
            after = int(qs.get("after", ["0"])[0])
            lines = job.log_lines[after:]
            self._send_json({"lines": lines, "count": len(job.log_lines)})
        elif path.startswith("/api/jobs/"):
            job = MANAGER.get(path.split("/")[3])
            if not job:
                self._send_json({"error": "no such job"}, 404)
                return
            self._send_json(job.to_dict())
        elif path == "/api/compare":
            self._handle_compare(qs)
        elif path == "/api/open":
            self._handle_open(qs)
        else:
            self._send_json({"error": "not found"}, 404)

    # ---- POST ----
    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/jobs":
            try:
                body = self._read_json()
            except Exception:
                self._send_json({"error": "bad json"}, 400)
                return
            pdf_path = body.pop("pdf_path", None)
            start = body.pop("start", False)
            if not pdf_path or not os.path.isfile(pdf_path):
                self._send_json({"error": "pdf not found"}, 400)
                return
            if body.get("pages"):
                pg = body["pages"]
                body["pages"] = [int(pg[0]),
                                 (int(pg[1]) if pg[1] is not None else None)]
            job = MANAGER.create(pdf_path, body)
            if start:
                MANAGER.enqueue(job)
            self._send_json(job.to_dict())

        elif path == "/api/jobs/start-all":
            MANAGER.enqueue_all()
            self._send_json({"ok": True})

        elif path == "/api/jobs/clear-finished":
            n = MANAGER.clear_finished()
            self._send_json({"ok": True, "removed": n})

        elif path.startswith("/api/jobs/") and path.endswith("/start"):
            job = MANAGER.get(path.split("/")[3])
            if not job:
                self._send_json({"error": "no such job"}, 404)
                return
            if job.status not in ("draft", "queued"):
                self._send_json({"error": "job not startable"}, 400)
                return
            MANAGER.enqueue(job)
            self._send_json(job.to_dict())

        elif path.startswith("/api/jobs/") and path.endswith("/stop"):
            job = MANAGER.get(path.split("/")[3])
            if not job:
                self._send_json({"error": "no such job"}, 404)
                return
            job._stop.set()
            if job.status in ("draft", "queued"):
                job.status = "stopped"
                job.stage = "stopped"
            self._send_json({"ok": True})

        elif path.startswith("/api/jobs/") and path.endswith("/remove"):
            job = MANAGER.get(path.split("/")[3])
            if not job:
                self._send_json({"error": "no such job"}, 404)
                return
            if job.status == "running":
                self._send_json({"error": "任务正在运行，请先停止"}, 400)
                return
            MANAGER.remove(job.id)
            self._send_json({"ok": True})

        elif path == "/api/upload":
            self._handle_upload()

        elif path == "/api/pick":
            self._handle_pick()

        else:
            self._send_json({"error": "not found"}, 404)

    # ---- compare ----
    def _handle_compare(self, qs):
        job_id = qs.get("job", [None])[0]
        page_no = int(qs.get("page", ["1"])[0])
        width = int(qs.get("width", ["1500"])[0])
        job = MANAGER.get(job_id) if job_id else None
        if not job or not job.result:
            self._send_json({"error": "任务尚未完成"}, 400)
            return
        try:
            import cv2
            import numpy as np
            import pymupdf as fitz

            def render_page(pdf, idx, tw):
                d = fitz.open(pdf)
                idx = max(0, min(idx, d.page_count - 1))
                pg = d[idx]
                z = tw / pg.rect.width
                pix = pg.get_pixmap(matrix=fitz.Matrix(z, z),
                                    colorspace=fitz.csRGB, alpha=False)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(
                    pix.height, pix.width, 3)
                out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                d.close()
                return out

            count = job.total_pages or 1
            page_no = max(1, min(page_no, count))
            orig_idx = job.page_start + (page_no - 1)
            orig = render_page(job.pdf_path, orig_idx, width)
            enh = render_page(job.result["out_path"], page_no - 1, width)

            def to_b64(img):
                ok, buf = cv2.imencode(".png", img,
                                       [cv2.IMWRITE_PNG_COMPRESSION, 3])
                return "data:image/png;base64," + base64.b64encode(
                    buf.tobytes()).decode("ascii")

            self._send_json({
                "orig": to_b64(orig),
                "enh": to_b64(enh),
                "page": page_no,
                "count": count,
                "name": job.name,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    # ---- open in explorer ----
    def _handle_open(self, qs):
        path = qs.get("path", [None])[0]
        mode = qs.get("mode", ["open"])[0]
        if not path:
            self._send_json({"error": "no path"}, 400)
            return
        ap = os.path.abspath(path)
        if not ap.startswith(PROJECT_ROOT):  # 只允许项目内路径
            self._send_json({"error": "forbidden"}, 403)
            return
        if not os.path.exists(ap):
            self._send_json({"error": "not found"}, 404)
            return
        try:
            if mode == "reveal" and os.path.isfile(ap):
                subprocess.Popen(["explorer", "/select,", ap])
            else:
                os.startfile(ap)  # noqa: 仅 Windows
            self._send_json({"ok": True})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    # ---- upload ----
    def _handle_upload(self):
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._send_json({"error": "expect multipart"}, 400)
            return
        boundary = ctype.split("boundary=")[1].strip().strip('"')
        parts = data.split(b"--" + boundary.encode())
        for part in parts:
            if b"filename=" in part and b"Content-Type" in part:
                header, _, content = part.partition(b"\r\n\r\n")
                content = content.rsplit(b"\r\n", 1)[0]
                fn = header.split(b'filename="')[1].split(b'"')[0]
                try:
                    fn = fn.decode("utf-8")
                except Exception:
                    fn = fn.decode("latin-1")
                safe = os.path.basename(fn)
                dest = os.path.join(INPUT_DIR, safe)
                with open(dest, "wb") as f:
                    f.write(content)
                self._send_json({"pdf_path": dest, "name": safe})
                return
        self._send_json({"error": "no file"}, 400)

    # ---- native file picker (optional) ----
    def _handle_pick(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            paths = filedialog.askopenfilenames(
                title="选择扫描版 PDF",
                initialdir=INPUT_DIR,
                filetypes=[("PDF 文件", "*.pdf"), ("所有文件", "*.*")])
            root.destroy()
            self._send_json({"paths": list(paths)})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)


# --------------------------------------------------------------------------
def main(port=8765, open_browser=True):
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(message)s",
                            datefmt="%H:%M:%S")

    MANAGER.ensure_worker()

    httpd = None
    last_err = None
    for candidate in range(port, port + 10):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            port = candidate
            break
        except OSError as e:
            last_err = e
    if httpd is None:
        print(f"[ERROR] 无法绑定端口 {port}: {last_err}")
        return

    url = f"http://127.0.0.1:{port}"
    print("=" * 52)
    print(f"  PDF 增强 GUI 已启动")
    print(f"  地址: {url}")
    print("  按 Ctrl+C 停止")
    print("=" * 52)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    import sys
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    _port = int(positional[0]) if positional else 8765
    main(_port, open_browser=("--no-browser" not in flags))
