# 扫描版 PDF AI 画质增强工具

处理扫描版教材 / 论文 / 技术书：**文档修复（DocRes）+ 超分辨率 + 自适应压缩**，  
全部资源隔离在项目目录内 —— 删除本项目文件夹即完全卸载。

## 当前状态

- [x] Phase 1 调研与方案（见 `PLAN.md`）
- [x] Phase 2 baseline：PDF → 渲染 → DocRes 修复 → 超分 → PDF 重建
- [x] Phase 3 OCR 隐藏层（RapidOCR，Windows 原生，可 Ctrl+F 搜索中文）
- [x] Phase 4 页面六分类 + 黑白页 1-bit/JBIG2 压缩 + 字体子集化
- [x] Phase 5 模型可切换架构扩展（backend 注册制，SwinIR / RealESRGAN 可切换）
- [x] Phase 6 对比测试报告（`output/benchmark_report.md`）
- [x] 混合路由 SR（`routing: by-type`，文字页→RealESRGAN、插图页→SwinIR）
- [x] auto DPI 渲染（按每页内嵌扫描图名义 DPI 零插值，防无意义放大）
- [x] MRC 分层压缩（v3：JPEG 背景 + 多色 1-bit 前景层，省 2.3~3.1 倍）
- [x] 页数范围参数（`--pages N` / `--pages a-b`）
- [x] 本地 Web GUI（`run_gui.bat`，拖入/队列/进度/对比图，OCR 默认关闭）

## 1. 安装（一次性）

```bat
scripts\install.bat
```

自动完成：~~项目内 venv（`env\`）→ PyTorch cu126 → PyMuPDF/OpenCV 等（约 3.5GB）。  
无需管理员权限；不修改系统 Python、不写注册表、不装任何全局包~~。

模型权重已包含在 `models\`：

| 文件                         | 用途                     | 大小    |
| -------------------------- | ---------------------- | ----- |
| `docres.pkl`               | DocRes 文档修复（CVPR 2024） | 183MB |
| `realesr-general-x4v3.pth` | 轻量文档友好超分               | 5MB   |
| `rapidocr/*.onnx`          | OCR 检测/识别/方向分类（中英） | 13.5MB |
| `../tools/fonts/simsun.ttc`| 隐藏文字层 CJK 字体（宋体）    | 18MB  |
| `../tools/jbig2/`          | jbig2.exe + DLL（实验性）  | 5.1MB |

## 2. 运行

```bat
run.bat input\book.pdf          # 处理单个 PDF
run.bat                         # 处理 input\ 下所有 PDF
run.bat --scale 4               # 放大 4 倍（默认 2）
run.bat --quality 90            # JPEG 质量（默认 85）
run.bat --task deshadowing      # DocRes 任务：appearance | deshadowing | deblurring
run.bat --model none --no-sr    # 只做文档修复
run.bat --device cpu            # 强制 CPU
run.bat --save-images           # 同时导出页图 PNG
run.bat --ocr                   # 强制开启 OCR 隐藏层（默认已开）
run.bat --no-ocr                # 关闭 OCR
run.bat --dpi auto              # 按每页内嵌扫描图名义 DPI 零插值渲染（默认）
run.bat --dpi 300               # 固定渲染 DPI（矢量页 / 无内嵌图时用）
run.bat --routing by-type       # 混合路由 SR：文字页→RealESRGAN、插图页→SwinIR
run.bat --list-backends         # 列出已注册的修复/超分后端
run.bat --pages 50              # 只处理前 50 页
run.bat --pages 10-59           # 只处理第 10~59 页（1-based 含两端）
```

输出：`output\<书名>_enhanced.pdf`；中间页图（默认保留）在 `temp\<书名>\`，  
日志在 `logs\<书名>.log`（含每页耗时与显存峰值）。

## 2.5 图形界面（GUI）

不想记命令行参数时，双击 `run_gui.bat` 启动本地 Web 界面（零额外依赖，  
纯 Python 标准库 + 原生 HTML/JS，**服务就绪后浏览器自动打开** `http://127.0.0.1:8765`）：

- **任务队列**：拖入（或点「选择文件」）多本 PDF 排队，实时**阶段进度**
  （渲染 → OCR → 修复·超分 → 重建 → 完成）+ 逐页日志；
- 每本单独设置：**页数范围**、SR 倍数、JPEG 质量，以及可展开的**高级参数**
  （修复/SR 后端、混合路由、渲染 DPI、CPU/GPU、单色编码、fp16、OCR）；
- **全局操作**：开始全部 / 全部停止 / 清空已完成；任务串行执行（GPU 独占）；
- **对比查看**：选任务+页码 → **拖动滑块左右对比**原书与增强效果，可缩放、下载对比拼图；
- **高级设置**：默认参数存本地，新任务自动套用；
- 完成后可一键在资源管理器中**打开/定位输出文件**。

OCR 默认**关闭**：本项目专注图像增强（DocRes + 超分 + MRC 压缩），文字层建议  
用 ABBYY FineReader 等专业 OCR 工具手动添加（识别精度更高，且避免了内置  
RapidOCR 的漏检/错位问题）。

```
run_gui.bat            # 默认端口 8765（被占用时自动向后找空闲端口）
run_gui.bat 9000       # 指定端口
```

### OCR 隐藏文字层（Phase 3）

默认开启：RapidOCR（PP-OCRv3 系，ONNX，Windows 原生）在 200 DPI 下识别，  
把文字以**不可见文字层**写回 PDF（`render_mode=3`），因此：

- 增强后的 PDF 可用 **Ctrl+F 搜索**中文/英文/数字；
- 文字层不影响页面视觉（完全隐藏）；
- 中文用内置宋体 `tools\fonts\simsun.ttc` 嵌入，非中文环境也能正确显示。

控制：`config\config.yaml` → `ocr:`（`enabled` / `dpi` / `use_cuda`）。  
CPU 推理约 2 秒/页，一般无需 GPU 版 onnxruntime。

## 3. 更换模型

编辑 `config\config.yaml`：

```yaml
document_restoration:
  backend: docres        # none = 关闭
  task: appearance       # appearance / deshadowing / deblurring

super_resolution:
  backend: realesrgan-general   # realesrgan-general | swinir | none
  scale: 2                      # 2 或 4
  # routing: by-type            # 混合路由：文字页→realesrgan、插图页→swinir
```

新模型权重放入 `models\`，在 `src\backends\` 添加对应 backend 类并注册即可，  
流水线其余部分无需改动。

## 4. 黑白 / 灰度 / 彩色 PDF（Phase 4）

流水线在**增强后**逐页自动分类为六类，分别编码：

| 页面类型 | 判定 | 编码 |
| --- | --- | --- |
| `bw-text` 纯黑白文字 | 无彩色 + 几乎无中间调 | 自适应二值化 → **1-bit**（FlateDecode，无损） |
| `gray-text` / `gray-image` 灰度 | 有中间调、无彩色 | 单通道灰度 JPEG（省 1/3） |
| `color-text` / `mixed` 图文混排 | 有彩色 + 有文字 | **MRC 分层 / 整页 JPEG 双编码取小**（见 §5） |
| `color-image` 纯图 | 有彩色、无文字 | 彩色 JPEG |

- 分类在 DocRes 修复**之后**进行：光照不均修复后，黑白文字页才呈现真实二值特征。
- 黑白页 1-bit 是默认方案：约 112KB/页 @300DPI×2，所有阅读器兼容。
- **JBIG2（实验性）**：`config.yaml` → `compression.monochrome: jbig2` 可切换
  （`tools/jbig2/jbig2.exe`，符号模式，实测约 32KB/页，压缩率再降 3.5 倍）。
  但有两个代价：① 符号合并有"相似字替换"的幻觉风险；② PyMuPDF/MuPDF 不支持
  JBIG2 解码，保存时会剥掉 JBIG2Decode 滤镜——流水线会自动检测损坏页并整册
  回退到 1-bit，因此该选项当前实际不生效，留待引入 pikepdf 后启用。

## 5. MRC 分层压缩（图文混排页）

对 `mixed` / `color-text` / `gray-text` 页，不再整页压一张 JPEG，而是
**混合栅格内容（Mixed Raster Content）** 分层编码，取体积更小者（`render.py`）：

1. **背景层**：彩色图/照片（把文字区域涂白）压 JPEG，q85；
2. **前景层**：文字笔画用 `jbig2enc -O` 阈值分割成连通域，按实测墨色分桶
   （黑组 + 最多 3 个彩色组），每组一张 **1-bit 蒙版（stencil）** + 一个实测
   填充色（FlateDecode 无损）。

这样文字保持锐利、彩色标注（如正文里的蓝字批注）不会被二值化抹掉或涂黑，
而照片/插图仍走有损 JPEG。逐页与整页 JPEG 双编码比较后取小，因此：

- 文字密集页 → MRC 胜（实测省 **2.3~3.1 倍**）；
- 图片占主导 / 无有效文字的页 → 整页 JPEG 胜（自动回退）。

关键实现细节（踩坑记录见项目记忆）：

- **笔画不加粗**：涂色用原始 mask + 自适应灰度带修剪，涂白单独用加宽 mask，
  避免二值化把抗锯齿边缘吞进前景导致笔画变胖（实测墨迹覆盖 11.25%→9.25%）。
- **彩色文字保护**：小彩色笔画放行进前景，只排除大块彩色连通域（保护截图/插图）。
- **MuPDF ImageMask 陷阱**：stencil 必须 `update_stream(compress=1)` 让 PyMuPDF
  自己压缩（手写 zlib + `/Filter /FlateDecode` 会渲染全黑）；极性钉死为
  ink 位=0 + `/Decode [0 1]` + 黑填充。

## 6. 控制输出质量与体积

- 清晰度：`--scale 4` + `--task deblurring`（更激进）；保守用默认。
- 体积：`--quality`（60–95）；灰度页自动省 1/3。
- `--dpi` 控制渲染分辨率（默认 300，极高 DPI 会显著增加耗时与显存）。

## 7. 处理扫描教材建议

扫描质量差（阴影/泛黄/模糊）→ 默认配置（appearance 修复 + x2 超分）。  
扫描质量尚可、只想放大阅读 → `run.bat --no-restore --scale 4`。

## 8. 完全卸载

直接删除项目文件夹 `E:\04_Projects\PDF_AI_Enhancer`。  
本项目所有 Python 包、模型、缓存、临时文件、日志都在文件夹内，  
系统（注册表 / 全局 site-packages / 用户缓存目录）**零残留**。

## 目录结构

```
run.bat            主入口（自动设置全部缓存环境变量）
run_gui.bat        图形界面入口（本地 Web GUI，浏览器操作）
config/config.yaml 全部参数
scripts/           安装/测试/环境脚本
src/               流水线代码（渲染/分类/修复/超分/重建/OCR）
src/gui.py         本地 Web GUI 后端（零依赖，标准库 http.server）
src/web/           GUI 前端（原生 HTML/JS）
third_party/       vendored 模型定义（DocRes-Restormer, SRVGG）
env/               独立 Python venv
models/            模型权重
tools/             第三方可执行程序（jbig2.exe，phase 4）
input/ output/ temp/ cache/ logs/
```

## 产物示例（output/）

| 文件 | 说明 |
| --- | --- |
| `云计算导论_样张_p14-34_enhanced.pdf` | 21 页样张，55.3MB → 25.9MB，可 Ctrl+F 搜索 |
| `医疗信息管理平台-需求规格说明书_enhanced.pdf` | 完整跑通的一本 |
| `计算机组成原理_p17_彩色增强.pdf` | 彩色文字标注验证页（黑+蓝两层前景） |
| `benchmark_report.md` | Phase 6 边缘锐度/耗时/显存对比报告 |
| `compare_*.png` | 各页增强前后 / 编码方案对比图 |

> 目前 output 里多为样张/单页验证产物；**全本**（如 340 页《云计算导论》）
> 尚未从头到尾跑完，属待办（见项目记忆 `MEMORY.md` → 待办）。
