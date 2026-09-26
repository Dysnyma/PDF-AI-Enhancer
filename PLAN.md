# PLAN.md — 扫描版 PDF AI 画质增强工具

> 状态：第一阶段（技术调研 + 环境检测 + 方案设计）
> 日期：2026-09-25
> 本阶段**未安装任何软件、未下载任何模型、未修改任何系统配置**。

---

## 1. 项目目标回顾

处理对象：扫描版教材 / 论文 / 技术书（黑白、灰度、彩色扫描 PDF）。

核心目标：
1. 提高扫描页面整体视觉清晰度，**重点是文字区域**；
2. 彩色内容不得二值化；
3. 文字区域与图片区域采用**不同的处理与编码策略**；
4. 输出 PDF 兼顾：清晰度、放大阅读效果、文件大小、彩色保留、可搜索（OCR 隐藏层）、PDF 阅读器兼容性；
5. 全部资源隔离在项目目录内，删除项目文件夹即完全卸载。

---

## 2. 环境检测结果（2026-09-25 实测）

| 项目 | 检测结果 | 结论 |
|---|---|---|
| GPU | NVIDIA RTX 4060 Laptop, 8188MB | WDDM 模式 |
| 系统占用 VRAM | 约 1.3GB（桌面程序常驻） | **实际可用约 6.5–6.8GB**，模型推理需按 ≤6GB 规划 |
| 驱动 / CUDA | 驱动 610.74，CUDA 运行时 13.3 | 可运行任何 PyTorch cu121–cu126 版本 |
| E 盘剩余空间 | 143GB | 充足（本项目全量预计 ≤12GB） |
| 工作目录 | `E:\04_Projects\PDF_AI_Enhancer`（空目录） | 干净起点 |
| OS | Windows 11 + bash shell | jbig2.exe 有官方预编译版，无需 MSYS2 编译 |

硬件关键约束：**8GB VRAM 中约 1.3GB 被系统占用**。所有模型推理必须支持"分块（tiling）+ FP16 + OOM 自动降级"，绝不能假设 8GB 全部可用。

---

## 3. 技术调研结论

### 3.1 候选方案评估总表

评估维度对应你的 A–M 标准。✅ 满足 / ⚠️ 部分满足 / ❌ 不满足。

#### AI 模型类

| 方案 | A 针对文档 | B 扫描书 | C/D/E 黑白/灰度/彩色 | F/G/H GPU与8GB显存 | I/J 批量/自动化 | K 破坏文字 | L 幻觉字符 | 结论 |
|---|---|---|---|---|---|---|---|---|
| **DocRes** (CVPR 2024) | ✅ 专为文档设计 | ✅ | ✅/✅/✅（二值化任务仅输出 1bpp） | ✅ CNN 架构，权重小，≤4GB 即可 | ✅ Python 推理脚本 | ✅ 不破坏 | ✅ 非生成式，无幻觉 | **采用（修复层主力）** |
| **Real-ESRGAN** | ❌ 照片导向 | ⚠️ | ⚠️/✅/✅ | ✅ 轻量模型 1–2GB | ✅ | ⚠️ 笔画可能风格化 | ⚠️ **中风险**：小字可能伪影/变形 | 采用（baseline + 对比项） |
| **Upscayl** | ❌ 即 Real-ESRGAN 的 GUI | ⚠️ | ⚠️/✅/✅ | ✅ | ❌ GUI 不利自动化 | ⚠️ 同上 | ⚠️ 同上 | 不安装，用 Real-ESRGAN 等效对比 |
| **SwinIR** | ⚠️ 通用但 PSNR 导向、保真 | ✅ | ✅/✅/✅ | ⚠️ 原始实现慢且吃显存，需 tiling | ✅ | ✅ 保真好 | ✅ 风险低 | 采用（对比项 + 备选 backend） |
| **TADiSR** (NeurIPS 2025) | ✅ 文字感知超分 | ⚠️ 面向自然场景文字 | ⚠️/✅/✅ | ❌ **SD 扩散底座，8GB 仅能小 patch 慢速推理** | ⚠️ 每页需切块多次推理，很慢 | ✅ 专门保护文字结构 | ⚠️ 扩散模型存在轻微风险 | 暂缓：仅第六阶段对比实验时再决定是否下载（约 +5GB） |
| **TextSR** (AAAI 2024) | ⚠️ 面向**自然场景文字**（街拍、招牌），非文档版式 | ❌ | ⚠️ | ⚠️ 扩散底座 | ⚠️ | ⚠️ | ⚠️ | **不采用**：训练数据分布与扫描书不匹配 |

**关键判断**：没有任何单一模型同时解决"修复 + 超分"。正确做法是**分层**：
- **修复层**用 DocRes（去阴影 / 外观增强 / 去模糊 / 二值化，非生成式、无幻觉）；
- **超分层**用可切换 backend（Real-ESRGAN / SwinIR / 未来的 TADiSR），文字保真度在第六阶段用实测数据说话。

#### 传统工具类

| 工具 | 用途 | Windows 原生 | 结论 |
|---|---|---|---|
| **PyMuPDF (fitz)** | PDF 渲染 / 重建 / 隐藏文字层写入 / DPI 控制 | ✅ 纯 pip | **主线 PDF 引擎**，一个库覆盖渲染+重建+OCR层 |
| **OCRmyPDF** | OCR + PDF/A 输出 | ❌ **官方明确不支持原生 Windows**，只能 WSL 或 Docker | 采用为**可选 backend**（你已有 WSL）；主力 OCR 不押在它身上 |
| **Tesseract** | OCRmyPDF 默认引擎 | ⚠️ | 中文准确率约 75–85%，明显弱于 Paddle 系 |
| **RapidOCR**（PP-OCR 的 ONNX 版） | OCR | ✅ 纯 pip，无 Paddle 框架依赖 | **主力 OCR**：中文+英文+数字混合识别强，模型仅约 15MB，CPU 也可接受、CUDA 可加速；隐藏层由 PyMuPDF 写入 |
| **jbig2enc** | 黑白页 JBIG2 压缩（比 G4 再小 30–60%） | ✅ **hank-ai/hankpdf 项目提供 Windows 预编译 jbig2.exe**（portable，绿色解压到 tools/，无需管理员） | 采用；注意：**禁用 refinement 模式**（官方已知 bug 会导致 Acrobat 崩溃） |
| CCITT G4 | 黑白页兜底压缩 | ✅ | JBIG2 不可用时的 fallback，体积约大 10–20% |
| **JPEG2000 (OpenJPEG)** | 灰度/彩色压缩 | ✅ 有二进制 | **默认不启用**：JPXDecode 在部分 PDF 阅读器（老版本/移动端）兼容性差；保留为 config 可选项 |
| **ScanTailor Advanced** | 交互式扫描处理（纠偏、分页等） | ⚠️ | **不集成**：GUI 交互式，无法自动化；如某本书扫描质量极差，建议先用它手工预处理再放入 input/ |
| Poppler / ImageMagick | 图像提取 / 转换 | ⚠️ | **不安装**：功能被 PyMuPDF + OpenCV 完全覆盖，减少依赖面 |
| OpenCV (Python) | 预处理：去噪、自适应二值化、倾斜校正 | ✅ 纯 pip | 采用（预处理层） |

### 3.2 明确不安装清单（第一阶段决策）

1. **TextSR** —— 训练分布是自然场景文字，不是文档；不适配扫描教材。
2. **Upscayl** —— 本质是 Real-ESRGAN 的 GUI 封装，无法命令行自动化；对比实验直接用 Real-ESRGAN 等效替代。
3. **ScanTailor Advanced** —— 交互式工具，无法进自动化流水线。
4. **Poppler / ImageMagick** —— 与 PyMuPDF/OpenCV 功能重叠。
5. **PaddleOCR 完整框架** —— 需要 PaddlePaddle 深度学习框架（数 GB）；改用 RapidOCR（ONNX 权重同源，pip 轻量安装）。
6. **TADiSR**（暂缓）—— 扩散模型底座，在 6.5GB 可用显存下只能小 patch 慢速推理，整本书批量处理不现实；第六阶段对比实验前再确认是否值得下载。

---

## 4. 总体架构设计

### 4.1 处理流水线

```
PDF (input/)
  ↓ ① 页面提取与渲染 (PyMuPDF, 可配置 DPI)
  ↓ ② 页面类型自动分类 (色彩统计 + 文字密度 + 图片区域检测)
  ↓ ③ 预处理 (OpenCV: 去噪 / 倾斜校正 / 裁边，可配置)
  ↓ ④ Document Restoration (DocRes: 去阴影/外观/去模糊，非生成式，无幻觉)
  ↓ ⑤ Super Resolution (可切换: realesrgan / swinir / none)
  ↓ ⑥ 分区域处理 (文字区域 vs 图片区域: 不同的二值化/保留策略)
  ↓ ⑦ OCR (可切换: rapidocr / ocrmypdf-wsl / none → 输出隐藏文字层)
  ↓ ⑧ PDF 重建 (PyMuPDF: 按页面类型嵌入图像)
  ↓ ⑨ 压缩 (黑白→JBIG2/G4; 灰度→JPEG; 彩色→JPEG; 可选 JP2)
  ↓ ⑩ 最终 PDF (output/xxx_enhanced.pdf [+ xxx_ocr.pdf 可选])
```

每一步都是独立的 Stage 类，由 `config/config.yaml` + 命令行参数驱动，**可单独开关、可替换实现**。

### 4.2 页面类型自动分类（第⑵步）

| 类型 | 判定特征（自动计算） | 处理策略 |
|---|---|---|
| 纯黑白文字页 | 无彩色通道差异；灰度直方图双峰；文字覆盖率中高 | DocRes 二值化（或自适应二值化）→ **JBIG2** |
| 灰度文字页 | 无彩色；直方图连续（纸张泛黄/灰色） | DocRes 外观增强 → SR → 灰度 JPEG（或可配置转二值+JBIG2） |
| 彩色文字页 | 有彩色但集中在文字/标题（文字区饱和度高） | 增强 → SR → 彩色 JPEG |
| 彩色图片页 | 大面积照片区域（饱和度+局部方差高） | 增强 → SR → **禁止二值化**，彩色 JPEG 高质量 |
| 图文混排页 | 同时存在文字区和图片区 | **MRC 混合策略**：文字区域二值 JBIG2 + 背景图片区域 JPEG（进阶项，见 4.4） |
| 公式密集页 | 符号密度高 / OCR 平均置信度低 / 特殊字符比例高 | 跳过激进二值化；SR 保守放大；JPEG 质量上调 |

### 4.3 压缩策略矩阵（第⑼步）

```yaml
compression:
  monochrome: jbig2        # 可选: jbig2 / g4 / png
  grayscale:  jpeg         # 可选: jpeg / jp2
  color:      jpeg         # 可选: jpeg / jp2
  jpeg_quality: 85
  jbig2:
    symbol_mode: true      # 符号字典压缩（体积最优）
    refinement: false      # 必须关闭：已知导致 Acrobat 崩溃
```

### 4.4 MRC 混合光栅内容（第四阶段进阶项）

图文混排页的理想编码：拆成"前景 1bpp 文字层（JBIG2）+ 背景彩色层（降采样 JPEG）"，在 PDF 内用 SMask 合成显示。参考 hankpdf 的 MRC 实现思路。第四阶段先实现**整页单一策略**的稳定版，区域级 MRC 作为其后的增强（风险：PDF 内部结构复杂度上升，需逐页验证渲染正确性）。

### 4.5 目录结构与环境隔离

```
PDF_AI_Enhancer/
├── run.bat                  # 主入口（自动调用 scripts/setenv.bat 后启动）
├── config/config.yaml       # 全部可调参数
├── scripts/                 # 全部脚本（含 setenv.bat 环境隔离）
├── env/                     # 项目独立 Python venv（全部 pip 依赖）
├── models/                  # 全部模型权重
├── tools/                   # 第三方可执行程序（jbig2.exe 等，绿色便携）
├── input/                   # 待处理 PDF
├── output/                  # 结果（xxx_enhanced.pdf / xxx_ocr.pdf / xxx_images/）
├── temp/                    # 中间图像（处理完自动清理，可保留调试）
├── cache/                   # pip 缓存 / HF 缓存
├── logs/                    # 运行日志（每本书一个）
└── PLAN.md / README.md
```

**环境隔离方案（scripts/setenv.bat，run.bat 自动 source，无需手动配置）**：

```bat
set HF_HOME=%~dp0..\cache\huggingface
set TRANSFORMERS_CACHE=%~dp0..\cache\huggingface
set TORCH_HOME=%~dp0..\cache\torch
set PIP_CACHE_DIR=%~dp0..\cache\pip
set TEMP=%~dp0..\temp
set TMP=%~dp0..\temp
set XDG_CACHE_HOME=%~dp0..\cache\xdg
set GRPC_VERBOSITY=ERROR
```

验证标准：处理一本书后，`C:\Users\CKC\.cache\` 与用户全局 site-packages **零新增**；删除项目文件夹 = 完整卸载。

### 4.6 GPU 资源管理策略

1. 启动时通过 `nvidia-smi` + `torch.cuda.mem_get_info()` 检测实际可用显存（扣除系统占用）；
2. 所有 SR 模型推理默认 **tiling（分块）+ FP16**；
3. 捕获 CUDA OOM：自动缩小 tile 尺寸（1024→768→512→384）重试，**不允许直接崩溃**；
4. 批量处理按"页"为单位串行（8GB 显存下页间并行无收益且有 OOM 风险），CPU 阶段（OCR/压缩）可多进程并行；
5. `--device cpu` 强制 CPU 回退（慢但可用）。

### 4.7 命令行接口（第二阶段实现）

```
run.bat input\book.pdf                    # 处理单个 PDF
run.bat                                   # 处理 input\ 下全部 PDF
run.bat --scale 2 --ocr --quality 90 --model realesrgan
run.bat --no-ocr --model swinir --scale 4
run.bat --device cpu
```

---

## 5. 分阶段实施计划

### 第一阶段（本阶段，已完成）
调研 + 环境检测 + 本方案 + 目录骨架。**零安装**。

### 第二阶段：Baseline 端到端
- 创建 `env/` venv；安装 PyTorch (cu126) + PyMuPDF + OpenCV + numpy（约 3.5GB，全部在项目内）；
- 下载 **DocRes** 权重（约 100–300MB，放 `models/`）+ **realesr-general-x4v3**（约 5MB）；
- 实现：PDF → 渲染 → DocRes 外观增强 → SR ×2 → JPEG → PDF 重建；
- 用一份小型测试 PDF 验证全链路 + 显存峰值记录。
- **验收**：`output/xxx_enhanced.pdf` 肉眼清晰度提升、无 OOM、`~/.cache` 零污染。

### 第三阶段：OCR 隐藏层
- 主力：RapidOCR（中文/英文/数字，ONNX，约 15MB）+ PyMuPDF 写入不可见文字层；
- 可选：OCRmyPDF via WSL backend（你已有 WSL；需在 WSL 内 apt 安装，**这是唯一一处触碰 WSL 环境的操作，实施前会再次向你确认**）；
- **验收**：Ctrl+F 可搜索中文关键词；OCR 不改变页面视觉；输出体积增量 <5%。

### 第四阶段：页面分类 + 自适应压缩
- 实现 4.2 的六类页面分类器；
- 黑白页 → 二值化 + JBIG2（下载 jbig2.exe 约 2MB 到 tools/）；
- 灰度/彩色页 → 对应 JPEG 策略；公式页保守策略；
- **验收**：同一本混合类型书，黑白页体积明显下降且文字更锐；彩色页色彩保留。

### 第五阶段：模型可切换架构
- Backend 注册机制：`super_resolution.backend` 与 `document_restoration.backend` 均可配置切换；
- 接入 SwinIR、Real-ESRGAN x4plus（各约 64–200MB）；
- **验收**：改一行 config 即换模型，无需改代码。

### 第六阶段：对比测试报告
- 对比对象：原始 PDF / Real-ESRGAN ×4（≈Upscayl 等效）/ DocRes 修复 / SwinIR / （视显存实测决定是否加 TADiSR）；
- 指标：文字边缘锐度（客观）、400% 放大目视、OCR 字符错误率 CER（客观）、文件大小、单页处理耗时、显存峰值；
- 输出 `output/benchmark_report.md` + 并排对比图。

---

## 6. 资源预算

| 资源 | 阶段 1–4 | 含阶段 5/6 全量 |
|---|---|---|
| 磁盘 | venv ~3.5GB + 模型 ~0.5GB + 工具 ~10MB ≈ **4GB** | +SwinIR/x4plus ~0.3GB（+TADiSR 约 5GB，装前确认） |
| 显存峰值（预估） | DocRes ≤4GB；realesr-general ≤2GB；均 tiling 安全 | SwinIR ≤6GB（tiling）；TADiSR 仅小 patch |
| 单页耗时（预估，RTX 4060） | 渲染+增强+SR×2 约 3–8 秒/页 | — |

---

## 7. 风险与未决问题

| # | 风险/问题 | 对策 |
|---|---|---|
| 1 | Real-ESRGAN 对小号文字可能产生笔画变形（幻觉） | 第二阶段起所有输出页存入 `temp/` 可抽查；对比报告量化；DocRes 非生成式路径始终可用 |
| 2 | jbig2.exe 符号模式对某些字体书压缩率不稳定 | G4 自动 fallback；体积对比后择优 |
| 3 | MRC 混合编码 PDF 兼容性 | 第四阶段先用整页策略稳定交付，MRC 作为可选增强并逐页渲染校验 |
| 4 | WDDM 模式下显存报告不准 | 用实际分配测试而非仅看空闲值；OOM 自动降 tile |
| 5 | OCRmyPDF 依赖 WSL | 默认 backend 是 RapidOCR（原生 Windows）；WSL 方案仅在明确同意后安装 |
| 6 | 中文教材竖排/古文 | 超出当前范围，config 中预留 `ocr.lang` 参数 |

---

## 8. 下一步（待你确认后执行第二阶段）

第二阶段将首次安装内容（预先声明，符合你的安全要求）：

1. **安装到 `env/`**：PyTorch 2.x cu126、torchvision、PyMuPDF、opencv-python、numpy、Pillow（约 3.5GB 磁盘，全部项目内，无需管理员权限，不修改系统环境）；
2. **下载到 `models/`**：DocRes 官方权重（HuggingFace/GitHub Release，约 100–300MB）+ realesr-general-x4v3（约 5MB）；
3. **不安装**：本阶段不装 OCR、不装 jbig2、不装 TADiSR/SwinIR；
4. **产生的文件**：`env/`、`models/`、`scripts/`、`config/config.yaml`、`run.bat`、测试输出；
5. **执行的命令**：仅 `python -m venv`、项目内 pip、模型下载到指定目录——全部在项目目录内执行。

> 确认后回复"开始第二阶段"即可。
