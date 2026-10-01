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
- [x] MRC 分层压缩（v3：JPEG 背景 + 多色 1-bit 前景层，背景层降采样+中值去噪）
- [x] 页数范围参数（`--pages N` / `--pages a-b`）
- [x] 本地 Web GUI（`run_gui.bat`，拖入/队列/进度/对比图，OCR 默认关闭）
- [x] 断点续跑（逐页落盘 + 配置签名）与 `--rebuild-only` 快速重建
- [x] 编码进程池并行（`encode_workers`）+ 不抢占桌面的优先级（`encode_priority`）
- [x] MRC 前景模板改 JBIG2（`mrc_stencil`）：整本再瘦 31.6%，渲染逐位一致
- [x] 340 页全本跑通：328.4 MB → **40.2 MB**（8.17×），整本约 26 min（旧 104 min）

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
run.bat --ocr                   # 开启 OCR 隐藏层（默认关闭）
run.bat --no-ocr                # 明确关闭 OCR
run.bat --dpi auto              # 按每页内嵌扫描图名义 DPI 零插值渲染（默认）
run.bat --dpi 300               # 固定渲染 DPI（矢量页 / 无内嵌图时用）
run.bat --routing by-type       # 混合路由 SR：文字页→RealESRGAN、插图页→SwinIR
run.bat --list-backends         # 列出已注册的修复/超分后端
run.bat --pages 50              # 只处理前 50 页
run.bat --pages 10-59           # 只处理第 10~59 页（1-based 含两端）
run.bat --no-resume             # 忽略断点，全部重跑
run.bat --rebuild-only          # 不增强，只用断点缓存重建 PDF（调压缩参数用）
run.bat --clear-checkpoint      # 先清掉本书的断点缓存
run.bat -o out/mine.pdf         # 显式指定输出路径
```

> `--pages` 跑出来的文件名会自动带 `_p<a>-<b>` 后缀（如
> `<书名>_p43-44_enhanced.pdf`），**不会再覆盖整本成品**。这个坑踩过两次
> （一次是 50 页成品被单页 GUI 测试覆盖），所以现在默认就不会同名。

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

```
run_gui.bat            # 默认端口 8765（被占用时自动向后找空闲端口）
run_gui.bat 9000       # 指定端口
```

OCR 默认**关闭**：本项目专注图像增强（DocRes + 超分 + MRC 压缩），文字层建议  
用 ABBYY FineReader 等专业 OCR 工具手动添加（识别精度更高，且避免了内置  
RapidOCR 的漏检/错位问题）。需要 Ctrl+F 搜索时加 `--ocr`（340 页实测多花 39.6 分钟，  
结果也会进断点缓存，不会重复支付）。

## 2.6 断点续跑与快速重建

整本 340 页旧版实测 **104 分钟**（其中重建 45.5 min 是单线程编码）。
打开断点续跑、编码并行、并按推荐关掉 OCR 后，同样一本降到
**增强 19 min + 重建约 17 min ≈ 36 min**。为防止中断白跑，**每处理完一页就落盘**
（`temp\<书名>\ckpt\`，无损 PNG + 元数据 JSON），下次运行自动从断点继续：

- **中断即可续**：GUI 的「停止」/Ctrl-C/断电后重新运行同一命令即可，
  已完成页显示 `cached, skipped`。两次已发生的中断曾白烧约 1.5 小时 GPU，
  这是本项目最贵的一课。
- **`--rebuild-only`**：不重跑增强，直接用断点缓存重建 PDF。
  只调压缩参数（`jpeg_quality` / `mrc_bg_scale` / `mrc_bg_denoise` / `monochrome`）时，
  从 1.7 小时降到几分钟。缺页会**明确报错**而不是静默输出残本。
- **缓存签名**：渲染 DPI、DocRes 任务、SR 后端/倍数、路由、fp16 组成增强签名；
  OCR DPI/语言单独一个签名。改这些会另起一份缓存（不会混用不同参数的页）；
  改压缩参数**不会**使缓存失效——这正是 `--rebuild-only` 能成立的原因。
- **体积**：2878×4010 页约 5MB/页，整本约 1.7GB，位于 `temp\<书名>\ckpt\`。
  不需要时删掉该目录或用 `--clear-checkpoint`；`debug.keep_checkpoint: false`
  可在每次跑完后自动清理（那样就失去 `--rebuild-only` 能力）。
- **内存**：逐页渲染 + 逐页编码，峰值只占 1 页（不再把整本读进内存）。
  旧写法整本约 15GB 常驻，SR×4 的整本任务**根本跑不起来**。
- **编码并行**（`compression.encode_workers`，默认 `auto` = **物理核数 − 2**）：
  逐页编码（MRC 分割 + JPEG 双编码比较）是纯 CPU、逐页独立的，旧版却是
  单线程串行。改为进程池后的实测账（340 页全本）：

  | 阶段 | 旧全本 104 min | 现在 |
  | --- | --- | --- |
  | OCR 识别 | 39.6 min | **0**（默认关闭） |
  | OCR 文字层写入 | ~24.8 min | **0**（同上，文字层交给 FineReader） |
  | GPU 增强（渲染+DocRes+SR） | 19.0 min | 20.4 min（未动） |
  | 重建**编码** | ~20.7 min（单线程） | **5.0 min**（实测 301.6s / 340 页） |
  | **合计** | **~104 min** | **~31 min** |

  > 注意：早期版本把"重建阶段 45.5 min"整体记成"单线程编码"，这是**错的**。
  > 全本实测编码只要 301.6s（8 进程）/ 单线程约 20.7 min，剩下的 ~24.8 min
  > 其实是 **OCR 文字层写回**（逐 span 排版 + 字体子集化）。所以旧账里
  > **与 OCR 相关的开销合计约 64 min = 62%**，编码只占约 20%。

  24 页真实页基准（同一批缓存页，`--rebuild-only`）：编码
  `1→4→8` 进程 = 2.20→0.87→0.81 s/页（**2.70×**），且输出**字节级一致**；
  含 MuPDF 写盘的整条重建路径 87.7s → 33.5s（**2.6×**，8 进程）。
  MuPDF 那半边保持串行（`fitz.Document` 不能跨进程共享），
  页面数组若以内存形式传入（脚本单页演示）则自动退回串行，避免 pickle。
- **别把整机跑满**（`compression.encode_priority`，默认 `below-normal`）：
  worker 会把分到的核跑满，NORMAL 优先级下它和你的前台程序**平分时间片**，
  整本跑起来电脑会明显发卡。两处改进：
  1. `auto` 不再用"逻辑核一半"——本机是 8 核 16 线程，16/2 = 8 **正好等于全部
     物理核**，整机被占死。现在按 **物理核数**（`GetLogicalProcessorInformation`）
     算，并**留 2 个物理核给桌面** → 本机 6 个 worker。实测 4→8 进程只快 7%
     （0.87→0.81 s/页），这点余量几乎不亏。
  2. worker 进程优先级降到 **below-normal**：你不动电脑时它仍是唯一可运行
     线程（速度几乎无损），你一操作系统立刻把 CPU 让给你。
     可选 `idle` / `normal` / `null`。
- **编码临时文件不落在 output/**：MRC/jbig2 会写 pid 标记的中间 PNG，以前写在
  `dirname(out_path)` 也就是 **output/ 里**（8 个 worker 留 16 个文件 ≈ 10MB，
  清理时还要一次性删一堆、触发批量删除保护导致进程异常退出）。现在改到
  `temp/pdfenhance_enc/`，并且**每个 worker 退出时用 atexit 只删自己那一对**
  （每次运行 2×N 次删除，而不是 2×页数）。

### OCR 隐藏文字层（Phase 3）

**本项目不负责 OCR。** 内置 RapidOCR 的识别质量明显不如 ABBYY FineReader
（漏检/错位更多），因此文字层的正确做法是：先用本流水线产出**增强后**的 PDF，
再交给 FineReader 做 OCR。`--ocr` 只是"想立刻 Ctrl+F 一下"的临时替代，
默认关闭，也不参与任何体积/清晰度结论。

`--ocr` 打开后：RapidOCR（PP-OCRv3 系，ONNX，Windows 原生）在 200 DPI 下识别，  
把文字以**不可见文字层**写回 PDF（`render_mode=3`），因此：

- 增强后的 PDF 可用 **Ctrl+F 搜索**中文/英文/数字；
- 文字层不影响页面视觉（完全隐藏）；
- 中文用内置宋体 `tools\fonts\simsun.ttc` 嵌入，保存时做**字体子集化**
  （否则整包 18MB 会进 PDF；一个 2 页样例因此从 177KB 涨到 9.9MB）。
  子集化用 `os.replace` 落地，Windows 上杀软扫描新文件会造成瞬时占用，
  已改为带退避重试，失败会明确告警而不是静默带着整包字体发布。

控制：`config\config.yaml` → `ocr:`（`enabled` / `dpi` / `use_cuda`）。  
CPU 推理约 2~5 秒/页，一般无需 GPU 版 onnxruntime。

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
   填充色。蒙版默认用 **JBIG2** 编码（`compression.mrc_stencil`），无损。

这样文字保持锐利、彩色标注（如正文里的蓝字批注）不会被二值化抹掉或涂黑，
而照片/插图仍走有损 JPEG。逐页与整页 JPEG 双编码比较后取小，因此：

- 文字密集页 → MRC 胜；
- 图片占主导 / 无有效文字的页 → 整页 JPEG 胜（自动回退）。

**背景层两项优化**（`compression.mrc_bg_scale` / `mrc_bg_denoise`，默认 `auto`）：

- **降采样**：背景层只承载低频（纸面/渐变/光照/大插图），文字由 1bit 模板
  全分辨率保留 → 降到 1/2 几乎无损，体积按倍数平方下降。`auto=2` 的含义是
  "回到扫描件原始分辨率"（渲染本就跑在源有效 DPI 上，再高都是 SR 插值出来的）。
- **中值去噪**：抠掉文字后背景里剩的几乎全是 SR 注入的逐像素噪声——占字节
  不含信息。`auto` = gray-text 页用 5、mixed/color-text 用 0（保插图）。

实测（同一批真实页，与旧编码全本成品逐页对比）：

| 页 | 旧（整页 JPEG） | 新（MRC 分层） | 倍数 |
| --- | --- | --- | --- |
| p43 | 1406.3 KB | 211.4 KB | 6.65× |
| p44 | 1079.1 KB | 191.8 KB | 5.63× |
| p45 | 1171.5 KB | 189.3 KB | 6.19× |
| 30 页样张（p43–72） | 32.43 MB | 5.53 MB | 5.86× |

**整本 340 页：前景模板换 JBIG2 的收益**（`mrc_stencil: jbig2`，逐层实测）：

| 层 | Flate 版 | JBIG2 版（现默认） | 倍数 |
| --- | --- | --- | --- |
| 1-bit 前景模板 | 30.63 MiB | **12.71 MiB** | **2.41×** |
| JPEG 背景层 | 25.16 MiB | 25.16 MiB（未动） | 1.00× |
| **整本文件** | 56.04 MiB | **38.35 MiB** | **1.46×** |

对照旧编码全本 baseline（313.2 MiB）：**8.17× 更小**（943.2 → 115.5 KiB/页）。
**渲染逐位一致**：Ghostscript（p5/p129/p200/p330）与 MuPDF（p5/p200/p330）
`maxdiff` 与 `diff_px` **全为 0**。编码耗时没有变差（285.0s vs 301.6s）。

> 独立交叉验证：成熟工具 **pdfsizeopt**（908★，内部用的就是同一个 jbig2enc）
> 处理同一份 Flate 成品得到 **38.25 MiB / 模板 12.71 MiB**，与本项目原生实现
> 相差 **0.25%**。两条路径数字一致 → 这 31.6% 是编解码器本身的收益，
> 不是参数调出来的（也没有引入任何新依赖）。

关键实现细节（踩坑记录见项目记忆）：

- **笔画不加粗**：涂色用原始 mask + 自适应灰度带修剪，涂白单独用加宽 mask，
  避免二值化把抗锯齿边缘吞进前景导致笔画变胖（实测墨迹覆盖 11.25%→9.25%）。
- **细笔画不能被"去噪"掉**（用户抓包的"少笔画"）：`split_mrc` 里的 3×3 开运算
  会删掉一切 <3px 的结构，而 2878px 页宽上 CJK 细横画正好 2px → 整根消失。
  已删除该开运算（噪点交给基于面积的 `_drop_specks`），灰度上限 130→140。
  同网格量化：丢失墨迹 1.02% → **0.09%**（旧版整页 JPEG 为 0.04%）。
- **彩色文字保护**：小彩色笔画放行进前景，只排除大块彩色连通域（保护截图/插图）。
- **前景层用 JBIG2，不用 Flate**（`compression.mrc_stencil`，默认 `jbig2`）：
  这些 1-bit 模板是整本书的**大头——30.6 MB / 54.7 %**，而 Flate 对文字位图
  就是错的编解码器。换成 jbig2enc 的 **generic coder**（`-p`，非符号模式
  `-s`，故无字形合并幻觉）后模板小 **2.4×**（30.6 → 12.7 MB），栅格逐位一致。
  `flate` 仅作逃生开关；`jbig2.exe` 缺失或失败会自动退回 Flate。
- **PyMuPDF 会吞掉 `/Filter` 键**（本次挖出的真 bug）：`update_stream()` 会把
  对象上的 `/Filter` 删掉，随后的 `save(deflate=True)` 于是把这些流重压成
  Flate 并把 `/Filter` 改标成 `/FlateDecode` → JBIG2 数据被当 Flate 解，
  整页**全黑**。**必须在 `update_stream()` 之后用 `xref_set_key()` 重新写回
  `/Filter`**（`embed_mrc` / `embed_jbig2` 均已修）。
  实测同一份 1032 B 载荷：不写回 → 落盘只剩 294 B 且标成 Flate（数据损毁）；
  写回 → 1032 B、`/JBIG2Decode`、完好。
  这也是本项目长期误判"**MuPDF 不支持 JBIG2**"的真正原因——此前产出的文件
  从来就不是合法 JBIG2。**MuPDF 其实能正常解码 JBIG2**（已用 Ghostscript 与
  MuPDF 双向渲染比对确认）。
- **ImageMask 极性**：ink 位=0 + `/Decode [0 1]` + 黑填充（两者都渲染一致）。

## 6. 控制输出质量与体积

- 清晰度：`--scale 4` + `--task deblurring`（更激进）；保守用默认。
- 体积：`--quality`（60–95）；灰度页自动省 1/3。
- `--dpi` 控制渲染分辨率（默认 `auto`：按每页内嵌扫描图的名义 DPI 零插值；
  矢量页回退 300，上限 `render.max_dpi`）。
- 只想调压缩参数时用 `--rebuild-only`（见 §2.6），不必重跑增强。

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
scripts/           安装/测试/诊断脚本
src/               流水线代码（渲染/分类/修复/超分/重建/OCR）
src/checkpoint.py  断点缓存（逐页落盘 + 配置签名）
src/imgio.py       中文路径安全的图像读写（cv2 在 Windows 走 ANSI 代码页）
src/gui.py         本地 Web GUI 后端（零依赖，标准库 http.server）
src/web/           GUI 前端（原生 HTML/JS）
third_party/       vendored 模型定义（DocRes-Restormer, SRVGG）
env/               独立 Python venv
models/            模型权重
tools/             第三方可执行程序（jbig2.exe，phase 4）
input/ output/ cache/ logs/
temp/<书名>/ckpt/  断点缓存（增强后的页图 + 元数据，见 §2.6）
```

## 产物示例（output/）

| 文件 | 说明 |
| --- | --- |
| `<书名>_enhanced.pdf` | 成品：按页类型自适应编码（+ 可选隐藏文字层） |
| `云计算导论_第3版_新编码_340页_JBIG2.pdf` | **340 页全本，前景模板 JBIG2：40.2 MB / 115.5 KiB/页**（当前最优） |
| `…云计算导论  第3版 ---1_enhanced.pdf` | 同一本的 Flate 模板版：58.8 MB / 168.8 KiB/页（换 JBIG2 前的对照） |
| `云计算导论_第3版_旧编码_340页_baseline.pdf` | 340 页全本**旧编码**成品（328.4 MB / 943.2 KiB/页），对比基线 |
| `云计算导论_样张_p14-34_enhanced.pdf` | 21 页样张，27.2MB（1.30MB/页，旧编码） |
| `医疗信息管理平台-需求规格说明书_enhanced.pdf` | 完整跑通的一本 |
| `计算机组成原理_p17_彩色增强.pdf` | 彩色文字标注验证页（黑+蓝两层前景） |
| `benchmark_report.md` | Phase 6 边缘锐度/耗时/显存对比报告 |
| `compare_*.png` | 各页增强前后 / 编码方案对比图 |

> **整本 340 页实测：328.4 MB → 40.2 MB（8.17×）**，逐页 943.2 → 115.5 KiB。
> 其中最后一步（前景模板 Flate→JBIG2）独占 31.6%，且渲染逐位一致。
> 耗时：增强 20.4 min + 编码 4.8 min ≈ 26 min（旧 104 min）。
> 所有数字以 `output/` 里的实物为准；引用前先量，别用外推值。
