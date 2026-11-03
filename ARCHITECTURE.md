# MOMIA3 架构规格文档

**版本**：v0.1 · 2026-07-16
**定位**：单细菌显微图像表征与谱系追踪平台 —— AI-ready、agent-native、GPU（PyTorch）加速
**前身**：MOMIA2（`/Users/jzrolling/Desktop/Projects/Playground_kimi/momia2`，保留为只读参考实现，不再修改）
**对标**：CellProfiler 4（模块协议/测量算子集）、MicrobeJ（细菌表征算子）、TrackMate/Mastodon（追踪与谱系图）、napari（GUI 生态）、OME-NGFF/Arrow（数据标准）

---

## 0. 设计原则

1. **表 + 数组 + 边表**三件套取代对象图：一切状态存于 Arrow 表（细胞）、Zarr 数组（图像/掩码）、边表（谱系）。
2. **纯函数操作**取代可变对象方法：每个算子 `f(输入, 参数) → 输出`，无副作用、可并行、可缓存。
3. **一次定义，三处界面**：每个操作在注册表定义一次（pydantic schema + 描述），自动生成 CLI、napari GUI 表单、MCP 工具 schema。
4. **数据自描述**：单位、语义、provenance（参数/代码版本/上游哈希）写入数据 metadata，agent 无需读源码即可理解。
5. **GPU 统一 PyTorch**：`torch` 为唯一 GPU 后端（CUDA→MPS→CPU 自动回落），禁止 TensorFlow 进入依赖树。
6. **保住微生物专精优势**：亚像素轮廓、迭代中线、回溯纠错、V-snapping 处理为一级公民，不因重构丢失。

---

## 1. 项目布局

```
momia3/
├── ARCHITECTURE.md                # 本文档
├── pyproject.toml                 # hatchling; extras: [gpu] [train] [gui] [dev]
├── src/momia3/
│   ├── __init__.py
│   ├── schema.py                  # Arrow/parquet schema 定义（单一事实源）
│   ├── keys.py                    # 复合主键编码/解码
│   ├── storage/
│   │   ├── zarr_store.py          # OME-Zarr 图像/掩码读写（懒加载）
│   │   ├── tables.py              # cells/edges/tracks parquet 读写（polars）
│   │   └── cache.py               # 内容寻址缓存 + provenance
│   ├── backend.py                 # numpy/torch 双后端分派 + device 选择
│   ├── core/
│   │   ├── field.py               # Field：不可变帧束（zarr 引用 + 元数据）
│   │   ├── cellset.py             # CellSet：细胞表 + 几何访问视图
│   │   └── lineage.py             # 边表 ↔ networkx 视图
│   ├── ops/                       # 纯函数算子（注册表项）
│   │   ├── preprocess.py          # 照明校正/背景/去噪/漂移/漂白/归一化
│   │   ├── segment.py             # 传统阈值 + AI 分割后端协议
│   │   ├── measure.py             # regionprops/形态/强度/纹理
│   │   ├── outline.py             # 亚像素轮廓 + 曲率
│   │   ├── midline.py             # 中线/极点/宽度轮廓
│   │   ├── polygon.py             # inside_polygon：numpy/PIL 双后端
│   │   ├── characterize.py        # 坐标投影/生长动力学/群体结构
│   │   └── track.py               # LAP 链接/间隙闭合/谱系/纠错
│   ├── models/                    # AI 模型注册表（Omnipose/Cellpose-SAM/自训 U-Net-torch）
│   ├── pipeline/
│   │   ├── registry.py            # 工具注册表（pydantic schema → CLI/GUI/MCP）
│   │   ├── engine.py              # JSON DAG 执行器（分片/并行/缓存/失效）
│   │   └── cli.py                 # typer CLI：`momia3 run pipeline.json`
│   ├── gui/                       # napari 插件（npe2）
│   └── compat/                    # momia2 适配器（Patch↔Field 双向）
├── tests/
│   ├── synth.py                   # 合成数据生成器（杆菌生长/分裂时序，含 ground truth）
│   ├── test_*.py
│   └── benchmarks/                # pyinstrument 基准（2048²×100 帧标准集）
└── examples/
```

---

## 2. 数据层规格

### 2.1 复合主键（`keys.py`）

`cell_key: uint64 = fov_id << 40 | frame << 20 | label`
- fov_id ≤ 2²⁴⁻¹，frame ≤ 2²⁰⁻¹，label ≤ 2²⁰⁻¹，全程无字符串解析。
- 提供 `encode_key(fov, frame, label)` / `decode_key(key)`，向量化（numpy）与标量两版。

### 2.2 OME-Zarr 存储（`storage/zarr_store.py`）

```
experiment.zarr/
├── fov_{id}/
│   ├── images/{channel}   # (t, y, x) uint16，chunks=(1, 512, 512)
│   ├── labels             # (t, y, x) uint32，实例分割结果
│   └── prob               # (t, c, y, x) float16，AI 分割概率图（可选）
└── .zattrs                # pixel_microns、通道名、时间间隔、provenance
```
- 读取一律懒加载（dask.array 或 zarr 句柄切片），禁止全栈入内存。
- 兼容导入：`compat` 层把 ND2/TIFF（momia2 ImageLoader 逻辑）写入 zarr。

### 2.3 细胞表 `cells.parquet`（`schema.py` 唯一定义）

**标识列**（每行 = 一个细胞实例）：
| 列 | 类型 | 说明 |
|---|---|---|
| cell_key | uint64 | 复合主键 |
| fov_id / frame / label | uint32 | 解码冗余列（分区键：fov_id） |
| track_id | uint32? | 追踪后回填 |
| parent_track | uint32? | 谱系 |
| touching_edge / include | bool | 过滤标记 |

**几何列**（Arrow 嵌套类型）：
| 列 | 类型 | 说明 |
|---|---|---|
| bbox | struct{x0:i32,y0:i32,x1:i32,y1:i32} | |
| centroid | struct{x:f32,y:f32} | |
| outline | struct{x:list\<f32\>,y:list\<f32\>} | 亚像素闭合轮廓（全局坐标） |
| midline | list\<struct{x:list\<f32\>,y:list\<f32\>}\> | 多分支中线 |
| width_profile | list\<f32\> | 沿中线宽度（µm） |
| coords | struct{x:list\<i32\>,y:list\<i32\>} | 掩码像素（可选，默认同 labels zarr 推导） |

**标量形态列**（f32，metadata 含 unit/semantic）：
`area, perimeter, major_axis_length, minor_axis_length, aspect_ratio, eccentricity, solidity, extent, circularity, compactness, hu_0..hu_6, curv_min, curv_max, curv_mean, curv_q1, curv_q3, length_um, width_median_um, width_std_um, width_max_um, width_q1_um, width_q3_um, sinuosity, branched, pole1_x/y, pole2_x/y, septum_pos?`

**强度列**（每通道一簇）：`{ch}_mean, {ch}_max, {ch}_min, {ch}_std, {ch}_p25, {ch}_p75, {ch}_integrated`
**扩展列**：`embedding: list<f32>?`，`annotation: i8?`，`puncta_{ch}: list<struct{x:f32,y:f32}>?`（FISH）

### 2.4 谱系边表 `edges.parquet` 与轨迹表 `tracks.parquet`

- edges：`(parent_key:u64, child_key:u64, event:categorical[continue|division|merge|appear|disappear|gap], confidence:f32)`
- tracks：`(track_id, fov_id, birth_frame, last_frame, division_frame?, parent_track?, n_frames, fate:categorical[divide|disappear|edge|incomplete])`
- `core/lineage.py` 提供 `edges_to_nx()` / `nx_to_edges()`；算法内部可用 networkx，落盘永远是边表。

### 2.5 缓存与 provenance（`storage/cache.py`）

- 操作输出落盘键：`sha256(op_name + version + params_json + 上游键)`，存 `.momia3_cache/`。
- 每份产物附 `provenance.json`：`{op, version, params, upstream:[...], timestamp, git_hash}`。
- 参数变更 → 键变更 → 下游自动失效；重复运行 → 命中缓存零成本。

---

## 3. 计算层规格

### 3.1 操作协议（`pipeline/registry.py`）

```python
class Operation(Protocol):
    name: str                    # "segment.threshold"
    version: str                 # semver；进入缓存键
    params: type[BaseModel]      # pydantic 参数模型
    inputs: dict[str, str]       # {"images": "zarr/images", "mask": "zarr/labels?"} 类型声明
    outputs: dict[str, str]      # {"labels": "zarr/labels"}
    def run(self, inputs, params, ctx) -> dict: ...   # 纯函数；ctx 含 device/worker/cache
```

- `@register` 装饰器入注册表；`registry.to_cli()` 生成 typer 命令；`registry.to_magicgui()` 生成 napari 表单；`registry.to_mcp()` 生成 MCP tool schema。
- 参数模型字段带 `Field(description=..., ge=..., le=...)` —— 描述同时用于 CLI help、GUI tooltip、agent 提示。

### 3.2 分片与并行（`pipeline/engine.py`）

| 操作类别 | 分片键 | 并行方式 |
|---|---|---|
| 前处理 / 分割 | (fov, frame) | ProcessPool，尴尬并行 |
| 表征算子（轮廓/中线/宽度/投影） | (fov, frame) 内按细胞批 | 进程池 + 细胞级向量化 |
| 追踪 | fov | 时间维串行、FOV 间并行 |
- 引擎默认 `ProcessPoolExecutor`；可选 `dask.distributed`（extras `[distributed]`）。
- regionprops 只算一次写表；后续算子从表 + zarr 按需读取，**禁止全量重算**。

### 3.3 后端（`backend.py`）

```python
def get_device() -> torch.device:        # cuda → mps → cpu
def to_xp(arr, device=None): ...         # numpy ↔ torch 零拷贝互转（CPU）
class Ops:                               # 需 GPU 的算子统一入口
    conv2d / gaussian / hessian / edt / fft_shift ...
```
- 规则：整幅图像级操作（滤波、形态学、EDT、FFT、相位相关）→ torch on device；逐细胞小 ROI 几何（样条/求交）→ numpy/numba on CPU。
- 无 torch 环境：Ops 自动回落 numpy/scipy，结果一致（测试保证容差内等价）。

---

## 4. 算法层规格（ops/）

### 4.1 preprocess.py
- `illumination.basic`（BaSiC 平场/暗场校正）、`background.rolling_ball_fft`（torch FFT 近似，替代 momia2 逐像素版）、`denoise.n2v?`（可选 extras）、`drift.phase_corr`（**参考通道算一次 shift，全通道应用**；时序累积漂移 + 单次 Fourier-shift）、`bleach.exp_fit`、`normalize`（统一入口，显式 dtype 策略）。

### 4.2 segment.py
- 传统：`threshold.{isodata,sauvola,local,composite}`（移植 momia2，纯函数化）。
- AI：模型注册表协议 `Segmenter.predict(image, params) -> prob (c,y,x)`；实现 `omnipose`、`cellpose_sam`、`unet_torch`（旧 AttResUNet 权重可转）；后处理 `prob_to_labels`（seed/foreground/分水，移植 prediction2seed/foreground）。

### 4.3 measure.py
- `regionprops`：skimage `regionprops_table` 一次算清，只取 schema 声明列；加权取向矩（momia2 已实现，移植）；`intensity_stats` 向量化（`ndimage.sum_labels` 等，禁逐细胞 Python 循环）；`texture.haralick`、`texture.lbp`（新增，对标 CellProfiler MeasureTexture）。

### 4.4 outline.py（重点优化，详见 §5.1）
- 亚像素轮廓：mask 高斯 σ=1 → marching squares(0.5) → 像素等距降采样（N = perimeter / downsample_factor，默认 2 px/点）→ 闭合 B 样条（s = smooth_scale·N）。
- 默认开启一轮 edge+intensity 精化：Sobel 法向 subpixel 吸附（momia v1 风格）+ 强度阈值收腰（notch），逐轮轻轻平滑；参数 refine_outline_* / constriction_image_* 全部 opt-in 可调。
- 曲率：窗口化 bend angle（移植 momia2）+ 椭圆傅里叶描述子 EFD；输出 schema 几何列与 curv_* 标量列。
- 多边形包含测试/栅格化统一走 momia3.ops.polygon.inside_polygon（numpy Path + PIL 双后端）。

### 4.5 midline.py（重点优化，详见 §5.2）
- 默认仍为 pole-first 低自由度样条（spline）路线；优化路线 equidistant 为 opt-in（pipeline 可用 midline_method 选择）。
- equidistant：outline-SDF / mask skeleton 轮廓切割 → 极点对、主干、分支 → 自适应重采样/平滑/Taubin → 贴边 clamp；近圆细胞走 degenerate 直轴 + 弦中点精修；guard 失败保留候选而非回退。
- 另保留 energy 实验路线；宽度 profile、极点、分支、质量标志均写入 cells 表。
### 4.6 characterize.py
- `projection.demographic`（中线弧长 l∈[0,1] × 横向 d 坐标投影，任意通道；Coli-Coords/BacStalk 式，升级 momia2 fish.create_canvas）、`growth.rates`（伸长速率/分裂间期/adder-sizer-timer 判别）、`division.septum`（中线曲率拐点 + 膜通道信号定位）、`poles.age`（跨帧新旧极继承）、`population.cording`（取向张量/局部取向熵/接触网络）、`fisher.puncta`（LoG + 亚像素，移植 fish.py 并去除字符串存储）。

### 4.7 track.py
- 链接：`link.lap`——代价矩阵（IoU + 质心距 + 面积差 + Hu 矩相似，权重为参数）+ `scipy.optimize.linear_sum_assignment`；Kalman 恒速预测缩小门限；`link.gap_close`（跨 ≤2 帧）。
- 谱系：`lineage.build`（边表 → tracks 表）、`lineage.refine`（momia2 回溯纠错泛化为图手术：merge 节点对 / split 节点，种子证据来自祖先/后代）。
- 基准：`benchmark.ctc`（TRA/SEG/AOGM 指标，用合成数据 ground truth）。
- V-snapping：显式分裂候选事件（短轴取向突变 + 面积守恒检验）。

---

## 5. midline/outline 优化规格（对标 MicrobeJ/Oufti/BacStalk）

### 5.1 outline
1. 生成：mask 高斯平滑（σ=1，修复 momia2 忽略 sigma 的 bug）→ marching squares（level=0.5）→ 可选 RDP 简化 → 像素等距降采样（N = perimeter / downsample_factor，默认 2.0）→ 闭合 B 样条平滑（s = 0.03·N）→ 输出 N 个约 2 px 间隔的点。
2. 默认精化（refine_outline_iters=1，refine_outline_mode=edge+intensity）：
   - edge 步：法向 ±0.5 px 采 Sobel 图，三点抛物线 subpixel，弱边（cutoff=0.2）不动，位移 cap 1.5 px，越界回滚；先外扩 0.3 px（momia v1 contour_optimization 语义）。
   - intensity 步：阈值暗细胞体生成 refined ring，仅把比局部圆窗中位数更深（>0.2 px）的凹点向内移动，cap 3 px。
   - 每轮轻样条平滑；0 关闭，可多轮。
3. 兼容保留：snake/active_contour（opt-in）、mask-target constriction_iters（deprecated）。
4. 曲率：窗口化 bend angle + 椭圆傅里叶描述子 EFD（Kuhl-Giardina 归一化）。
5. 全部输出写入 schema 几何列与 curv_*、ellipse_dev、sagitta_ratio 标量列；后两者即 degenerate/equidistant 路由门控所用的形状特征（fill_cells_geometry 统一计算）。

### 5.2 midline（性能与精度双升级）
- **路由**：midline_method 选择 spline（默认，pole-first 低自由度样条）/ energy（实验）/ equidistant（优化路线）。
- **equidistant 路线**：
  1. 简单形状门控：solidity ≥ 0.88 且 ellipse_dev ≤ 0.12 且 sagitta_ratio ≤ 0.08（或 AR < 1.25）→ degenerate：直主轴 + 1 轮横截面弦中点精修（末端 2 点只作样条约束），输出 32 点。
  2. 其余：SDF skeleton（outline 以自适应 4× 栅格化；PIL 快速多边形填充）或长细胞（bbox > 400 px）改 mask skeleton；极点对、主干、分支阈值（max(px, μm/pixel_microns)）。
  3. 自适应采样：axis/n_points 按 2 px 间距（上限 2048）；主干样条平滑 + Taubin；长细胞贴边点投影回主干；分支坐标作为 midline[1:]，分支末端平滑到顶点。
  4. guard：候选偏离弦/过弯时拒绝并保留失败候选（不静默回退 spline），质量写 midline_quality。
- **性能**：轮廓↔中线最近点/垂直求交全部 cKDTree；骨架邻域一次卷积；SDF 栅格化用 PIL（比 skimage.draw.polygon 快 50–100×）；Taubin 用 convolve1d。
- **宽度/极点/分支**：沿重采样中线法向与轮廓求交得 width_profile（µm）；极点延伸至轮廓；多分支中线保留，主分支 = 最长测干。
- **实测**（18 个全幅场，SDF-all + 4a88696 门控，ms/cell）：SP_003 2.69、murA 1.72、5I 1.78、PAO1 1.64、A22 2.05、PSVB 2.65、bthai 1.98、francisella 2.00、Serratia 2.89、vibrio 2.31、wiggins 2.33、dnaA 5.04、caulo 9.93、Hpylori 13.56、cex 10.73、ftsN 10.02、Az_branch 10.75、streptomyces 36.37。

### 5.3 验证标准（可用性测试的一部分）
- 合成直杆菌：中线长度误差 < 2%，宽度中位误差 < 1 px；
- 合成弯杆/分枝：中线位于掩码内比例 ≥ 99%，极点定位误差 < 2 px；
- 与 momia2 输出对照回归（同一批真实数据，长度/宽度相关系数 > 0.99）。

---

### 5.4 参考文献
[1] Ducret, A., Quardokus, E. M. & Brun, Y. V. MicrobeJ, a tool to automate high-content screening for bacterial cell biology. Nat. Microbiol. 1, 16209 (2016).
[2] Zhu, J. MOMIA: Mycobacteria-Optimized Microscopy Image Analysis (PyPI, 2021); https://pypi.org/project/momia.
[3] Zhu, J. momia2: Mycobacteria-optimized microscopy image analysis version 2 (PyPI, 2023); https://pypi.org/project/momia2.
[4] Taubin, G. Curve and surface smoothing without shrinkage. In Proc. Fifth Int. Conf. Comput. Vis. 852–857 (IEEE, 1995).

---

## 6. 兼容与迁移

- `compat/patch_adapter.py`：`Patch → Field + CellSet` 单向导入；`CellSet → regionprops DataFrame` 供旧绘图代码消费。
- momia2 算法移植清单与"原文件 → 新模块"映射表随重构 PR 维护；momia2 本体不改。

---

## 7. 测试与验收

1. `tests/synth.py`：合成杆菌时序生成器——参数化长度生长速率、宽度、弯曲、分裂（V-snapping 可选）、漂移、噪声、光漂白；输出图像 + ground truth（labels、谱系边表、长度/宽度真值）。
2. 单测：schema 往返、key 编解码、各算子（含 GPU/CPU 等价性容差 1e-4）、追踪基准（合成集 TRA ≥ 0.95）。
3. 端到端：`examples/pipeline_demo.json` 从 ND2 导入到 tracks 表全链路，CLI 一键运行。
4. 基准：`tests/benchmarks/` 对比 momia2（2048²×100 帧）：目标端到端 ≥ 10×，rolling-ball ≥ 50×，中线 ≥ 10×。

---

## 8. GUI（napari，Phase 5 详设）

- npe2 插件；dock widget：数据集浏览（FOV/通道/帧）、流水线编辑器（registry 自动生成表单 + JSON 预览）、结果查看器。
- 图层映射：images→Image、labels→Labels（可手改，改动触发增量重算）、outline/midline→Shapes、tracks→Tracks 层（track_id 着色）。
- 交互纠错：Tracks 层选择假分裂/合并 → 调用 `lineage.refine` → 局部刷新。
- 标注闭环：困难样本（refine 检出）→ Points 层标注 → 导出训练集。

---

## 9. 里程碑（对应执行顺序）

| Phase | 交付 | 验收 |
|---|---|---|
| 1 | 本文档 | 评审通过 |
| 2 | 数据层 + 脚手架 + 合成数据生成器 | schema 往返测试通过 |
| 3 | 算法层纯函数化 + 流水线引擎 + CLI | 端到端 demo 跑通 |
| 4 | 可用性测试 + 基准报告 | §7 全部通过 |
| 5 | midline/outline 优化 + 新表征算子 | §5.3 验证标准通过 |
| 6 | napari GUI | 可视化加载/运行/纠错闭环 |
