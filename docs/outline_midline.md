# Outline / midline pipeline (momia3.ops)

本文档记录 momia3 中已经默认启用的 outline 与 midline 实现、可调参数、pipeline 调用方式与实测性能。算法来源在文末统一按 Nature 风格编号引用。

## 1. 模块边界

- momia3/src/momia3/ops/outline.py：亚像素闭合轮廓、曲率、EFD。
- momia3/src/momia3/ops/midline.py：中线、极点、宽度 profile、分支、质量标志。
- momia3/src/momia3/ops/polygon.py：inside_polygon 多边形包含/栅格化（numpy 精确测试 + PIL 快速栅格化）。
- momia3/src/momia3/pipeline/steps.py：OutlineParams / MidlineParams pydantic 模型与 pipeline 适配。

omni_lite 只放分割与评测脚本；outline/midline 的实现与默认参数都在 momia3。

## 2. outline 流程

1. mask 高斯平滑 sigma=1，marching squares level=0.5。
2. 可选 RDP 简化；默认关闭。
3. 像素等距降采样：N = perimeter / downsample_factor（默认 2.0 px/点），再用闭合 B 样条拟合（s = smooth_scale * N，smooth_scale=0.03）。
4. 默认一轮 edge+intensity 精化（refine_outline_iters=1）：
   - edge 步：Sobel 法向 ±0.5 px 三点采样，抛物线 subpixel；cutoff=0.2，单轮位移上限 1.5 px，超限回滚；先外扩 0.3 px。
   - intensity 步：阈值暗细胞体生成 refined ring，只把比局部圆窗中位数更深（>0.2 px）的凹点向内移动，上限 3 px。
   - 每轮轻样条平滑，可多轮。
5. 曲率（窗口 bend angle）与 EFD 描述子一并输出。

关键参数（pipeline OutlineParams，默认值与 ops 一致）：

| 参数 | 默认 | 说明 |
|---|---|---|
| sigma | 1.0 | mask 高斯 |
| level | 0.5 | marching squares 等值 |
| downsample_factor | 2.0 | 像素/点 |
| smooth_scale | 0.03 | B 样条刚度（×N） |
| n_points | null | null 时按 perimeter/downsample_factor |
| refine_outline_iters | 1 | 0 关闭 |
| refine_outline_mode | edge+intensity | edge / intensity / edge+intensity |
| refine_edge_step | 0.5 | 法向采样步长 px |
| refine_edge_cutoff | 0.2 | 弱边阈值 |
| refine_edge_max_shift | 1.5 | 单轮位移上限 px |
| refine_edge_pre_expand | 0.3 | 首轮前外扩 px |
| constriction_image_iters | 0 | 单独跑 intensity 收腰，0 关闭 |

## 3. midline 路由

midline_method 三个取值：

- spline：默认，pole-first + 测地中心线 + 低自由度样条（历史路线）。
- energy：实验路线（dense polyline 曲率/间距正则），默认不使用。
- equidistant：优化路线，MicrobeJ 风格 contour-cut 中线 [1]。

equidistant 的步骤：

1. 简单形状门控：solidity >= 0.88 且 ellipse_dev <= 0.12 且 sagitta_ratio <= 0.08（或 AR < 1.25）走 degenerate：直主轴 + 1 轮横截面弦中点精修（末端 2 点只参与样条约束），输出 32 点。
2. 其他细胞：outline 以自适应 4x 栅格化后 skeletonize（SDF skeleton）；bbox 最长边 > 400 px 的超长细胞改走 mask skeleton。
3. 极点对、主干、分支；分支最短长度 = max(equidistant_branch_min_px, equidistant_branch_min_um / pixel_microns)。
4. 自适应采样：axis 与 midline 点距 2 px（上限 2048）；主干样条平滑 + Taubin [4]；长细胞越界点投影回主干。
5. guard：候选偏离极点弦或过度弯曲时拒绝（保留失败候选，不回退 spline），质量写 midline_quality。

关键参数（pipeline MidlineParams，默认值与 ops 一致）：

| 参数 | 默认 | 说明 |
|---|---|---|
| midline_method | spline | spline / energy / equidistant |
| method | zhang | skeleton 算法 |
| min_branch_length_um | 0.2 | spline 路线分支阈值 |
| pixel_microns | null | null 读 store |
| degenerate_ar | 1.4 | 近圆门控 |
| degenerate_refine | true | 弦中点精修 |
| degenerate_refine_points | 12 | 精修折线点数 |
| degenerate_solidity_min | 0.85 | 仅凸形 mask 走 degenerate |
| equidistant_guard | true | 拒绝坏候选 |
| equidistant_skeleton_from_outline | true | 从 outline 几何建 SDF |
| equidistant_sdf_max_dim | 400 | 超长细胞切 mask skeleton |
| equidistant_sdf_max_pixels | 4e6 | SDF 栅格预算 |
| equidistant_n_points_auto | true | 按弧长自适应采样 |
| equidistant_inside_clamp | true | 长细胞越界点投影回主干 |
| advanced_params | {} | 其余 ops 参数透传（如 equidistant_axis_points_max） |

## 4. pipeline 调用

outline.extract 与 midline.extract 的默认参数已经与 ops 默认一致。SP_003 的优化组合可以直接写成：

    {
      "id": "outlines",
      "op": "outline.extract",
      "params": {"refine_outline_iters": 1, "refine_outline_mode": "edge+intensity"},
      "inputs": {"experiment": "declump.experiment", "cells": "cells.cells"},
      "outputs": {"cells": "tables/cells_outlines.parquet"}
    }

    {
      "id": "midlines",
      "op": "midline.extract",
      "params": {"midline_method": "equidistant", "equidistant_guard": true,
                 "advanced_params": {"equidistant_axis_points_max": 2048}},
      "inputs": {"experiment": "declump.experiment", "cells": "outlines.cells"},
      "outputs": {"cells": "tables/cells.parquet"}
    }

advanced_params 用于透传尚未提升为一等字段的 ops 参数；同名字段以 advanced_params 为准。

## 5. inside_polygon

momia3.ops.polygon.inside_polygon(ring, points=None, *, x0, y0, nx, ny, scale, backend="auto")：

- 传 points：matplotlib Path 精确点包含测试。
- 传 nx/ny：栅格化；auto 先试 PIL，失败退 numpy。网格约定：cell (row, col) = (x0+(col+0.5)/scale, y0+(row+0.5)/scale)。

SDF skeleton 栅格化内部即调用该函数。

## 6. 性能（18 个全幅场，SDF-all + 4a88696 门控，ms/cell）

SP_003 2.69、murA 1.72、5I 1.78、PAO1 1.64、A22 2.05、PSVB 2.65、bthai 1.98、francisella 2.00、Serratia 2.89、vibrio 2.31、wiggins 2.33、dnaA 5.04、caulo 9.93、Hpylori 13.56、cex 10.73、ftsN 10.02、Az_branch 10.75、streptomyces 36.37。

outline 精化单轮额外成本约 0.9 ms/cell（SP_003 0.24 -> 1.20 ms/cell，含 edge+intensity）。

## 7. 参考文献

[1] Ducret, A., Quardokus, E. M. & Brun, Y. V. MicrobeJ, a tool to automate high-content screening for bacterial cell biology. Nat. Microbiol. 1, 16209 (2016).
[2] Zhu, J. MOMIA: Mycobacteria-Optimized Microscopy Image Analysis (PyPI, 2021); https://pypi.org/project/momia.
[3] Zhu, J. momia2: Mycobacteria-optimized microscopy image analysis version 2 (PyPI, 2023); https://pypi.org/project/momia2.
[4] Taubin, G. Curve and surface smoothing without shrinkage. In Proc. Fifth Int. Conf. Comput. Vis. 852-857 (IEEE, 1995).

## 8. 验证

- tests/test_outline.py、tests/test_midline.py、tests/test_polygon.py 覆盖轮廓、中线、inside_polygon。
- omni_lite/scripts/qc_sdf_fullfield.py 复现全幅场统计；omni_lite/results/sdf_fullfield_sdfall/ 保存可视化。
