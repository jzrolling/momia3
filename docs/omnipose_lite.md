# Omnipose-lite 推理后端

momia3.models.omnipose_lite 是 omni_lite 框架合并进来的推理层，围绕参考 Omnipose CPnet
权重做轻量编排：不做 RGB flow 可视化，Euler 积分走 torch（CUDA/MPS/CPU 均可）。可选返回
原始 flow/boundary，并可启用原版 flow-error QC。不包含小模型/蒸馏网络，默认就是原始 CPnet 权重。

## 安装

    pip install "momia3[inference]"
    # 需要 Omnipose 包及其 cellpose_omni 依赖（按 Omnipose 官方安装说明）
    pip install omnipose fastremap

momia3 核心安装不依赖 torch；只有导入该子模块时才需要。

## 权重

权重不随仓库分发，解析顺序：

1. 调用时显式传入的 weights_path
2. 环境变量 MOMIA3_OMNIPOSE_WEIGHTS
3. ~/.cellpose/models/bact_phase_omnitorch_0
4. ~/.omnipose/models/bact_phase_omnitorch_0

模型下载用 Omnipose 自身的模型管理（python -m omnipose）或上游发布文件，然后用环境变量
指到本地路径。

## 用法

    from momia3.models.omnipose_lite import FastOmnipose
    fast = FastOmnipose(device="mps")        # 自动解析权重
    masks = fast.segment(image, niter=200)   # uint32 labels（兼容接口）
    result = fast.segment_result(image)       # labels + foreground + distance + qc
    print(fast.timings.totals)               # preprocess/network/euler/...

`SegmentationResult` 也是概率图 `prob_to_result` 和距离图 `declump_result` 的统一返回类型；
三条路径共享实例清理，概率图与距离图共享 watershed 重建。仅需要标签时仍可调用
`segment`、`prob_to_labels` 或 `declump`。

CUDA 默认使用 BF16 autocast；FP16 在 CPnet 中会溢出，不应使用。可显式关闭混合精度：

    fast = FastOmnipose("/path/to/bact_phase_omnitorch_0", device="cuda", half=False)

### 推理尺度

`rescale` 与原版 Omnipose 一致，在归一化后、网络推理前缩放图像；输入严格使用
`scipy.ndimage.zoom(..., order=3)`。由细胞直径估计时为 `diam_mean / diameter`，细菌模型的
`diam_mean=30`。默认 `rescale=1` 不改变输入尺寸。

    # 半尺寸网络推理，输出插值回原尺寸后再做 Euler 与实例重建
    labels = fast.segment(image, rescale=0.5, resample=True)

    # Euler、实例重建和 cleanup 也在半尺寸执行，标签最终最近邻放大回原尺寸
    labels = fast.segment(image, rescale=0.5, resample=False)

`resample=True` 保留原尺寸轮廓重建，但小细胞可能触发较慢的 DBSCAN 分支；`resample=False`
更快，但可能增加过度分割和边界量化误差。当前真实拼接基准中 `rescale=0.5` 明显降低精度，
因此两种缩放模式均为可选项，默认保持原尺寸推理。

### Flow QC 与原始输出

    result = fast.segment_result(
        image, flow_threshold=0.4, return_flow=True, return_boundary=True
    )
    result.flow       # (2, H, W)，网络原始 flow
    result.boundary   # (H, W)，网络原始 boundary

`flow_threshold=0` 默认关闭 QC 以保持速度；原版默认值为0.4。开启后在实例重建后、cleanup
前调用原版 `remove_bad_flow_masks`，会明显增加耗时。请求的 flow/boundary 与 labels、distance
一样始终保持原图尺寸；`resample=False` 时使用双线性插值恢复。

### Tiling

lite 当前采用整图推理，不启用 tiling。原版对照中，224 px tiles 在2048×2048密集细胞图上没有
带来质量提升，overlap 0.1/0.3 均增加耗时，因此不作为默认路径； tiling 主要仍适用于显存受限
或整图无法放入设备的情况。

大图重建使用 OpenCV 连通域（未安装时回退 SciPy）和 Numba JIT 孔洞清理；这些路径不改变
网络权重、Euler 默认 200 步或实例尺寸/孔洞阈值。

## Pipeline 用法

    {
      "id": "segment",
      "op": "segment.omnipose",
      "params": {"channel": "phase", "device": "mps", "niter": 200,
                 "mask_threshold": 0.0, "min_size": 15},
      "inputs": {"experiment": "import.experiment"},
      "outputs": {"experiment": "experiment.zarr"}
    }

segment.omnipose 逐帧写回 store 的 labels；GPU 设备下会在主进程串行执行，
避免每个 ProcessPool worker 复制一份模型。单帧处理可用 segment.omnipose_frame
（输入 image array，输出 labels array）。权重缺失时 runner 会抛 FileNotFoundError
并提示下载路径。

## 预处理建议

明场/相差图像先做背景压制再送网络：momia3.ops.preprocess.background_correct
（rolling_ball / tophat / tile_percentile / tile_median）或 dual_bandpass，
随后 normalize。实测在 DRT7 明场数据上 rolling-ball 50 + 还原均值 + normalize 最稳；
dual-bandpass 会增强边缘，容易加重 oversegmentation，不作为默认分割预处理。

## 参考

[1] Ducret, A., Quardokus, E. M. & Brun, Y. V. MicrobeJ, a tool to automate
    high-content screening for bacterial cell biology. Nat. Microbiol. 1,
    16209 (2016).
[2] Cutler, K. J. et al. Omnipose: a high-precision morphology-independent
    solution for bacterial cell segmentation. Nat. Methods 19, 1438-1448
    (2022).
