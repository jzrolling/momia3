# Omnipose-lite 推理后端

momia3.models.omnipose_lite 是 omni_lite 框架合并进来的推理层，围绕参考 Omnipose CPnet
权重做轻量编排：不做 RGB flow 可视化、不做 boundary/flow error 后处理，Euler 积分走
torch（CUDA/MPS/CPU 均可）。不包含小模型/蒸馏网络，默认就是原始 CPnet 权重。

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
    masks = fast.segment(image, niter=200)   # uint32 labels
    print(fast.timings.totals)               # preprocess/network/euler/...

也支持显式路径和 FP16/bf16（CUDA）：

    fast = FastOmnipose("/path/to/bact_phase_omnitorch_0", device="cuda", half=True)

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
