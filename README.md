# BReD 训练与加速推理环境

本目录仅整理 BReD 两阶段训练、标准评测和 v8.5 加速推理重测直接使用的代码。数据集、训练结果、普通 checkpoint、运行缓存和日志未复制；运行前需按各入口脚本中的路径准备或挂载这些资产。

## 代码范围

- `stage1/`：BReD Stage 1 训练、评测和模型实现。
- `bridge/`：从 Stage 1 冻结导出 Stage 2 所需高程与几何先验。
- `stage2/`：BReD Stage 2 训练与独立评测。
- `vendor/`：BEVDepth 与三套 Gaussian CUDA 扩展源码。
- `bench_opt/`：v2 → v5 → v7 → v8.5 加速推理实际调用链、CUDA matcher、TensorRT Python 运行库和 engine。

## 实测环境版本

以下版本来自本机训练与加速推理重测所用解释器 `/root/autodl-tmp/envs/bevdepth_rogs/bin/python`，用于复现实验时对齐 ABI 和数值环境。

### 系统、GPU 与编译工具

| 组件 | 实测版本 |
|---|---|
| OS / 架构 | Linux x86_64，kernel 5.15.0-94-generic |
| glibc | 2.35 |
| GPU | 2 × NVIDIA GeForce RTX 4090 D |
| NVIDIA Driver | 595.71.05 |
| CUDA Toolkit / nvcc | 11.7 |
| GCC | 11.4.0 |
| Ninja | 1.11.1 |

GPU 驱动可向后兼容 CUDA 11.7。TensorRT engine 与 GPU 架构、CUDA、TensorRT 和插件 ABI 相关；跨机器重测时若这些条件变化，应重新构建 engine，不能假定已有 engine 可直接复用。

### Python 与训练依赖

| 组件 | 实测版本 |
|---|---|
| Python | 3.8.20 |
| PyTorch | 1.13.1+cu117 |
| torchvision | 0.14.1+cu117 |
| PyTorch 内置 cuDNN | 8.5.0（`torch.backends.cudnn.version() == 8500`） |
| PyTorch3D | 0.7.8 |
| mmcv-full | 1.6.0 |
| mmdet | 2.25.0 |
| PyTorch Lightning | 1.6.0 |
| torchmetrics | 0.7.3 |
| NumPy | 1.23.5 |
| SciPy | 1.10.1 |
| numba | 0.58.1 |
| nuscenes-devkit | 1.1.11 |
| OpenCV | 4.8.1 |
| Pillow | 10.4.0 |
| PyYAML | 6.0.3 |
| pyquaternion | 0.9.9 |
| plyfile | 1.0.3 |
| matplotlib | 3.5.3 |
| tensorboard | 2.14.0 |

其余 Python 依赖及固定版本见 `requirements.txt`。不要将高版本 PyTorch、mmcv 或 NumPy 直接混入原 checkpoint 复现环境。

### 加速推理依赖

| 组件 | 实测版本或位置 |
|---|---|
| TensorRT Python / runtime | 8.6.0，位于 `bench_opt/trt_target/` |
| cuDNN runtime | `nvidia-cudnn-cu11==8.5.0.96` |
| CUDA matcher | `bench_opt/matcher_ext/build_matcher/matcher_ext.so` |
| Stage 1 FP32 backbone engine | `bench_opt/goal/method_version/v7/engine/resnet50_neck_fp32_notf32.engine` |
| Stage 1 FP16 backbone engine | `bench_opt/goal/method_version/v8/engine/resnet50_neck_fp16.engine` |
| DepthNet FP32 engine | `bench_opt/goal/method_version/v8/engine/depthnet_predcn_fp32_notf32.engine` |
| DepthNet FP16 engine | `bench_opt/goal/method_version/v8/engine/depthnet_predcn_fp16.engine` |

本地 TensorRT binding 加载时需要找到 `libcudnn.so.8`。当前重测环境可使用：

```bash
export LD_LIBRARY_PATH=/root/autodl-tmp/envs/rogs/lib/python3.8/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$PWD/stage1:$PWD/stage2:$PWD/bench_opt:$PWD/bench_opt/goal/method_version/v7:$PWD/bench_opt/goal/method_version/v8:$PWD/bench_opt/trt_target"
```

上述设置下已验证 `import tensorrt` 返回 8.6.0，CUDA matcher 也可正常导入。

## 安装与扩展构建

先创建与上述版本一致的 Python/CUDA 环境并安装 `requirements.txt`，再在项目根目录执行：

```bash
cd /root/autodl-tmp/BReD
PYTHON=/root/autodl-tmp/envs/bevdepth_rogs/bin/python bash setup.sh
```

`setup.sh` 会原地构建：

1. BEVDepth voxel pooling inference/train 扩展；
2. 标准 Gaussian rasterizer；
3. 正交 Gaussian rasterizer；
4. label rasterizer。

预编译 `.so`、CUDA matcher 和 TensorRT engine 只能在兼容 ABI 下使用。若 Python、PyTorch、CUDA、GPU 架构或 TensorRT 版本变化，应重新编译扩展并重建相应 engine。

## 训练与评测入口

```bash
# Stage 1
PYTHON=/root/autodl-tmp/envs/bevdepth_rogs/bin/python bash train_stage1.sh
PYTHON=/root/autodl-tmp/envs/bevdepth_rogs/bin/python bash evaluate_stage1.sh

# 冻结桥接与 Stage 2
PYTHON=/root/autodl-tmp/envs/bevdepth_rogs/bin/python bash prepare_stage2.sh
PYTHON=/root/autodl-tmp/envs/bevdepth_rogs/bin/python bash train_stage2.sh
PYTHON=/root/autodl-tmp/envs/bevdepth_rogs/bin/python bash evaluate_stage2.sh
```

训练和标准评测依赖的数据、metadata、官方权重、训练 checkpoint 与 Stage 2 run directory 未包含在本整理目录中，需要按脚本约定放入 `assets/`、`results/`，或通过环境变量指定。

## v8.5 加速推理重测

v8.5 没有单独 runner，实际入口是 `bench_opt/goal/method_version/v7/run_v7.sh`。调用链为：

```text
v7/run_v7.sh
  -> v5/eval_v5.py
     -> v2/eval_v2.py
     -> v5/v7 优化补丁
     -> v8/j81_depthnet_trt.py
  -> v5/evaluate_stage2_v56.py
```

runner 已改为根据自身位置解析 BReD 根目录，不再硬编码旧项目路径。示例：

```bash
cd /root/autodl-tmp/BReD
export LD_LIBRARY_PATH=/root/autodl-tmp/envs/rogs/lib/python3.8/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

V7VER=v8.5 \
V7_TRT_BACKBONE=1 \
V8_TRT_FP16=1 \
V8_DN_TRT=1 \
V8_DN_TRT_FP16=1 \
bash bench_opt/goal/method_version/v7/run_v7.sh <run_tag> 16
```

具体开关应以原重测命令为准，不应为追求速度额外启用 INT8、剪枝、再训练或改变评测协议。图像预缓存和 Stage 2 dataset cache 未复制；启用相应开关前需要重新生成或从外部挂载。

## 已完成的静态检查

- 545 个 Python 文件通过 `py_compile`；
- v5/v7 四个 runner 通过 `bash -n`；
- CUDA matcher 可从整理目录导入；
- TensorRT 8.6.0 在补充 cuDNN library path 后可导入；
- 整理目录中未保留旧根路径 `/root/autodl-tmp/BReD_validation_20261115`。
