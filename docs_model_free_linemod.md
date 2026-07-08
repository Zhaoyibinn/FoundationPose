# LINEMOD Model-Free 流程

本文档记录单物体 `ob_0000001` 的流程：先用 BundleSDF 从参考视图重建 mesh，再用 FoundationPose 做位姿估计。

## 1. 环境

```bash
conda activate foundationpose
cd /home/zhaoyibin/3DRE/MVS/FoundationPose

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/home/zhaoyibin/miniforge3/envs/foundationpose/lib/python3.11/site-packages/torch/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD:$PWD/bundlesdf:$PYTHONPATH
export MPLCONFIGDIR=/tmp/matplotlib-foundationpose
```

## 2. BundleSDF 重建

输入参考视图目录：

```text
linemod/ref/ob_0000001/
  K.txt
  select_frames.yml
  rgb/
  depth_enhanced/
  mask/ 或 mask_refined/
  cam_in_ob/
```

开始重建：

```bash
python bundlesdf/run_nerf_one.py \
  --ref_view_dir linemod/ref \
  --ob_id 1 \
  --dataset linemod
```

其中 `--ref_view_dir linemod/ref` 是 BundleSDF 参考视图父目录，内部应有 `ob_0000001/`；`--ob_id 1` 表示只重建 `ob_0000001`。

重建输出：

```text
linemod/ref/ob_0000001/model/model.obj
linemod/ref/ob_0000001/nerf/model_latest.pth
```

## 3. 位姿估计

待估计序列目录：

```text
linemod/all/000001/
  scene_camera.json
  scene_gt.json
  rgb/
  depth/
  mask_visib/
```

运行 model-free 位姿估计：

```bash
python run_linemod.py \
  --linemod_video_dir linemod/all/000001 \
  --use_reconstructed_mesh 1 \
  --ref_view_dir linemod/ref \
  --ob_id 1 \
  --debug_dir debug_linemod_ob1_model_free
```

其中 `--linemod_video_dir linemod/all/000001` 是要估计位姿的序列目录；`--use_reconstructed_mesh 1` 表示使用 BundleSDF 重建 mesh；`--ref_view_dir linemod/ref` 指向 `ob_0000001/model/model.obj` 所在父目录；`--ob_id 1` 表示只处理 `ob_0000001`。

运行中会持续保存：

```text
debug_linemod_ob1_model_free/
  linemod_res.yml        # 汇总位姿结果
  ob_in_cam/*.txt        # 每帧 4x4 位姿矩阵
  track_vis/*.png        # 每帧 bbox + 坐标轴可视化
  rgb/*.png              # 输入 RGB
  mask/*.png             # 使用的 mask
```

如果不想保存输入 RGB 和 mask：

```bash
python run_linemod.py \
  --linemod_video_dir linemod/all/000001 \
  --use_reconstructed_mesh 1 \
  --ref_view_dir linemod/ref \
  --ob_id 1 \
  --debug_dir debug_linemod_ob1_model_free \
  --save_frame_inputs 0
```
