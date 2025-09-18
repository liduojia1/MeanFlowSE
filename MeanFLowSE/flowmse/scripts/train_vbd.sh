#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/xmudata/djli/se/flowmse/flowmse/vbd"   # VB-DMD 根目录（需包含 train/valid/test 子目录及 clean/noisy 结构）
NPROC=4                                           
BATCH_PER_GPU=2                                   
LOG_DIR="lightning_logs"                           

# ==== 建议的环境变量（NCCL/线程） ====
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

# ==== 启动训练 ====
torchrun --standalone --nproc_per_node="${NPROC}" \
  ../../train.py \
  --backbone ncsnpp \
  --ode flowmatching \
  --base_dir "${DATA_DIR}" \
  --batch_size "${BATCH_PER_GPU}" \
  --num_workers 8 \
  --max_epochs 150 \
  --precision 32 \
  --gradient_clip_val 1.0 \
  --t_eps 0.03 \
  --T_rev 1.0 \
  --sigma_min 0.0 \
  --sigma_max 0.487 \
  --use_mfse \
  --mf_weight_final 0.25 \
  --mf_warmup_frac 0.5 \
  --mf_delta_gamma_start 8.0 \
  --mf_delta_gamma_end 1.0 \
  --mf_delta_warmup_frac 0.7 \
  --mf_r_equals_t_prob 0.1 \
  --mf_jvp_clip 5.0 \
  --mf_jvp_eps 1e-3 \
  --mf_jvp_impl fd \
  --mf_jvp_chunk 1 \
  --mf_skip_weight_thresh 0.05 \
  --val_metrics_every_n_epochs 1 \
  --log_every_n_steps 10 \
  --default_root_dir "${LOG_DIR}"

# 说明：
# 1) 单机 4 卡；torchrun 负责多进程，每个 rank 读取 --devices 1，仅绑定到自己这张卡。
# 2) 日志与 ckpt 均在 lightning_logs/<exp_name>/ 下；TensorBoard 直接读取该根目录。
# 3) FD-JVP + 按样本分块 + 课程早期跳过 MF，兼顾稳定性与显存占用。
# 4) 本地监控： tensorboard --logdir lightning_logs --port 6006 ，浏览器访问 http://localhost:6006
