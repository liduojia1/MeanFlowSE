#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# FlowSE / MeanFlow-SE — VBD 推理脚本（自动选 ckpt + 三种模式）
#   模式：
#     - multistep      : 5-step 瞬时场（Euler）
#     - multistep_mf   : 5-step 平均场（Euler-MF，每一步用平均速度）
#     - onestep        : 1-step 平均场（Euler-MF，极快）
# ==============================================================================

#############################
# 1) 用户需确认/可修改的配置
#############################

# 【A】数据与输出
TEST_DATA_DIR="${TEST_DATA_DIR:-}"        # 必须包含 test/{clean,noisy}
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

# 【B】检查点：可填“文件”或“目录”
# - 若是 .ckpt 文件，直接使用；
# - 若是 checkpoints 目录，脚本会自动在其中按 `pesq=`（其次 `si_sdr=`）挑选最佳；再不行回退到最后一次 `*last*.ckpt`。
CKPT_INPUT="${CKPT_INPUT:-}"

# 【C】默认推理模式与步数
#   MODE=multistep      -> Euler（瞬时场）
#   MODE=multistep_mf   -> Euler-MF（平均场，每一步都用平均速度）
#   MODE=onestep        -> Euler-MF（一步）
MODE="${MODE:-multistep_mf}"
STEPS="${STEPS:-5}"

# 【D】ODE 时间端点（保持与训练/论文一致）
REVERSE_STARTING_POINT="${REVERSE_STARTING_POINT:-1.0}"   # t_N
LAST_EVAL_POINT_MULTI="${LAST_EVAL_POINT_MULTI:-0.03}"    # 多步默认 0.03
LAST_EVAL_POINT_MULTI_MF="${LAST_EVAL_POINT_MULTI_MF:-${LAST_EVAL_POINT_MULTI}}"  # 多步-MF 末端，默认同上
LAST_EVAL_POINT_ONE="${LAST_EVAL_POINT_ONE:-0.0}"         # 单步建议 0.0（严格 1→0）

# 线程等（可选）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

##########################################
# 2) 解析 ckpt：若传目录则自动挑“最佳”
##########################################
pick_ckpt() {
  local input="$1"
  if [[ -f "$input" ]]; then
    echo "$input"
    return 0
  fi
  if [[ -d "$input" ]]; then
    local best
    best="$(CKPT_DIR="$input" python - <<'PY'
import os, re, glob, sys
ckpt_dir = os.environ['CKPT_DIR']
cands = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
def pick_by(metric):
    best, smax = None, -1e18
    pat = re.compile(rf"{metric}=([0-9.]+)")
    for p in cands:
        m = pat.search(os.path.basename(p))
        if m:
            s = float(m.group(1))
            if s > smax: smax, best = s, p
    return best, smax
best, _ = pick_by("pesq")
if best is None:
    best, _ = pick_by("si_sdr")
if best is None:
    last = sorted([p for p in cands if "last" in os.path.basename(p)],
                  key=os.path.getmtime)
    best = last[-1] if last else None
if best is None:
    sys.exit("No checkpoint found under dir: "+ckpt_dir)
print(best)
PY
)"
    echo "$best"
    return 0
  fi
  echo "❌ CKPT_INPUT 不存在：$input" >&2
  exit 1
}

##########################################
# 3) Sanity Check + 生成输出目录名
##########################################
if [[ "$CKPT_INPUT" == "path/to/your/best/model.ckpt" ]]; then
  echo "❌ 请先设置 CKPT_INPUT 为 .ckpt 文件或 checkpoints 目录"; exit 1
fi
CKPT_PATH="$(pick_ckpt "$CKPT_INPUT")"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_tag="$(basename "$(dirname "$CKPT_PATH")")"   # version_x
run_root="$(basename "$(dirname "$(dirname "$CKPT_PATH")")")"  # dataset_xxx
OUT_DIR_MODE="${MODE}"
[[ "$MODE" == "multistep" || "$MODE" == "multistep_mf" ]] && OUT_DIR_MODE="${MODE}_N${STEPS}"
OUT_DIR="${OUTPUT_ROOT}/${run_root}/${run_tag}/${OUT_DIR_MODE}_${timestamp}"
mkdir -p "$OUT_DIR"

echo "✅ 推理设置确认："
echo "  - CKPT:                 $CKPT_PATH"
echo "  - TEST_DATA_DIR:        $TEST_DATA_DIR"
echo "  - OUTPUT_DIR:           $OUT_DIR"
echo "  - MODE:                 $MODE"
echo "  - STEPS:                $STEPS"
echo "  - REVERSE_STARTING_PT:  $REVERSE_STARTING_POINT"

##########################################
# 4) 选择模式并调用 evaluate.py
##########################################
if [[ "$MODE" == "multistep" ]]; then
  # 多步：Euler（瞬时场），如需 5 步：STEPS=5
  echo "🚀 运行多步推理（Euler, N=${STEPS}) ..."
  python evaluate.py \
    --test_dir "$TEST_DATA_DIR" \
    --folder_destination "$OUT_DIR" \
    --ckpt "$CKPT_PATH" \
    --odesolver euler \
    --reverse_starting_point "$REVERSE_STARTING_POINT" \
    --last_eval_point "$LAST_EVAL_POINT_MULTI" \
    --N "$STEPS"
  echo "✅ 多步推理完成：$OUT_DIR"

elif [[ "$MODE" == "multistep_mf" ]]; then
  # 多步：Euler-MF（平均场），每一步都使用平均速度
  echo "🚀 运行多步推理（Euler-MF, N=${STEPS}) ..."
  python evaluate.py \
    --test_dir "$TEST_DATA_DIR" \
    --folder_destination "$OUT_DIR" \
    --ckpt "$CKPT_PATH" \
    --odesolver euler_mf \
    --reverse_starting_point "$REVERSE_STARTING_POINT" \
    --last_eval_point "$LAST_EVAL_POINT_MULTI_MF" \
    --N "$STEPS"
  echo "✅ 多步-MF 推理完成：$OUT_DIR"

elif [[ "$MODE" == "onestep" ]]; then
  # 单步：Euler-MF（平均场，一步到位），建议 t_eps=0.0
  echo "🚀 运行单步推理（Euler-MF, 1-NFE） ..."
  python evaluate.py \
    --test_dir "$TEST_DATA_DIR" \
    --folder_destination "$OUT_DIR" \
    --ckpt "$CKPT_PATH" \
    --odesolver euler_mf \
    --reverse_starting_point "$REVERSE_STARTING_POINT" \
    --last_eval_point "$LAST_EVAL_POINT_ONE" \
    --one_step
  echo "✅ 单步推理完成：$OUT_DIR"

else
  echo "❌ MODE 仅支持 {multistep|multistep_mf|onestep}，当前：$MODE"; exit 1
fi
