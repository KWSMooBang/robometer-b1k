#!/usr/bin/env bash
# BEHAVIOR-1K -> Robometer pipeline driver.
#
# Run each stage separately so you can inspect the output before paying for the
# next one. Every stage is re-runnable; conversion skips clips whose mp4 already
# exists and preprocessing skips datasets that are already cached.
#
#   ./scripts/b1k_pipeline.sh check      # what can be converted right now
#   ./scripts/b1k_pipeline.sh convert    # annotations -> clips + HF dataset
#   ./scripts/b1k_pipeline.sh preview    # contact sheet, LOOK AT THIS
#   ./scripts/b1k_pipeline.sh preprocess # clips -> npz cache
#   ./scripts/b1k_pipeline.sh verify     # cache sanity before GPU time
#   ./scripts/b1k_pipeline.sh smoke      # 20-step training run
#   ./scripts/b1k_pipeline.sh train      # the real run
#   ./scripts/b1k_pipeline.sh all        # check -> verify (stops before training)
#
# Must be run from the repo root: the relative output path decides the cache key
# that name_mapping.py and DATASET_MAP are registered against.
#
# See docs/b1k/06-runbook.md.

set -euo pipefail

: "${B1K_ROOT:?set B1K_ROOT to the 2026-challenge-demos directory}"
: "${ROBOMETER_PROCESSED_DATASETS_PATH:=$PWD/processed_datasets}"
export ROBOMETER_PROCESSED_DATASETS_PATH

TRAIN_KEY="datasets_b1k_rbm_b1k_skill_train_b1k_skill_train"
VAL_KEY="datasets_b1k_rbm_b1k_skill_val_b1k_skill_val"
EXP_NAME="${EXP_NAME:-rbm4b_lora_b1k_skill}"
RUN="${RUN:-uv run}"

stage="${1:-}"

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

do_check() {
  banner "availability + invariants"
  $RUN python scripts/b1k_check_alignment.py "$B1K_ROOT" --ready-only
}

do_convert() {
  banner "convert TRAIN split"
  $RUN python -m dataset_upload.generate_hf_dataset \
    --config_path dataset_upload/configs/data_gen_configs/b1k_skill.yaml \
    --dataset.dataset_path="$B1K_ROOT"

  banner "convert VAL split"
  $RUN python -m dataset_upload.generate_hf_dataset \
    --config_path dataset_upload/configs/data_gen_configs/b1k_skill_val.yaml \
    --dataset.dataset_path="$B1K_ROOT"

  banner "conversion reports"
  cat datasets/b1k_rbm/*_conversion_report.json
}

do_preview() {
  banner "contact sheet"
  $RUN python scripts/b1k_preview_segments.py "$B1K_ROOT" \
    -o datasets/b1k_rbm/preview.png --rows 12 --truncated-ratio 0.5
  echo "open datasets/b1k_rbm/preview.png -- the LAST frame of each positive row"
  echo "must show that subtask completed, and truncated rows must not."
}

do_preprocess() {
  banner "preprocess -> $ROBOMETER_PROCESSED_DATASETS_PATH"
  $RUN python -m robometer.data.scripts.preprocess_datasets \
    --config robometer/configs/preprocess_b1k.yaml \
    --cache_dir="$ROBOMETER_PROCESSED_DATASETS_PATH"
}

do_verify() {
  banner "verify train cache"
  $RUN python scripts/b1k_verify_cache.py "$TRAIN_KEY"
  banner "verify val cache"
  $RUN python scripts/b1k_verify_cache.py "$VAL_KEY"
}

# data.max_frames=8 matches how Robometer-4B was pretrained; see docs/b1k/02 §5.1.1.
train_args() {
  cat <<EOF
model.base_model_id=Qwen/Qwen3-VL-4B-Instruct
model.use_peft=true
model.train_progress_head=true
model.train_preference_head=true
model.train_success_head=true
data.train_datasets=[b1k]
data.eval_datasets=[b1k]
data.max_frames=8
data.traj_same_source_prob=0.8
data.dataset_preference_ratio=0.3
training.load_from_checkpoint=robometer/Robometer-4B
training.per_device_train_batch_size=8
training.learning_rate=2e-5
training.warmup_ratio=0.1
training.weight_decay=0.01
training.output_dir=./logs
training.overwrite_output_dir=True
custom_eval.eval_types=[reward_alignment,policy_ranking]
custom_eval.reward_alignment=[b1k]
custom_eval.policy_ranking=[b1k]
EOF
}

do_smoke() {
  banner "smoke training run (20 steps)"
  # shellcheck disable=SC2046
  $RUN python train.py $(train_args) \
    training.max_steps=20 \
    training.eval_steps=10 \
    training.custom_eval_steps=10 \
    training.exp_name="${EXP_NAME}_smoke" \
    logging.log_to=[]
}

do_train() {
  banner "training: $EXP_NAME"
  # shellcheck disable=SC2046
  $RUN python train.py $(train_args) \
    training.max_steps="${MAX_STEPS:-1000}" \
    training.eval_steps=50 \
    training.custom_eval_steps=50 \
    training.exp_name="$EXP_NAME" \
    logging.log_to=[wandb] \
    logging.save_best.metric_names=[eval_rew_align/pearson_b1k_skill_val,eval_p_rank/kendall_last_b1k_skill_val] \
    logging.save_best.greater_is_better=[true,true]
}

case "$stage" in
  check)      do_check ;;
  convert)    do_convert ;;
  preview)    do_preview ;;
  preprocess) do_preprocess ;;
  verify)     do_verify ;;
  smoke)      do_smoke ;;
  train)      do_train ;;
  all)        do_check; do_convert; do_preview; do_preprocess; do_verify
              banner "stopped before training on purpose"
              echo "look at datasets/b1k_rbm/preview.png, then: ./scripts/b1k_pipeline.sh smoke" ;;
  *)          sed -n '3,20p' "$0"; exit 1 ;;
esac
