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
#   ./scripts/b1k_pipeline.sh wandb      # check the wandb entity works
#   ./scripts/b1k_pipeline.sh smoke      # 20-step training run
#   ./scripts/b1k_pipeline.sh train      # the real run
#   ./scripts/b1k_pipeline.sh all        # check -> verify (stops before training)
#   ./scripts/b1k_pipeline.sh names      # the names this resolution resolves to
#
# Must be run from the repo root: the relative output path decides the cache key
# that name_mapping.py and DATASET_MAP are registered against.
#
# RES=<pixels> picks the frame resolution and threads it through every stage --
# conversion writes to its own directory, preprocessing to its own cache key, and
# training reads the matching DATASET_MAP entry:
#
#   RES=240 ./scripts/b1k_pipeline.sh all     # default, matches RBM-1M pretraining
#   RES=480 ./scripts/b1k_pipeline.sh all
#   RES=720 ./scripts/b1k_pipeline.sh all
#
# Resolutions never share a directory: create_trajectory_video_optimized() skips
# clips whose mp4 already exists, so reusing one output_dir would silently keep
# the old resolution's clips. Names come from robometer/data/b1k_variants.py.
#
# See docs/b1k/06-runbook.md.

set -euo pipefail

: "${B1K_ROOT:?set B1K_ROOT to the 2026-challenge-demos directory}"
: "${ROBOMETER_PROCESSED_DATASETS_PATH:=$PWD/processed_datasets}"
export ROBOMETER_PROCESSED_DATASETS_PATH

RES="${RES:-240}"
EXP_NAME="${EXP_NAME:-rbm4b_lora_b1k_skill}"
RUN="${RUN:-uv run}"

# Quieten third-party import noise. RBM_VERBOSE=1 turns it all back on.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export GRPC_VERBOSITY="${GRPC_VERBOSITY:-ERROR}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# TRANSFORMERS_CACHE is deprecated; HF_HUB_CACHE is the direct replacement, so
# carry the value over instead of losing the existing model cache.
if [ -n "${TRANSFORMERS_CACHE:-}" ]; then
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$TRANSFORMERS_CACHE}"
  unset TRANSFORMERS_CACHE
fi

# Conversion, preview and (with precompute_embeddings=false) preprocessing are
# CPU-only. Hiding the GPUs stops CUDA/cuDNN registration errors at the source.
cpu_only() { CUDA_VISIBLE_DEVICES="" "$@"; }

stage="${1:-}"

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# One source of truth for every resolution-dependent name. Sets B1K_DATASET_DIR,
# B1K_TRAIN_PATH, B1K_TRAIN_KEY, B1K_VAL_KEY, B1K_MAP_KEY, B1K_ALIGNED_RES, ...
if ! b1k_names="$(cpu_only $RUN python -m robometer.data.b1k_variants "$RES" --shell)"; then
  echo "  ❌ could not resolve names for RES=$RES" >&2
  exit 1
fi
eval "$b1k_names"

TRAIN_KEY="$B1K_TRAIN_KEY"
VAL_KEY="$B1K_VAL_KEY"

# The experiment name carries the resolution so runs at different resolutions do
# not overwrite each other's checkpoints.
if [ "$RES" != "240" ]; then
  EXP_NAME="${EXP_NAME}_${RES}"
fi

# Visual tokens per frame grow with the square of the resolution, so the batch
# that fits at 240 does not fit at 480 or 720. A starting point, not a measured
# optimum -- override with BATCH_SIZE=<n>.
DEFAULT_BATCH_SIZE=$(( 8 * 240 * 240 / (RES * RES) ))
if [ "$DEFAULT_BATCH_SIZE" -lt 1 ]; then
  DEFAULT_BATCH_SIZE=1
fi

res_banner() {
  printf '\033[2mres %s -> %s | cache %s | DATASET_MAP[%s]\033[0m\n' \
    "$RES" "$B1K_DATASET_DIR" "$TRAIN_KEY" "$B1K_MAP_KEY"
  # 240 is also unaligned (the processor rounds it to 256), but it is the
  # pretraining resolution and the default, so warning about it on every run
  # would be noise. Only speak up when a resolution was chosen deliberately.
  if [ "$B1K_PATCH_ALIGNED" != "1" ] && [ "$RES" != "240" ]; then
    printf '\033[33m  ! %s is not a multiple of 32; the processor rounds frames to %sx%s,\033[0m\n' \
      "$RES" "$B1K_ALIGNED_RES" "$B1K_ALIGNED_RES"
    printf '\033[33m    so you would store pixels the model never sees. Prefer RES=%s.\033[0m\n' "$B1K_ALIGNED_RES"
  fi
}
res_banner

do_check() {
  banner "availability + invariants"
  cpu_only $RUN python scripts/b1k_check_alignment.py "$B1K_ROOT" --ready-only
}

do_convert() {
  banner "convert TRAIN split (res $RES)"
  cpu_only $RUN python -m dataset_upload.generate_hf_dataset \
    --config_path dataset_upload/configs/data_gen_configs/b1k_skill.yaml \
    --dataset.dataset_path="$B1K_ROOT" \
    --output.shortest_edge_size="$RES" \
    --output.output_dir="$B1K_DATASET_DIR" \
    --hub.hub_repo_id="$B1K_HUB_REPO_ID"

  banner "convert VAL split (res $RES)"
  cpu_only $RUN python -m dataset_upload.generate_hf_dataset \
    --config_path dataset_upload/configs/data_gen_configs/b1k_skill_val.yaml \
    --dataset.dataset_path="$B1K_ROOT" \
    --output.shortest_edge_size="$RES" \
    --output.output_dir="$B1K_DATASET_DIR" \
    --hub.hub_repo_id="$B1K_HUB_REPO_ID"

  banner "conversion reports"
  cat "$B1K_DATASET_DIR"/*_conversion_report.json

  banner "clip resolution"
  # Cheap proof that the clips really carry the resolution that was asked for:
  # the converter leaves square input untouched on one of its two code paths, so
  # do not take shortest_edge_size on faith.
  cpu_only $RUN python scripts/b1k_check_clip_resolution.py "$B1K_DATASET_DIR" --expect "$RES"
}

do_preview() {
  banner "contact sheet"
  cpu_only $RUN python scripts/b1k_preview_segments.py "$B1K_ROOT" \
    -o "$B1K_DATASET_DIR/preview.png" --rows 12 --truncated-ratio 0.5
  echo "open $B1K_DATASET_DIR/preview.png -- the LAST frame of each positive row"
  echo "must show that subtask completed, and truncated rows must not."
}

do_preprocess() {
  banner "preprocess res $RES -> $ROBOMETER_PROCESSED_DATASETS_PATH"
  if [ "$RES" != "240" ]; then
    echo "  npz grows with the square of the resolution: a 240 cache costs ~5.5 MB"
    echo "  per trajectory, so $RES is ~$(( RES * RES / (240 * 240) ))x that. Check disk first."
  fi
  # The dataset paths are overridden on the command line rather than in the yaml,
  # which keeps one config for every resolution. Confirm the override actually
  # took: if it were dropped, preprocessing would quietly rebuild the 240 cache
  # and the mistake would only surface once training could not find its data.
  $RUN python -m robometer.data.scripts.preprocess_datasets \
    --config robometer/configs/preprocess_b1k.yaml \
    --cache_dir="$ROBOMETER_PROCESSED_DATASETS_PATH" \
    --train_datasets="[$B1K_TRAIN_PATH]" \
    --train_subsets="[[$B1K_TRAIN_SUBSET]]" \
    --eval_datasets="[$B1K_EVAL_PATH]" \
    --eval_subsets="[[$B1K_EVAL_SUBSET]]"

  for key in "$TRAIN_KEY" "$VAL_KEY"; do
    if [ ! -d "$ROBOMETER_PROCESSED_DATASETS_PATH/$key" ]; then
      echo "  ❌ preprocessing did not produce $key"
      echo "     the --train_datasets/--eval_datasets override did not reach the config;"
      echo "     set the paths in robometer/configs/preprocess_b1k.yaml by hand instead:"
      echo "       train_datasets: [\"$B1K_TRAIN_PATH\"]"
      echo "       eval_datasets:  [\"$B1K_EVAL_PATH\"]"
      exit 1
    fi
  done
  echo "  ✅ cache written: $TRAIN_KEY, $VAL_KEY"
}

do_wandb() {
  banner "wandb entity check"
  # Runs in seconds, before the 16 GB checkpoint download. wandb rejects runs
  # logged to an organization root, and guessing an entity name (an email
  # prefix, say) fails with "entity ... not found".
  $RUN python - <<'PY'
import os, sys, wandb

try:
    api = wandb.Api()
    viewer = api.viewer
    print(f"  username        : {viewer.username}")
    print(f"  teams           : {viewer.teams}")
    print(f"  default_entity  : {api.default_entity}")
except Exception as exc:
    print(f"  ! could not query wandb: {exc}")
    print("  run `wandb login` first")
    sys.exit(1)

entity = os.environ.get("WANDB_ENTITY") or api.default_entity
print(f"\n  trying wandb.init(entity={entity!r}) ...")
try:
    run = wandb.init(project="robometer", entity=entity, name="b1k-entity-check",
                     mode="online", settings=wandb.Settings(silent=True))
    url = run.url
    run.finish()
    print(f"  ✅ works: {url}")
    print(f"  use: WANDB_ENTITY={entity} ./scripts/b1k_pipeline.sh train")
except Exception as exc:
    print(f"  ❌ {type(exc).__name__}: {exc}")
    print("\n  pick one of the teams listed above:")
    print("    WANDB_ENTITY=<team> ./scripts/b1k_pipeline.sh wandb")
    print("  or skip logging entirely:")
    print("    WANDB=off ./scripts/b1k_pipeline.sh train")
    sys.exit(1)
PY
}

do_verify() {
  banner "verify train cache"
  $RUN python scripts/b1k_verify_cache.py "$TRAIN_KEY" --expect-resolution "$RES"
  banner "verify val cache"
  $RUN python scripts/b1k_verify_cache.py "$VAL_KEY" --expect-resolution "$RES"
}

# data.max_frames=8 matches how Robometer-4B was pretrained; see docs/b1k/02 §5.1.1.
#
# data.predict_last_frame_partial_progress=true keeps the paper's rule that failed
# trajectories carry no frame-level progress target (they are "p=None", used only
# through the preference objective). Our truncated negatives set partial_success,
# and only the SUBOPTIMAL strategy swaps them into the rejected slot -- the other
# three leave them as the chosen trajectory, where progress loss *is* applied.
# Without this flag they would be trained toward a linear 0->1 progress curve,
# i.e. "this failure completed the subtask". See docs/b1k/NOTION.md §5.3.
#
# data.max_image_side must be >= RES or the collator downscales the frames back
# and the whole point of converting at a higher resolution is lost. It warns when
# that happens, but the run would already be wasted.
#
# BATCH_SIZE: visual tokens scale with the square of the resolution (roughly 64
# per frame at 240, 225 at 480, 484 at 720, times 8 frames times two trajectories
# per preference pair), so the batch that fits at 240 will not fit at 720.
train_args() {
  cat <<EOF
model.base_model_id=Qwen/Qwen3-VL-4B-Instruct
model.use_peft=true
model.train_progress_head=true
model.train_preference_head=true
model.train_success_head=true
data.train_datasets=[$B1K_MAP_KEY]
data.eval_datasets=[$B1K_MAP_KEY]
data.max_frames=8
data.max_image_side=$RES
data.max_image_pixels=$(( RES * RES ))
data.traj_same_source_prob=0.8
data.dataset_preference_ratio=0.3
data.predict_last_frame_partial_progress=true
training.load_from_checkpoint=robometer/Robometer-4B
training.per_device_train_batch_size=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
training.learning_rate=2e-5
training.warmup_ratio=0.1
training.weight_decay=0.01
training.output_dir=./logs
training.overwrite_output_dir=True
custom_eval.eval_types=[reward_alignment,policy_ranking]
custom_eval.reward_alignment=[$B1K_MAP_KEY]
custom_eval.policy_ranking=[$B1K_MAP_KEY]
EOF
}

# The cache directory name is derived from the preprocess dataset_path, so running
# preprocess from another cwd (or with an absolute path, or at another RES) produces
# a key that DATASET_MAP[$B1K_MAP_KEY] does not match. Catch that here rather than
# after the model has loaded.
require_cache() {
  local missing=0
  for key in "$TRAIN_KEY" "$VAL_KEY"; do
    if [ ! -d "$ROBOMETER_PROCESSED_DATASETS_PATH/$key" ]; then
      echo "  ❌ missing cache: $ROBOMETER_PROCESSED_DATASETS_PATH/$key"
      missing=1
    fi
  done
  if [ "$missing" = "1" ]; then
    echo
    echo "  caches that DO exist:"
    ls -1 "$ROBOMETER_PROCESSED_DATASETS_PATH" 2>/dev/null | sed 's/^/    /' || echo "    (none)"
    echo
    echo "  The cache key is derived from the preprocess dataset_path, which the"
    echo "  resolution decides. Run preprocessing from the repo root at RES=$RES:"
    echo "    cd $(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    echo "    export ROBOMETER_PROCESSED_DATASETS_PATH=\$PWD/processed_datasets"
    echo "    RES=$RES ./scripts/b1k_pipeline.sh preprocess"
    exit 1
  fi
}

do_smoke() {
  banner "smoke training run (20 steps)"
  require_cache
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
  require_cache
  # WANDB=off skips logging entirely; WANDB_ENTITY picks the team/user to log
  # under (unset = the logged-in account's default entity).
  local log_args="logging.log_to=[wandb]"
  if [ "${WANDB:-on}" = "off" ]; then
    log_args="logging.log_to=[]"
  elif [ -n "${WANDB_ENTITY:-}" ]; then
    log_args="$log_args logging.wandb_entity=$WANDB_ENTITY"
  fi
  # shellcheck disable=SC2046,SC2086
  $RUN python train.py $(train_args) \
    training.max_steps="${MAX_STEPS:-1000}" \
    training.eval_steps=50 \
    training.custom_eval_steps=50 \
    training.exp_name="$EXP_NAME" \
    $log_args \
    logging.save_best.metric_names=[eval_rew_align/pearson_$B1K_SHORT_NAME_VAL,eval_p_rank/kendall_last_$B1K_SHORT_NAME_VAL] \
    logging.save_best.greater_is_better=[true,true]
}

case "$stage" in
  check)      do_check ;;
  convert)    do_convert ;;
  preview)    do_preview ;;
  preprocess) do_preprocess ;;
  verify)     do_verify ;;
  wandb)      do_wandb ;;
  smoke)      do_smoke ;;
  train)      do_train ;;
  names)      $RUN python -m robometer.data.b1k_variants "$RES" ;;
  all)        do_check; do_convert; do_preview; do_preprocess; do_verify
              banner "stopped before training on purpose"
              echo "look at $B1K_DATASET_DIR/preview.png, then: RES=$RES ./scripts/b1k_pipeline.sh smoke" ;;
  *)          sed -n '3,33p' "$0"; exit 1 ;;
esac
