# 03. preprocessor / 레지스트리 / 학습 명세

## 1. 전체 순서

```
generate_hf_dataset (02)  →  [HF repo <user>/b1k_rbm : b1k_skill_train, b1k_skill_val]
        ↓
preprocess_datasets       →  $ROBOMETER_PROCESSED_DATASETS_PATH/<user>_b1k_rbm_b1k_skill_train/
        ↓                       ├── processed_dataset/   (HF Dataset)
        ↓                       ├── frames/*.npz         (32프레임 배열)
        ↓                       └── index_mappings.json
train.py                  →  LoRA on robometer/Robometer-4B
```

## 2. preprocessor 변경

### 2.1 [필수] 로컬 디렉터리 로드 경로 수정

`robometer/data/scripts/preprocess_datasets.py:_load_dataset_from_path`의 로컬 분기는 현재 이렇다:

```python
else:
    # Load from local disk
    dataset = load_dataset(dataset_path)
    return dataset
```

문제 두 가지:
1. HF Hub 분기에서만 `frames_video` 컬럼을 채운다(`patch_path`). 로컬 분기에는 그게 없어서
   이후 `_process_dataset_videos_threaded`의 `example.get("frames_video")`가 항상 `None` →
   **모든 row가 드롭되고 빈 데이터셋이 된다.**
2. `generate_hf_dataset`이 로컬 저장 시 `dataset.save_to_disk()`를 쓰는데 `load_dataset()`으로는 못 읽는다.

Hub에 푸시하는 흐름만 쓸 거면 손댈 필요는 없지만, B1K 클립은 수십 GB라 **로컬에서 먼저 돌려보는 경로가
필요하다.** 다음으로 교체:

```python
else:
    # Load from local disk (save_to_disk output or a directory of parquet/arrow)
    from datasets import load_from_disk

    if os.path.exists(os.path.join(dataset_path, "dataset_info.json")) or os.path.exists(
        os.path.join(dataset_path, "state.json")
    ):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path, name=subset, split="train")

    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]

    # Hub 분기와 동일하게 상대 mp4 경로를 절대 경로로 승격시킨다.
    root = os.environ.get("ROBOMETER_DATASET_PATH", "") or os.path.dirname(os.path.abspath(dataset_path))

    def _patch(rel_path: str) -> str:
        return rel_path if os.path.isabs(rel_path) else os.path.join(root, rel_path)

    dataset = dataset.map(
        lambda x: {"frames_video": _patch(x["frames"]), "frames_path": _patch(x["frames"])}
    )
    return dataset
```

> 이 수정은 다른 데이터셋의 Hub 경로 동작을 바꾸지 않는다 (로컬 분기만 건드림).
> 수정 후 기존 데이터셋 하나(예: robofac 캐시)로 회귀 확인 — [04 G5](04-validation-plan.md#g5-preprocess).

### 2.2 [선택] 불량 trajectory 필터 등록

`preprocess_datasets.py` 상단 `filters` 딕셔너리는 `"<dataset_path>/<subset>"` 키로 드롭 조건을 받는다.
변환기 필터(02 §3.2)를 통과했더라도 재인코딩 후 프레임이 부족한 클립이 남을 수 있으므로 안전망을 건다:

```python
filters = {
    ...
    "<HF_USERNAME>/b1k_rbm/b1k_skill_train": lambda x: x["frames_shape"][0] < 8,
    "<HF_USERNAME>/b1k_rbm/b1k_skill_val":   lambda x: x["frames_shape"][0] < 8,
}
```

`data.min_frames_per_trajectory: 5`(config.yaml)가 학습 시 한 번 더 거른다.

### 2.3 `robometer/configs/preprocess_b1k.yaml`

```yaml
# uv run python -m robometer.data.scripts.preprocess_datasets \
#   --config robometer/configs/preprocess_b1k.yaml --cache_dir=$ROBOMETER_PROCESSED_DATASETS_PATH
#
# 사전 조건: export ROBOMETER_DATASET_PATH=<b1k_rbm 를 내려받은 부모 디렉터리>
#            huggingface-cli download <HF_USERNAME>/b1k_rbm --repo-type dataset \
#              --local-dir $ROBOMETER_DATASET_PATH/b1k_rbm

train_datasets:
  - "<HF_USERNAME>/b1k_rbm"
train_subsets:
  - ["b1k_skill_train"]

eval_datasets:
  - "<HF_USERNAME>/b1k_rbm"
eval_subsets:
  - ["b1k_skill_val"]

max_frames_for_preprocessing: 32
video_frame_sampling: "uniform"
num_proc: 1
num_threads: 16            # I/O 바운드. 디스크가 느리면 8로
force_reprocess: false
cache_dir: /path/to/robometer/processed_datasets

precompute_embeddings: false
embeddings_cache_dir: "embeddings"
dinov2_model: "facebook/dinov2-base"
sentence_model: "sentence-transformers/all-MiniLM-L12-v2"
embedding_batch_size: 64
```

**용량 주의:** npz는 압축이 약하다. 32프레임 × 240×240×3 ≈ 5.5 MB/traj (비압축 기준).
3.6만 traj → **약 200 GB**. 1단계는 task당 episode 수를 줄여 traj 수를 1만 이하로 유지하거나
`max_frames_for_preprocessing`을 16으로 낮춰라. 캐시 디스크 여유를 먼저 확인할 것.

## 3. success cutoff

`robometer/data/dataset_success_cutoff.txt` 끝에 추가 (파일 형식: `data_source,비율`):

```
b1k_skill,1.0
```

- 이 값은 **`data_source` 기준**으로 조회된다 (`robometer/data/samplers/base.py:632`).
  02 §5.5대로 train/val 모두 `b1k_skill`이므로 한 줄이면 된다.
- 1.0인 근거: skill segment는 annotation 경계로 정확히 잘렸고, 시뮬레이션 데이터라 종료 시점이 깔끔하다.
  (`FINETUNE_ROBOMETER.md`도 sim 데이터셋은 1.0을 권장.)
- **episode 단위(`b1k_task`)를 나중에 추가한다면 1.0을 쓰면 안 된다.** episode의 유효 구간 비율이
  0.785~1.0으로 흩어져 있어 단일 스칼라로 근사 불가 → 변환 시점에 `valid_duration`으로 자르고 1.0을 쓸 것
  ([01 §3.4](01-dataset-spec.md#34-valid_duration-과-trailing-idle)).

## 4. 레지스트리 등록

### 4.1 `robometer/data/datasets/name_mapping.py`

캐시 키(= `<hub_user>_<repo>_<subset>`) → 짧은 이름. custom eval 메트릭 이름에 쓰인다.

```python
DS_SHORT_NAME_MAPPING = {
    ...
    # BEHAVIOR-1K
    "<HF_USERNAME>_b1k_rbm_b1k_skill_train": "b1k_skill",
    "<HF_USERNAME>_b1k_rbm_b1k_skill_val": "b1k_skill_val",
}
```

### 4.2 `robometer/data/dataset_category.py`

```python
ALL_DATASOURCES = [
    ...
    "b1k_skill",
]

DATASET_MAP = {
    ...
    "b1k": {
        "train": ["<HF_USERNAME>_b1k_rbm_b1k_skill_train"],
        "eval":  ["<HF_USERNAME>_b1k_rbm_b1k_skill_val"],
    },
}
```

`DATA_SOURCE_CATEGORY`에는 **넣지 않는다**:
- `success`/`failure` — 우리 데이터는 두 라벨이 섞여 있고 row별 `quality_label`로 이미 구분된다.
- `preference_only` — progress 학습이 핵심 목적이므로 제외.
- `paired`, `suboptimal_fail` — 해당 없음.

등록 후 `data.train_datasets=[b1k]` 같은 키로도, 전체 캐시 키로도 지정할 수 있다.

## 5. 학습

### 5.1 LoRA 파인튜닝 (권장 시작점)

```bash
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets

uv run python train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true \
  model.train_preference_head=true \
  model.train_success_head=true \
  data.train_datasets=[<HF_USERNAME>_b1k_rbm_b1k_skill_train] \
  data.eval_datasets=[<HF_USERNAME>_b1k_rbm_b1k_skill_val] \
  data.max_frames=8 \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.warmup_ratio=0.1 \
  training.weight_decay=0.01 \
  training.max_steps=1000 \
  training.output_dir=./logs \
  training.exp_name=rbm4b_lora_b1k_skill \
  logging.log_to=[wandb] \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[<HF_USERNAME>_b1k_rbm_b1k_skill_val] \
  custom_eval.policy_ranking=[<HF_USERNAME>_b1k_rbm_b1k_skill_val] \
  logging.save_best.metric_names=[eval_rew_align/pearson_b1k_skill_val,eval_p_rank/kendall_last_b1k_skill_val] \
  logging.save_best.greater_is_better=[true,true] \
  training.overwrite_output_dir=True \
  training.eval_steps=50 \
  training.custom_eval_steps=50
```

메트릭 이름의 접미사는 §4.1의 짧은 이름과 정확히 일치해야 한다. 틀리면 `save_best`가 조용히 동작하지 않는다.

### 5.2 데이터 관련 하이퍼파라미터 (기본값에서 바꿀 것)

| 키 | 기본 | B1K 권장 | 이유 |
|----|------|---------|------|
| `data.sample_type_ratio` | `[1,0,0]` | 그대로 | 한 번의 forward로 3개 head 전부 학습 |
| `data.traj_same_source_prob` | 0.5 | **0.8** | 같은 소스(=다른 subtask) 네거티브가 FSM 오전이 방지에 직결 |
| `data.dataset_preference_ratio` | 0.7 | **0.3** | 데이터셋 내장 선호쌍이 없으므로 생성 전략 비중을 높임 |
| `data.preference_strategy_ratio` | `[1,1,1,1]` | **그대로** | 아래 정정 참고 |
| `data.progress_strategy_ratio` | `[1,1,1,1]` | 무효 | `sample_type_ratio=[1,0,0]`이면 ProgressSampler가 생성되지 않는다 |

> **정정.** 초안에서 `preference_strategy_ratio`를 `[1,0,2,1]`로(=`SUBOPTIMAL`을 0으로) 권했으나
> **틀렸다.** truncated 네거티브가 `_has_suboptimal`을 `True`로 만들고, truncated가 학습에 반영되는
> 주 경로가 바로 `SUBOPTIMAL`이다(`partial_success` 비교 후 자동 스왑). 0으로 주면 네거티브가 죽는다.
> 기본값 `[1,1,1,1]`로 baseline을 먼저 잡을 것.
| `data.max_frames` | 16 | **8** | Robometer-4B가 8프레임으로 학습됨 ([02 §5.1.1](02-converter-spec.md#511-왜-8인가)). config 기본값 16을 그대로 쓰면 사전학습 분포와 어긋난다 |
| `data.predict_last_frame_partial_progress` | False | **true** | 논문은 실패 궤적에 progress 타깃을 주지 않는다(`p = None`). 끄면 truncated 네거티브가 선형 0→1.0 타깃으로 학습된다 ([02 §6.1](02-converter-spec.md#61-truncated)) |
| `data.min_frames_per_trajectory` | 5 | 그대로 | |

> 위 ratio 순서는 `config.yaml`의 주석 기준:
> preference = `[rewind, suboptimal_same_task, different_task, reverse_progress]`,
> progress = `[different_task, forward_progress, reverse_progress, rewind]`.
> **코드에서 실제 인덱스 순서를 한 번 확인하고 적용할 것** (`samplers/pref.py:_execute_strategy`).

### 5.3 멀티 GPU / 풀 파인튜닝

`FINETUNE_ROBOMETER.md`의 명령과 동일하되 데이터셋 이름만 교체:

```bash
uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=$N_GPUS train.py model.use_peft=false ... (위와 동일)
```

풀 파인튜닝은 B1K 도메인에 과적합되어 일반 로봇 도메인 성능이 붕괴할 수 있다.
챌린지가 B1K 전용이면 문제없지만, LoRA 결과와 반드시 비교할 것.

### 5.4 베이스라인 (비교용, 필수)

1. **Robometer-4B zero-shot** — 파인튜닝 없이 `b1k_skill_val`에서 평가. 개선폭의 분모.
2. **base Qwen3-VL-4B + LoRA** (`training.load_from_checkpoint` 생략) — 사전학습 체크포인트의 기여 측정.

세 조건 모두 같은 val 셋, 같은 메트릭으로 [04 G7](04-validation-plan.md#g7-학습) 표에 기록한다.

## 6. 추론 / 서빙

```bash
uv run python scripts/example_inference_local.py \
  --model-path ./logs/rbm4b_lora_b1k_skill/checkpoint-best \
  --video /path/to/segment_clip.mp4 \
  --task "pick up the radio from the coffee table"
```

FSM 통합용 서버 모드와 임계값은 [05-runtime-contract.md](05-runtime-contract.md) 참고.
