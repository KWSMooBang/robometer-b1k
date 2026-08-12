# 06. 서버 실행 런북

task-0000(`turning_on_radio`) 200 episode로 파이프라인 전체를 관통 검증한 뒤, task를 늘리는 순서.
모든 단계는 재실행 가능하다 (변환은 이미 있는 mp4를 건너뛰고, 전처리는 캐시가 있으면 건너뛴다).

> **반드시 repo 루트에서 실행할 것.** 상대 출력 경로가 전처리 캐시 키를 결정하고,
> 그 키가 `name_mapping.py` / `DATASET_MAP`에 등록된 값과 일치해야 한다.

## 0. 환경

```bash
cd /path/to/robometer-b1k
uv sync

export B1K_ROOT=/path/to/behavior-1k/2026-challenge-demos
export ROBOMETER_PROCESSED_DATASETS_PATH=$PWD/processed_datasets
export HF_TOKEN=...            # Hub에 올릴 때만 필요
```

드라이버 스크립트로 한 단계씩:

```bash
./scripts/b1k_pipeline.sh check       # 1
./scripts/b1k_pipeline.sh convert     # 2
./scripts/b1k_pipeline.sh preview     # 3  ← 눈으로 확인
./scripts/b1k_pipeline.sh preprocess  # 4
./scripts/b1k_pipeline.sh verify      # 5
./scripts/b1k_pipeline.sh smoke       # 6
./scripts/b1k_pipeline.sh train       # 7
```

`./scripts/b1k_pipeline.sh all`은 1~5를 이어서 돌리고 **학습 직전에 멈춘다**(3번을 사람이 봐야 하므로).
아래는 각 단계가 실제로 실행하는 명령과 확인할 것.

---

## 1. 무엇을 변환할 수 있는가

```bash
uv run python scripts/b1k_check_alignment.py $B1K_ROOT --ready-only
```

task별로 annotation과 비디오가 **둘 다** 있는 episode 수(`ready`)를 출력하고, 불변식
(skill end ≤ episode length, skill_idx 정렬)을 검사한다. 다운로드가 부분적이면 `--missing`으로
빠진 비디오 파일 목록을 받아 그것만 추가로 받으면 된다.

**확인:** 쓰려는 task의 `ready`가 기대한 episode 수와 같은지. 불변식 위반이 나오면 **중단**하고
[01 §6](01-dataset-spec.md#6-변환기가-신뢰해도-되는-불변식)을 다시 볼 것 — 프레임 계산이 틀렸다는 뜻이다.

## 2. 변환

```bash
uv run python -m dataset_upload.generate_hf_dataset \
  --config_path dataset_upload/configs/data_gen_configs/b1k_skill.yaml \
  --dataset.dataset_path=$B1K_ROOT

uv run python -m dataset_upload.generate_hf_dataset \
  --config_path dataset_upload/configs/data_gen_configs/b1k_skill_val.yaml \
  --dataset.dataset_path=$B1K_ROOT
```

**task 범위를 바꾸는 법** — yaml을 고치거나 CLI로 덮어쓴다:

```bash
# 여러 task
--dataset.b1k.task_whitelist="['turning_on_radio','picking_up_trash']"
# 전체 task
--dataset.b1k.task_whitelist=null
# task당 상한 (첫 스케일업에 권장)
--dataset.b1k.max_episodes_per_task=20
# 아주 작게 한 번 돌려보기
--dataset.b1k.max_episodes_per_task=2
```

산출물:

```
datasets/b1k_rbm/
├── b1k_skill_train/                          # HF Dataset (save_to_disk)
│   └── batch_0000/trajectory_XXXX.mp4        # 240x240, 64 frames
├── b1k_skill_val/
└── b1k_skill_train_conversion_report.json
```

**확인 (`*_conversion_report.json`):**
- `segments_kept + Σsegments_dropped == segments_total` (로더가 런타임에 assert한다)
- `episodes_skipped_no_video`가 0인지 — 0이 아니면 다운로드가 덜 된 것
- `trajectories_emitted.truncated / positive ≈ truncated_negative_ratio`
- `segments_dropped`에 `no_template`이 몰려 있으면 템플릿 문제
  → `uv run python scripts/b1k_template_coverage.py $B1K_ROOT`

## 3. 육안 확인 — 건너뛰지 말 것

```bash
uv run python scripts/b1k_preview_segments.py $B1K_ROOT \
  -o datasets/b1k_rbm/preview.png --rows 12 --truncated-ratio 0.5
```

**프레임 오프셋 버그는 숫자 게이트를 전부 통과하면서 라벨 전체를 무의미하게 만든다.** 유일한 방어선이다.

- positive 행: **마지막 프레임에서 해당 subtask가 완료**되어 있어야 한다
  (`pick up ...`이면 그리퍼에 물체가 잡혀 있어야 함)
- truncated 행: 마지막 프레임에서 **아직 완료되지 않았어야** 한다
- 스크립트는 `global_start`가 큰 clip을 우선 뽑는다 (파일 첫 episode는 오프셋 0이라 버그가 안 드러남)

### 3-b. 변환 결과 + 라벨을 함께 보기

위 `b1k_preview_segments.py`는 **원본 비디오에서** 잘라 보여주므로 로더를 검증한다.
아래는 **변환 산출물**(HF dataset의 mp4, 또는 전처리 npz)을 읽어서 **학습이 실제로 계산할
progress/success 라벨과 나란히** 그린다 — 라벨 함수를 그대로 호출한다.

```bash
# 변환된 데이터셋
uv run python scripts/b1k_visualize_dataset.py datasets/b1k_rbm/b1k_skill_train -n 12 -o viz.png

# 네거티브만 (이게 제일 중요하다)
uv run python scripts/b1k_visualize_dataset.py datasets/b1k_rbm/b1k_skill_train \
  --quality failure -n 12 -o viz_neg.png

# 전처리 캐시를 재생 가능한 HTML로 (scp 한 방)
uv run python scripts/b1k_visualize_dataset.py \
  $ROBOMETER_PROCESSED_DATASETS_PATH/datasets_b1k_rbm_b1k_skill_train_b1k_skill_train \
  -n 12 --html viz.html

# 특정 skill만
uv run python scripts/b1k_visualize_dataset.py datasets/b1k_rbm/b1k_skill_train --task "pick up"
```

읽는 법:
- 프레임 테두리와 하단 막대가 **초록 = 그 프레임의 success 라벨이 1**
- `p=` 값이 그 프레임의 progress 타깃
- positive는 `p` 0 → 1.00, 마지막 프레임만 초록
- truncated는 `p`가 전부 0이고 **마지막 프레임만 α**, 초록이 하나도 없어야 한다
- 캡션의 `cutoff=`이 1.0이 아니면 `dataset_success_cutoff.txt` 등록이 안 된 것

> `--progress-pred-type`은 학습 설정의 `data.progress_pred_type`과 같아야 한다 (기본
> `absolute_first_frame`). 다르게 주면 그림과 학습이 달라진다.

## 4. 전처리

```bash
uv run python -m robometer.data.scripts.preprocess_datasets \
  --config robometer/configs/preprocess_b1k.yaml \
  --cache_dir=$ROBOMETER_PROCESSED_DATASETS_PATH
```

clip mp4 → 32프레임 npz + 인덱스 캐시. **trajectory당 약 5.5 MB**이므로 스케일업 전에 디스크를 볼 것.

**확인:** 로그의 `Filtered out N trajectories`에서 N/전체 < 2%.

## 5. 캐시 검증 — GPU 시간 쓰기 전에

```bash
uv run python scripts/b1k_verify_cache.py datasets_b1k_rbm_b1k_skill_train_b1k_skill_train
uv run python scripts/b1k_verify_cache.py datasets_b1k_rbm_b1k_skill_val_b1k_skill_val
```

인자 없이 실행하면 캐시 목록을 출력한다. 검사 항목:
`quality_label`에 successful/failure가 모두 있는지, `data_source`가 정확히 하나인지,
cutoff이 1.0으로 잡히는지, npz가 `(32,240,240,3) uint8`인지, `RBMDataset[0]`이 8프레임 샘플과
progress/success 라벨을 내는지.

## 6. 스모크 학습 (20 스텝)

```bash
uv run python train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true model.train_preference_head=true model.train_success_head=true \
  data.train_datasets=[b1k] data.eval_datasets=[b1k] \
  data.max_frames=8 \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.per_device_train_batch_size=8 \
  training.max_steps=20 training.eval_steps=10 training.custom_eval_steps=10 \
  training.output_dir=./logs training.exp_name=rbm4b_lora_b1k_smoke \
  training.overwrite_output_dir=True \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[b1k] custom_eval.policy_ranking=[b1k] \
  logging.log_to=[]
```

`data.train_datasets=[b1k]`는 `DATASET_MAP`이 캐시 키로 풀어 준다 (`train.py:207`).

**확인:** loss가 NaN이 아니고, custom eval이 예외 없이 돌고,
`eval_rew_align/pearson_b1k_skill_val`가 로그에 찍히는지. 안 찍히면 `name_mapping.py`의 짧은 이름 불일치다.

## 7. 본 학습

```bash
./scripts/b1k_pipeline.sh train          # 기본 1000 스텝
MAX_STEPS=2000 EXP_NAME=my_run ./scripts/b1k_pipeline.sh train
```

멀티 GPU:

```bash
uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=$N_GPUS train.py <위와 동일한 인자>
```

데이터 관련 오버라이드는 [03 §5.2](03-preprocess-train-spec.md#52-데이터-관련-하이퍼파라미터-기본값에서-바꿀-것) 참고
(`data.max_frames=8`, `traj_same_source_prob=0.8`, `dataset_preference_ratio=0.3`).

베이스라인 2개도 같은 val로 반드시 측정할 것 ([03 §5.4](03-preprocess-train-spec.md#54-베이스라인-비교용-필수)):
Robometer-4B zero-shot, base Qwen3-VL + LoRA.

---

## Hub 경유로 갈 경우

로컬 경로 대신 Hub를 쓰면 캐시 키가 짧고 깔끔해진다.

1. 변환 시 `--hub.push_to_hub=true --hub.hub_repo_id=b1k_rbm` (그리고 `export HF_USERNAME=...`)
2. 내려받기:
   ```bash
   export ROBOMETER_DATASET_PATH=/path/holding/datasets
   huggingface-cli download $HF_USERNAME/b1k_rbm --repo-type dataset \
     --local-dir $ROBOMETER_DATASET_PATH/b1k_rbm
   ```
3. `preprocess_b1k.yaml`의 `train_datasets`를 `["<user>/b1k_rbm"]`, `train_subsets`를 `[["b1k_skill_train"]]`로
4. `name_mapping.py`에 `"<user>_b1k_rbm_b1k_skill_train": "b1k_skill"` 추가 (파일에 주석으로 자리 표시해 둠)
5. `dataset_category.py`의 `DATASET_MAP["b1k"]`도 같은 키로 교체

## 스케일업 순서

task-0000이 끝까지 통과한 뒤:

1. `--dataset.b1k.max_episodes_per_task=20`으로 **여러 task** (예: 10개) → 용량/시간 실측
2. 실측치로 전체 예산 계산 (clip당 mp4 ~250 KB, npz ~5.5 MB)
3. 목표 task 집합으로 본 변환

전체 100 task 무제한 변환은 clip 약 36만 개(mp4 ~90 GB, npz ~200 GB)다. 먼저 재보고 결정할 것.

## 트러블슈팅

### `ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'`

`from transformers import ...`를 하는 모든 스크립트(변환 포함)가 죽는다. 스택은 대개
`sentence_transformers → transformers → quantizers/auto.py → quantizer_torchao → import torchao`.

**원인:** `unsloth`이 `torchao`를 상한 없이 요구해서 uv가 최신 **torchao 0.18.0**을 잡는데,
이건 torch ≥ 2.11용이고 이 저장소는 `torch==2.8.0`을 핀한다. 같이 뜨는
`Skipping import of cpp extensions ... Please upgrade to torch >= 2.11.0 (found 2.8.0+cu128)`가 그 증거다.
transformers는 torchao가 **설치돼 있기만 하면** 임포트하므로
(`if is_torchao_available(): import torchao`), 양자화를 안 써도 터진다.
이 프로젝트는 `model.quantization: false`라 torchao가 아예 필요 없다.

**즉시 우회:**

```bash
uv pip uninstall torchao
RUN="uv run --no-sync" ./scripts/b1k_pipeline.sh convert
```

`--no-sync`가 없으면 `uv run`이 락 파일을 보고 torchao를 다시 깔아 놓는다.
(`b1k_pipeline.sh`는 `RUN` 환경변수로 실행기를 갈아끼울 수 있다.)

**영구 수정** — `pyproject.toml`의 `[tool.uv]`에 이미 넣어 뒀다:

```toml
constraint-dependencies = ["torchao<0.14"]
```

```bash
uv lock && uv sync
uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

이 명령이 통과하면 해결. 여전히 같은 에러면 상한을 더 내리거나(`torchao<0.13`),
그냥 torchao를 빼고 `--no-sync`로 가면 된다.

### `ffmpeg: error while loading shared libraries: libvpx.so.9`

`BrokenPipeError writing frame` + `❌ Error processing trajectory N`이 trajectory마다 반복된다.

**원인:** ffmpeg 바이너리의 동적 링크가 깨졌다(conda/apt ffmpeg인데 libvpx가 그 밑에서 바뀐 경우가 흔하다).
변환은 프레임을 ffmpeg stdin으로 밀어 넣어 mp4를 만들므로 **전부 실패**한다.

**진단:**

```bash
which ffmpeg && ffmpeg -version
ldd $(which ffmpeg) | grep "not found"
```

**수정** — 셋 중 하나. root가 없으면 첫 번째가 확실하다 (정적 빌드):

```bash
uv pip install imageio-ffmpeg
export FFMPEG_BINARY=$(uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
```
```bash
conda install -c conda-forge ffmpeg      # conda 환경이면
apt-get install -y ffmpeg                # root가 있으면
```

`FFMPEG_BINARY`는 변환기가 읽는다(`dataset_upload/helpers.py:get_ffmpeg_binary`).
변환 시작 전에 `check_ffmpeg()`가 한 번 검사하므로, 깨져 있으면 1071개를 다 돌기 전에 바로 멈춘다.

### `wandb.errors.CommError: ... 403 permission denied (upsertBucket)`

`config.yaml`의 `wandb_entity`가 논문 저자 엔트리티(`jesbu1`)로 하드코딩돼 있어서, 다른 계정으로는
그 엔트리티에 run을 만들 수 없다. `wandb_entity: null`로 바꿔 뒀다 — null이면 로그인한 계정의
기본 엔트리티(또는 `WANDB_ENTITY` 환경변수)를 쓴다.

```bash
wandb login                      # 로그인 상태 확인
./scripts/b1k_pipeline.sh train  # 기본 엔트리티로 기록
```

팀/조직 계정에 남기려면:

```bash
WANDB_ENTITY=<your-team> ./scripts/b1k_pipeline.sh train
```

로깅을 아예 끄거나 오프라인으로 돌리려면:

```bash
WANDB=off ./scripts/b1k_pipeline.sh train      # logging.log_to=[] 로 실행
WANDB_MODE=offline ./scripts/b1k_pipeline.sh train
```

`smoke` 단계는 원래부터 `logging.log_to=[]`라 wandb를 타지 않는다. 먼저 smoke로 학습 경로를
검증하고 나서 `train`으로 넘어가면 이 문제에 시간을 안 뺏긴다.

### 화면을 뒤덮는 warning들 — 처리 완료

전부 무해하지만 실제 실패 메시지를 묻어 버리므로 출처별로 막아 뒀다.

| 메시지 | 출처 | 조치 |
|---|---|---|
| `Unable to register cuDNN/cuBLAS factory`, `computation placer already registered`, `absl::InitializeLog` | `generate_hf_dataset.py`가 TensorFlow를 무조건 임포트했다. TF의 C++ 레이어가 뿜는 것이라 파이썬 필터로는 못 막는다 | **TF 임포트를 지연시켰다.** RLDS/TFDS 로더(OXE·SOAR·AgiBot)만 필요하고, B1K는 안 쓴다. 강제로 켜려면 `RBM_IMPORT_TF=1` |
| `Using TRANSFORMERS_CACHE is deprecated` | 환경변수가 설정돼 있음 | `HF_HUB_CACHE`로 **값을 옮기고** 옛 이름을 지운다. 캐시 위치가 유지되므로 모델 재다운로드가 없다 |
| `google.api_core ... Python 3.10 end of life` 등 FutureWarning | 전이 의존성 | 패턴 기반 warning 필터 |
| sentence-transformers 다운로드 진행바 | 최초 1회 모델 다운로드 | 캐시된 뒤로는 안 뜬다 |
| worker마다 반복 | spawn 워커가 각자 모듈을 재임포트 | 위 조치가 워커에도 그대로 적용된다 |

구현은 [dataset_upload/quiet.py](../../dataset_upload/quiet.py)이며 `generate_hf_dataset.py` 최상단에서
무거운 임포트보다 먼저 호출된다. `b1k_pipeline.sh`는 CPU만 쓰는 단계(check/convert/preview)를
`CUDA_VISIBLE_DEVICES=`로 실행해 CUDA 등록 에러 자체를 원천 차단한다.

**전부 되돌려 원래 로그를 보고 싶으면:**

```bash
RBM_VERBOSE=1 ./scripts/b1k_pipeline.sh convert
```

> 부수 변경: `TOKENIZERS_PARALLELISM`이 `true`에서 `false`로 바뀌었다. 변환기는 SentenceTransformer를
> 쓴 뒤 프로세스 풀을 띄우므로 `true`면 fork 경고가 워커마다 뜬다. 임베딩은 이미 풀 생성 전에
> 한 번에 계산되므로 성능 영향은 없다.

## 로컬(맥) 개발 참고

`decord`는 macOS arm64 휠이 없다. 두 곳에 폴백을 넣어 뒀다:
- `dataset_upload/dataset_loaders/b1k_video.py` — decord → PyAV → OpenCV
- `robometer/data/scripts/preprocess_datasets.py` — decord 임포트가 optional, 없으면 OpenCV

서버(Linux)에서는 `pyproject.toml`의 `decord>=0.6.0`이 설치되어 원래 경로 그대로 쓴다.
