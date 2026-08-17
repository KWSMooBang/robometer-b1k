# BEHAVIOR-1K → Robometer LoRA 파인튜닝

BEHAVIOR-1K 2026 Challenge 데모로 Robometer를 **subtask 완료 판정기**로 파인튜닝한다.
필요한 것은 episode 단위 task 성공 판정이 아니라 **subtask 단위 완료 판정**이다. 이 문서의 모든 설계가 여기서 나온다.

---

## 0. 목표

### 0.1 FSM에서 Progress Tracker의 역할

```
task → [사전 정의된 subtask sequence]
   ├── VLA(pi05)가 현재 subtask의 action 생성
   ├── Robometer가 매 스텝 현재 subtask의 progress/success 판정
   │     ├── success → 다음 subtask로 transition
   │     └── fail    → feedback 모듈 → recovery subtask
```

### 0.2 전체 흐름 한 장 요약

```
[Conversion]  skill segment ──▶ 최대 64프레임 mp4 (240×240)
                                + 50% 확률로 α∈[0.30,0.70] 절단본(failure)
                                        │
[Preprocess]                     최대 32프레임 npz
                                        │
[Training]     dataset[idx] = seed
                 → 전략 추첨 [REWIND, SUBOPTIMAL, DIFFERENT_TASK, REVERSE_PROGRESS]
                 → A/B 결정 (SUBOPTIMAL이면 partial_success 비교 후 스왑)
                 → A, B 각각: 앵커 3개 → 경로 → 8프레임
                 → preference(A,B) + progress(A) + success(A)
```

| 단계 | 프레임 | 해상도 |
|---|---|---|
| 원본 segment | 평균 515 | 720×720 |
| 변환 clip (mp4) | ≤ 64 | 240×240 |
| 전처리 캐시 (npz) | ≤ 32 | 240×240 |
| **학습 입력** | **8** | 240×240 |

학습 8프레임은 **Robometer-4B의 사전학습 설정**이다. `config.yaml` 기본값은 16이지만 논문의
RBM-1M 학습·평가 커맨드는 전부 `max_frames=8`이고, `use_per_frame_progress_token: true`라
프레임 수가 토큰 시퀀스 구조에 직접 반영된다.

---

## 1. 설계 기준

### 1.1 학습 단위 = subtask(skill) segment

BEHAVIOR-1K 어노테이션이 각 episode를 skill 단위로 정확한 프레임 경계와 함께 쪼개 놓았다.
**segment 하나 = trajectory 하나**로 변환한다.

```
episode 0 "turning_on_radio" (1,956 frames)
  skill 0  [   0,  265)  move to      → "go to the radio"
  skill 1  [ 265, 1162)  pick up from → "pick up the radio from the coffee table"
  skill 2  [1162, 1434)  press        → "press the radio"
  skill 3  [1434, 1776)  place on     → "place the radio back on the coffee table"
```

이렇게 하면 `progress == 1.0`이 **"이 subtask가 방금 끝났다"**가 되어 FSM 전이 신호와 일치한다.

| 항목 | 값 |
|---|---|
| episode / task | 20,000 / 100 |
| **총 skill segment** | **406,341** |
| episode당 skill | 최소 4, 중앙값 18, 최대 74 |
| segment 길이 | 평균 515 프레임(≈17초) |
| skill 종류 / object | 35종 / 573 id → 278 카테고리 |

task-0000은 episode당 4개로 **가장 작은 축**이다. `task-0026`은 68.7개 — 스케일업 시 clip 수가 17배까지 벌어진다.

### 1.2 라벨 정의 (논문)

[Robometer (RSS 2026)](https://arxiv.org/abs/2603.02115):

| 대상 | 라벨 |
|---|---|
| expert progress | `p_t = t / T` (선형 0 → 1) |
| success | `s_t = 0 (t < T)`, `s_t = 1 (t = T)` |
| **실패·준최적 궤적** | **`p = None` — progress 타깃을 부여하지 않는다** |

실패 궤적은 **preference 목적함수로만** 쓰인다. 구현도 이를 따른다 —
`rbm_heads_trainer.py:2476`에서 progress·success 손실은 **`A`(chosen)에만** 걸리고
`progress_loss_B`는 존재하지 않는다. 이 규정이 §3.4 플래그의 근거다.

---

## 2. Conversion

### 2.1 흐름

```
meta/episodes/*.parquet ─┐
annotations/*.json ──────┼─> [b1k_loader] ──> traj_dict (프레임 미독출)
videos/*.mp4 (경로만) ───┘                        │
                                                  ▼
                              [generate_hf_dataset 공용 변환기]
                          언어 임베딩 → 프로세스 풀 → 디코드 → mp4 인코딩
```

```
b1k demo (episode video + skill annotation)
  → annotation의 frame_duration으로 subtask segment 분절
  → subtask language instruction 합성 (skill_description + object_id + prefix)
  → segment에서 최대 64 frame 균등 추출
  → subtask trajectory video 저장 (240×240 mp4)
  → 50% 확률로 α∈[0.30,0.70] 절단본을 failure case로 추가 생성
```

**① episode 인덱스** — `meta/episodes`에서 `video_offset = round(from_timestamp × 30)`을 얻는다.
B1K의 mp4는 46분짜리 집합 파일이라 episode 하나가 파일 중간에 박혀 있다.

**② 선택** — task whitelist / split(`demo_index_within_task % 10 == 0` → val, **episode 단위**) / 개수 상한.
그다음 `(video_path, video_offset)`으로 정렬해 디코더가 앞으로만 seek하게 한다.

**③ 프레임 구간** — 어노테이션은 episode 로컬 좌표다:

```
frame_duration [387, 1313] + video_offset 68042 = 전역 [68429, 69355)
```

**④ 지연 로더** — 프레임을 읽지 않고 *읽는 방법*을 담은 picklable 객체를 만든다.
호출 시에만 64개를 균등 추출한다. 전부 디코드하면 최장 segment가 13GB다.

### 2.2 subtask instruction 합성

annotation의 skill + object를 **규칙 기반**으로 문장화한다.

```
{
  "skill_description": ["pick up from"],
  "object_id": [["radio_89", "coffee_table_koagbh_0"]],
  "memory_prefix": ["back"],
  "spatial_prefix": ["left"]
}
=> "pick up the radio from the left coffee table"
```

구성 요소:

| 요소 | 처리 |
|---|---|
| skill + arity | `(skill_description, arity)` 템플릿 테이블. 같은 skill이 다른 arity로 등장(`pour`는 1~11) |
| object id | 정규식이 아니라 **검수된 룩업 테이블**(573 id → 278 이름) |
| memory_prefix | 3종. `back`(부사), `the other`/`the same`(한정사 치환) |
| spatial_prefix | 35종을 5가지 kind로 렌더 — `det`(the **left** door) / `prep`(**to the edge of** the table) / `part`(the **right door of** the fridge) / `layer`(the **4th layer of** the shelf) / `pose`(… **and face it away**) |

- **렌더 실패 시 추측하지 않고 drop**한다. 잘못된 지시문은 없는 지시문보다 나쁘다.
- object 이름을 정규식으로 자르면 `<cat>_<6자해시>_<idx>`와 `<cat>_<번호>`가 문법적으로 같아서
  `allen_wrench_189`→"allen", `swiss_cheese_77`→"swiss" 처럼 17개가 망가진다. 그래서 룩업 테이블이다.
- 그리드 좌표 prefix(`2X2`, `3X1` …, 약 1,800 segment)는 자연어로 옮기는 게 추측이라 **의도적으로 drop**.

> **prefix는 버리지 않았다.** memory/spatial prefix 모두 현재 구현에서 문장에 반영된다
> (`b1k_skill_templates.py`). 남은 문제는 §5.1의 동종 object 중복 식별.

### 2.3 truncation trajectory — 유일한 네거티브

**BEHAVIOR-1K 데모에는 실패 궤적이 하나도 없다.** FSM이 방지해야 할 핵심 오류는 조기 전이인데,
그 판정 경계는 *"올바른 subtask를 올바르게 수행 중, 다만 아직 안 끝남"* 이다. 온라인 전략
(rewind/reverse/different task)은 전부 "뭔가 잘못된" 클립이라 이 경계를 못 만든다.
⇒ skill segment를 중간에 끊은 truncation trajectory를 생성한다.

```
positive segment [a, b) 를 α ~ U(0.30, 0.70) 지점에서 절단
  quality_label   = "failure"
  partial_success = α
```

`truncated_negative_ratio: 0.5` → positive 2개당 1개.

### 2.4 success cutoff = 1.0

segment 경계가 곧 완료 시점이라 `dataset_success_cutoff.txt`에 `b1k_skill,1.0`을 넣는다.
(논문의 *"10 trajectories per data source to determine the point at which the task actually ends"*가
이 파일의 정체다. episode 단위로 변환한다면 유효 구간 비율이 0.785~1.0으로 흩어져 단일 스칼라로 표현 불가.)

---

## 3. Preprocess & Training

### 3.1 Preprocess

```
subtask trajectory video
  → 최대 32 frame 추출하여 npz 저장   # (32, 240, 240, 3) uint8
  → quality / task / data_source 인덱스 생성
```

### 3.2 샘플링 — A(chosen) / B(rejected) 만들기

`sample_type_ratio=[1,0,0]` + `predict_pref_progress: true` → **모든 샘플이 preference 쌍**이고,
한 번의 forward로 세 헤드가 모두 학습된다.

```
item = dataset[idx]          # seed → 일단 chosen(A)
   ↓ 전략 추첨 [REWIND, SUBOPTIMAL, DIFFERENT_TASK, REVERSE_PROGRESS]  비율 [1,1,1,1]
   ↓ rejected(B) 생성
   ↓ A, B 각각 32프레임에서 창을 뽑아 8프레임
```

| 전략 | B (rejected) | A (chosen) |
|---|---|---|
| `REWIND` / `REVERSE_PROGRESS` | 같은 클립을 되감기/역방향 재샘플 | 그대로 |
| `DIFFERENT_TASK` | **다른 영상** + 원래 지시문 (progress·success를 0으로 덮어씀) | 그대로 |
| `SUBOPTIMAL` | 같은 task의 저품질 클립 | **`partial_success` 비교 후 스왑** |

| 손실 | A (chosen) | B (rejected) |
|---|---|---|
| preference | ✅ | ✅ |
| progress | ✅ | ❌ |
| success | ✅ | ❌ |

### 3.3 8프레임 추출과 progress 기준점

```
32프레임에서 앵커 3개 무작위 → start/middle/end (순서가 전략을 결정)
  → start→middle→end 경로 전체 열거 (25~29개)
  → linspace로 8개 절단
```

**progress의 0점은 창의 첫 프레임**이다. trajectory의 시작이 아니다:

```
전체    (0..31)   progress = 0.00 → 1.00
뒤 절반 (16..31)  progress = 0.00 → 1.00   ← 0.5가 아니라 0에서 시작
```

→ **런타임 계약:** 최근 N프레임 슬라이딩 윈도를 넣으면 기준점이 계속 움직여 **progress가 영원히
0 근처로 리셋된다.** 반드시 **subtask 진입 프레임을 첫 프레임으로 고정**하고 뒤쪽만 늘려야 한다.

### 3.4 반드시 켜야 하는 플래그

```bash
data.predict_last_frame_partial_progress=true
```

truncated 클립이 `partial_success`를 갖는 순간 progress 경로에 편입되는데, 자리를 rejected로 옮기는
것은 `SUBOPTIMAL` 분기뿐이라 **약 75%가 chosen에 남아 progress 손실을 받는다.** 그것도 창에 따라
`[0,…,α]`거나 `[0, 0.14, …, 1.0]`로 오락가락한다. 후자는 "이 실패 클립이 subtask를 100% 완료했다"를
가르치는 것이라 §1.2의 `p=None` 규정 위반이다.
이 플래그는 마스크로 그걸 무력화한다 — 창에 마지막 프레임이 있으면 그 프레임만, 없으면 전부 0.

### 3.5 설정 요약

| 키 | 값 | 근거 |
|---|---|---|
| `data.max_frames` | 8 | Robometer-4B 사전학습 설정 (§0.2) |
| `data.predict_last_frame_partial_progress` | true | 논문의 `p=None` (§3.4) |
| `data.sample_type_ratio` | `[1,0,0]` 유지 | `[0,1,0]`은 successful만 남겨 네거티브를 전부 버린다 |
| `data.preference_strategy_ratio` | `[1,1,1,1]` 유지 | truncated가 학습되는 주 경로가 `SUBOPTIMAL`이다. 0으로 주면 네거티브가 죽는다 |
| `model.use_peft` | true | LoRA로 시작 |
| `training.load_from_checkpoint` | `robometer/Robometer-4B` | |

> **무효 파라미터.** `dataset_preference_ratio`는 `_load_preference_dataset`이 스텁이라 죽은 코드다.
> `traj_same_source_prob`은 data_source가 `b1k_skill` 하나뿐이라 두 분기가 같은 집합을 낸다.
> 둘 다 값을 바꿔도 아무 일도 일어나지 않는다.

### 3.6 베이스라인

같은 val로 3종을 비교한다: **Robometer-4B zero-shot** / **base Qwen3-VL + LoRA** / **Robometer-4B + LoRA**.

### 3.7 해상도 변형 (240 / 480 / 720)

해상도는 conversion부터 training까지 관통하는 축이고, `RES` 하나로 전 단계가 따라간다.

```bash
RES=480 ./scripts/b1k_pipeline.sh all
RES=480 ./scripts/b1k_pipeline.sh smoke
```

| 단계 | 240 (기본) | 480 |
|---|---|---|
| 변환 출력 | `datasets/b1k_rbm` | `datasets/b1k_rbm_480` |
| npz 캐시 키 | `datasets_b1k_rbm_b1k_skill_train_…` | `datasets_b1k_rbm_480_b1k_skill_train_…` |
| `DATASET_MAP` | `b1k` | `b1k_480` |
| 학습 인자 | `data.train_datasets=[b1k]` | `data.train_datasets=[b1k_480]` |
| 기본 배치 | 8 | 2 |

이름은 전부 [b1k_variants.py](robometer/data/b1k_variants.py)에서 파생된다. 240은 접미사 없는
기존 이름을 그대로 유지하므로 이미 만들어둔 캐시는 계속 유효하다.
`./scripts/b1k_pipeline.sh names`로 해당 해상도가 어떤 이름들로 풀리는지 확인할 수 있다.

**해상도별로 디렉터리를 분리하는 이유:** `create_trajectory_video_optimized()`는 mp4가 이미 있으면
건너뛴다. 같은 `output_dir`에 다시 변환하면 **이전 해상도 클립이 그대로 남는다.**
`b1k_check_clip_resolution.py`가 변환 직후 실제 mp4 크기를 확인하고, 크기가 섞여 있으면 실패시킨다.

**`data.max_image_side`를 반드시 같이 올려야 한다.** collator는 `resized_height/width`가 없으면
프레임을 이 값으로 잘라내는데, 기본값이 480이라 720 캐시를 그대로 쓰면 480으로 축소된 채 학습된다.
파이프라인이 `RES`에서 자동으로 맞춰주고, 축소가 실제로 일어나면 경고를 남긴다.

**패치 정렬.** Qwen3-VL은 프레임을 32의 배수로 반올림한다. 480은 정확히 들어가지만 720은
704로 반올림되므로, 저장만 하고 모델은 못 보는 픽셀이 생긴다. 720급을 쓸 거면 **`RES=704`**가
낭비가 없다 (파이프라인이 경고한다). 240도 256으로 반올림되지만 이건 사전학습 때와 같은 조건이다.

**비용.** npz는 해상도 제곱에 비례한다 — 240 기준 480은 4배, 720은 9배 (§4의 3.3TB → 13TB / 30TB).
비주얼 토큰도 프레임당 대략 64 → 225 → 484로 늘어서 배치를 줄여야 한다. §3.5의 비교 실험을
돌리기 전에 §5.2의 판단 근거부터 확보하는 게 순서다.

---

## 4. 검증 현황 (task-0000, 200 episodes)

| 항목 | 결과 |
|---|---|
| 지시문 렌더율 | 99.49% (406,341 중 404,273). 실패 0.51%는 전부 의도적 드롭 |
| 프레임 정렬 | 육안 확인 통과 (파일 끝부분 오프셋 58k~81k clip) |
| 변환 | 720 segment 전부 emit, 드롭 0, 카운터 보존 assert 통과 |
| train/val | 180/20 episode, 교집합 0 |
| 전처리 | npz `(32, 240, 240, 3) uint8` |
| 통합 | `RBMDataset`이 8프레임 샘플 생성, 온라인 전략 동작 |
| 테스트 | 60개 통과 |

**스케일업 예산:** 전체 100 task 무제한 시 약 61만 clip → mp4 ~150GB, **npz ~3.3TB**.
npz가 병목이므로 task 부분집합 + `max_segments_per_task`로 제한해야 한다.
task-0000 관통 후 10개 task × 20 episode로 실측하고 재계산할 것.

---

## 5. 열린 질문 / 다음 단계

### 5.1 데이터

- **동종 object 중복.** id가 다른 같은 종류의 object가 둘 이상일 때(`shoe_1`, `shoe_2`) 룩업 테이블이
  둘 다 "the shoe"로 렌더한다. spatial_prefix가 있는 경우에만 구분되고, 없으면 지시문이 모호해진다.
  빈도 측정 → 필요시 서수/위치 기반 disambiguation 규칙 추가.
- **진짜 실패가 없다.** truncated는 "아직 안 끝남"만 가르친다. 물체를 놓치거나 충돌하는 실패는 데이터에
  없다. 2단계에서 FSM 롤아웃을 시뮬레이터 술어로 자동 라벨링해 수집하는 것이 **가장 큰 개선 여지**.

### 5.2 학습

- `DIFFERENT_TASK_INSTRUCTION`(**같은 영상** + 다른 지시문)은 ProgressSampler 소속이라
  `sample_type_ratio=[1,0,0]`에서 쓰이지 않는다. FSM 조기 전이를 가장 직접 겨냥하는 전략이므로
  활성화 경로를 검토할 것.
- **해상도가 병목인지 미확인.** 480/720 경로는 §3.7로 열어뒀지만, 240 학습 결과가 아직 없어서
  오차의 원인이 픽셀인지 라벨인지 데이터인지 모른다. 240을 먼저 끝까지 돌리고, false positive
  전이 프레임을 240에서 사람이 판정할 수 있는지 눈으로 확인한 뒤에 해상도를 올릴 것.
  사람도 못 하면 해상도 문제, 사람은 쉽게 하는데 모델이 틀리면 데이터·라벨 문제다.

---

# 부록 A. 코드 사용법

## A.1 환경

```bash
cd /path/to/robometer-b1k
uv sync
export B1K_ROOT=/path/to/behavior-1k/2026-challenge-demos
export ROBOMETER_PROCESSED_DATASETS_PATH=$PWD/processed_datasets
```

> **repo 루트에서 실행할 것.** 전처리 캐시 디렉터리 이름이 `preprocess_b1k.yaml`의 `dataset_path`
> **문자열 그대로**에서 만들어지고(`/`→`_`), 그 이름이 `DATASET_MAP`·`name_mapping` 등록값과
> 일치해야 한다. 절대 경로를 쓰면 `_workspace_home_...` 같은 키가 되어 학습이 캐시를 못 찾는다.
> 변환 결과가 repo 밖에 있다면 심볼릭 링크로 해결한다:
> `ln -sfn /path/to/b1k_rbm datasets/b1k_rbm`

## A.2 단계별 실행

```bash
./scripts/b1k_pipeline.sh check       # 변환 가능한 task 목록 + 불변식 검사
./scripts/b1k_pipeline.sh convert     # 어노테이션 → clip + HF dataset
./scripts/b1k_pipeline.sh preview     # contact sheet ← 반드시 눈으로 볼 것
./scripts/b1k_pipeline.sh preprocess  # clip → npz 캐시
./scripts/b1k_pipeline.sh verify      # GPU 시간 쓰기 전 캐시 점검
./scripts/b1k_pipeline.sh wandb       # wandb 엔트리티 확인 (5초)
./scripts/b1k_pipeline.sh smoke       # 20스텝 학습
./scripts/b1k_pipeline.sh train       # 본 학습
./scripts/b1k_pipeline.sh names       # 이 해상도가 풀리는 이름들 (즉시)
```

`all`은 check~verify를 이어서 돌리고 **학습 직전에 멈춘다**(preview를 사람이 봐야 하므로).

옵션: `RES=480` / `WANDB=off` / `WANDB_ENTITY=<team>` / `MAX_STEPS=2000` / `BATCH_SIZE=4` /
`SAVE_STEPS=100` / `RESUME=auto` / `EXP_NAME=...` / `RBM_VERBOSE=1`

### A.2.1 학습이 끊겼을 때 이어서 하기

```bash
./scripts/b1k_pipeline.sh checkpoints        # 뭘로 재개할 수 있는지 확인
RESUME=auto ./scripts/b1k_pipeline.sh train  # 가장 최근 체크포인트에서 재개
RESUME=./logs/<exp>/checkpoint-400 ./scripts/b1k_pipeline.sh train
```

재개하면 **step 카운터 + LR 스케줄 + optimizer 상태 + 데이터셋 샘플링 난수**가 복원된다.
마지막 항목이 이 리포에서 특히 중요한데, rewind/suboptimal 전략 추첨이 거기 걸려 있다
([train.py:319](train.py:319)).

**체크포인트 두 종류를 구분할 것:**

| | 내용 | 재개 |
|---|---|---|
| `checkpoint-<step>/` (HF Trainer) | 가중치 + `optimizer.pt` + `scheduler.pt` | 완전 재개 |
| `ckpt-latest-*` / `ckpt-best-*` (SaveBestCallback) | 가중치 + `trainer_state.json` | step·데이터 순서만, **optimizer는 초기화** |

`RESUME=auto`는 앞쪽을 우선 고르고, 없으면 뒤쪽으로 폴백하면서 어느 쪽인지 알려준다.
배포용 가중치는 `ckpt-best-*`를 쓰고, `checkpoint-*`는 재개 전용으로 볼 것.

**기본값이 바뀌었다.** `config.yaml`은 `save_strategy: "no"`라 중간 체크포인트가 아예 안 생겼다.
파이프라인이 이제 `SAVE_STEPS`(기본 100)로 `save_strategy=steps`를 켠다. `save_total_limit=2`가
[setup_utils.py:1309](robometer/utils/setup_utils.py:1309)에 하드코딩돼 있어 디스크는 안 터진다.
끄려면 `SAVE_STEPS=0`.

`SAVE_STEPS`는 eval 주기(train 50 / smoke 10)의 배수여야 `ckpt-latest-*`도 같이 생긴다 —
`SaveBestCallback`이 `on_evaluate`에서만 저장하기 때문. 어긋나면 스크립트가 경고한다.

> **`overwrite_output_dir` 함정 (수정됨).** `train.py`는 `output_dir`이 이미 있으면
> 지우거나(`True`) 에러를 냈다(`False`). 체크포인트는 바로 그 디렉터리 안에 있으므로
> **재개 대상을 읽기 전에 지워버리는** 구조였다. 이제 `resume_from_checkpoint`이 설정돼 있으면
> 디렉터리를 보존한다. `wandb_info.json`도 살아남아서 같은 wandb 런으로 이어진다.

`RES`는 모든 단계에 적용된다 (§3.7). 단계마다 같은 값을 줘야 한다 —
`RES=480 ./scripts/b1k_pipeline.sh convert` 후 `RES` 없이 `preprocess`를 돌리면 240 캐시를 찾는다.

## A.3 task 범위 조절

```bash
--dataset.b1k.task_whitelist="['turning_on_radio','picking_up_trash']"
--dataset.b1k.task_whitelist=null          # 전체
--dataset.b1k.max_episodes_per_task=20
--dataset.b1k.max_segments_per_task=500
--dataset.b1k.truncated_alpha_max=0.6
```

스케일업 시에는 `max_episodes_per_task`보다 **`max_segments_per_task`가 정확한 손잡이**다.
episode 수로 제한하면 task마다 clip 수가 17배까지 차이 나 데이터가 쏠린다.

## A.4 육안 확인 — 가장 중요한 게이트

**프레임 오프셋 버그는 모든 숫자 검사를 통과하면서 라벨 전체를 무의미하게 만든다.** 유일한 방어선이다.

```bash
./scripts/b1k_pipeline.sh preview                                   # 원본에서 자른 clip (로더 검증)
uv run python scripts/b1k_visualize_dataset.py datasets/b1k_rbm/b1k_skill_train -n 12 -o viz.png
uv run python scripts/b1k_visualize_dataset.py datasets/b1k_rbm/b1k_skill_train --quality failure -o neg.png
uv run python scripts/b1k_visualize_dataset.py <cache_dir> --html viz.html   # 재생 가능
```

- positive 행: **마지막 프레임에서 subtask 완료** (pick up이면 그리퍼에 물체가 있어야)
- truncated 행: 마지막 프레임에서 **아직 미완료**
- `visualize_dataset`은 학습 라벨을 실제 함수로 계산해 함께 그린다 — 초록 테두리 = success 1

## A.5 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `ImportError: ScalingType` | torchao 0.18이 torch 2.11용. `pyproject.toml`에 `torchao<0.14` 제약 추가함 |
| `ffmpeg: libvpx.so.9 not found` | `uv pip install imageio-ffmpeg` 후 `export FFMPEG_BINARY=$(...get_ffmpeg_exe())` |
| 전처리가 빈 캐시 생성 | 로컬 경로 분기가 `frames_video`를 안 채우던 버그. 수정 완료 |
| `No configured datasets are available in the cache` | 캐시 키 불일치 → §A.1 참고. `b1k_verify_cache.py`를 인자 없이 실행하면 실제 목록 출력 |
| wandb 403 / 400 / 404 | `./scripts/b1k_pipeline.sh wandb`로 엔트리티 확인. 급하면 `WANDB=off` |
| cuDNN/cuBLAS 에러 도배 | 불필요한 TF 임포트. 지연 임포트로 해결(`RBM_IMPORT_TF=1`로 강제 가능) |

## A.6 파일

**신규**

```
dataset_upload/dataset_loaders/  b1k_loader.py  b1k_video.py
                                 b1k_skill_templates.py  b1k_object_names.py
dataset_upload/quiet.py
dataset_upload/configs/data_gen_configs/  b1k_skill.yaml  b1k_skill_val.yaml
robometer/configs/preprocess_b1k.yaml
robometer/data/b1k_variants.py                  해상도별 이름의 단일 진실 소스
scripts/  b1k_pipeline.sh  b1k_check_alignment.py  b1k_preview_segments.py
          b1k_visualize_dataset.py  b1k_verify_cache.py
          b1k_check_clip_resolution.py
          b1k_build_object_names.py  b1k_template_coverage.py
tests/  test_b1k_templates.py  test_b1k_loader.py  test_quiet.py
        test_b1k_variants.py
```

**수정**

```
dataset_upload/generate_hf_dataset.py           b1k 분기, ffmpeg 사전 점검, TF 지연 임포트
dataset_upload/helpers.py                       FFMPEG_BINARY 지원
robometer/data/scripts/preprocess_datasets.py   로컬 경로 수정, decord optional
robometer/data/dataset_success_cutoff.txt       b1k_skill,1.0
robometer/data/datasets/name_mapping.py         짧은 이름 (해상도별 자동 등록)
robometer/data/dataset_category.py              DATASET_MAP b1k / b1k_480 / b1k_720
robometer/data/collators/rbm_heads.py           multi-image 해상도 제어 + 축소 경고
robometer/data/collators/base.py                max_image_side / max_image_pixels
robometer/configs/experiment_configs.py         위 두 키를 DataConfig에 추가
robometer/utils/setup_utils.py                  collator로 전달
robometer/configs/config.yaml                   wandb_entity: null, 이미지 상한
pyproject.toml                                  torchao 제약
```
