# 01. 원본 데이터셋 명세 (BEHAVIOR-1K 2026 Challenge demos)

> 본 문서의 모든 수치는 로컬 사본
> `/Users/woosung-kim/workspace/Physical-AI/datasets/behavior-1k/2026-challenge-demos` 를
> 실제로 파싱해서 얻은 값이다 (2026-08-11 기준). 상대 경로는 모두 이 디렉터리 기준.

## 1. 최상위 레이아웃

```
2026-challenge-demos/
├── meta/
│   ├── info.json                          # LeRobot v3.0 데이터셋 메타
│   ├── tasks.parquet                      # task_index → task (100개)
│   ├── stats.json
│   └── episodes/chunk-000/file-000.parquet   # episode 인덱스 (핵심)
├── data/chunk-000/file-{000,001}.parquet  # per-frame 저차원 데이터 (state/action)
├── videos/{video_key}/chunk-000/file-NNN.mp4 # 여러 episode가 이어붙은 집합 mp4
└── annotations/
    ├── skill_summary.csv, skill_type_summary.csv
    └── task-NNNN/episode_XXXXXXXX.json     # skill 분절 어노테이션 (핵심)
```

`meta/info.json` 요약:

| 필드 | 값 |
|------|-----|
| codebase_version | `v3.0` (LeRobot v3) |
| fps | 30 |
| total_episodes / total_tasks | 20000 / 100 |
| total_frames | 210,916,774 |
| robot_type | `R1Pro` |
| data_path | `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |
| video_path | `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4` |

## 2. 관측 스트림

### 2.1 features (info.json)

| key | dtype | shape |
|-----|-------|-------|
| `action` | float32 | [23] |
| `observation.state` | float32 | [61] |
| `next.reward` / `next.terminated` / `next.truncated` | float32/bool | [1] |
| `observation.rgb.zed_link_camera_0` | video | [720, 720, 3] |
| `observation.rgb.{left,right}_realsense_link_camera_0` | video | [480, 480, 3] |
| `observation.depth_linear.*` | video | [·, ·, 1] |
| `observation.robot2cam_pose.*` | float32 | [7] |

Robometer는 RGB 프레임 + 언어 지시문만 먹으므로 **`observation.rgb.zed_link_camera_0`(헤드캠) 하나만** 쓴다.
wrist(realsense)는 subtask 완료 판정에 유용할 수 있으나 별도 data_source로 2단계에서 추가한다 (전신 맥락이
navigation skill 판정에 필수라 head cam이 1순위).

### 2.2 비디오

집합(aggregated) mp4다. 한 파일에 여러 episode가 시간축으로 이어 붙어 있다:

```
videos/observation.rgb.zed_link_camera_0/chunk-000/file-000.mp4
  → 720x720, 30/1 fps, duration 2761.5s, nb_frames 82845, ~206 MB
```

episode의 위치는 `meta/episodes/...parquet`의
`videos/{video_key}/{chunk_index,file_index,from_timestamp,to_timestamp}`로 찾는다.

**검증 완료:** 로컬 200 episode 전부에서 `(to_timestamp − from_timestamp) × 30 == length` (오차 <1.5 프레임).
따라서 파일 내 전역 프레임 인덱스는

```
global_frame = round(from_timestamp * fps) + episode_local_frame
```

로 계산해도 안전하다.

### 2.3 data parquet

`data/chunk-000/file-000.parquet`: 229,565 row, 1 row group, 컬럼 =
`action, next.reward, next.terminated, next.truncated, observation.state, observation.robot2cam_pose.*,
timestamp, frame_index, episode_index, index, task_index`.

**변환에는 쓰지 않는다.** RBM trajectory 스키마에 action이 저장되지 않기 때문
(`dataset_upload/helpers.py:create_hf_trajectory`가 `actions` 키를 무시한다). episode 경계는 전부
`meta/episodes`에서 얻는다.

## 3. episode 인덱스와 annotation

### 3.1 meta/episodes 스키마

로컬 200 row. 관련 컬럼:

| 컬럼 | 예시 | 용도 |
|------|------|------|
| `episode_index` | 0 | 전역 episode id |
| `tasks` | `['turning_on_radio']` | task 이름 |
| `length` | 1956 | episode 프레임 수 |
| `data/{chunk,file}_index`, `dataset_{from,to}_index` | 0,0,0,1956 | parquet 위치 |
| `videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}` | 0,0,0.0,65.2 | **비디오 슬라이스 위치** |
| `task_index`, `demo_index_within_task` | 0, 0 | split 나눌 때 사용 |
| `raw_episode_id` | 10 | annotation 파일명과 일치 |
| `annotation_path` | `annotations/task-0000/episode_00000010.json` | **annotation 직결 링크** |

**검증 완료:** 200 row 전부에서 `basename(annotation_path) == f"episode_{raw_episode_id:08d}.json"`.
즉 annotation은 `annotation_path`를 그대로 열면 되고, 파일명 규칙을 다시 짤 필요 없다.

### 3.2 annotation JSON 스키마

```jsonc
{
  "task_name": "turning on radio",
  "data_folder": "",
  "meta_data": { "task_duration": 1776, "valid_duration": [0, 1776] },
  "skill_annotation": [
    {
      "skill_idx": 0,
      "skill_id": [1],
      "skill_description": ["move to"],
      "object_id": [["radio_89"]],
      "manipulating_object_id": [],
      "memory_prefix": [],            // 예: ["back"]
      "spatial_prefix": [],
      "frame_duration": [0, 265],     // [start, end) — episode-local 프레임 인덱스
      "mp_ef": [],
      "skill_type": ["navigation"]    // navigation | uncoordinated | coordinated
    }
    // ...
  ],
  "primitive_annotation": [           // skill들을 묶은 상위 단위. skill_idxes로 역참조
    { "primitive_idx": 0, "primitive_description": ["pick up from"],
      "frame_duration": [0, 1162], "skill_idxes": [0, 1], ... }
  ]
}
```

두 층위가 있다:
- `skill_annotation` — 세분 단위 (move to / pick up from / press / place on ...)
- `primitive_annotation` — 여러 skill을 묶은 단위 (navigation을 pick에 흡수하는 식)

**본 하네스는 `skill_annotation`을 기본 단위로 쓴다.** FSM의 subtask 입도가 여기에 가깝고,
"move to X"의 완료 판정(= 대상 앞에 도달)이 별도 전이 조건으로 필요하기 때문이다.
`primitive_annotation` 기반 변환은 컨피그 플래그(`annotation_level: skill|primitive`)로 열어 두되
1단계 기본값은 `skill`.

### 3.3 실측 통계 (annotation 17,998개 전수 파싱)

| 항목 | 값 |
|------|-----|
| annotation 파일 수 | 17,998 (task 디렉터리 91개) |
| 총 skill segment 수 | 369,306 (파싱 성공분) |
| episode당 skill 수 | 4 ~ 74, 최빈 10~21 |
| segment 길이 | 평균 515 프레임(≈17.2s), 최대 8,946, **최소 −42(불량)** |
| 15프레임 미만 segment | 142개 |
| skill_type 분포 | uncoordinated 207,553 / navigation 127,826 / coordinated 33,926 / None 1 |
| 상위 skill | move to 130,908 · pick up from 83,474 · place in 41,932 · place on 29,117 · push to 12,056 · open door 11,809 · close door 9,524 · place on next to 8,757 · chop 7,999 · sweep surface 4,803 |

`annotations/skill_summary.csv`(skill별 n/mean_s/min_s/max_s/n_tasks_present)와
`skill_type_summary.csv`도 그대로 제공되므로 커버리지 리포트에 재사용할 것.

### 3.4 valid_duration 과 trailing idle

- `valid_duration[0] != 0` 인 episode: 3,946개 → **앞쪽 무효 구간이 존재한다.**
- 로컬 200 episode에서 `valid_duration[1] / length`: 최소 0.785, 중앙값 0.960, 최대 1.0
  → episode 끝에 평균 4%의 유휴 프레임이 붙어 있고 편차가 크다.
- `task_duration`이 마지막 skill의 end와 항상 같지는 않다 (예: length 3227 / task_duration 2529 /
  마지막 skill end 3105).

**함의:** Robometer의 `dataset_success_cutoff.txt`는 **data_source당 스칼라 하나**만 지원한다
(`robometer/data/samplers/base.py:632`). episode마다 유효 비율이 다른 이 데이터셋에는 부족하다.
→ episode 단위 trajectory를 만들 거면 **변환 시점에 `valid_duration`으로 잘라서** cutoff을 1.0으로 만들어라.
skill segment 단위로 자르면 이 문제 자체가 사라진다 (경계가 곧 완료 시점).

**검증 완료:** 로컬 200 episode에서 모든 skill의 end ≤ `length`. 즉 annotation 프레임 인덱스는
episode-local 프레임 공간이며 별도 스케일 변환이 필요 없다. (episode `length`가 `task_duration`보다
중앙값 4.6% 큰 것은 스케일 차이가 아니라 뒤쪽 유휴 프레임 때문이다.)

### 3.5 반드시 처리해야 하는 예외

| 예외 | 규모 | 처리 |
|------|------|------|
| `frame_duration`이 중첩 리스트 (`[[2795,3007],[3209,3638]]`) — 분절된 skill | 724 segment | 각 구간을 별도 clip으로 만들거나 skip. 기본: **첫 구간만 사용하고 metadata에 `split_range: true` 기록** |
| 인접 segment 경계 불일치 | 전체 17,998파일 중 30,603건. 4,000파일 샘플 기준 exact 58,819 / gap 9,375 / overlap 208 | gap은 정상(전이 사이 유휴). 경계는 자기 segment의 `frame_duration`만 신뢰 |
| 길이 ≤ 0 인 segment (최소 −42) | 소수 | **무조건 drop** + 카운터 기록 |
| 15프레임 미만 segment | 142 | `min_segment_frames`(기본 15)로 drop |
| `skill_description`/`skill_type`에 `None` | 1건 확인 | drop + 로그 |
| `object_id`가 빈 리스트 | 존재 | 템플릿 폴백 (02 §4.3) |

## 4. object id 명명 규칙

`object_id` 원소는 다음 두 형태다:

```
radio_89                    # <카테고리>_<숫자 인스턴스 id>
coffee_table_koagbh_0       # <카테고리>_<6자 모델해시>_<인스턴스 idx>
```

자연어 지시문을 만들려면 정규화가 필요하다 → [02 §4.2](02-converter-spec.md#42-object-id-정규화).

## 5. 로컬 동기화 현황 (S0에서 해결)

| 자원 | 로컬 | 전체 |
|------|------|------|
| annotation | 17,998개 / 91 task 디렉터리 (224 MB) | 20,000 / 100 |
| `meta/episodes` | **200 row (task-0000만)** | 20,000 |
| data parquet | 2 파일 (148 MB) | — |
| zed RGB 비디오 | 6 파일 (~1.0 GB) | — |
| 전체 | 5.1 GB | 수 TB 규모 |

**`meta/episodes`가 200 row뿐이라 지금 상태로는 task-0000 이외의 episode를 비디오에 매핑할 수 없다.**
S0에서 최소한 다음을 받아야 한다:

```bash
# meta 전체 (작음) — 반드시 먼저
huggingface-cli download <b1k-challenge-repo> --repo-type dataset \
  --include "meta/**" --local-dir $B1K_ROOT

# 필요한 뷰의 비디오만 (depth/realsense 제외 → 용량 1/3)
huggingface-cli download <b1k-challenge-repo> --repo-type dataset \
  --include "videos/observation.rgb.zed_link_camera_0/**" --local-dir $B1K_ROOT
```

> 실제 repo id는 챌린지 페이지에서 확인해 채울 것. 다운로드 총량이 감당 안 되면
> `meta/episodes`에서 task별 `demo_index_within_task < N`인 episode만 골라 그 episode들이 참조하는
> `videos/.../file-NNN.mp4`만 받는 방식으로 서브셋을 정한다 (`scripts/b1k_check_alignment.py`가 필요 파일 목록을 출력).

## 6. 변환기가 신뢰해도 되는 불변식

구현 시 assert로 박아 둘 것 (검증 완료 항목):

1. `annotation_path`는 데이터셋 루트 기준 상대 경로이며 존재한다.
2. `(to_timestamp − from_timestamp) * fps ≈ length` (오차 < 1.5 프레임).
3. 모든 skill의 `frame_duration` end ≤ episode `length`.
4. `skill_annotation`은 `skill_idx` 오름차순이며 시간순이다.
5. `skill_id`는 원소 1개 (다중 id segment 0건).
