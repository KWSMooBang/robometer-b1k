# 02. 변환기(converter) 구현 명세

대상: `dataset_upload/dataset_loaders/b1k_loader.py`, `b1k_skill_templates.py`,
`dataset_upload/configs/data_gen_configs/b1k_skill.yaml`, `generate_hf_dataset.py` 분기.

## 1. 기존 파이프라인이 요구하는 계약

변환기는 아래 계약만 만족하면 나머지는 기존 코드가 처리한다.

```
load_b1k_dataset(...)  ->  dict[task_key, list[traj_dict]]
        │
        ├─ flatten_task_data()                       # dataset_upload/helpers.py
        ├─ convert_dataset_to_hf_format()            # generate_hf_dataset.py:199
        │     ├─ SentenceTransformer로 task 문장 임베딩 (unique 캐시)
        │     ├─ Pool(spawn) 병렬 → process_single_trajectory
        │     └─ create_hf_trajectory()              # dataset_upload/helpers.py:329
        │           └─ create_trajectory_video_optimized()  # frames가 callable이면 호출
        └─ HF Dataset + mp4 파일 트리
```

### 1.1 traj_dict 스키마 (loader가 만들어야 하는 것)

| 키 | 타입 | 필수 | 비고 |
|----|------|:---:|------|
| `frames` | callable → `np.ndarray (T,H,W,3) uint8` | ✔ | **picklable해야 함** (spawn 병렬). `LiberoFrameLoader` 패턴 참고 |
| `task` | str | ✔ | subtask 지시문 (§4) |
| `id` | str | ✔ | `generate_unique_id()` 또는 결정적 id (§5.4 권장) |
| `is_robot` | bool | ✔ | 항상 `True` |
| `quality_label` | str | ✔ | `"successful"` \| `"failure"` |
| `partial_success` | float | ✔ | 성공 1.0, truncated 네거티브는 α (§6) |
| `data_source` | str | ✔ | `"b1k_skill"` 고정 (§5.5 이유) |
| `preference_group_id` / `preference_rank` | None | – | 현재 BASE_FEATURES에서 주석 처리됨. `None`으로 둘 것 |
| `actions` | – | ✘ | `create_hf_trajectory`가 무시한다. 넣지 말 것 |

`create_hf_trajectory`가 반환하는 최종 row: `id, task, lang_vector, data_source, frames(상대 mp4 경로),
is_robot, quality_label, preference_group_id, preference_rank, partial_success`.

### 1.2 반드시 지킬 제약

- **frames 로더는 서브샘플된 프레임만 반환한다.** segment 최대 길이가 8,946 프레임이고 720×720×3이므로
  전체 디코드 시 ~13 GB. 로더 내부에서 `max_frames`(기본 64)로 균등 서브샘플한 뒤 반환할 것.
  이후 `downsample_frames`는 no-op이 된다.
- **로더는 `decord.VideoReader`를 `__call__` 안에서 열고 닫는다.** 인스턴스에 reader를 들고 있으면
  pickle 실패 + FD 고갈.
- 병렬 워커는 `spawn`이다. 로더 객체 필드는 str/int/float만.

## 2. 변환 단위

### 2.1 기본: skill segment 1개 = trajectory 1개

```
episode E, skill s_i with frame_duration [a, b)
  → clip  = E의 프레임 [a, b) 를 균등 서브샘플한 T프레임
  → task  = template(s_i)                       # "pick up the radio from the coffee table"
  → progress: 0 → 1.0 (프레임 인덱스 선형), 마지막 프레임 success=1
```

이렇게 하면 `progress == 1.0`의 의미가 **"이 subtask가 방금 끝났다"**가 되어 FSM 전이 신호와 정확히 일치한다.

Robometer의 progress/success 계산은 `robometer/data/samplers/base.py:632`와
`robometer/data/datasets/helpers.py:compute_progress_from_segment / compute_success_labels`에서 일어난다:

- `success_cutoff = dataset_success_cutoff_map.get(data_source, config.max_success)` → `b1k_skill = 1.0`
- 기본 `progress_pred_type = "absolute_first_frame"` → segment 내 위치 비율
- `compute_success_labels`: `quality_label`이 failure/suboptimal이면 **전 프레임 0.0**,
  아니면 `progress >= threshold`인 프레임만 1.0

### 2.2 옵션: episode 단위 trajectory (`b1k_task`)

전체 task 성공 판정용. 만들 거면 `[valid_duration[0], valid_duration[1])`로 잘라서 cutoff을 1.0으로
만들 것 ([01 §3.4](01-dataset-spec.md#34-valid_duration-과-trailing-idle)). 1단계 범위 밖 — 컨피그
`emit_episode_level: false` 기본값으로 두고 코드 경로만 열어 둔다.

## 3. `b1k_loader.py` 구조 — **구현 완료**

실제 시그니처는 아래와 같다 (본 절의 스케치와 다른 부분은 실제 코드가 기준):

```python
load_b1k_dataset(
    dataset_path, video_key=DEFAULT_VIDEO_KEY, annotation_level="skill",
    max_frames=64, min_segment_frames=15, max_segment_frames=3000,
    split="train", val_demo_ratio=0.1,
    task_whitelist=None,            # task 이름 / "task-0007" / 인덱스 혼용 가능
    max_episodes_per_task=None, max_segments_per_task=None,
    truncated_negative_ratio=0.0,   # S4에서 켠다
    skip_split_range=False, data_source="b1k_skill",
    include_task_context=False, fps=30.0, seed=42,
    report_path=None, verbose=True,
) -> dict[task_key, list[traj_dict]]
```

부수 API:
- `summarize_availability(root, video_key)` — task별 변환 가능 episode 수 (부분 다운로드 대응)
- `load_episode_index(root, video_key, fps)` — `meta/episodes` → `EpisodeRef` 목록
- `b1k_video.B1KSegmentFrameLoader` — picklable 프레임 로더
- `b1k_video.read_frames` — decord → PyAV → OpenCV 폴백 (decord는 macOS arm64 휠 없음)

### 3.0 task 일부만 변환하기

```python
# 이름으로
load_b1k_dataset(root, task_whitelist=["turning_on_radio", "picking_up_trash"])
# 폴더 id 또는 인덱스로
load_b1k_dataset(root, task_whitelist=["task-0007", 12])
# task당 상한
load_b1k_dataset(root, task_whitelist=[0], max_episodes_per_task=20, max_segments_per_task=500)
```

무엇을 고를 수 있는지는 먼저 확인한다:

```bash
python scripts/b1k_check_alignment.py $B1K_ROOT --ready-only
```

비디오나 annotation이 없는 episode는 **예외를 던지지 않고 건너뛰며 카운트**된다. 부분 다운로드가
정상 상태이기 때문이다.



```python
#!/usr/bin/env python3
"""BEHAVIOR-1K 2026 challenge demos → Robometer(RBM) trajectory loader."""

from dataclasses import dataclass

FPS = 30
DEFAULT_VIDEO_KEY = "observation.rgb.zed_link_camera_0"


class B1KSegmentFrameLoader:
    """picklable, on-demand frame loader for one skill segment.

    집합 mp4에서 [global_start, global_end) 구간을 균등 서브샘플해 반환한다.
    """

    def __init__(self, video_path: str, global_start: int, global_end: int,
                 max_frames: int = 64):
        self.video_path = video_path
        self.global_start = int(global_start)
        self.global_end = int(global_end)
        self.max_frames = int(max_frames)

    def __call__(self):
        import decord, numpy as np
        vr = decord.VideoReader(self.video_path, num_threads=1)
        n = len(vr)
        start = max(0, min(self.global_start, n - 1))
        end = max(start + 1, min(self.global_end, n))
        span = end - start
        if span <= self.max_frames:
            idx = list(range(start, end))
        else:
            idx = [start + int(i * span / self.max_frames) for i in range(self.max_frames)]
        frames = vr.get_batch(idx).asnumpy()      # (T, H, W, 3) uint8
        del vr
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8, copy=False)
        return frames


@dataclass
class SegmentRecord:
    episode_index: int
    task_index: int
    task_name: str            # 'turning_on_radio'
    skill_idx: int
    instruction: str
    global_start: int
    global_end: int
    video_path: str
    n_frames: int
    skill_type: str
    variant: str              # 'positive' | 'truncated'
    partial_success: float
    quality_label: str


def load_b1k_dataset(
    dataset_path: str,
    video_key: str = DEFAULT_VIDEO_KEY,
    annotation_level: str = "skill",        # 'skill' | 'primitive'
    max_frames: int = 64,
    min_segment_frames: int = 15,
    max_segment_frames: int = 3000,         # 이보다 길면 drop (mean 515, 상위 극단값 제거)
    split: str = "train",                   # 'train' | 'val' | 'all'
    val_demo_ratio: float = 0.1,
    task_whitelist: list[str] | None = None,
    max_episodes_per_task: int | None = None,
    truncated_negative_ratio: float = 0.5,  # §6
    seed: int = 42,
) -> dict[str, list[dict]]:
    ...
```

### 3.1 처리 흐름

1. `meta/info.json` 로드 → `fps`, `video_path` 템플릿 검증 (`codebase_version == "v3.0"` assert).
2. `meta/episodes/**/*.parquet` 전부 읽어 episode 인덱스 테이블 구성 (pyarrow).
   메모리: 20,000 row × 40 컬럼 → 무시 가능.
3. split 결정: episode를 `(task_index, demo_index_within_task)`로 정렬하고
   **task별로** `demo_index_within_task % round(1/val_demo_ratio) == 0` 인 것을 val로.
   → task 커버리지를 양쪽에 유지 (§5.3).
4. episode 순회:
   - `annotation_path` 로드. 없으면 skip + 카운터.
   - `video_root/{video_key}/chunk-{c:03d}/file-{f:03d}.mp4` 존재 확인. 없으면 skip + 카운터
     (부분 다운로드 상태 대응 — **에러로 죽이지 말 것**).
   - `ep_offset = round(from_timestamp * fps)`
   - `annotation_level`에 따라 segment 목록 추출 → 각각에 대해 §3.2 필터 → `SegmentRecord`
   - 네거티브 변형 생성 (§6)
5. `SegmentRecord` → traj_dict 로 변환하고 `task_data[task_key].append(...)`.
   `task_key`는 **`f"{task_name}/{skill_description}"`** 를 권장 — `flatten_task_data`가 `task_name`
   필드로 넣어 주며, 이후 커버리지 리포트에 유용하다.

### 3.2 segment 필터 (드롭 사유를 전부 카운트해서 마지막에 출력할 것)

| 조건 | 처리 |
|------|------|
| `frame_duration`이 중첩 리스트 | 첫 구간만 사용, `split_range=True` 기록 (또는 `skip_split_range=True`면 drop) |
| `end - start <= 0` | drop (`neg_len`) |
| `end - start < min_segment_frames` | drop (`too_short`) |
| `end - start > max_segment_frames` | drop (`too_long`) |
| `end > episode.length` | drop (`oob`) — 불변식 위반이므로 WARN 로그 |
| `skill_description` 또는 `object_id`에 `None` | drop (`bad_annotation`) |
| 템플릿 렌더 실패 | drop (`no_template`) |

## 4. Instruction 템플릿

`b1k_skill_templates.py`에 독립 모듈로 분리한다 (단위 테스트 대상).

### 4.1 공개 API

```python
def render_instruction(
    skill_description: str,          # "pick up from"
    object_ids: list[str],           # ["radio_89", "coffee_table_koagbh_0"]
    memory_prefix: list[str],        # ["back"] | []
    spatial_prefix: list[str],       # ["left"] | []
    task_name: str | None = None,    # "turning_on_radio" — 맥락 접미 옵션
    include_task_context: bool = False,
) -> str | None:
    """자연어 subtask 지시문. 렌더 불가하면 None."""
```

### 4.2 object id 정규화 — **정규식 금지, 룩업 테이블로 할 것**

전수 조사 결과 **unique object_id는 573개뿐**이고, 정규화 후 카테고리는 **291개**다. 닫힌 어휘이므로
룩업 테이블(`OBJECT_NAME_MAP: dict[str, str]`)을 생성해 두고 **291행을 사람이 검수**하는 것이 정답이다.

정규식만으로는 해결 불가능하다는 것이 실측으로 확인되었다. id 형태가 셋인데 서로 구분이 안 된다:

| 형태 | 예 | 정규화 |
|------|-----|--------|
| 순수 카테고리 | `floors`, `plate`, `wine_bottle` | 그대로 |
| `<cat>_<6자해시>_<idx>` | `coffee_table_koagbh_0` | `coffee table` |
| `<cat>_<번호>[_<idx>]` | `broom_172`, `half_beet_211_0` | `broom`, `half beet` |

`_[a-z]{6}$`를 해시로 보고 자르면 **다음 10개가 망가진다** (실측):

```
wine_bottle    → "wine"      wicker_basket  → "wicker"    toy_figure    → "toy"
ice_bucket     → "ice"       electric_kettle→ "electric"  half_banana   → "half"
grated_cheese  → "grated"    rubber_eraser  → "rubber"    toilet_tissue → "toilet"
electric_switch→ "electric"
```

`<cat>_<hash>_<idx>`와 `<6자카테고리>_<번호>_<idx>`(예: `half_banana_211_0`)는 문법적으로 동일해서
규칙으로 분리할 수 없다.

**절차:**

1. 생성 스크립트로 전체 573개 id를 뽑고, 아래 휴리스틱으로 **초안** 테이블을 만든다
   (`(?:_[a-z]{6})?(?:_\d+)+$` 제거 → 언더바를 공백으로).
2. 291행 초안을 **사람이 훑어보고** 잘못 잘린 항목을 고친다. 1회성 작업이고 30분이면 끝난다.
3. 확정된 테이블을 `b1k_skill_templates.py`에 리터럴 dict로 박는다.
4. 런타임에 테이블에 없는 id가 나오면 → **렌더 실패(None) + WARN**. 조용히 휴리스틱으로 넘어가지 말 것.

- 정규화 후 빈 문자열이면 렌더 실패(None) 처리.
- 테이블 검수 결과는 [04 §G1](04-validation-plan.md#g1-instruction-템플릿) 게이트의 증빙이다.

### 4.2.1 접두사 어휘 (실측)

`memory_prefix` — 3종뿐:

| 값 | 빈도 | 처리 |
|---|---|---|
| `back` | 2,817 | 동사구 뒤에 ` back`: `place the radio back on the coffee table` |
| `the other` | 2,825 | 대상 앞: `pick up the other cup from the table` |
| `the same` | 3 | 대상 앞: `the same` |

`spatial_prefix` — **35종이고 성격이 제각각**이다 (문서 초안에서 `left`/`right`만 가정한 것은 부족했다).
빈도순: `""`(3,225 — 빈 문자열이 최빈값이므로 반드시 스킵 처리), `to_the_edge_of`(510), `right`(361),
`high_level`(270), `left`(254), `right_door`(235), `left_door`(223), `low_level`(184), `layer_4`(146),
`in_front_of`(144), `layer_2`(128), `center`(120), `middle_level`(109), `layer_3`(91), `under`(89),
`layer_5`(76), `away`(52), `face`(34), `2X2`(33), `reorient`(26), `2X1`(23), `second_right_door`(22),
`face_away`(22), `3X1`(21), `first_left_door`(21), … (`3X3`, `4X1` 등 그리드 좌표까지)

부류별 처리:

| 부류 | 예 | 처리 |
|---|---|---|
| 빈 문자열 | `""` | 무시 |
| 방향 형용사 | `left`, `right`, `center` | 대상 명사 앞: `the left door` |
| 전치사구 | `to_the_edge_of`, `in_front_of`, `under`, `near`, `away` | 대상 앞 전치사구로: `to the edge of the table` |
| 부품 지정 | `right_door`, `first_left_door`, `left_door` | 대상의 부위: `the right door of the cabinet` |
| 높이/층 | `high_level`, `layer_4`, `middle_level` | `the 4th layer of the {obj}` 식 매핑 |
| 그리드 좌표 | `2X2`, `3X1` | **드롭**(총 ~150건). 자연어화 이득 대비 위험이 큼 |
| 자세 | `reorient`, `face`, `face_away` | 동사 수식: `and face it away` — 또는 드롭 |

이것도 닫힌 35개 어휘이므로 **`SPATIAL_PREFIX_MAP` 룩업 테이블**로 처리한다.

### 4.2.2 skill별 arity가 가변이다

같은 `skill_description`이라도 `object_id` 그룹 크기가 다르다 (1,500 episode 샘플):

```
pour              {1: 11,  3: 306}
sweep surface     {1: 78,  2: 310}
place on next to  {2: 2,   3: 806}
move to           {1: 10797}          # 단일 arity
```

따라서 템플릿 키는 `skill_description` 단독이 아니라 **`(skill_description, arity)`** 여야 한다.
소수 arity(예: `pour` arity 1)는 템플릿을 안 만들고 드롭해도 되지만, **드롭 카운터에 기록**할 것.
`object_id`가 중첩 리스트인 경우도 있으므로(1,500 샘플 중 330건) 평탄화 후 arity를 센다.

### 4.3 템플릿 테이블

`skill_description`은 자유 텍스트가 아니라 사실상 닫힌 어휘다 (`annotations/skill_summary.csv`에 전량 존재).
상위 15개는 명시 템플릿을 쓰고 나머지는 폴백 규칙을 쓴다.

| skill_description | arity | 템플릿 |
|---|---|---|
| `move to` | 1 | `go to the {0}` |
| `pick up from` | 2 | `pick up the {0} from the {1}` |
| `place in` | 2 | `place the {0} in the {1}` |
| `place on` | 2 | `place the {0} on the {1}` |
| `place on next to` | 3 | `place the {0} on the {1} next to the {2}` |
| `push to` | 2 | `push the {0} to the {1}` |
| `open door` / `close door` | 1 | `open the {0} door` / `close the {0} door` |
| `open lid` / `close lid` | 1 | `open the lid of the {0}` / `close the lid of the {0}` |
| `turn on switch` / `turn off switch` | 1 | `turn on the {0}` / `turn off the {0}` |
| `press` | 1 | `press the {0}` |
| `chop` | 1 | `chop the {0}` |
| `sweep surface` | 1~2 | `sweep the surface of the {0}` |
| `pour` | 2 | `pour the {0} into the {1}` |
| `hand over` | 1~2 | `hand over the {0}` |

폴백: `f"{skill_description} the {obj0}"` + 잔여 오브젝트를 ` with the {objN}`로 붙이는 대신
**arity 불일치 시 None을 반환**한다 (조용히 이상한 문장을 만들지 말 것). 폴백 미커버 skill 비율은
G1 게이트에서 5% 미만이어야 한다.

접두사 처리는 §4.2.1의 실측 어휘 표를 따른다 (memory 3종 / spatial 35종, 둘 다 룩업 테이블).

`include_task_context=True`면 `f"{instruction}. (task: {task_name.replace('_',' ')})"` 형태로 붙인다.
**기본값 False.** 이유: 런타임에 FSM이 주는 것은 subtask 지시문 하나이며, 학습 시 task 맥락을 넣어두면
분포가 어긋난다. 맥락 포함 버전은 별도 ablation data_source로 만들어 비교할 것.

### 4.4 지시문 중복

같은 지시문이 수천 개 segment에 반복된다 (`move to the table` 등). 이는 문제가 아니라 **자산**이다 —
`different_task` 네거티브가 "같은 소스 내 다른 subtask 지시문"을 뽑을 때 진짜 헷갈리는 후보를 준다
(`data.traj_same_source_prob: 0.5`). 다만:

- unique 지시문 수를 G2에서 리포트할 것 (너무 적으면 템플릿이 뭉갠 것).
- `convert_dataset_to_hf_format`이 unique task 문장만 임베딩하므로 임베딩 비용은 걱정 없다.

## 5. 세부 결정

### 5.1 프레임 예산

| 단계 | 값 | 근거 |
|------|-----|------|
| 로더 반환 | 64 | segment 평균 515프레임 → 8배 압축 |
| `output.max_frames` (mp4 인코딩) | 64 | 로더와 동일 (no-op) |
| `output.fps` | 30 | 재인코딩 fps. 어차피 프레임 시퀀스로만 쓰인다 |
| `output.shortest_edge_size` | 240 | 720×720 → 240×240 |
| preprocess `max_frames_for_preprocessing` | 32 | 기존 데이터셋(`preprocess.yaml`)과 동일 |
| train `data.max_frames` | **8** | Robometer-4B가 8프레임으로 학습됨 (§5.1.1) |

#### 5.1.1 왜 8인가

`config.yaml`의 기본값은 16이지만, README의 RBM-1M 학습 커맨드와 모든 평가 커맨드는 예외 없이
`data.max_frames=8` / `max_frames=8`을 쓴다. 즉 **`robometer/Robometer-4B` 체크포인트는 8프레임 입력으로
학습된 모델**이다. LoRA 파인튜닝은 그 표현을 재사용하는 것이 목적이므로 프레임 수를 바꾸면
사전학습 분포에서 벗어난다 (`use_per_frame_progress_token: true`라 프레임 수가 토큰 시퀀스 구조에 직접 반영됨).

- 8프레임 × 평균 17초 segment ≈ 2.1초 간격 관측. B1K skill segment에는 다소 성긴 편이다.
- 16으로 올리는 것은 **ablation으로만** 시도하고, 반드시 8 조건과 같은 val 셋에서 비교할 것.
  (풀 파인튜닝이라면 16이 유리할 수 있으나 LoRA에서는 8이 기본.)

`center_crop: false` (이미 정사각).

### 5.2 성능

- segment 하나당 mp4 하나를 굽는다. 17,998 episode × 평균 20 skill ≈ **36만 clip**.
  240×240×64프레임 H.264 ≈ 250 KB → **약 90 GB**. 디스크와 시간을 감안해 1단계는
  `max_episodes_per_task`로 task당 20 episode(≈ 3.6만 clip, 9 GB)부터 시작하라.
- 집합 mp4에 대한 랜덤 seek이 병목이다. **같은 `(video file)`을 참조하는 segment들이 연속으로 처리되도록
  trajectory 리스트를 `(file_index, global_start)` 순으로 정렬**해 디코더 캐시 지역성을 살릴 것.
  (단, `convert_dataset_to_hf_format`은 `imap_unordered`라 워커별로 흩어진다 — `num_workers`를
  코어 수보다 낮게(예: 8) 두고 정렬을 유지하는 편이 낫다.)

### 5.3 train/val 분할

- **episode 단위로** 나눈다 (같은 episode의 segment가 양쪽에 걸치면 누수).
- task별 stratified: 각 task에서 `demo_index_within_task`가 10의 배수인 episode → val.
- 별도 `dataset_name`(=HF config name)으로 굽는다: `b1k_skill_train`, `b1k_skill_val`.
- **두 subset 모두 `data_source="b1k_skill"`로 통일**한다 (§5.5).

### 5.4 결정적 id

`id`는 재실행/증분 변환에서 안정적이어야 한다:

```python
traj_id = f"b1k-{episode_index:06d}-{skill_idx:03d}-{variant}"
```

`generate_unique_id()`(uuid4)를 쓰면 재변환 시 캐시 무효화와 중복 검출이 불가능해진다.

### 5.5 `data_source`를 고정하는 이유

Robometer는 세 레지스트리를 **`data_source` 문자열**로 조회한다:
`dataset_success_cutoff.txt`(cutoff), `DATA_SOURCE_CATEGORY`(샘플링 카테고리),
custom eval 메트릭 이름. train/val subset이 서로 다른 data_source를 가지면 cutoff과 카테고리를
두 번 등록해야 하고 실수가 난다. `create_hf_trajectory`는 `traj_dict["data_source"]`가 있으면
`dataset_name` 대신 그것을 쓰므로 loader에서 명시적으로 넣어 준다.

## 6. 네거티브 변형 생성

목적: 런타임 FSM의 두 오류를 직접 겨냥한다.
- **조기 전이(false success)** — subtask가 안 끝났는데 완료로 판정
- **미검출(missed success)** — 끝났는데 전이 안 함

1단계에서는 변환 시점 네거티브 **1종**만 만든다.

### 6.1 `truncated`

```
positive segment [a, b) 에 대해
  α ~ U(0.30, 0.85)
  clip = [a, a + round(α*(b-a)))
  quality_label   = "failure"
  partial_success = α
```

이 조합이 학습 라벨에서 어떻게 되는지 (코드 확인 완료):
- `compute_progress_from_segment`: `partial_success != 1.0`이면 progress를 전부 0으로 만들고
  **원본 마지막 프레임에 해당하는 위치에만 α**를 넣는다 (`helpers.py:625` 이후 블록).
- `compute_success_labels`: `quality_label`이 failure면 **전 프레임 success=0**.

즉 "이 클립은 subtask의 α 지점까지만 왔고 아직 성공이 아니다"가 정확히 학습된다.

생성 비율: `truncated_negative_ratio` (기본 0.5 → positive 2개당 truncated 1개).
같은 segment에서 최대 1개만 만든다. α는 `seed` 고정 RNG로 뽑아 재현성 유지.

### 6.2 온라인 전략에 맡기는 것 (변환기에서 만들지 않음)

`robometer/data/samplers/pref.py` / `progress.py`가 학습 중 생성한다
(`data.preference_strategy_ratio`, `data.progress_strategy_ratio`):

| 전략 | 만들어지는 네거티브 | FSM에서 방지하는 오류 |
|------|--------------------|----------------------|
| `DIFFERENT_TASK` / `DIFFERENT_TASK_INSTRUCTION` | 다른 subtask 지시문 + 현재 영상 | 잘못된 subtask에 대한 오전이 |
| `REWIND` | 진행하다 되돌아가는 클립 | 되감김/미끄러짐을 진행으로 오인 |
| `REVERSE_PROGRESS` | 역재생 | progress 방향성 |
| `SUBOPTIMAL` | 같은 task의 저품질 궤적 | (본 데이터셋엔 실패 데모가 없어 미작동) |

**본 데이터셋에는 실패 데모가 없다.** 따라서 `SUBOPTIMAL` 전략은 사실상 놀고,
실패 신호는 전부 §6.1 + `DIFFERENT_TASK` + `REWIND`에서 나온다. 2단계에서 정책 롤아웃 실패 궤적을
수집해 `quality_label="failure"`로 추가하는 것이 가장 큰 개선 여지다 ([05 §4](05-runtime-contract.md#4-2단계-실패-데이터-수집)).

### 6.3 2단계 후보 (지금 구현하지 말 것)

- `wrong_object`: 같은 장면의 다른 오브젝트로 지시문 치환 → `failure`, `partial_success=0.0`
- `overrun`: `[a, b + 0.15*(b-a))` → `successful` (완료 후에도 success가 유지되는지)
- `cross_boundary`: `[a + 0.5*(b-a), b + 0.5*(next-b))` → 경계 혼동 케이스

## 7. `generate_hf_dataset` 연결

`main()`의 dispatch 체인(파일 하단)에 **`robofac` 분기 앞**에 추가한다:

```python
elif "b1k" in cfg.dataset.dataset_name.lower():
    from dataset_upload.dataset_loaders.b1k_loader import load_b1k_dataset

    print(f"Loading BEHAVIOR-1K demos from: {cfg.dataset.dataset_path}")
    task_data = load_b1k_dataset(
        cfg.dataset.dataset_path,
        split="val" if cfg.dataset.dataset_name.lower().endswith("_val") else "train",
        max_frames=cfg.output.max_frames,
        # 나머지는 yaml에서 기본값 사용
    )
    trajectories = flatten_task_data(task_data)
```

> `DatasetConfig`에 b1k 전용 필드(video_key, max_episodes_per_task 등)를 추가하고 싶으면
> `@dataclass DatasetConfig`에 옵셔널 필드를 더하는 편이 pyrallis CLI 오버라이드와 잘 맞는다.
> 필드를 추가하지 않는다면 yaml 대신 loader 기본값 + 코드 상수로 관리할 것.

### 7.1 `b1k_skill.yaml`

```yaml
# dataset_upload/configs/data_gen_configs/b1k_skill.yaml
dataset:
  dataset_path: /Users/woosung-kim/workspace/Physical-AI/datasets/behavior-1k/2026-challenge-demos
  dataset_name: b1k_skill_train        # HF config name. loader가 split을 이름으로 판별

output:
  output_dir: datasets/b1k_rbm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 30
  shortest_edge_size: 240
  center_crop: false
  num_workers: 8                        # seek 지역성 때문에 코어 수보다 낮게

hub:
  push_to_hub: true
  hub_repo_id: <HF_USERNAME>/b1k_rbm
```

`b1k_skill_val.yaml`은 `dataset_name: b1k_skill_val`만 다르고 나머지 동일 (같은 `hub_repo_id`,
같은 `output_dir` → 하나의 repo에 두 config).

### 7.2 실행

```bash
export HF_TOKEN=...
export B1K_ROOT=/Users/woosung-kim/workspace/Physical-AI/datasets/behavior-1k/2026-challenge-demos

uv run python -m dataset_upload.generate_hf_dataset \
  --config_path dataset_upload/configs/data_gen_configs/b1k_skill.yaml \
  --dataset.dataset_path=$B1K_ROOT \
  --output.max_trajectories=200          # 첫 스모크 런
```

## 8. 로깅 요구사항

변환 종료 시 다음을 stdout + `output_dir/b1k_conversion_report.json`에 남긴다. G2/G3 게이트가 이걸 읽는다.

```json
{
  "episodes_seen": 0, "episodes_skipped_no_video": 0, "episodes_skipped_no_annotation": 0,
  "segments_total": 0,
  "segments_dropped": {"neg_len":0,"too_short":0,"too_long":0,"oob":0,"bad_annotation":0,"no_template":0,"split_range":0},
  "trajectories_emitted": {"positive":0,"truncated":0},
  "unique_instructions": 0,
  "skill_description_coverage": {"move to": 0, "...": 0},
  "uncovered_skill_descriptions": [],
  "frames_per_traj": {"min":0,"median":0,"max":0},
  "tasks": 0, "episodes_per_task": {"min":0,"median":0,"max":0}
}
```
