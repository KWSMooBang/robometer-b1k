# 04. 검증 계획 (Acceptance Gates)

각 단계는 아래 게이트를 통과해야 다음 단계로 간다. 게이트는 전부 **자동 검사 가능**하게 적었다.
검증 스크립트 2개(`scripts/b1k_check_alignment.py`, `scripts/b1k_preview_segments.py`)를 함께 만든다.

---

## G0. 데이터 범위

`scripts/b1k_check_alignment.py --root $B1K_ROOT` 가 출력해야 하는 것:

- `meta/episodes` 총 row 수와 `annotations/` 파일 수의 일치 여부
- 각 episode에 대해 `annotation_path` / 비디오 파일 존재 여부
- **필요한 비디오 파일 목록**(`videos/observation.rgb.zed_link_camera_0/chunk-XXX/file-NNN.mp4`)과
  누락분 → 이걸로 추가 다운로드 범위를 정한다

**통과 조건**
- 학습에 쓸 episode 집합이 확정되고, 그 집합의 annotation·비디오가 100% 존재
- episode 수 ≥ 2,000, task 수 ≥ 50 (그보다 적으면 skill 어휘 커버리지가 부족)

> 현재 로컬 상태는 `meta/episodes` 200 row(task-0000)뿐이라 **이 게이트를 통과하지 못한다.**
> `meta/**` 전체 다운로드가 선행되어야 한다 ([01 §5](01-dataset-spec.md#5-로컬-동기화-현황)).

## G1. instruction 템플릿

`pytest tests/test_b1k_templates.py` (신규):

```python
def test_normalize_object():
    assert normalize_object("radio_89") == "radio"
    assert normalize_object("coffee_table_koagbh_0") == "coffee table"

def test_six_letter_category_not_mangled():
    # 정규식 휴리스틱이 깨지는 실측 케이스 — 룩업 테이블이 이걸 막아야 한다
    assert normalize_object("wine_bottle") == "wine bottle"
    assert normalize_object("toy_figure") == "toy figure"
    assert normalize_object("wicker_basket") == "wicker basket"
    assert normalize_object("half_beet_211_0") == "half beet"

def test_unknown_id_returns_none():
    assert normalize_object("totally_unseen_thing_zzz") is None

def test_render_basic():
    assert render_instruction("pick up from", ["radio_89", "coffee_table_koagbh_0"], [], []) \
        == "pick up the radio from the coffee table"

def test_memory_prefix_back():
    assert render_instruction("place on", ["radio_89", "coffee_table_koagbh_0"], ["back"], []) \
        == "place the radio back on the coffee table"

def test_arity_mismatch_returns_none():
    assert render_instruction("pick up from", ["radio_89"], [], []) is None
```

전수 커버리지 검사 (annotation 전체를 훑는 스크립트):

**통과 조건**
- 렌더 실패(None) segment 비율 **< 5%**
- `skill_summary.csv` 상위 20개 skill이 `(skill, arity)` 조합까지 전부 커버됨
- **`OBJECT_NAME_MAP` 전수 검수 완료** (검수자/일시를 커밋 메시지에 남길 것).
  잘못 잘린 이름 0건 — 정규식 휴리스틱은 `wine_bottle`→"wine" 등 17건을 망친다
- `SPATIAL_PREFIX_MAP` 처리 방침 확정 (드롭 항목 명시)
- unique 지시문 수 ≥ 1,500

### G1 실측 결과 (2026-08-11, PASS)

```
annotation files        : 20,000
object groups (segments): 406,341
rendered                : 404,273  (99.49%)
failed                  :   2,068  ( 0.51%)
unique instructions     :   1,722

failure reasons:
   1,802  0.44%  dropped_spatial_prefix(grid)      # 2X2/3X1 등 그리드 좌표 — 의도적 드롭
     200  0.05%  unsupported_arity(pour|1)
      65  0.02%  unsupported_arity(place on next to|2)
       1  0.00%  unsupported_arity(move to|1)
```

`OBJECT_NAME_MAP`은 573 id → 278 카테고리이며, 해시로 판정된 6자 토큰 84개가 전부 무의미 문자열임을
확인했다(`abzvij`, `koagbh`, … 영어 단어 0건). 재생성·재검수 커맨드:

```bash
python scripts/b1k_build_object_names.py $B1K_ROOT
```

> unique 지시문 기준을 2,000 → 1,500으로 낮췄다. 2,000은 측정 전에 임의로 잡은 값이었고,
> 실제 조합 공간이 100 task × 278 오브젝트 카테고리 × 35 skill에서 나오는 1,722개로 수렴한다.
> 목표는 지시문 암기가 아니라 skill×object 일반화이므로 이 수준이면 충분하다.

## G2. loader (mp4 생성 없이) — task-0000 기준 PASS (2026-08-12)

`load_b1k_dataset(..., max_episodes_per_task=2)` 를 직접 호출해서:

**통과 조건**
- 반환된 traj_dict가 [02 §1.1](02-converter-spec.md#11-traj_dict-스키마-loader가-만들어야-하는-것) 스키마를 만족
- `pickle.dumps(traj["frames"])` 성공 (spawn 병렬 전제)
- `traj["frames"]()` 호출 → `(T,H,W,3) uint8`, `T <= max_frames`, `T >= 8`
- 반환 프레임의 첫/마지막이 실제 segment 경계와 맞음 → `b1k_preview_segments.py`로 12개 샘플을
  contact sheet(PNG)로 뽑아 **육안 확인**. 마지막 프레임에서 해당 subtask가 완료된 상태여야 한다.
- 드롭 카운터 합 + emit 수 == 전체 segment 수 (누수 없음)
- 같은 입력·같은 seed로 두 번 실행 시 `id` 목록이 동일 (결정성)

> **육안 확인이 이 프로젝트에서 가장 중요한 게이트다.** frame offset이 한 episode만큼 어긋나도
> 숫자 게이트는 전부 통과하면서 라벨이 전부 무의미해진다. 반드시 `from_timestamp` 오프셋이
> 다른 여러 episode(파일 중간에 있는 episode)를 골라서 볼 것.
> `b1k_preview_segments.py`는 `global_start`가 큰 clip을 우선 뽑도록 되어 있다 (오프셋 버그는
> 파일 첫 episode에서는 드러나지 않기 때문).

### G2 실측 결과 (task-0000, 200 episodes)

```bash
python scripts/b1k_check_alignment.py $B1K_ROOT --check-limit 200
python scripts/b1k_preview_segments.py $B1K_ROOT -o preview.png --rows 12
B1K_ROOT=$B1K_ROOT pytest tests/test_b1k_loader.py -q
```

- 불변식 검사 200 episode 통과 (skill end ≤ length, skill_idx 정렬)
- 720 segment 전부 emit, 드롭 0건, **카운터 보존 검사 통과**
  (`segments_kept + Σsegments_dropped == segments_total`, 로더가 런타임에 assert)
- train/val = 180/20 episode, episode 단위로 분리되어 교집합 0
- clip 디코딩 `(64, 720, 720, 3) uint8`, `pickle.dumps(traj["frames"])` 성공
- 육안 확인: `global_start` 58,448~80,956 (파일 끝부분) clip 12개에서
  `go to the radio`는 다른 방에서 출발해 라디오 앞에서 종료, `pick up`은 마지막 프레임에
  그리퍼에 라디오가 잡혀 있음, `place back`은 테이블에 놓고 손이 빠짐 → **오프셋 정렬 정확**
- 테스트 50개 통과 (템플릿 33 + 로더 17)

## G3. 변환 (스모크 → 풀)

**스모크** (`--output.max_trajectories=200`)

- 종료 코드 0, `b1k_conversion_report.json` 생성
- 생성된 mp4 200개 중 무작위 10개: `ffprobe`로 `nb_frames >= 8`, 240×240
- HF Dataset row 수 == mp4 수
- `partial_success` 분포: positive는 전부 1.0, truncated는 0.30~0.85

**풀 변환**

- 실패(=None 반환) trajectory 비율 < 1%
- `quality_label` 비율이 `truncated_negative_ratio`와 일치 (±2%p)
- 디스크 사용량이 예산 내 ([02 §5.2](02-converter-spec.md#52-성능))

## G4. 네거티브

`b1k_skill_train`에서 무작위 truncated 20개를 뽑아:

**통과 조건**
- 클립 마지막 프레임에서 subtask가 **아직 완료되지 않았음**을 육안 확인 (α 상한 0.85가 너무 높지 않은지)
- 라벨 시뮬레이션: `compute_progress_from_segment(...)` + `compute_success_labels(...)`를
  실제 값으로 호출해 `progress`가 마지막 프레임만 α이고 나머지 0, `success`가 전부 0인지 확인
- positive에 대해서는 `progress[-1] == 1.0`, `success[-1] == 1.0`

이 게이트는 코드로 확인할 수 있다:

```python
from robometer.data.datasets.helpers import compute_progress_from_segment, compute_success_labels
p = compute_progress_from_segment(num_frames_total=T, frame_indices=list(range(T)),
                                  progress_pred_type="absolute_first_frame",
                                  success_cutoff=1.0, partial_success=alpha)
s = compute_success_labels(p, "b1k_skill", {"b1k_skill": 1.0}, 1.0, "failure")
```

## G5. preprocess

**회귀 확인 먼저:** `_load_dataset_from_path` 수정([03 §2.1](03-preprocess-train-spec.md#21-필수-로컬-디렉터리-로드-경로-수정)) 후
기존 데이터셋 하나(robofac 등)를 Hub 경로로 전처리해 **row 수가 수정 전과 동일**한지 확인.

B1K:
- `$CACHE/<user>_b1k_rbm_b1k_skill_train/processed_dataset` 생성, row 수 == 변환 row 수 − 필터 드롭
- `index_mappings.json`의 `quality_indices`에 `successful`과 `failure`가 모두 존재
- `source_indices` 키가 `b1k_skill` **하나뿐** (02 §5.5 확인)
- `frames/*.npz` 하나를 로드해 shape `(<=32, 240, 240, 3)` uint8
- 로그의 `Filtered out N trajectories`에서 N/전체 < 2%

## G6. 레지스트리

파이썬 한 줄 검사:

```python
from robometer.data.datasets.helpers import load_dataset_success_percent
assert load_dataset_success_percent("robometer/data/dataset_success_cutoff.txt")["b1k_skill"] == 1.0

from robometer.data.datasets.name_mapping import DS_SHORT_NAME_MAPPING
assert "<HF_USERNAME>_b1k_rbm_b1k_skill_train" in DS_SHORT_NAME_MAPPING

from robometer.data.dataset_category import DATASET_MAP
assert "b1k" in DATASET_MAP and DATASET_MAP["b1k"]["eval"]
```

추가: `data.train_datasets=[b1k]`로 dataset 객체를 만들어 `len(ds) > 0`, `ds[0]`이 정상 샘플인지.

## G7. 학습

**스모크** (`training.max_steps=20`, `custom_eval_steps=10`)
- loss가 NaN이 아니고 감소 추세
- custom eval이 예외 없이 돌고 `eval_rew_align/pearson_b1k_skill_val`가 로그에 찍힘
  (안 찍히면 name_mapping 접미사 불일치)

**본 학습** — 아래 표를 채운다.

| 조건 | pearson (reward align) | kendall_last (policy rank) | success F1 @0.5 | 비고 |
|------|---|---|---|---|
| Robometer-4B zero-shot | | | | 분모 |
| base Qwen3-VL-4B + LoRA | | | | 체크포인트 기여 |
| Robometer-4B + LoRA (b1k) | | | | 목표 |

**통과 조건**
- `Robometer-4B + LoRA` 가 zero-shot 대비 pearson **+0.10 이상**
- val에서 positive의 최종 progress 평균 > 0.9, truncated의 최종 progress 평균 < α+0.15

## G8. 런타임 임계값

[05-runtime-contract.md](05-runtime-contract.md) §3의 캘리브레이션 절차를 돌려
전이 임계값 θ와 hysteresis 창을 확정하고, val 셋 기준
**조기 전이율(false success) < 5%**, **중앙값 전이 지연 < 1.0초(30프레임)** 를 만족시킨다.

---

## 회귀 방지

기존 데이터셋 파이프라인을 건드리는 변경은 `preprocess_datasets.py` 한 곳뿐이다.
PR에는 반드시 다음을 포함할 것:

1. 수정 전/후 robofac(또는 임의 기존 데이터셋) 전처리 row 수 비교 로그
2. `filters` 딕셔너리에 추가한 키가 기존 키와 충돌하지 않음
3. `dataset_success_cutoff.txt` diff가 **추가 한 줄뿐**임
   (기존 마지막 줄에 개행이 없으므로 `roboarena,0.90b1k_skill,1.0` 로 붙지 않게 주의)
