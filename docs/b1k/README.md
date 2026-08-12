# BEHAVIOR-1K → Robometer 파인튜닝 하네스

BEHAVIOR-1K 2026 Challenge demo 데이터셋으로 Robometer를 **subtask(skill) 완료 판정기**로 파인튜닝하기 위한
구현 하네스 문서 모음. 각 문서는 "무엇을 구현할지"가 아니라 "어떤 파일에 어떤 함수를 어떤 계약으로 만들지"까지 적어 두었다.

## 문서 목록

| # | 문서 | 내용 |
|---|------|------|
| 01 | [dataset-spec.md](01-dataset-spec.md) | 원본 데이터셋 해부 — LeRobot v3 레이아웃, annotation 스키마, 실측 통계, 함정 |
| 02 | [converter-spec.md](02-converter-spec.md) | `b1k_loader` 구현 명세 — segment→trajectory 변환, instruction 템플릿, 프레임 로더, 네거티브 생성 |
| 03 | [preprocess-train-spec.md](03-preprocess-train-spec.md) | preprocessor / 레지스트리 변경, cutoff, 학습·평가 커맨드 |
| 04 | [validation-plan.md](04-validation-plan.md) | 각 단계 acceptance gate와 검증 스크립트 |
| 05 | [runtime-contract.md](05-runtime-contract.md) | 학습된 모델이 subtask FSM에서 쓰이는 방식 (라벨 설계를 구속하는 계약) |
| 06 | [runbook.md](06-runbook.md) | **서버 실행 순서** — 명령어와 각 단계 확인 사항 |

## 대상 시스템 (구상 중인 모델)

```
task → [고정된 subtask sequence]
         ├── VLA(pi05)가 현재 subtask의 action 생성
         ├── Robometer가 매 스텝 현재 subtask의 progress/success 판정
         │     ├── success  → 다음 subtask로 transition
         │     └── fail 판정 → feedback 모듈 → recovery subtask로 transition
```

즉 Robometer에 요구되는 것은 **episode 단위 task 성공 판정이 아니라 subtask 단위 완료 판정**이다.
이 하네스의 모든 설계 결정은 여기서 파생된다.

## 핵심 설계 결정 (요약)

1. **학습 단위 = skill segment.** episode 하나를 통째로 한 trajectory로 넣지 않는다.
   `annotations/task-XXXX/episode_XXXXXXXX.json`의 `skill_annotation[i].frame_duration`으로 잘라
   segment 하나 = RBM trajectory 하나로 만든다. 이렇게 해야 progress 1.0 = "이 subtask 완료" 가 된다.
   → [02](02-converter-spec.md#2-변환-단위)

2. **task 지시문 = 템플릿으로 합성한 subtask 문장.** `skill_description`("pick up from") + `object_id`(["radio_89",
   "coffee_table_koagbh_0"])를 자연어("pick up the radio from the coffee table")로 조립한다.
   → [02](02-converter-spec.md#4-instruction-템플릿)

3. **success cutoff = 1.0.** segment 경계로 이미 정확히 잘랐으므로 `dataset_success_cutoff.txt`에 1.0을 넣는다.
   대신 episode 단위 trajectory를 추가로 만들 때는 episode마다 유효 구간 비율이 0.785~1.0으로 편차가 크므로
   (중앙값 0.96) **cutoff 파일에 의존하지 말고 변환 시점에 `valid_duration`으로 잘라서** cutoff을 1.0으로 만든다.
   → [01](01-dataset-spec.md#34-valid_duration-과-trailing-idle), [03](03-preprocess-train-spec.md#3-success-cutoff)

4. **네거티브는 2계층.** (a) 변환 시점에 굽는 `truncated` 네거티브(중간에서 끊고 `quality_label="failure"`,
   `partial_success=α`) + (b) 학습 시 온라인 전략(`different_task`, `rewind`, `reverse_progress`).
   같은 data_source 안에 subtask가 수백 종 섞여 있으므로 `different_task` 네거티브가 자동으로
   "같은 데이터 소스의 다른 subtask 영상"을 뽑는다 — FSM 오전이(false transition) 방지에 도움이 되나,
   영상 자체가 바뀌므로 모델이 지시문이 아니라 장면 차이로 구분할 여지가 있다. "같은 영상 + 틀린 지시문"을
   만드는 `DIFFERENT_TASK_INSTRUCTION`은 ProgressSampler 소속이라 현재 설정에서는 쓰이지 않는다.
   → [02](02-converter-spec.md#6-네거티브-변형-생성)

5. **뷰는 zed head camera RGB 단일.** `observation.rgb.zed_link_camera_0` (720×720@30fps). depth와 wrist는 1단계 제외.
   → [01](01-dataset-spec.md#22-비디오)

## 구현 순서 (의존 순)

| 단계 | 산출물 | 문서 | Gate |
|------|--------|------|------|
| S0 | 데이터 다운로드 범위 확정 (현재 로컬은 부분 동기화 상태) | [01 §5](01-dataset-spec.md#5-로컬-동기화-현황) | G0 |
| S1 | `b1k_skill_templates.py` — instruction 템플릿 + object id 정규화 | [02 §4](02-converter-spec.md#4-instruction-템플릿) | G1 |
| S2 | `b1k_loader.py` — episode index → segment 목록 → trajectory dict | [02 §3,5](02-converter-spec.md) | G2 |
| S3 | `generate_hf_dataset.py` 분기 + `b1k_skill.yaml` 컨피그 | [02 §7](02-converter-spec.md#7-generate_hf_dataset-연결) | G3 |
| S4 | 네거티브 변형 (`truncated`) | [02 §6](02-converter-spec.md#6-네거티브-변형-생성) | G4 |
| S5 | preprocessor 로컬 경로 지원 패치 + `preprocess_b1k.yaml` | [03 §2](03-preprocess-train-spec.md#2-preprocessor-변경) | G5 |
| S6 | 레지스트리 3곳 등록 (cutoff / name_mapping / dataset_category) | [03 §3-4](03-preprocess-train-spec.md#3-success-cutoff) | G6 |
| S7 | LoRA 파인튜닝 + custom eval | [03 §5](03-preprocess-train-spec.md#5-학습) | G7 |
| S8 | subtask 전이 임계값 캘리브레이션 | [05](05-runtime-contract.md) | G8 |

각 Gate의 통과 조건은 [04-validation-plan.md](04-validation-plan.md)에 있다.
실제 실행 명령은 [06-runbook.md](06-runbook.md).

### 진행 상황 (2026-08-12)

| 단계 | 상태 |
|---|---|
| S1 템플릿 | ✅ 렌더율 99.49% (406,341 segment), 테스트 33개 |
| S2 로더 | ✅ task 부분 변환 지원, 정렬 육안 확인 통과, 테스트 17개 |
| S3 변환기 연결 | ✅ task-0000으로 관통 (12 clip, 240×240×64) |
| S4 네거티브 | ✅ truncated 라벨이 실제 라벨 함수에서 의도대로 계산됨 |
| S5 전처리 | ✅ 로컬 경로 버그 수정, npz `(32,240,240,3)` 생성 |
| S6 레지스트리 | ✅ cutoff / name_mapping / DATASET_MAP 등록, `RBMDataset` 샘플 생성 확인 |
| S7 학습 | ⬜ 서버에서 실행 |
| S8 캘리브레이션 | ⬜ |

## 건드리는 파일 목록

신규:
```
dataset_upload/dataset_loaders/b1k_loader.py
dataset_upload/dataset_loaders/b1k_skill_templates.py
dataset_upload/configs/data_gen_configs/b1k_skill.yaml
dataset_upload/configs/data_gen_configs/b1k_skill_val.yaml
dataset_upload/dataset_guides/BEHAVIOR1K.md
robometer/configs/preprocess_b1k.yaml
scripts/b1k_check_alignment.py        # 가용성 + 불변식
scripts/b1k_preview_segments.py       # 원본에서 자른 clip (로더 검증)
scripts/b1k_visualize_dataset.py      # 변환 산출물 + 학습 라벨 (PNG / HTML)
scripts/b1k_verify_cache.py           # 전처리 캐시 검증
scripts/b1k_build_object_names.py     # object 이름 테이블 생성
scripts/b1k_template_coverage.py      # 템플릿 커버리지 (G1)
scripts/b1k_pipeline.sh               # 단계별 드라이버
```

수정:
```
dataset_upload/generate_hf_dataset.py           # main()에 b1k 분기 추가
robometer/data/scripts/preprocess_datasets.py   # 로컬 디렉터리 로드 경로 수정 (필수, 03 §2)
robometer/data/dataset_success_cutoff.txt       # b1k_skill,1.0
robometer/data/datasets/name_mapping.py         # 짧은 이름 등록
robometer/data/dataset_category.py              # ALL_DATASOURCES / DATASET_MAP 등록
```

## 전제와 미확정 사항

- 로컬 데이터셋 경로: `/Users/woosung-kim/workspace/Physical-AI/datasets/behavior-1k/2026-challenge-demos`
  (본 문서의 모든 상대 경로는 이 디렉터리 기준).
- 현재 로컬은 **부분 다운로드 상태**다. annotation은 17,998/20,000개(91 task)가 있으나 `meta/episodes`는
  200 row(task-0000)만, zed RGB 비디오는 6개 파일만 받아져 있다. S0에서 범위를 확정해야 한다.
- 챌린지 규정상 학습에 쓸 수 있는 데이터/모델 제한은 확인하지 않았다. 제출 전에
  <https://behavior.stanford.edu/challenge/index.html> 규정을 직접 확인할 것.
- pi05(VLA) 쪽 인터페이스는 이 하네스 범위 밖이다. Robometer가 노출해야 하는 계약만 [05](05-runtime-contract.md)에 적었다.
