# 05. 런타임 계약 — subtask FSM에서의 Robometer

데이터 설계가 런타임 사용법에 의해 구속되므로, 변환기를 짜기 전에 이 계약을 먼저 고정한다.
(구현 자체는 이 하네스 범위 밖이지만, **여기서 벗어나면 앞의 라벨 설계가 무의미해진다.**)

## 1. FSM 구조

```
subtask_seq = [s0, s1, ..., sN]          # task별 사전 정의
state = (i, t_i)                          # 현재 subtask 인덱스, 진입 후 경과 스텝

매 스텝:
  a_t   = VLA(obs_t, instruction(s_i))
  p_t, σ_t = Robometer(frames[t-W:t], instruction(s_i))   # progress, success prob
  if transition_condition(p, σ):  i ← i+1, t_i ← 0
  elif failure_condition(p, σ, t_i):  i ← recovery(s_i)
```

Robometer에 들어가는 **instruction은 학습 때와 정확히 같은 템플릿 문장이어야 한다**
([02 §4](02-converter-spec.md#4-instruction-템플릿)). FSM의 subtask 정의를
`(skill_description, object_ids)` 튜플로 들고 있다가 같은 `render_instruction()`을 호출해 문자열을
만들 것. 손으로 쓴 다른 문장을 넣으면 분포가 어긋난다.

## 2. 관측 창(window)

- 학습 시 한 샘플은 **subtask 진입 시점부터 현재까지**를 균등 서브샘플한 8프레임이다
  (`data.max_frames=8` — Robometer-4B의 사전학습 설정과 동일, `progress_pred_type: absolute_first_frame`).
- 따라서 런타임에서도 **subtask 진입 프레임을 첫 프레임으로 고정**하고, 그 이후 구간을 8프레임으로
  균등 서브샘플해서 넣어야 한다. 최근 N프레임 슬라이딩 윈도만 넣으면 progress 기준점이 사라져
  값이 무의미해진다. ← 이게 가장 흔한 통합 버그다.
- 뷰는 학습과 동일하게 head camera(zed) RGB 1개, 240×240.

## 3. 임계값 캘리브레이션 (G8)

val 셋의 positive/truncated 클립으로 오프라인 캘리브레이션한다. 온라인 튜닝 금지(누수).

1. val positive 각각에 대해, 클립을 앞에서부터 잘라가며(prefix) progress/success 시계열을 계산한다.
2. 두 분포를 만든다:
   - **완료 시점 분포**: 실제 segment 끝에서의 σ
   - **미완료 분포**: 끝나기 전 임의 지점(= truncated와 동일)에서의 σ
3. 전이 규칙을 정하고 파라미터를 스윕한다:

```
transition_condition:  σ_t ≥ θ  가 연속 K 스텝(hysteresis) 유지
failure_condition:     t_i > T_max(s_i)  또는  progress가 M 스텝 동안 ε 미만 증가
```

- θ: 조기 전이율(false success) 5% 이하가 되는 최소값에서 출발
- K: 30fps 기준 5~10 (0.2~0.3초) — 단발 스파이크 억제
- `T_max(s_i)`: `annotations/skill_summary.csv`의 skill별 `mean_s`, `max_s`를 사용
  (예: `T_max = min(max_s, 3 * mean_s)`). skill 종류별로 다르게 잡는 게 중요하다 —
  `move to`는 평균 18.6초, 최대 298초로 편차가 극단적이다.

**보고 지표** (val 기준):
- 조기 전이율: 실제 완료 전에 전이한 비율 < 5%
- 전이 지연: 실제 완료 프레임 → 전이 프레임, 중앙값 < 30프레임(1초)
- 미검출률: `T_max` 안에 전이 못 한 비율

## 4. 2단계: 실패 데이터 수집

현재 B1K demo에는 **실패 궤적이 하나도 없다.** 실패 판정 능력은 전적으로 합성 네거티브
(truncated / different_task / rewind)에서 나오므로, 진짜 실패 모드(물체를 놓침, 잘못 잡음, 충돌)에
대한 판정은 검증되지 않은 상태다.

가장 효과가 큰 후속 작업은 **FSM을 돌려서 나온 실제 롤아웃을 라벨링해 되먹이는 것**:

```
rollout 수집 → subtask 경계에서 시뮬레이터 성공 조건으로 자동 라벨
  ├─ 성공한 subtask 구간  → quality_label="successful", partial_success=1.0
  └─ 실패한 subtask 구간  → quality_label="failure",   partial_success=(도달한 progress 추정)
→ b1k_skill_rollout 이라는 별도 data_source로 변환해 함께 학습
```

BEHAVIOR-1K는 시뮬레이터 안에서 술어(predicate) 기반 성공 조건을 제공하므로 자동 라벨링이 가능하다.
이 데이터가 들어오면 `SUBOPTIMAL` 선호 전략도 비로소 작동하고
`data.preference_strategy_ratio`를 기본값으로 되돌릴 수 있다.

## 5. 서빙

기존 자산 두 개를 그대로 쓴다:

- `robometer/evals/eval_server.py` — 배치 추론 서버 (`compute_batch_outputs`, `process_batch_helper`)
- `scripts/example_libero_robometer_wrapper.py` — 환경 루프에 리워드 모델을 붙이는 참조 구현.
  프레임 버퍼(`deque`) 관리 방식이 §2의 요구(진입 프레임 고정)와 다르므로 **그대로 복사하지 말고
  진입 시점 고정 로직으로 바꿔서** B1K용 래퍼를 새로 만들 것.

지연 예산: 4B 모델 + 16프레임이면 A100 기준 수백 ms 수준이다. 매 스텝 호출은 비현실적이므로
**N스텝마다(예: 10스텝 = 0.33초) 호출**하고 그 사이는 직전 값을 유지하는 게 기본.
캘리브레이션(§3)의 K도 이 호출 주기 기준으로 세야 한다.
