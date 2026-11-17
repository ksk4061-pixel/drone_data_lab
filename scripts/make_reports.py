"""제출물 (2) 분석 보고서와 (3) 2026-01 드론 재점검 계획을 만든다.

실행 (drone_data_lab 폴더):  python scripts/make_reports.py     (eda.py → train.py → score.py를 먼저 실행해 둔다)
보고서의 숫자는 전부 아래 파일에서 읽는다. 직접 쓴 숫자가 없고, 만든 뒤 제출 결과와 다시 대조한다.
  artifacts/eda_summary.json · artifacts/model_card.json · artifacts/feature_compare.csv
  artifacts/validation_by_month.csv · artifacts/threshold_cost.csv · reports/score_2026-01.json · reports/test_predictions_2025.csv
"""
import datetime as dt
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drone_lib import ART, FEATURE_KO, FEATURES, REPORTS  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

E = json.loads((ART / "eda_summary.json").read_text(encoding="utf-8"))
C = json.loads((ART / "model_card.json").read_text(encoding="utf-8"))
SC = json.loads((REPORTS / "score_2026-01.json").read_text(encoding="utf-8"))
COMP = pd.read_csv(ART / "feature_compare.csv", encoding="utf-8-sig")
BACK = pd.read_csv(ART / "validation_by_month.csv", encoding="utf-8-sig")
THR = pd.read_csv(ART / "threshold_cost.csv", encoding="utf-8-sig")
PRED = pd.read_csv(REPORTS / "test_predictions_2025.csv", encoding="utf-8")
P = E["patterns"]
V = C["validation"]
TD = C["threshold_detail"]
CAP = C["capacity_per_month"]


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


pct = lambda x: f"{x * 100:.1f}%"                                    # noqa: E731
num = lambda x: f"{int(x):,}"                                        # noqa: E731
limits = "\n".join(f"- {x}" for x in C["limits"])


def rate_rows(key, label_map=None):
    return [(label_map.get(k, k) if label_map else k, pct(v["rate"]), num(v["rows"])) for k, v in P[key].items()]


def extreme(key):
    d = P[key]
    hi, lo = max(d, key=lambda k: d[k]["rate"]), min(d, key=lambda k: d[k]["rate"])
    return hi, pct(d[hi]["rate"]), lo, pct(d[lo]["rate"])


# =========================================================== 분석 보고서
S, MS, TS = E["size"]["train"], E["size"]["month"], E["size"]["test"]
miss = E["missing"]
cc = E["clean_check"]
lc = E["label_check"]
cons = E["consistency"]
hi_t, hi_tr, lo_t, lo_tr = extreme("equipment_type")
hi_s, hi_sr, lo_s, lo_sr = extreme("season")
tbs = E["patterns"]["type_by_season"]
best_row = lambda k: max(tbs[k], key=tbs[k].get)                     # noqa: E731
worst_row = lambda k: min(tbs[k], key=tbs[k].get)                    # noqa: E731
comp_rows = [(r.feature_set, r.n_features, f"{r.pr_auc:.3f}", f"{r.roc_auc:.3f}", r.top20_hits,
              "선택" if r.feature_set == C["feature_choice"].split(" (")[0] else ("비교용(선택에서 제외)" if not r.selectable else "")) for r in COMP.itertuples()]
back_rows = [(r.month, r.candidates, r.picked, r.model, r.oldest, f"{r.random:.1f}") for r in BACK.itertuples()]
t3, t6 = TD["300"], TD["600"]
pol = TD["policy"]
sp = C["split"]
mean_p = PRED["prob"].mean()
heavy = max(SC["rows"], key=lambda r: (r["month_maintenance_count"], r["prob"]))     # 이번 달 정비 횟수가 가장 많은 드론
fit_range = lambda s: s.split(" (")[0]                                              # noqa: E731  "2023-01-01 ~ 2024-12-01"
fit_count = lambda s: s.split(" (")[1].split(")")[0]                                # noqa: E731  "2,223건"

analysis = f"""# 드론 예측정비 분석 보고서

- 작성: AI 에이전트 초안 · 검토: 정비 담당자 (검토 전)
- 자료: 교육용 가상 데이터 (`data/drone_train.csv` · `drone_test.csv` · `drone_this_month.csv`). 실제 부대 · 기종 정보 아님
- 규칙 원본: `AGENTS.md` · 요구사항 원본: `README.txt`
- 숫자는 모두 직접 실행한 결과에서 가져옴. 각 절 끝에 나온 코드 · 파일을 적음

## 1. 분석 목적과 조건
- 정비를 마친 드론 가운데 **30일 안에 다시 고장 날 드론**을 미리 골라 2026-01 재점검 {CAP}대를 정함
- 예측 대상: `failure_within_30days` (정비 뒤 30일 안 사후정비 · 긴급정비가 있으면 1). 예측 시점: 정비를 마친 직후
- 업무 조건(`README.txt`): 월 재점검 {CAP}대. 놓친 재고장 1건 300만원, 재점검 1대 60만원(가정)

## 2. 데이터 살펴보기
{md(['파일', '기간', '정비 기록', '드론', '부대'], [
    ('drone_train.csv (학습, 정답 있음)', f"{S['from']} ~ {S['to']}", num(S['rows']), S['drones'], S['units']),
    ('drone_test.csv (시험, 정답 없음)', f"{TS['from']} ~ {TS['to']}", num(TS['rows']), TS['drones'], TS['units']),
    ('drone_this_month.csv (이번 달)', f"{MS['from']} ~ {MS['to']}", num(MS['rows']), MS['drones'], MS['units'])])}

- 같은 드론 같은 날 중복 행: {E['duplicate_drone_date']}건. 이번 달 파일은 드론 {MS['drones']}대에 정비 기록 {MS['rows']}건으로, 드론당 정비 횟수는 {', '.join(f'{k}회 {v}대' for k, v in E['month_rows_per_drone'].items())}임
- 학습 데이터의 재고장 비율 {pct(E['base_rate'])}
- 출처: `python scripts/eda.py` → `artifacts/eda_summary.json`

### 2-1. 결측: 왜 비었는지 확인
{md(['컬럼', '학습', '시험', '이번 달', '판단'], [
    ('fault_code · fault_category', num(miss['train']['fault_category']), num(miss['test']['fault_category']), num(miss['month']['fault_category']),
     f"전부 예방정비(고장 없음). 사후 · 긴급정비의 결측은 0건 (학습: 예방정비 {num(E['fault_missing_by_type']['예방정비']['missing'])} / {num(E['fault_missing_by_type']['예방정비']['rows'])}건 결측)"),
    ('days_since_last_maintenance', num(miss['train']['days_since_last_maintenance']), 0, 0,
     f"드론별 첫 정비 기록 (학습 첫 기록 {E['days_since_missing']['first_records']}건이 모두 결측: {E['days_since_missing']['all_first']})"),
    ('availability_rate', 0, 0, num(miss['month']['availability_rate']), '이번 달 파일에는 컬럼 자체가 없음 → 모델에 쓸 수 없음')])}

### 2-2. 이상치
{md(['컬럼', '이상 값', '건수 (학습 / 시험 / 이번 달)', '판단'], [
    ('downtime_hours', '999.0', f"{E['downtime_999']['train']} / {E['downtime_999']['test']} / {E['downtime_999']['month']}",
     f"미입력 표시값. 999를 뺀 최댓값이 {E['downtime_next_max']}이라 값이 아님"),
    ('maintenance_cost', f"{min(r['cost'] for r in E['cost_over_1000']['rows']):,} ~ {max(r['cost'] for r in E['cost_over_1000']['rows']):,}",
     f"{E['cost_over_1000']['by_file']['train']} / {E['cost_over_1000']['by_file']['test']} / {E['cost_over_1000']['by_file']['month']}",
     f"자릿수 입력 오류(모두 사후정비). 정상 행 최댓값 {E['cost_over_1000']['normal_max']:.0f}. 100으로 나누면 {E['cost_fixed_range'][0]:.0f} ~ {E['cost_fixed_range'][1]:.0f}으로 사후정비 정상 범위(중앙값 {E['cost_post_normal']['median']:.0f}, 99% {E['cost_post_normal']['p99']:.0f}, 최대 {E['cost_post_normal']['max']:.0f})에 듦")])}

- 비용 오입력 행: {', '.join(f"{r['equipment_id']}({r['date']}, {r['cost']:,})" for r in E['cost_over_1000']['rows'])}
- 기록 일관성(학습 · 시험 · 이번 달을 이어서 확인): `days_since_last_maintenance`가 날짜 차이와 일치 {cons['days_since_matches_date_gap']}, 누적 비행시간 감소 {cons['cum_hours_decrease']}건, 경과연수 감소 {cons['age_decrease']}건, 시험 · 이번 달 드론 중 앞 기간 기록이 있는 비율 {cons['test_drones_with_prior_records']:.0%} · {cons['month_drones_with_prior_records']:.0%}
- 출처: `python scripts/eda.py`

### 2-3. 정답 정의 확인
- ‘같은 드론에 이 정비 뒤 30일 안 사후 · 긴급정비가 있으면 1’을 학습 기록만으로 다시 만들면 정답과 {lc['match_rate']:.2%} 일치함. 어긋난 {lc['mismatch']}건은 전부 {', '.join(lc['mismatch_months'])} 정비이고 정답이 1임. 2025년 1월 기록을 봐야 알 수 있는 재고장이어서임
- 그래서 학습 · 검증에서는 정답 확인 구간(30일)이 다음 기간에 걸치는 행을 뺌(5절)
- 출처: `python scripts/eda.py`

### 2-4. 패턴 (학습 데이터)
{md(['구분', '값', '재고장률', '건수'], [(g, k, r, n) for g, key, mp in [
    ('드론 유형', 'equipment_type', None), ('정비 구분', 'maintenance_type', None), ('고장 유형', 'fault_category', None),
    ('계절', 'season', None), ('정비 성공 여부', 'maintenance_success', {'0': '0 (실패)', '1': '1 (성공)'}),
    ('경과연수', 'age_bin', None), ('최근 90일 고장 수', 'fail_count_90d', {'4': '4 이상'})]
    for k, r, n in rate_rows(key, mp)])}

{md(['드론 유형', '봄', '여름', '가을', '겨울'], [(t, *[pct(tbs[t][s]) for s in ['봄', '여름', '가을', '겨울']]) for t in tbs])}

- 드론 유형: {hi_t} {hi_tr}로 가장 높고 {lo_t} {lo_tr}로 가장 낮음
- 계절: {hi_s} {hi_sr}, {lo_s} {lo_sr}. 유형별로는 수송드론이 봄 {pct(tbs['수송드론']['봄'])}에서 겨울 {pct(tbs['수송드론']['겨울'])}까지 차이가 가장 큼
- 정비 구분 · 고장 유형: 예방정비(고장 없음) {pct(P['maintenance_type']['예방정비']['rate'])}에 비해 사후정비 {pct(P['maintenance_type']['사후정비']['rate'])}, 긴급정비 {pct(P['maintenance_type']['긴급정비']['rate'])}. 정비가 실패한 경우 {pct(P['maintenance_success']['0']['rate'])} (성공 {pct(P['maintenance_success']['1']['rate'])})
- 최근 90일 고장이 쌓일수록 재고장률이 단계적으로 올라감 (0건 {pct(P['fail_count_90d']['0']['rate'])} → 4건 이상 {pct(P['fail_count_90d']['4']['rate'])})
- 그 달 가동률(`availability_rate`) 평균은 재고장 1인 정비 {P['availability_rate_mean']['재고장 1']}, 0인 정비 {P['availability_rate_mean']['재고장 0']}. 재고장 뒤에 낮아지는 결과 값이라 예측에 쓰면 안 됨
- 출처: `python scripts/eda.py` → `artifacts/eda_summary.json`

## 3. 데이터 처리 기준과 근거
{md(['항목', '건수', '처리', '근거'], [
    ('downtime_hours = 999', cc['downtime_to_nan'], '결측(NaN)으로 바꿈. 평균 · 중앙값으로 채우지 않음', '미입력 표시값. 평균으로 채우면 값을 지어내게 됨. 모델(LightGBM)이 결측을 처리'),
    ('maintenance_cost > 1,000', cc['cost_divided'], '100으로 나눔. 나눈 행은 `cost_fixed`=1로 표시', '자릿수 입력 오류. 나눈 값이 사후정비 정상 범위에 듦'),
    ('fault_category · fault_code 결측', num(cc['fault_category_filled']), '"N/A" 범주로 채움', '전부 예방정비(고장 없음). 최빈값으로 채우면 없는 고장 유형이 생김. fault_code(40종)는 모델에 쓰지 않음'),
    ('days_since_last_maintenance 결측', cc['is_first_maintenance'], '비워 두고 `is_first_maintenance`(1/0) 열 추가', '드론별 첫 정비. 0으로 채우면 방금 정비한 것처럼 보임')])}

- **행을 지우지 않음.** 처리 전 {num(cc['rows_before'])}건 → 처리 후 {num(cc['rows_after'])}건(학습 {num(S['rows'])} + 시험 {num(TS['rows'])} + 이번 달 {num(MS['rows'])}). 건수 대조: `AGENTS.md`의 처리 기준과 일치 = {cc['matches_agents_md']}
- 처리 뒤 `downtime_hours` 최댓값 {cc['downtime_max_after']}, `maintenance_cost` 최댓값 {cc['cost_max_after']:.0f}
- 출처: `python scripts/eda.py` (`drone_lib.clean`)

## 4. 만든 피처와 넣지 않은 컬럼
같은 드론의 **그 정비일까지의 기록**만 사용함(같은 날짜 포함, 이후 기록은 사용하지 않음). 시험 · 이번 달 피처는 학습 → 시험 → 이번 달 기록을 이어서 계산함.

{md(['피처', '정의'], [
    ('최근 30 · 90 · 365일 고장 수', '그 정비일 기준 최근 30 · 90 · 365일의 사후 · 긴급정비 수(이번 정비 포함)'),
    ('직전 고장 뒤 경과일', '직전 사후 · 긴급정비로부터 경과일 (고장 이력이 없으면 비움)'),
    ('예방정비 뒤 비행시간', '마지막 예방정비 이후 누적 비행시간 증가분'),
    ('직전 정비 뒤 비행시간', '직전 정비 이후 누적 비행시간 증가분 (첫 정비는 비움)')])}

최종 피처 {len(FEATURES)}개: {', '.join(FEATURE_KO.get(f, f) for f in FEATURES)}

**넣지 않은 컬럼과 이유**

{md(['컬럼', '이유'], list(C['excluded'].items()))}

**피처 묶음 비교**: 검증 기간에서만 비교함. 더 복잡한 묶음은 검증 PR-AUC가 0.005 이상 좋아질 때만 고름

{md(['피처 묶음', '피처 수', '검증 PR-AUC', '검증 ROC-AUC', '월 20대 적중(6개월 합)', '비고'], comp_rows)}
- 선택: {C['feature_choice']}
- 출처: `python scripts/train.py` → `artifacts/feature_compare.csv`

## 5. 나눈 기간과 성능
{md(['용도', '정비일', '건수', '비고'], [
    ('학습(검증용 모델)', sp['fit'].split(' (')[0], sp['fit'].split('(')[1].rstrip(')'), '무작위로 나누지 않고 기간으로 나눔'),
    ('제외', '2024-06', sp['dropped_between'].split(' (')[0], '정답 확인 구간(30일)이 검증 기간에 걸침'),
    ('검증', sp['valid'].split(' (')[0], sp['valid'].split('(')[1].rstrip(')'), '위험 표시 기준 · 피처 묶음 선택에 사용'),
    ('시험 예측용 모델 학습', fit_range(C['trained_on_test_model']), fit_count(C['trained_on_test_model']), '12월 2일 이후는 정답이 2025년 기록에 걸쳐 제외'),
    ('이번 달 예측용 모델 학습', fit_range(C['trained_on']), fit_count(C['trained_on']), '학습 데이터 전체')])}

**검증 기간(2024-07 ~ 12) 성능**
- PR-AUC **{V['pr_auc']}** (재고장 비율 기준선 {V['base_rate']}), ROC-AUC {V['roc_auc']}
- 매달 {CAP}대를 골랐을 때 실제 재고장 수(6개월 {V['hits']['picked']}대 중): **모델 {V['hits']['model']}대 / 경과연수 오래된 순 {V['hits']['oldest']}대 / 무작위(기대값) {V['hits']['random']}대**
- 후보는 달마다 드론별 가장 최근 정비 기록 하나씩이며, 이 가운데 확률이 높은 {CAP}대를 고름. 정확도는 보고하지 않음

{md(['월', '후보(드론)', '선정', '모델 적중', '오래된 순 적중', '무작위(기대값)'], back_rows)}

- 출처: `python scripts/train.py` → `artifacts/validation_by_month.csv`, `artifacts/model_card.json`

## 6. 위험 표시 기준
- 검증 기간에서 **‘놓친 재고장 수 × 300 + 기준 이상 건수 × 60’(만원)이 가장 작은 확률 기준**을 고름. 시험 데이터(2025)로 고르지 않음

{md(['놓친 재고장 비용', '고른 기준', '기준 이상 건수', '놓친 재고장', '총 비용(만원)'], [
    ('300만원 (선택)', f"{t3['threshold']:.2f}", num(t3['flagged']), t3['missed'], num(t3['cost_300'])),
    ('600만원 (비교)', f"{t6['threshold']:.2f}", num(t6['flagged']), t6['missed'], num(t6['cost_600']))])}

- 비용을 300 → 600만원으로 올리면 기준이 {t3['threshold']:.2f} → {t6['threshold']:.2f}까지 내려가고, 기준 이상이 {num(t3['flagged'])} → {num(t6['flagged'])}건으로 늘며 놓친 재고장은 {t3['missed']} → {t6['missed']}건으로 줆. 놓치는 비용이 클수록 더 많이 점검하는 쪽으로 기준이 움직임
- 비교 정책(검증 {num(TD['valid_rows'])}건 중 재고장 {TD['valid_failures']}건): 아무도 점검하지 않으면 300만원 기준 {num(pol['아무도 점검 안 함']['cost_300'])}만원, 전부 점검하면 {num(pol['전부 점검']['cost_300'])}만원, 기준 {t3['threshold']:.2f}로 고르면 {num(t3['cost_300'])}만원
- 출처: `python scripts/train.py` → `artifacts/threshold_cost.csv`, `artifacts/model_card.json`

## 7. 시험 기간 예측 제출
- `reports/test_predictions_2025.csv`: drone_test.csv {num(len(PRED))}건 모두에 재고장 확률을 매김(열: equipment_id, maintenance_date, prob). 확률 평균 {mean_p:.3f}, 범위 {PRED['prob'].min():.3f} ~ {PRED['prob'].max():.3f}
- 시험용 모델은 정비일 ≤ 2024-12-01 행으로 학습함. 시험 피처는 학습 기록과 이어서 계산함
- **시험(2025) 정답은 없으므로 시험 기간 PR-AUC · 적중 수는 계산하지 못함.** 정답을 기록에서 다시 만들지 않았음. 위 성능은 검증 기간 값이며, 시험 성능은 제출한 확률로 강사가 채점함
- 출처: `python scripts/train.py`

## 8. 모델의 한계 (`artifacts/model_card.json`의 limits)
{limits}

- 추가: 이번 달 예측용 모델은 정답이 있는 2024-12까지로 학습함. 정답이 없는 2025년 기록은 학습에 쓰지 못해 2025년의 변화는 반영되지 않음. **이는 교육 과제의 규칙(시험 정답은 강사 보유)에 따른 의도적 제약임**(실제 운용 시 보완 방법은 9절)
- 추가: 이 분석은 2026-01 정비 기록(교육용 시나리오) 기준이며, 확률은 ‘정비 직후부터 30일 안’의 값이라 시간이 지나면 의미가 없음. 점검 결과의 날짜는 앱을 사용한 실제 날짜라 시나리오 날짜와 다를 수 있음
- 추가: 정비를 극단적으로 자주 하는 드론이 있음({heavy['equipment_id']}는 이번 달에만 {heavy['month_maintenance_count']}회 정비, 확률 {pct(heavy['prob'])}). 이런 드론은 시간 피처 값이 학습 데이터 대부분보다 훨씬 커서 확률이 매우 높게 나올 수 있고, 정비 이력 확인이 필요함. 상세 분석은 `reports/{heavy['equipment_id']}_정비이력분석.md`

## 9. 실제 운용에서 보완할 점
1. **2025년 기록으로 다시 학습**: 이 과제에서는 하지 않았으나, 실제 부대에서 쓴다면 2025년 정답을 기록에서 다시 만들어 학습해야 함. 정답의 정의(같은 드론에 정비 뒤 30일 안 사후 · 긴급정비가 있으면 1)는 학습 기록만으로 {lc['match_rate']:.2%} 재현됨(2-3절). 마지막 30일 기록은 정답을 확정할 수 없어 제외하고, 재학습 뒤에는 검증 · 점검(B, A)과 위험 표시 기준 선택을 다시 해야 함
2. **점수의 유효 기간**: 확률은 정비 후 30일 안의 값이므로 새 정비 기록이 들어올 때마다 점수를 다시 매겨야 함
3. **비용 가정 교체**: 놓친 재고장 · 재점검 비용을 실제 값(기체 손실, 임무 차질, 2차 피해 포함)으로 바꾸고 위험 표시 기준을 다시 골라야 함(재점검 계획 6절)
4. **단순 규칙 대비 우위 점검**: 모델이 단순 규칙보다 낫다는 근거가 아직 없으므로(`reports/모델점검보고서.md` B3) 정기적으로 비교해야 함
5. **점검 결과에 점검 항목 기록**: 반복 고장 기체는 ‘이상 없음’만으로는 판단할 수 없으므로 무엇을 점검했는지 남겨야 함
6. **극단 이력 기체는 별도 판정**: 정비 횟수가 매우 많은 기체는 모델 확률보다 정비 이력 분석과 지휘 판단이 필요함
"""
(REPORTS / "분석보고서.md").write_text(analysis, encoding="utf-8")

# =========================================================== 2026-01 재점검 계획
rows = SC["rows"]
top = rows[:CAP]
rest = [r for r in rows[CAP:] if r["risk_high"]]
by_type, by_unit = {}, {}
for r in top:
    by_type[r["equipment_type"]] = by_type.get(r["equipment_type"], 0) + 1
    by_unit[r["unit_code"]] = by_unit.get(r["unit_code"], 0) + 1
multi_units = sorted(u for u, n in by_unit.items() if n >= 2)
n_600 = sum(1 for r in rows if r["prob"] >= C["threshold_600"])


def advice(r):
    tier = "즉시 재점검" if r["prob"] >= 0.7 else "우선 재점검" if r["prob"] >= 0.4 else "재점검"
    extra = []
    if "정비 성공 여부(0)" in r["reasons"]:
        extra.append("직전 정비 실패 기록 확인")
    if r["month_maintenance_count"] >= 2:
        extra.append(f"이번 달 {r['month_maintenance_count']}회 정비")
    return tier + (" · " + ", ".join(extra) if extra else "")


common = {k: sum(1 for r in top if k in r["reasons"]) for k in ["직전 고장 뒤 경과일(0)", "최근 90일 고장 수", "정비 성공 여부(0)"]}

# ---- 피드백 반영용 자료 (모두 결과 파일에서 읽거나 결과 파일로 다시 계산함)
RV = json.loads((ART / "rule_vs_model.json").read_text(encoding="utf-8")) if (ART / "rule_vs_model.json").exists() else None
CASE_P = ART / f"drone_case_{heavy['equipment_id']}.json"
CASE = json.loads(CASE_P.read_text(encoding="utf-8")) if CASE_P.exists() else None
COST_CHECK = 60                                                                   # 재점검 1대 비용(만원, README의 가정)
sens = []                                                                         # 놓친 재고장 비용별 위험 표시 기준 (검증 기간 비용 최소)
for mc in (100, 200, 300, 450, 600, 800, 1000):
    cst = THR["missed"] * mc + THR["flagged"] * COST_CHECK
    i = int(cst.idxmin())
    thr_i = float(THR.loc[i, "threshold"])
    sens.append((mc, thr_i, int(THR.loc[i, "flagged"]), int(THR.loc[i, "missed"]), sum(1 for r in rows if r["prob"] >= thr_i)))
_s = {s[0]: s[1] for s in sens}
assert abs(_s[300] - C["threshold"]) < 1e-9 and abs(_s[600] - C["threshold_600"]) < 1e-9, "민감도 재계산이 model_card와 다름"
first_maint, last_maint = min(r["maintenance_date"] for r in rows), max(r["maintenance_date"] for r in rows)
valid_until = (dt.date.fromisoformat(last_maint) + dt.timedelta(days=30)).isoformat()
tier_days = lambda p: 3 if p >= 0.7 else 7 if p >= 0.4 else 14                    # 점검 기한 제안(일). 정책 제안값이며 모델이 계산한 값이 아님  # noqa: E731
sched = {}
for r in top:
    sched.setdefault(r["unit_code"], []).append(r)
sched_rows = [(u, len(l), ", ".join(f"{r['rank']}위 {r['equipment_id']}" for r in l), f"배정 후 {min(tier_days(r['prob']) for r in l)}일 이내", "(정비반장 지정 필요)")
              for u, l in sorted(sched.items(), key=lambda kv: (min(tier_days(r['prob']) for r in kv[1]), min(r['rank'] for r in kv[1])))]
rule_in_rest = [r["equipment_id"] for r in rest if RV and r["equipment_id"] in {d["equipment_id"] for d in RV["rule_only"]}]

rv_sec = ""
if RV:
    hdr = ['드론', '모델 순위', '모델 확률', '규칙 순위', '최근 정비', '최근 90일 고장', '판단 근거']
    row_of = lambda d: (d['equipment_id'], d['model_rank'], pct(d['model_prob']), d['rule_rank'], d['maintenance_type'], d['fail_90d'], d['basis'])   # noqa: E731
    rv_sec = f"""## 4. 단순 규칙으로 뽑은 20대와의 비교
- 단순 규칙: {RV['rule']}
- 모델 점검 B3에서 이 규칙이 월 20대 적중에서 모델과 비슷하거나 약간 더 많이 맞혔으므로, 이번 달 20대도 같은 규칙으로 뽑아 모델 목록과 비교함
- **두 방법이 모두 고른 드론 {RV['n_both']}대**: {', '.join(RV['both'])}
- 모델만 고른 {len(RV['model_only'])}대와 규칙만 고른 {len(RV['rule_only'])}대는 아래 표에 한 대씩 판단 근거를 적음

**모델만 고른 드론 (규칙으로는 20위 밖)**

{md(hdr, [row_of(d) for d in RV['model_only']])}

**규칙만 고른 드론 (모델로는 20위 밖)**

{md(hdr, [row_of(d) for d in RV['rule_only']])}

- 판단 원칙: 두 방법이 모두 고른 드론을 먼저 점검하고, 한쪽만 고른 드론은 위 근거를 보고 정비반장이 점검 여부를 정함
- 모델만 고른 드론에는 최근 정비가 예방정비이지만 최근 고장이 잦은 드론이 포함됨. 규칙은 최근 정비 구분을 우선하므로 이런 드론을 뒤로 보냄
- 이번 달 결과에서도 **모델이 단순 규칙보다 낫다는 근거는 없음**(B3 한계). 모델의 가치는 확률(위험 표시 기준 · 비용 계산)에 있음
- 출처: `python scripts/compare_rule.py` → `artifacts/rule_vs_model.json`

"""

plan = f"""# 2026-01 드론 재점검 계획 (안)

- 작성: AI 에이전트 초안 · 검토: 정비 담당자 (검토 전)
- 자료: 교육용 가상 데이터. 실제 부대 · 기종 정보 아님
- 근거: `reports/score_2026-01.json` (`python scripts/score.py` 출력), `artifacts/model_card.json`

## 1. 요약
- 2026-01 정비 기록 {num(SC['maintenance_rows'])}건, 드론 {SC['drones']}대 가운데 재점검 대상 **{CAP}대**를 확률 순으로 선정함(월 재점검 가능 대수 기준)
- 위험 표시 기준({SC['threshold']:.2f}) 이상인 드론은 **{SC['above_threshold']}대**이며, {CAP}대를 넘는 **{SC['above_threshold_over_capacity']}대**는 **추가 점검**으로 처리함(5절). 이 기준은 놓친 재고장 비용 가정에 크게 좌우되므로 6절에서 근거와 한계를 밝힘
- 모델이 단순 규칙보다 낫다는 근거는 아직 없음. 단순 규칙으로 뽑은 20대와 {RV['n_both'] if RV else '—'}대가 겹치며, 다른 드론은 4절에 한 대씩 근거를 적음
- {heavy['equipment_id']}는 재점검이 아니라 **지휘 판단 대상**임(8절)
- 한 드론이 이번 달에 여러 번 정비한 경우({SC['multi_maintenance_drones']}대)는 가장 최근 정비 기록으로 판단함
- 선정 {CAP}대의 재고장 확률은 {pct(top[-1]['prob'])} ~ {pct(top[0]['prob'])}. 1위 {top[0]['equipment_id']}({pct(top[0]['prob'])})
- 출처: `python scripts/score.py`

## 2. 기준일과 확률의 유효 기간
- 이 계획은 **2026-01 정비 기록({first_maint} ~ {last_maint})을 기준으로 한 교육용 시나리오**임. 실제 부대의 계획이 아님
- 모델의 확률은 ‘정비를 마친 직후부터 30일 안에 다시 고장 날 확률’임. 정비일로부터 30일이 지나면(가장 늦은 정비 기준 {valid_until}) 이 확률은 더 이상 의미가 없음
- 앱의 점검 결과에 남는 날짜는 입력한 실제 날짜이므로 시나리오 날짜(2026-01)와 다를 수 있음. 교육용 시연에서는 그대로 두되, 실제 운용에서는 최신 정비 기록으로 점수를 다시 매겨야 함

## 3. 재점검 대상 {CAP}대
{md(['순위', '드론', '유형', '부대', '최근 정비일', '정비 구분', '확률', '위험을 올린 요인', '권고'], [
    (r['rank'], r['equipment_id'], r['equipment_type'], r['unit_code'], r['maintenance_date'], r['maintenance_type'],
     pct(r['prob']), r['reasons'], advice(r)) for r in top])}

- 권고 구분(참고용 분류): 확률 70% 이상 ‘즉시 재점검’, 40% 이상 ‘우선 재점검’, 위험 표시 기준 이상 ‘재점검’. 최종 점검 순서와 방법은 정비 담당자가 정함
- 괄호 안 숫자는 해당 요인의 실제 값이며, 위험을 올린 요인은 확률 상승에 기여가 큰 3개임
- 출처: `python scripts/score.py` 출력 표 (`reports/score_2026-01.json`)

{rv_sec}## 5. 위험 표시 기준을 넘은 나머지 {len(rest)}대의 처리
- 위험 표시 기준 {SC['threshold']:.2f} 이상 {SC['above_threshold']}대 중 선정 {CAP}대를 뺀 **{len(rest)}대**(확률 {pct(rest[-1]['prob'])} ~ {pct(rest[0]['prob'])})임

{md(['순위', '드론', '유형', '부대', '확률', '위험을 올린 요인'], [
    (r['rank'], r['equipment_id'], r['equipment_type'], r['unit_code'], pct(r['prob']), r['reasons']) for r in rest])}

**처리 방안(안): 추가 점검** — 선정 {CAP}대의 점검이 끝나는 대로 순위 순서({rest[0]['rank']}위부터)로 같은 달 안에 점검함
1. 확률이 선정 {CAP}대의 최저({pct(top[-1]['prob'])})와 이어지는 경계 구간이라 순위 차이가 작음
2. 놓친 재고장 비용을 600만원으로 가정하면 기준이 {C['threshold_600']:.2f}까지 낮아져 이 {len(rest)}대는 기준을 크게 넘음(이번 달 기준 이상 {n_600}대, 6절)
3. 다음 달로 미루면 정비일로부터 30일이 지나 확률이 의미를 잃고 점수를 다시 매겨야 함(2절)
4. **비행 제한은 건의하지 않음**: 임무 영향 · 대체 기체 정보가 없음. 운용 영향이 큰 경우의 비행 제한은 지휘관이 판단함
- 점검 여력이 부족하면 이 {len(rest)}대 중 단순 규칙도 고른 {len(rule_in_rest)}대({', '.join(rule_in_rest) or '없음'})를 먼저 점검함
- 출처: `reports/score_2026-01.json` · `artifacts/rule_vs_model.json`

## 6. 비용 가정의 근거와 한계
- 놓친 재고장 300만원 · 재점검 60만원은 실습 지침(`README.txt`)이 정한 가정이며, 실제 비용에서 나온 값이 아님
- 드론이 비행 중 추락하면 기체 손실 · 임무 차질 · 2차 피해가 따름. 이는 반영되어 있지 않아 **300만원은 실제보다 낮을 수 있음**
- 위험 표시 기준은 이 가정에 크게 좌우됨. 아래는 같은 검증 기간(2024-07 ~ 12)에서 비용만 바꿔 다시 계산한 값임(재점검 비용 60만원 고정)

{md(['놓친 재고장 1건 비용(만원)', '위험 표시 기준', '기준 이상 건수(검증)', '놓친 재고장(검증)', '이번 달 기준 이상 드론'], [
    (s[0] if s[0] not in (300, 600) else f"{s[0]} ({'현재 가정' if s[0] == 300 else '비교'})", f"{s[1]:.2f}", s[2], s[3], f"{s[4]}대") for s in sens])}

- 비용이 클수록 기준이 낮아져 더 많은 드론을 점검 대상으로 봄. 점검 여력(월 {CAP}대)이 고정이면 기준을 낮춘 만큼 점검이 부족해짐
- 실제 운용에서는 (기체 손실 + 임무 차질 + 2차 피해) 추정값을 정비 · 재정 부서에서 받아 입력해야 함. 앱의 ‘④ 위험 표시 기준’ 탭에서 슬라이더로 바로 다시 계산할 수 있음
- 출처: `artifacts/threshold_cost.csv`(검증 기간)로 재계산, 300 · 600만원 값은 `artifacts/model_card.json`과 일치함을 확인함

## 7. 점검 일정(제안)과 남은 대상
- 점검 완료 현황은 앱의 ‘② 점검 결과’ 탭이 실시간으로 보여 줌. 이 문서는 점검 진행 상태를 담지 않음
- 아래는 재점검 대상 {CAP}대 전체의 점검 기한 **제안**임. 기한은 정책 제안값이며 정비반장이 조정함(확률 70% 이상은 배정 후 3일, 40% 이상은 7일, 그 외는 14일 이내)
- 인력을 줄이기 위해 부대별로 묶었음. **담당 반은 반 편제 · 인원 정보가 없어 정하지 않았으며** 정비반장이 지정해야 함

{md(['부대', '대상 수', '대상 드론(순위)', '점검 기한(제안)', '담당 반'], sched_rows)}

## 8. 특이 드론: {heavy['equipment_id']} (지휘 판단 대상)
{(f"- 정비 {CASE['rows']}건 중 고장 정비 {CASE['fails']}건({pct(CASE['fail_ratio'])}), 직전 365일 고장 정비 {CASE['recent365']}건(같은 유형 중앙값 {CASE['recent365_median_type']:g}건)" + chr(10) +
  f"- 고장 정비 뒤 30일 안 재고장 비율 {pct(CASE['repeat30'])}(나머지 드론 {pct(CASE['repeat30_fleet'])}), 고장 간격 중앙값 {CASE['gap_median']:g}일(나머지 드론 {CASE['gap_median_fleet']:g}일)" + chr(10) +
  f"- 이번 달 정비 {heavy['month_maintenance_count']}회, 확률 {pct(heavy['prob'])}") if CASE else '- 상세 분석 파일이 없음(`python scripts/analyze_drone.py` 실행 필요)'}
- 이 드론은 일반적인 ‘재점검’이 아니라 **운용 중지 · 창정비 의뢰 · 도태** 중에서 지휘관이 판단해야 하는 기체임. 근본 원인 분석과 선택지 비교는 `reports/{heavy['equipment_id']}_정비이력분석.md`에 있음
- 건의안: 운용 중지 후 정밀 진단(창정비 의뢰 검토). 도태는 교체 비용 · 잔존가치 · 대체 기체 정보가 있어야 판단할 수 있음
- 모델 확률(30일 안 재고장 가능성)과 점검 결과(‘이상 없음’)는 서로 다른 질문이므로 점검에서 이상이 없어도 확률이 낮아지지 않음. 반복 고장 기체는 점검 항목이 증상을 놓쳤을 가능성도 확인해야 함

## 9. 눈여겨볼 점
- 선정 {CAP}대의 드론 유형: {', '.join(f'{k} {v}대' for k, v in sorted(by_type.items(), key=lambda x: -x[1]))}
- 2대 이상 선정된 부대: {', '.join(f"{u}({by_unit[u]}대)" for u in multi_units)}. 점검 인력을 묶어서 배정할 때 참고할 수 있음
- 공통 요인: {CAP}대 중 ‘직전 고장 뒤 경과일(0)’(정비일에 고장 정비)이 위험 요인으로 나타난 드론 {common['직전 고장 뒤 경과일(0)']}대, ‘최근 90일 고장 수’ {common['최근 90일 고장 수']}대, ‘정비 성공 여부(0)’ {common['정비 성공 여부(0)']}대
- 정비 횟수가 가장 많은 드론은 {heavy['equipment_id']}(이번 달 {heavy['month_maintenance_count']}회 정비, {heavy['rank']}위, 확률 {pct(heavy['prob'])})임. {'최근 365일 고장 수가 위험 요인으로 나타남. ' if '최근 365일 고장 수' in heavy['reasons'] else ''}같은 고장이 반복되는 것이 아닌지 정비 이력 확인이 필요함
- 집계 값(유형 · 부대 · 공통 요인 대수)은 `reports/score_2026-01.json`의 상위 {CAP}대를 센 값이며 스크립트가 직접 출력한 숫자는 아님

## 10. 모델 성능 요약
- 검증(2024-07 ~ 12) PR-AUC {V['pr_auc']} (재고장 비율 기준선 {V['base_rate']})
- 매달 {CAP}대를 골랐을 때 실제 재고장 수(6개월 합): 모델 {V['hits']['model']}대 / 경과연수 오래된 순 {V['hits']['oldest']}대 / 무작위 {V['hits']['random']}대
- 출처: `artifacts/model_card.json` (`python scripts/train.py`). 자세한 방법은 `reports/분석보고서.md`

## 11. 모델의 한계 (`artifacts/model_card.json`의 limits)
{limits}

- 추가: 이번 달 예측용 모델은 정답이 있는 2024-12까지로 학습함. 2025년 기록은 정답이 없어 쓰지 못함. 이는 교육 과제의 규칙(시험 정답은 강사 보유)에 따른 **의도적 제약**이며, 실제 운용에서는 2025년 기록에서 정답을 재구성해 다시 학습해야 함(분석 보고서 9절)
- 추가: 이 계획은 2026-01 정비 기록 기준의 교육용 시나리오이며 확률은 정비 후 30일 안의 값이므로 시간이 지나면 의미가 없음(2절)
"""
(REPORTS / "2026-01_드론_재점검계획.md").write_text(plan, encoding="utf-8")


# =========================================================== 보고서 숫자 ↔ 제출 결과 대조
def verify():
    text = (REPORTS / "2026-01_드론_재점검계획.md").read_text(encoding="utf-8")
    truth = {r["equipment_id"]: round(r["prob"] * 100, 1) for r in rows}
    pairs = re.findall(r"\| (DR-\d{4}) \|[^\n]*?\| (\d{1,3}\.\d)% \|", text)
    wrong = [(e, p, truth.get(e)) for e, p in pairs if truth.get(e) is None or abs(truth[e] - float(p)) > 0.05]
    raw = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "drone_test.csv", encoding="utf-8-sig")
    keys_ok = set(zip(PRED["equipment_id"], PRED["maintenance_date"])) == set(zip(raw["equipment_id"], raw["maintenance_date"].astype(str)))
    print(f"대조: 재점검 계획의 드론 확률 {len(pairs)}개 중 스킬 출력과 다른 것 {len(wrong)}개 {wrong[:3]}")
    print(f"대조: 시험 예측 {len(PRED):,}건 = drone_test.csv {len(raw):,}건, (드론, 정비일) 키 일치 {keys_ok}, prob 범위 {PRED['prob'].min():.3f} ~ {PRED['prob'].max():.3f}")
    return not wrong and keys_ok and len(PRED) == 1241


ok = verify()
print("저장: reports/분석보고서.md, reports/2026-01_드론_재점검계획.md")
print("대조 결과:", "통과" if ok else "불일치 있음")
