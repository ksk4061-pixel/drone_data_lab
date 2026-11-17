"""특이 드론 한 대의 정비 이력을 근본 원인 중심으로 분석하고, 운용 중지 · 창정비 의뢰 · 도태 판단 자료를 만든다.

실행 (drone_data_lab 폴더, data/ 필요):  python scripts/analyze_drone.py            (기본 DR-0115)
                                         python scripts/analyze_drone.py --drone DR-0017
출력: 화면 + artifacts/drone_case_<번호>.json + reports/<번호>_정비이력분석.md
이 분석은 정비 기록(고장 유형 · 심각도 · 부품 · 비용 · 수리 성공 여부)만 본다. 부품 상태 · 운용 환경 · 임무 부하 · 창정비 비용 · 잔존가치는 데이터에 없다.
기록에서 고장 간격을 세는 것은 이 드론의 이력을 설명하기 위한 것이며 모델 학습 · 시험 평가에는 쓰지 않는다.
최종 결정(운용 중지 · 창정비 · 도태)은 지휘관이 하며, 이 문서는 건의안이다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drone_lib import ART, REPORTS, TARGET, prepare  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FAIL = ["사후정비", "긴급정비"]


def md(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def pct_rank(series, value):
    """series에서 value 이하인 비율(%)."""
    return float((series <= value).mean() * 100)


def fail_gaps(g):
    """고장 정비끼리의 간격(일)."""
    f = g[g["maintenance_type"].isin(FAIL)].sort_values("maintenance_date")["maintenance_date"]
    return f.diff().dt.days.dropna()


def repeat_within_30(g):
    """각 고장 정비 뒤 30일 안에 또 고장 정비가 있었는가 (마지막 고장 정비는 제외)."""
    f = g[g["maintenance_type"].isin(FAIL)].sort_values("maintenance_date")
    nxt = f["maintenance_date"].shift(-1) - f["maintenance_date"]
    ok = nxt.notna()
    return pd.DataFrame({"success": f.loc[ok, "maintenance_success"], "again": (nxt[ok].dt.days <= 30)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone", default="DR-0115")
    did = ap.parse_args().drone
    df = prepare()
    df["maintenance_date"] = pd.to_datetime(df["maintenance_date"])
    d = df[df["equipment_id"] == did].sort_values("maintenance_date")
    if d.empty:
        sys.exit(f"{did} 기록이 없습니다.")
    F = d[d["maintenance_type"].isin(FAIL)]
    dtype, unit = d["equipment_type"].iloc[0], d["unit_code"].iloc[0]
    last = d.iloc[-1]
    last_date = last["maintenance_date"]

    # ---------------------------------------------------------------- 플릿 비교
    # 비교 대상은 이 드론을 뺀 나머지 드론이다(이 드론의 기록이 평균을 끌어올리지 않게).
    others = df[df["equipment_id"] != did]
    last_rows = others.sort_values("maintenance_date").groupby("equipment_id").tail(1).set_index("equipment_id")
    r365_all, r365_type = last_rows["fail_count_365d"], last_rows[last_rows["equipment_type"] == dtype]["fail_count_365d"]
    cnt = others.groupby("equipment_id").agg(rows=("maintenance_type", "size"), fails=("maintenance_type", lambda s: s.isin(FAIL).sum()))
    cnt["ratio"] = cnt["fails"] / cnt["rows"]
    gaps_d, gaps_all = fail_gaps(d), pd.concat([fail_gaps(g) for _, g in others.groupby("equipment_id")])
    rep_d = repeat_within_30(d)
    rep_all = pd.concat([repeat_within_30(g) for _, g in others.groupby("equipment_id")])
    Fall = others[others["maintenance_type"].isin(FAIL)]

    S = {"id": did, "type": dtype, "unit": unit, "rows": len(d), "first": f"{d['maintenance_date'].min():%Y-%m-%d}", "last": f"{last_date:%Y-%m-%d}",
         "by_type": d["maintenance_type"].value_counts().to_dict(), "fails": len(F), "fail_ratio": round(len(F) / len(d), 3),
         "fleet_fail_ratio_median": round(float(cnt["ratio"].median()), 3), "fleet_rows_median": float(cnt["rows"].median()),
         "recent365": int(last["fail_count_365d"]), "recent365_pct_all": round(pct_rank(r365_all, last["fail_count_365d"]), 1),
         "recent365_pct_type": round(pct_rank(r365_type, last["fail_count_365d"]), 1),
         "recent365_median_all": float(r365_all.median()), "recent365_median_type": float(r365_type.median()),
         "age": float(last["equipment_age_years"]), "hours": float(last["cumulative_operating_hours"]),
         "category": F["fault_category"].value_counts().to_dict(), "codes_top": F["fault_code"].value_counts().head(5).to_dict(),
         "gap_median": float(gaps_d.median()), "gap_median_fleet": float(gaps_all.median()),
         "gap_le7_share": round(float((gaps_d <= 7).mean()), 3), "gap_le7_share_fleet": round(float((gaps_all <= 7).mean()), 3),
         "repeat30": round(float(rep_d["again"].mean()), 3), "repeat30_fleet": round(float(rep_all["again"].mean()), 3),
         "repeat30_success1": round(float(rep_d[rep_d["success"] == 1]["again"].mean()), 3), "repeat30_n": int(len(rep_d)),
         "success_rate": round(float(F["maintenance_success"].mean()), 3), "success_rate_fleet": round(float(Fall["maintenance_success"].mean()), 3),
         "severity": round(float(F["severity"].mean()), 2), "severity_fleet": round(float(Fall["severity"].mean()), 2),
         "parts": int(F["parts_replaced_count"].sum()), "repair_h": round(float(F["repair_duration_hours"].sum()), 1),
         "downtime_h": round(float(F["downtime_hours"].sum()), 1), "cost": round(float(d["maintenance_cost"].sum()), 0),
         "cost_per_fail": round(float(F["maintenance_cost"].mean()), 1), "cost_per_fail_fleet": round(float(Fall["maintenance_cost"].mean()), 1)}

    yearly = (d.assign(year=d["maintenance_date"].dt.year).pivot_table(index="year", columns="maintenance_type", values="severity", aggfunc="size", fill_value=0)
              .reindex(columns=["예방정비", "사후정비", "긴급정비"], fill_value=0))
    S["yearly"] = {int(y): {k: int(v) for k, v in r.items()} for y, r in yearly.iterrows()}
    months = d.assign(m=d["maintenance_date"].dt.to_period("M").astype(str))
    last13 = [str(p) for p in pd.period_range(end=last_date.to_period("M"), periods=13, freq="M")]
    S["monthly_fail"] = {m: int(((months["m"] == m) & months["maintenance_type"].isin(FAIL)).sum()) for m in last13}
    S["month_rows_last"] = int((months["m"] == last13[-1]).sum())

    # 부대 효과 vs 개체 효과 (학습 데이터의 정답 비율)
    tr = df[df["_src"] == "train"].assign(**{TARGET: lambda x: x[TARGET].astype(int)})
    u = tr[tr["unit_code"] == unit]
    S["unit_rate"], S["unit_rate_wo"] = round(float(u[TARGET].mean()), 3), round(float(u[u["equipment_id"] != did][TARGET].mean()), 3)
    S["unit_rows"], S["unit_rows_drone"] = len(u), int((u["equipment_id"] == did).sum())
    S["train_rate"] = round(float(tr[TARGET].mean()), 3)
    (ART / f"drone_case_{did}.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- 보고서
    top_cat = max(S["category"], key=S["category"].get)
    pr = lambda x: f"{x * 100:.1f}%"                                                                    # noqa: E731
    persist = S["repeat30"] > S["repeat30_fleet"]
    text = f"""# {did} 정비 이력 분석 (근본 원인 중심) — 지휘 판단 자료

- 작성: AI 에이전트 초안 · 검토: 정비 담당자 (검토 전)
- 자료: 교육용 가상 데이터. {dtype} · {unit}, 기록 {S['first']} ~ {S['last']}
- 이 문서는 **건의안**이며 운용 중지 · 창정비 의뢰 · 도태의 결정은 지휘관이 함
- 출처: `python scripts/analyze_drone.py` → `artifacts/drone_case_{did}.json`

## 1. 결론
- {did}는 3년간 정비 {S['rows']}건 중 {S['fails']}건({pr(S['fail_ratio'])})이 고장 정비임. 나머지 드론의 고장 정비 비율 중앙값은 {pr(S['fleet_fail_ratio_median'])}임(비교 기준은 모두 이 드론을 뺀 나머지 드론)
- 기준일(마지막 기록) 직전 365일의 고장 정비는 **{S['recent365']}건**으로 같은 유형({dtype}) 중앙값 {S['recent365_median_type']:g}건, 나머지 드론 중앙값 {S['recent365_median_all']:g}건보다 훨씬 많음(유형 내 {S['recent365_pct_type']:.0f}백분위, 나머지 전체 {S['recent365_pct_all']:.0f}백분위)
- 고장 정비 후 30일 안에 다시 고장 정비가 난 비율은 {pr(S['repeat30'])}(나머지 드론 {pr(S['repeat30_fleet'])}). 수리 성공으로 기록된 정비 뒤에도 {pr(S['repeat30_success1'])}가 재고장임{' → 현장 수리로 해결되지 않고 반복되는 패턴' if persist else ''}
- 마지막 달({last13[-1]})에 정비를 {S['month_rows_last']}번 받음
- **건의: 임무에서 배제(운용 중지)하고 현장 수리가 아닌 정밀 진단(창정비 의뢰 검토)으로 전환**. 도태는 데이터만으로 결정할 수 없고(교체 비용 · 잔존가치 · 대체 기체 정보 없음) 지휘관 판단 사항임(8절)

## 2. 정비 이력 개요
{md(['구분', '건수'], [(k, v) for k, v in S['by_type'].items()])}

{md(['연도', '예방정비', '사후정비', '긴급정비'], [(y, v['예방정비'], v['사후정비'], v['긴급정비']) for y, v in S['yearly'].items()])}

최근 13개월의 월별 고장 정비 수: {', '.join(f"{m} {n}건" for m, n in S['monthly_fail'].items())}

## 3. 어떤 고장이 반복되는가
{md(['고장 유형', '건수'], [(k, v) for k, v in S['category'].items()])}

- 가장 많은 유형은 **{top_cat}**({S['category'][top_cat]}건). 자주 나온 고장 코드: {', '.join(f"{k} {v}건" for k, v in S['codes_top'].items())}
- 고장 정비 사이의 간격 중앙값은 {S['gap_median']:g}일(나머지 드론 {S['gap_median_fleet']:g}일), 간격이 7일 이하인 비율은 {pr(S['gap_le7_share'])}(나머지 드론 {pr(S['gap_le7_share_fleet'])})

## 4. 수리가 효과가 있었는가
{md(['지표', f'{did}', '나머지 드론'], [
    ('고장 정비의 수리 성공 비율', pr(S['success_rate']), pr(S['success_rate_fleet'])),
    (f"고장 정비 뒤 30일 안 재고장 비율(표본 {S['repeat30_n']}건)", pr(S['repeat30']), pr(S['repeat30_fleet'])),
    ('수리 성공 정비 뒤 30일 안 재고장 비율', pr(S['repeat30_success1']), '—'),
    ('고장 정비 평균 심각도(0~5)', S['severity'], S['severity_fleet']),
    ('고장 정비 평균 비용(만원)', S['cost_per_fail'], S['cost_per_fail_fleet'])])}

- 누적: 고장 정비 {S['fails']}건에서 교체 부품 {S['parts']}개, 수리 시간 {S['repair_h']}시간, 운용중단 {S['downtime_h']}시간, 정비 비용 합계 {S['cost']:,.0f}만원(전체 기록 기준)
- {'수리 후 재고장이 나머지 드론보다 훨씬 잦아, 개별 수리가 원인을 제거하지 못했을 가능성이 있음(표본이 작은 비교이므로 단정하지 않음)' if persist else '수리 후 재고장 비율이 나머지 드론과 비슷해 수리 효과가 특별히 낮다고 보기 어려움'}

## 5. 같은 유형 · 부대와 비교
- 학습 데이터에서 {unit}의 재고장 비율은 {pr(S['unit_rate'])}임. 이 드론을 뺀 {unit}은 {pr(S['unit_rate_wo'])}이며 전체 평균은 {pr(S['train_rate'])}임
- {unit}의 학습 기록 {S['unit_rows']}건 중 {S['unit_rows_drone']}건이 이 드론의 기록임. {'부대 전체의 문제라기보다 이 **드론 개체의 문제**일 가능성이 큼' if S['unit_rate'] - S['unit_rate_wo'] > 0.05 else '부대 수준의 요인도 함께 봐야 함'}
- 누적 비행시간 {S['hours']:,.0f}시간, 경과연수 {S['age']:.1f}년

## 6. 근본 원인에 대해 데이터가 보여 주는 것과 알 수 없는 것
**보여 주는 것**
- 고장이 {top_cat}({S['category'][top_cat] / S['fails'] * 100:.0f}%)에 가장 많고, 모든 고장 유형({len(S['category'])}종)에서 고장이 반복됨
- 수리 후 재고장이 {'나머지 드론보다 잦음' if persist else '나머지 드론과 비슷함'}
- 이상 징후는 이 드론 개체에 집중되어 있고 부대 전체의 일반적 현상이 아님(5절)

**알 수 없는 것 (현장 확인 필요)**
- 부품 · 구조의 실제 상태, 운용 환경(기후 · 지형), 임무 부하, 정비 품질(같은 부품을 반복 교체하는지), 창정비 비용 · 기간, 잔존수명
- 위 기록에는 점검 항목이 없어 ‘무엇을 점검했는가’를 알 수 없음. 근본 원인은 부품별 진단(분해 점검)으로만 확인할 수 있음

## 7. 모델 확률과 점검 결과의 차이
- 모델 확률은 ‘정비를 마친 직후부터 30일 안에 **다시 고장 정비가 날 가능성**’이며, 현장의 점검 결과(이상 없음)는 **점검 시점의 상태**임. 서로 다른 질문이므로 “이상 없음”이 재고장 확률을 낮추지 않음
- 다만 이 드론처럼 고장이 짧은 간격으로 반복되는 경우, 점검에서 이상이 안 보이는 것은 **점검 항목이 증상을 포착하지 못했을 가능성**을 의미함. 점검 결과에 ‘점검 항목’을 함께 기록하도록 보완이 필요함
- 앱에 입력된 {did}의 ‘이상 없음’ 점검 기록은 같은 시각에 거의 같은 기록이 중복되어 있어 입력 오류 또는 시험 입력일 가능성이 있음. 실제 현장 점검 결과인지 확인한 뒤에 판단해야 함

## 8. 선택지 비교와 건의 (결정은 지휘관)
{md(['선택지', '이 자료가 보여 주는 근거', '결정에 더 필요한 정보', '비고'], [
    ('운용 중지(임무 배제)', f"최근 365일 고장 정비 {S['recent365']}건, 수리 후에도 반복, 마지막 달 정비 {S['month_rows_last']}회", '대체 기체 가용성, 해당 기체의 임무 중요도', '위험을 즉시 낮추고 되돌리기 쉬움. 건의'),
    ('창정비 의뢰(정밀 진단 · 개조)', f"{top_cat} 고장이 가장 많고 수리 후에도 재고장이 반복되는 패턴", '창정비 비용 · 기간, 기체의 보증 · 잔존수명', '운용 중지와 함께 진행하는 것을 건의'),
    ('도태(퇴역)', f"누적 비행시간 {S['hours']:,.0f}시간, 경과연수 {S['age']:.1f}년, 누적 정비 비용 {S['cost']:,.0f}만원", '교체 비용, 잔존가치, 대체 기체, 정비 이력의 비용 효과 분석', '데이터만으로 결정할 수 없음. 지휘관 판단')])}

**건의안**: 운용 중지 → 정밀 진단(창정비 의뢰 검토) → 진단 결과와 비용을 보고 수리 · 개조 · 도태를 결정. 도태는 위 ‘추가로 필요한 정보’가 확보된 뒤에 판단함

## 9. 한계
- 정비 기록만으로 본 분석이며 부품 단위 원인은 알 수 없음
- 같은 드론의 고장 간격 · 재고장 비율은 표본이 작은 기술 통계임(재고장 비율 표본 {S['repeat30_n']}건)
- 이 문서는 AI 에이전트가 작성한 초안이며 결정은 사람이 함
"""
    (REPORTS / f"{did}_정비이력분석.md").write_text(text, encoding="utf-8")
    print(json.dumps({k: S[k] for k in ("rows", "fails", "fail_ratio", "fleet_fail_ratio_median", "recent365", "recent365_median_type", "recent365_pct_type", "repeat30", "repeat30_fleet",
                                         "repeat30_success1", "success_rate", "success_rate_fleet", "unit_rate", "unit_rate_wo", "unit_rows", "unit_rows_drone", "gap_median", "gap_median_fleet", "category", "codes_top")},
                     ensure_ascii=False, indent=1))
    print(f"\n저장: artifacts/drone_case_{did}.json, reports/{did}_정비이력분석.md")


if __name__ == "__main__":
    main()
