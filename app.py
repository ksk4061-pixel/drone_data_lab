"""드론 예측정비 결과 앱 — 분석 결과 파일(reports/ · artifacts/)을 읽어 보여 준다.

실행 (drone_data_lab 폴더):  streamlit run app.py
이 앱은 모델을 다시 학습하지 않고 숫자를 새로 계산하지 않는다. 숫자는 아래 파일에서 읽는다.
  reports/score_2026-01.json (python scripts/score.py) · artifacts/model_card.json · validation_by_month.csv
  threshold_cost.csv · feature_compare.csv (python scripts/train.py) · reports/*.md (python scripts/make_reports.py)
데이터는 교육용 가상 데이터다.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS, ART = ROOT / "reports", ROOT / "artifacts"
SCORE, CARD = REPORTS / "score_2026-01.json", ART / "model_card.json"
BACK, THR, COMP = ART / "validation_by_month.csv", ART / "threshold_cost.csv", ART / "feature_compare.csv"
COST_CHECK = 60                                   # 재점검 1대 비용(만원, 교육용 가정)

st.set_page_config(page_title="드론 예측정비", page_icon="🛠️", layout="wide")
st.title("🛠️ 드론 예측정비 — 2026-01 재점검 계획")
st.caption("교육용 가상 데이터입니다. 실제 부대 · 기종 정보가 아닙니다. 최종 판단은 정비 담당자가 합니다.")

missing = [p.name for p in (SCORE, CARD, BACK, THR, COMP) if not p.exists()]
if missing:
    st.error("결과 파일이 없습니다: " + ", ".join(missing) + "  → python scripts/train.py, python scripts/score.py 를 먼저 실행하세요.")
    st.stop()


@st.cache_data
def load():
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    card = json.loads(CARD.read_text(encoding="utf-8"))
    rows = pd.DataFrame(score["rows"])
    return (score, card, rows, pd.read_csv(BACK, encoding="utf-8-sig"), pd.read_csv(THR, encoding="utf-8-sig"),
            pd.read_csv(COMP, encoding="utf-8-sig"))


score, card, rows, back, thr_tab, comp = load()
cap0, thr0 = card["capacity_per_month"], float(card["threshold"])

with st.sidebar:
    st.header("설정")
    cap = st.slider("이번 달 재점검 가능 대수", 5, 40, cap0)
    thr = st.slider("위험 표시 기준(확률)", 0.05, 0.95, thr0, 0.01)
    st.caption(f"모델: {card['model']} · 학습 {card['trained_on']}")

rows = rows.assign(재점검=rows["rank"] <= cap, 위험=rows["prob"].ge(thr).map({True: "높음", False: "-"}), **{"확률(%)": (rows["prob"] * 100).round(1)})

t1, t2, t3, t4, t5 = st.tabs(["① 재점검 목록", "② 검증 성능", "③ 위험 표시 기준", "④ 보고서", "⑤ 모델 정보"])

with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"이번 달 정비 드론 (정비 기록 {score['maintenance_rows']}건)", f"{len(rows)}대")
    c2.metric("위험 높음 (기준 이상)", f"{int(rows['prob'].ge(thr).sum())}대")
    c3.metric("재점검 대상", f"{min(cap, len(rows))}대")
    f1, f2 = st.columns(2)
    types = f1.multiselect("드론 유형", sorted(rows["equipment_type"].unique()))
    units = f2.multiselect("부대", sorted(rows["unit_code"].unique()))
    mask = pd.Series(True, index=rows.index)                       # 필터를 고르지 않으면 모든 행을 보여 준다
    if types:
        mask &= rows["equipment_type"].isin(types)
    if units:
        mask &= rows["unit_code"].isin(units)
    view = rows[mask]
    show = view[["rank", "재점검", "equipment_id", "equipment_type", "unit_code", "maintenance_date", "maintenance_type",
                 "확률(%)", "위험", "reasons"]].rename(columns={
        "rank": "순위", "equipment_id": "드론", "equipment_type": "유형", "unit_code": "부대", "maintenance_date": "최근 정비일",
        "maintenance_type": "정비 구분", "reasons": "위험을 올린 요인"})
    st.dataframe(show, hide_index=True, width="stretch", height=460)
    st.download_button("재점검 목록 받기 (CSV)", show[show["재점검"]].to_csv(index=False).encode("utf-8-sig"),
                       file_name="재점검_목록.csv", mime="text/csv")
    st.subheader("부대별 요약")
    unit = (rows.assign(_r=rows["위험"].eq("높음").astype(int), _c=rows["재점검"].astype(int)).groupby("unit_code")
            .agg(**{"정비 드론 수": ("equipment_id", "nunique"), "재점검 대상 수": ("_c", "sum"), "위험 높음 수": ("_r", "sum")})
            .reset_index().rename(columns={"unit_code": "부대"})
            .sort_values(["재점검 대상 수", "위험 높음 수", "부대"], ascending=[False, False, True]))
    st.dataframe(unit, hide_index=True, width="stretch")
    st.write(f"재점검 대상 합계: **{int(unit['재점검 대상 수'].sum())}대** ({len(unit)}개 부대, 정비 드론 {int(unit['정비 드론 수'].sum())}대 중)")

with t2:
    v = card["validation"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("검증 PR-AUC", v["pr_auc"], f"기준선 {v['base_rate']}")
    m2.metric("월 20대 적중 — 모델", v["hits"]["model"])
    m3.metric("오래된 순", v["hits"]["oldest"])
    m4.metric("무작위(기대값)", v["hits"]["random"])
    st.write(f"검증 기간(2024-07 ~ 12) 6개월, 매달 {cap0}대를 골랐을 때 실제 재고장 수. 후보는 달마다 드론별 가장 최근 정비 기록입니다.")
    st.bar_chart(back.rename(columns={"model": "모델", "oldest": "오래된 순", "random": "무작위(기대값)"}),
                 x="month", y=["모델", "오래된 순", "무작위(기대값)"])
    st.dataframe(back, hide_index=True, width="stretch")
    st.write("**피처 묶음 비교** (검증 기간)")
    st.dataframe(comp, hide_index=True, width="stretch")
    st.caption(f"선택: {card['feature_choice']}")
    st.caption("시험(2025) 정답이 없어 시험 성능은 계산하지 못했습니다. 위 값은 검증 기간 기준입니다.")

with t3:
    st.write("검증 기간에서 **놓친 재고장 수 × 놓친 비용 + 기준 이상 건수 × 60**(만원)이 가장 작은 확률 기준을 고릅니다. 놓친 비용을 바꿔 보세요.")
    miss_cost = st.slider("놓친 재고장 1건 비용(만원)", 100, 1000, 300, 50)
    tab = thr_tab.assign(cost=thr_tab["missed"] * miss_cost + thr_tab["flagged"] * COST_CHECK)
    best = tab.loc[tab["cost"].idxmin()]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("고른 기준", f"{best['threshold']:.2f}")
    b2.metric("기준 이상 건수", int(best["flagged"]))
    b3.metric("놓친 재고장", int(best["missed"]))
    b4.metric("총 비용(만원)", f"{int(best['cost']):,}")
    st.line_chart(tab, x="threshold", y="cost")
    st.caption("출처: artifacts/threshold_cost.csv (python scripts/train.py). 재점검 비용 60만원은 고정입니다.")

with t4:
    r1, r2 = st.tabs(["분석 보고서", "2026-01 재점검 계획"])
    for tab_, name in ((r1, "분석보고서.md"), (r2, "2026-01_드론_재점검계획.md")):
        with tab_:
            p = REPORTS / name
            st.markdown(p.read_text(encoding="utf-8") if p.exists() else f"`{name}` 파일이 없습니다.")

with t5:
    st.write("**예측 대상**:", card["target"])
    st.write("**위험 표시 기준 고른 방법**:", card["threshold_rule"])
    s = card["split"]
    st.write(f"**기간 분할**: 학습 {s['fit']}, 검증 {s['valid']}, 제외 {s['dropped_between']}")
    st.write("**뺀 컬럼**")
    st.table(pd.DataFrame(card["excluded"].items(), columns=["컬럼", "이유"]))
    st.write("**한계**")
    for lim in card["limits"]:
        st.write("- " + lim)
