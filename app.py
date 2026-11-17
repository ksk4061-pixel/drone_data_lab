"""드론 예측정비 결과 앱 — 분석 결과 파일(reports/ · artifacts/)을 읽어 보여 준다.

실행 (drone_data_lab 폴더):  streamlit run app.py
이 앱은 모델을 다시 학습하지 않고 숫자를 새로 계산하지 않는다. 숫자는 아래 파일에서 읽는다.
  reports/score_2026-01.json (python scripts/score.py) · artifacts/model_card.json · validation_by_month.csv
  threshold_cost.csv · feature_compare.csv (python scripts/train.py) · reports/*.md (python scripts/make_reports.py)
데이터는 교육용 가상 데이터다.
'점검 결과' 탭만 Supabase(Postgres)에 기록을 저장한다 (store.py, 연결 주소는 secrets에만 둔다).
"""
import datetime as dt
import hmac
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import store

ROOT = Path(__file__).resolve().parent
REPORTS, ART = ROOT / "reports", ROOT / "artifacts"
SCORE, CARD = REPORTS / "score_2026-01.json", ART / "model_card.json"
BACK, THR, COMP = ART / "validation_by_month.csv", ART / "threshold_cost.csv", ART / "feature_compare.csv"
COST_CHECK = 60                                   # 재점검 1대 비용(만원, 교육용 가정)
REPORT_FILES = {"분석 보고서": "분석보고서", "2026-01 재점검 계획": "2026-01_드론_재점검계획"}   # reports/ 안의 파일 이름(확장자 제외)
# 보고서 HTML을 앱 안에 넣을 때 바깥 여백 · 테두리를 없앤다 (보고서 자체의 디자인은 그대로)
EMBED_CSS = ("<style>body{background:transparent!important;padding:0 4px!important}"
             ".sheet{border:0!important;max-width:none!important;background:transparent!important;padding:8px 28px 28px!important}</style>")

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



def render_inspection(rows, period):
    """재점검 대상의 점검 결과를 Supabase에 기록하고 보여 준다."""
    status = st.empty()                                            # 저장소 연결 상태 한 줄 (아래에서 채운다)
    st.write("재점검 대상 드론의 점검 결과를 기록합니다. 기록은 **Supabase(Postgres)** 에 저장되어 앱을 다시 시작해도 남습니다.")
    if not store.configured():
        status.markdown(":gray[●] 저장소: **연결 안 됨** · Secrets에 연결 주소를 넣으면 저장할 수 있습니다")
        st.info("저장소가 연결되어 있지 않습니다. 내 PC는 `.streamlit/secrets.toml`, 배포 앱은 Streamlit Secrets에 "
                "`[connections.sql]`의 `url`을 넣으세요. 자세한 방법은 README.md에 있습니다.")
        return
    try:
        conn = store.connect()
        store.init_db()
    except Exception as e:
        status.markdown(f":red[●] 저장소: **{store.backend_label()} 연결 실패**")
        st.error("점검 결과 저장소(Supabase)에 연결하지 못했습니다. 무료 프로젝트는 1주일 접속이 없으면 멈춥니다. "
                 "Supabase 대시보드에서 다시 켠 뒤 새로고침하세요.")
        st.caption(f"오류 종류: {type(e).__name__}")
        return

    targets = rows[rows["재점검"]]
    label = {r["equipment_id"]: f"{r['equipment_id']} · {r['equipment_type']} · {r['unit_code']} · 확률 {r['확률(%)']}%"
             for _, r in targets.iterrows()}
    pw = store.write_password()
    top = st.container()                                           # 현황은 저장 처리 뒤에 채운다

    with st.form("inspection_form", clear_on_submit=True):
        st.write("**점검 결과 기록**")
        eid = st.selectbox("드론", list(label), format_func=lambda e: label[e])
        a, b = st.columns(2)
        day = a.date_input("점검일", value=dt.date.today())
        res = b.selectbox("점검 결과", store.RESULTS)
        crew = st.text_input("점검반 (선택)", max_chars=store.MAX_CREW, placeholder="예: 정비1반")
        note = st.text_area("메모 (선택)", max_chars=store.MAX_NOTE)
        st.caption("개인 이름 · 실제 부대 정보 · 연락처는 입력하지 마세요. 교육용 가상 데이터입니다.")
        pwd = st.text_input("쓰기 비밀번호", type="password") if pw else ""
        saved = st.form_submit_button("저장")
    if not pw:
        st.caption("⚠ 쓰기 비밀번호가 설정되어 있지 않아 누구나 기록할 수 있습니다. 공개 앱이면 Secrets에 `write_password`를 넣으세요.")
    if saved:
        if pw and not hmac.compare_digest(pwd.encode(), pw.encode()):
            st.error("쓰기 비밀번호가 맞지 않습니다.")
        else:
            try:
                new_id = store.add_result(conn, period, eid, day, res, crew, note)
                st.success(f"저장했습니다. (번호 {new_id}, {eid}, {res})")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"저장하지 못했습니다. ({type(e).__name__})")

    rec = store.list_results(conn, period)
    status.markdown(f":green[●] 저장소: **{store.backend_label()}** · 연결됨 · {period} 기록 {len(rec)}건")
    done = set(rec["equipment_id"]) & set(targets["equipment_id"])
    with top:
        c1, c2, c3 = st.columns(3)
        c1.metric("점검 완료", f"{len(done)} / {len(targets)}대")
        c2.metric("기록 건수", f"{len(rec)}건")
        c3.metric("이상이 확인된 드론", f"{rec.loc[rec['result'] != '이상 없음', 'equipment_id'].nunique()}대")
        st.progress(len(done) / max(len(targets), 1))
        waiting = [e for e in label if e not in done]
        st.caption("아직 점검 기록이 없는 대상: " + (", ".join(waiting) if waiting else "없음 (모두 기록됨)"))

    st.subheader("기록 목록")
    if rec.empty:
        st.info("아직 기록이 없습니다. 위 양식으로 첫 점검 결과를 저장해 보세요.")
        return
    info = rows.set_index("equipment_id")
    view = rec.assign(유형=rec["equipment_id"].map(info["equipment_type"]), 부대=rec["equipment_id"].map(info["unit_code"]),
                      순위=rec["equipment_id"].map(info["rank"]), **{"확률(%)": rec["equipment_id"].map(info["확률(%)"])})
    view = view.rename(columns={"id": "번호", "equipment_id": "드론", "inspected_on": "점검일", "result": "점검 결과",
                                "crew": "점검반", "note": "메모", "created_at": "기록 시각"})
    pick = st.multiselect("점검 결과 필터", store.RESULTS)
    if pick:
        view = view[view["점검 결과"].isin(pick)]
    cols = ["번호", "점검일", "드론", "유형", "부대", "순위", "확률(%)", "점검 결과", "점검반", "메모", "기록 시각"]
    st.dataframe(view[cols], hide_index=True, width="stretch")
    st.download_button("점검 결과 받기 (CSV)", view[cols].to_csv(index=False).encode("utf-8-sig"),
                       file_name="점검_결과.csv", mime="text/csv")
    st.bar_chart(rec["result"].value_counts().reindex(store.RESULTS, fill_value=0))

    with st.expander("잘못 기록한 항목 삭제"):
        del_id = st.selectbox("삭제할 번호", list(rec["id"]),
                              format_func=lambda i: f"{i} · {rec.loc[rec['id'] == i, 'equipment_id'].iloc[0]} · "
                                                    f"{rec.loc[rec['id'] == i, 'result'].iloc[0]}")
        pwd2 = st.text_input("쓰기 비밀번호", type="password", key="del_pw") if pw else ""
        if st.button("삭제", disabled=not st.checkbox("이 기록을 삭제합니다")):
            if pw and not hmac.compare_digest(pwd2.encode(), pw.encode()):
                st.error("쓰기 비밀번호가 맞지 않습니다.")
            else:
                st.success(f"{store.delete_result(conn, del_id)}건을 삭제했습니다. 새로고침하면 목록에 반영됩니다.")


t1, tI, t2, t3, t4, t5 = st.tabs(["① 재점검 목록", "② 점검 결과", "③ 검증 성능", "④ 위험 표시 기준", "⑤ 보고서", "⑥ 모델 정보"])

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

with tI:
    render_inspection(rows, score["period"])

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
    names = list(REPORT_FILES)
    pick = st.segmented_control("보고서 선택", names, default=names[0], label_visibility="collapsed") or names[0]
    stem = REPORT_FILES[pick]
    html_p, md_p = REPORTS / f"{stem}.html", REPORTS / f"{stem}.md"
    v, top1 = card["validation"], score["rows"][0]                 # 핵심 수치는 결과 파일의 값을 그대로 쓴다
    k1, k2, k3, k4 = st.columns(4)
    if pick == names[0]:
        k1.metric("검증 PR-AUC", v["pr_auc"], f"기준선 {v['base_rate']}", delta_color="off", delta_arrow="off")
        k2.metric("월 20대 적중 (모델)", v["hits"]["model"], f"오래된 순 {v['hits']['oldest']} · 무작위 {v['hits']['random']}",
                  delta_color="off", delta_arrow="off")
        k3.metric("위험 표시 기준", f"{thr0:.2f}", "놓친 비용 300만원", delta_color="off", delta_arrow="off")
        k4.metric("비용 600만원일 때", f"{card['threshold_600']:.2f}", delta_color="off", delta_arrow="off")
    else:
        k1.metric("재점검 대상", f"{cap0}대")
        k2.metric("위험 높음", f"{score['above_threshold']}대", f"기준 {score['threshold']:.2f} 이상", delta_color="off", delta_arrow="off")
        k3.metric("20대 초과", f"{score['above_threshold_over_capacity']}대", "다음 달 이월 검토", delta_color="off", delta_arrow="off")
        k4.metric("1위", f"{top1['prob'] * 100:.1f}%", top1["equipment_id"], delta_color="off", delta_arrow="off")
    if html_p.exists():
        st.iframe(html_p.read_text(encoding="utf-8").replace("</head>", EMBED_CSS + "</head>", 1), height=820, alt=f"{pick} 보고서")
    elif md_p.exists():
        st.markdown(md_p.read_text(encoding="utf-8"))
    else:
        st.warning(f"`{stem}` 보고서 파일이 없습니다.")
    d1, d2, _ = st.columns([1, 1, 4])
    if html_p.exists():
        d1.download_button("HTML 받기", html_p.read_bytes(), file_name=html_p.name, mime="text/html")
    if md_p.exists():
        d2.download_button("마크다운 받기", md_p.read_bytes(), file_name=md_p.name, mime="text/markdown")

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
