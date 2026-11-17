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
REPORT_FILES = {"분석 보고서": "분석보고서", "2026-01 재점검 계획": "2026-01_드론_재점검계획",
                "모델 점검 B": "모델점검보고서", "모델 점검 A": "모델점검A보고서"}          # reports/ 안의 파일 이름(확장자 제외)
CHECK = ART / "check_summary.json"                # python scripts/check_model.py 의 점검 결과
chk0 = json.loads(CHECK.read_text(encoding="utf-8")) if CHECK.exists() else None
CHECK_A, TRACE = ART / "check_features_summary.json", ART / "trace_targets.json"    # python scripts/check_features.py 의 점검 결과
fs0 = json.loads(CHECK_A.read_text(encoding="utf-8")) if CHECK_A.exists() else None
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


def _save_result(period):
    """저장 버튼 콜백. 화면을 다시 그리기 전에 실행되어 저장 직후의 기록으로 드롭다운 · 현황이 그려진다.
    결과 안내는 session_state['inspect_msg']에 남긴다."""
    ss, pw = st.session_state, store.write_password()
    if pw and not hmac.compare_digest(str(ss.get("f_pwd", "")).encode(), pw.encode()):
        ss["inspect_msg"] = ("error", "쓰기 비밀번호가 맞지 않습니다.")
        return
    try:
        conn, eid = store.connect(), ss["f_drone"]
        before = int((store.list_results(conn, period)["equipment_id"] == eid).sum())
        new_id = store.add_result(conn, period, eid, ss["f_day"], ss["f_res"], ss.get("f_crew", ""), ss.get("f_note", ""))
        msg = f"저장했습니다. (번호 {new_id}, {eid}, {ss['f_res']})"
        if before:
            msg += f" {eid}의 {before + 1}번째 기록이라 '점검한 드론' 수는 그대로입니다."
        ss["inspect_msg"] = ("success", msg)
    except ValueError as e:
        ss["inspect_msg"] = ("error", str(e))
    except Exception as e:
        ss["inspect_msg"] = ("error", f"저장하지 못했습니다. ({type(e).__name__})")


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
    rec = store.list_results(conn, period)                         # 저장 콜백이 먼저 실행되므로 저장 직후의 기록이다
    visits = rec["equipment_id"].value_counts()                    # 드론별 기록 횟수
    label = {r["equipment_id"]: (f"✔ 점검됨 ({int(visits[r['equipment_id']])}회) · " if r["equipment_id"] in visits.index else "")
             + f"{r['equipment_id']} · {r['equipment_type']} · {r['unit_code']} · 확률 {r['확률(%)']}%"
             for _, r in targets.iterrows()}
    options = sorted(label, key=lambda e: e in visits.index)       # 아직 점검하지 않은 드론을 먼저(각각 순위 순)
    pw = store.write_password()

    status.markdown(f":green[●] 저장소: **{store.backend_label()}** · 연결됨 · {period} 기록 {len(rec)}건")
    done = set(rec["equipment_id"]) & set(targets["equipment_id"])
    latest = rec.sort_values(["inspected_on", "id"]).drop_duplicates("equipment_id", keep="last")   # 드론별 가장 최근 기록
    c1, c2, c3 = st.columns(3)
    c1.metric("점검한 드론", f"{len(done)} / {len(targets)}대",
              help="드론 기준입니다. 같은 드론의 점검 결과를 여러 번 기록해도 1대로 셉니다.")
    c2.metric("기록 건수", f"{len(rec)}건", help="점검 결과를 저장한 횟수입니다. 같은 드론을 다시 기록해도 늘어납니다.")
    c3.metric("이상이 확인된 드론", f"{int((latest['result'] != '이상 없음').sum())}대",
              help="각 드론의 가장 최근 기록이 '이상 없음'이 아닌 드론 수입니다.")
    st.progress(len(done) / max(len(targets), 1))
    waiting = [e for e in options if e not in done]
    st.caption("아직 점검 기록이 없는 대상: " + (", ".join(waiting) if waiting else "없음 (모두 기록됨)")
               + " · 점검한 드론은 드론 수 기준이라 같은 드론을 다시 기록하면 기록 건수만 늘어납니다.")

    # 저장은 버튼 콜백(_save_result)이 화면을 다시 그리기 전에 처리한다. 그래서 저장 직후의 드롭다운 · 현황이 같은 실행에서 일관된다.
    with st.form("inspection_form", clear_on_submit=True):
        st.write("**점검 결과 기록**")
        st.selectbox("드론 (아직 점검하지 않은 드론이 먼저 나옵니다)", options, format_func=lambda e: label[e], key="f_drone")
        a, b = st.columns(2)
        a.date_input("점검일", value=dt.date.today(), key="f_day")
        b.selectbox("점검 결과", store.RESULTS, key="f_res")
        st.text_input("점검반 (선택)", max_chars=store.MAX_CREW, placeholder="예: 정비1반", key="f_crew")
        st.text_area("메모 (선택)", max_chars=store.MAX_NOTE, key="f_note")
        st.caption("개인 이름 · 실제 부대 정보 · 연락처는 입력하지 마세요. 교육용 가상 데이터입니다.")
        if pw:
            st.text_input("쓰기 비밀번호", type="password", key="f_pwd")
        st.form_submit_button("저장", on_click=_save_result, args=(period,))
    msg = st.session_state.pop("inspect_msg", None)                # 콜백이 남긴 저장 결과 안내
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])
    if not pw:
        st.caption("⚠ 쓰기 비밀번호가 설정되어 있지 않아 누구나 기록할 수 있습니다. 공개 앱이면 Secrets에 `write_password`를 넣으세요.")

    st.subheader("대상 드론별 점검 현황")
    cnt, lat = rec.groupby("equipment_id").size(), latest.set_index("equipment_id")
    stat = pd.DataFrame({
        "순위": targets["rank"].to_numpy(), "드론": targets["equipment_id"].to_numpy(),
        "유형": targets["equipment_type"].to_numpy(), "부대": targets["unit_code"].to_numpy(),
        "확률(%)": targets["확률(%)"].to_numpy(),
        "점검 상태": ["✔ 점검됨" if e in cnt.index else "대기" for e in targets["equipment_id"]],
        "기록 횟수": [int(cnt.get(e, 0)) for e in targets["equipment_id"]],
        "최근 점검일": [lat["inspected_on"].get(e, "-") for e in targets["equipment_id"]],
        "최근 결과": [lat["result"].get(e, "-") for e in targets["equipment_id"]],
    })
    st.dataframe(stat, hide_index=True, width="stretch")
    st.caption("같은 드론의 기록은 한 줄로 합쳐 '기록 횟수'로 보여 줍니다. 위 '점검한 드론'은 이 표에서 ✔ 점검됨인 드론 수입니다. "
               "기록 하나하나는 아래 '기록 목록'에 있습니다.")

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


t1, tI, t2, t3, t4, t5, t7, t8 = st.tabs(["① 재점검 목록", "② 점검 결과", "③ 검증 성능", "④ 위험 표시 기준", "⑤ 보고서", "⑥ 모델 정보",
                                          "⑦ 모델 점검 B", "⑧ 모델 점검 A"])

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
    elif pick == names[1]:
        k1.metric("재점검 대상", f"{cap0}대")
        k2.metric("위험 높음", f"{score['above_threshold']}대", f"기준 {score['threshold']:.2f} 이상", delta_color="off", delta_arrow="off")
        k3.metric("20대 초과", f"{score['above_threshold_over_capacity']}대", "다음 달 이월 검토", delta_color="off", delta_arrow="off")
        k4.metric("1위", f"{top1['prob'] * 100:.1f}%", top1["equipment_id"], delta_color="off", delta_arrow="off")
    elif pick == names[3] and fs0:
        sa = fs0["stats"]
        k1.metric("점검 통과", f"{sum(c['verdict'] == '통과' for c in fs0['checks'])} / {len(fs0['checks'])}")
        k2.metric("전수 대조 기록", f"{fs0['rows']:,}건", f"불일치 {sa['all'][1]}건", delta_color="off", delta_arrow="off")
        k3.metric("경계 사례", f"{sa['boundary'][0]}건", f"불일치 {sa['boundary'][1]}건", delta_color="off", delta_arrow="off")
        k4.metric("파일 경계(시험 · 이번 달)", f"{sa['nonTrain'][0]:,}건", f"불일치 {sa['nonTrain'][1]}건", delta_color="off", delta_arrow="off")
    elif pick == names[2] and chk0:
        tt = chk0["totals"]
        k1.metric("점검 통과", f"{sum(c['verdict'] == '통과' for c in chk0['checks'])} / {len(chk0['checks'])}")
        k2.metric("월 20대 적중 (모델)", int(tt["model"]), f"최대치 {int(tt['upper_bound'])}", delta_color="off", delta_arrow="off")
        k3.metric("오래된 순", int(tt["oldest"]), f"무작위 {tt['random']}", delta_color="off", delta_arrow="off")
        k4.metric("단순 규칙", int(tt["rule"]), "비교용", delta_color="off", delta_arrow="off")
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

with t7:
    picks_p = ART / "check_monthly_picks.csv"
    if chk0 is None or not picks_p.exists():
        st.info("점검 결과 파일이 없습니다. `python scripts/check_model.py`를 먼저 실행하세요.")
    else:
        st.subheader("B. 과거로 되돌려 채점 (검증 기간 2024-07 ~ 12)")
        st.caption("시험(2025) 정답이 없어 시험 성능은 점검하지 못했습니다. 이 점검은 모델 · 피처 · 기준을 고치지 않고 결과만 보여 줍니다.")
        n_ok = sum(c["verdict"] == "통과" for c in chk0["checks"])
        st.write(f"**종합: 점검 {len(chk0['checks'])}개 중 {n_ok}개 통과**")
        color = {"통과": "green", "주의": "orange", "실패": "red"}
        for c in chk0["checks"]:
            st.markdown(f":{color[c['verdict']]}[●] **{c['id']} {c['name']}** · :{color[c['verdict']]}[{c['verdict']}]  \n{c['result']}")
            st.caption(f"합격 기준: {c['criterion']}")

        st.divider()
        st.subheader("달별로 고른 20대 확인")
        mon = pd.DataFrame(chk0["monthly"])
        picks = pd.read_csv(picks_p, encoding="utf-8-sig")
        month = st.selectbox("월 선택", list(mon["month"]))
        r = mon[mon["month"] == month].iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("모델 적중", f"{int(r['model'])}대", help="모델이 고른 20대 중 실제로 30일 안에 재고장 난 드론 수")
        m2.metric("오래된 순", f"{int(r['oldest'])}대")
        m3.metric("단순 규칙", f"{int(r['rule'])}대", help="최근 정비가 사후 · 긴급인 드론부터, 최근 90일 고장 수가 많은 순")
        m4.metric("무작위(기대값)", f"{r['random']}대")
        m5.metric("달성 가능한 최대", f"{int(r['upper_bound'])}대", help="그달 후보 중 실제 재고장 수와 20 중 작은 값")
        sel = picks[picks["month"] == month]
        st.dataframe(pd.DataFrame({"순위": sel["rank"], "드론": sel["equipment_id"], "정비일": sel["maintenance_date"],
                                   "정비 구분": sel["maintenance_type"], "예측 확률(%)": (sel["prob"] * 100).round(1),
                                   "실제 30일 안 재고장": sel["failure"].map({1: "O 재고장", 0: "X"})}),
                     hide_index=True, width="stretch")
        st.caption(f"후보는 그달 드론별 가장 최근 정비 기록({int(r['candidates'])}대)이며, 이 중 확률이 높은 20대를 보여 줍니다. "
                   f"그달 실제 재고장은 {int(r['failures'])}대였습니다.")

        st.write("**6개월 비교**")
        st.bar_chart(mon.rename(columns={"model": "모델", "oldest": "오래된 순", "rule": "단순 규칙", "random": "무작위(기대값)",
                                         "upper_bound": "최대치"}), x="month", y=["모델", "오래된 순", "단순 규칙", "무작위(기대값)", "최대치"])

        st.divider()
        st.subheader("예측 확률이 맞는가 (보정)")
        cal = pd.DataFrame(chk0["calibration"])
        st.bar_chart(cal.rename(columns={"predicted": "평균 예측", "actual": "실제 비율"}), x="bin", y=["평균 예측", "실제 비율"])
        st.dataframe(cal.rename(columns={"bin": "예측 확률 구간", "n": "건수", "predicted": "평균 예측", "actual": "실제 비율", "gap": "차이(실제 − 예측)"}),
                     hide_index=True, width="stretch")
        st.caption("두 막대가 비슷하면 확률을 믿을 수 있다는 뜻입니다. 건수가 적은 구간(0.7~0.8는 12건)은 차이가 커도 우연일 수 있습니다.")

        st.divider()
        st.subheader("걸어가며 다시 학습해도 같은가")
        st.dataframe(pd.DataFrame(chk0["walk_forward"]).rename(columns={
            "month": "월", "train_rows": "학습 건수", "walk_forward_hits": "걸어가며 적중", "single_split_hits": "단일 분할 적중", "pr_auc": "그달 PR-AUC"}),
            hide_index=True, width="stretch")
        st.caption("각 달을 그 이전 데이터로만 학습한 모델로 예측한 결과입니다. 단일 분할(앱의 기본 모델)과 비슷하면 결과가 한 번의 분할 덕분이 아니라는 뜻입니다.")

with t8:
    if fs0 is None or not TRACE.exists():
        st.info("점검 A 결과 파일이 없습니다. `python scripts/check_features.py`를 먼저 실행하세요(원본 data/ 필요).")
    else:
        tr = json.loads(TRACE.read_text(encoding="utf-8"))
        st.subheader("A. 드론 한 대씩 근거 따라가기")
        st.caption(f"원본 기록 {fs0['rows']:,}건 전체의 시간 피처를 별도 코드로 다시 세어 대조한 결과와, 재점검 대상 드론의 근거 기록입니다. "
                   "이 점검은 코드를 고치지 않고 결과만 보여 줍니다. 피처의 '정의'(이번 정비 포함, 30 · 90 · 365일 이하)가 업무에 맞는지는 사람이 판단합니다.")
        n_ok_a = sum(c["verdict"] == "통과" for c in fs0["checks"])
        st.write(f"**종합: 점검 {len(fs0['checks'])}개 중 {n_ok_a}개 통과**")
        for c in fs0["checks"]:
            cc = {"통과": "green", "주의": "orange", "실패": "red"}[c["verdict"]]
            st.markdown(f":{cc}[●] **{c['id']} {c['name']}** · :{cc}[{c['verdict']}]  \n{c['result']}")
            st.caption(f"합격 기준: {c['criterion']}")

        st.divider()
        st.subheader("대상 드론의 근거 기록")
        opts = {f"{x['rank']}위 · {x['equipment_id']} · 확률 {x['prob'] * 100:.1f}%": x for x in tr["drones"]}
        d = opts[st.selectbox("드론 선택", list(opts))]
        z1, z2, z3, z4 = st.columns(4)
        z1.metric("재고장 확률", f"{d['prob'] * 100:.1f}%")
        z2.metric("순위", f"{d['rank']}위", f"{d['equipment_type']} · {d['unit_code']}", delta_color="off", delta_arrow="off")
        z3.metric("판단한 정비일", d["date"])
        z4.metric("정비 구분", d["type"])

        FE = ["fail_count_30d", "fail_count_90d", "fail_count_365d", "days_since_last_failure", "hours_since_last_pm", "hours_since_last_maintenance"]
        show_v = lambda v: "없음" if v is None else v                                              # noqa: E731
        agree = lambda a, b: (a is None and b is None) or (a is not None and b is not None and abs(float(a) - float(b)) <= 0.051)   # noqa: E731
        st.write("**모델 입력값과 별도 재계산 비교**")
        st.dataframe(pd.DataFrame({"피처": [tr["labels"][f] for f in FE], "모델 입력값": [show_v(d["features"][f]) for f in FE],
                                   "별도 재계산": [show_v(d["recount"][f]) for f in FE],
                                   "일치": ["✔" if agree(d["features"][f], d["recount"][f]) else "✖" for f in FE]}),
                     hide_index=True, width="stretch")

        st.write("**근거 기록 (이 드론의 고장 정비, 판단한 정비일 기준 400일 이내)**")
        ev = pd.DataFrame(d["events"])
        if ev.empty:
            st.info("400일 이내 고장 정비 기록이 없습니다.")
        else:
            st.dataframe(pd.DataFrame({"정비일": ev["date"], "정비 구분": ev["type"], "며칠 전": ev["age_days"],
                                       "30일 안": ev["in30"].map({True: "O", False: "-"}), "90일 안": ev["in90"].map({True: "O", False: "-"}),
                                       "365일 안": ev["in365"].map({True: "O", False: "-"})}), hide_index=True, width="stretch")
            st.caption(f"각 열의 O를 세면 위 표의 모델 입력값과 같습니다: 30일 안 {int(ev['in30'].sum())}건 · 90일 안 {int(ev['in90'].sum())}건 · "
                       f"365일 안 {int(ev['in365'].sum())}건. 판단한 정비일 당일의 고장 정비도 포함합니다(0일 전).")
        pm, pv = d["last_pm"], d["prev"]
        st.caption(f"마지막 예방정비: {pm['date'] or '없음'}" + (f" (그 뒤 비행시간 {pm['hours_since']}h)" if pm["date"] else "")
                   + f" · 직전 정비: {pv['date'] or '없음(첫 기록)'}" + (f" (그 뒤 비행시간 {pv['hours_since']}h)" if pv["date"] else ""))

        st.write("**확률을 올리고 내린 요인 (기여도 상위 10)**")
        cb = pd.DataFrame({"요인": [tr["labels"].get(k, k) for k in d["contrib"]], "기여도": list(d["contrib"].values())})
        cb = cb.reindex(cb["기여도"].abs().sort_values(ascending=False).index).head(10)
        st.bar_chart(cb, x="요인", y="기여도", horizontal=True, sort=False)
        st.caption("오른쪽(+)은 재고장 확률을 올린 요인, 왼쪽(−)은 낮춘 요인입니다(로그 오즈 단위). 재점검 목록의 '위험을 올린 요인'은 이 중 위 3개입니다.")
