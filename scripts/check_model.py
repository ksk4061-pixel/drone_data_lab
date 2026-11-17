"""[모델 점검 B] 과거로 되돌려 채점해 본다. 점검 대상은 검증 기간(2024-07 ~ 12)이다. 시험(2025) 정답은 쓰지 않는다.

실행 (drone_data_lab 폴더, python scripts/train.py를 먼저 실행):  python scripts/check_model.py
출력: 화면 + artifacts/check_*.csv · check_summary.json + reports/모델점검보고서.md
점검 결과가 나빠도 모델 · 피처 · 기준을 이 스크립트가 고치지 않는다(검증 기간에 또 맞추는 것을 막기 위해). 보고만 한다.

합격 기준(결과를 보기 전에 정해 둔 값):
  B1 재계산 대조      보고서(model_card.json)의 값과 별도 코드로 다시 계산한 값이 일치(AUC 0.001, 적중 수 정확히)
  B2 월별 적중        모델 적중이 오래된 순과 무작위(기대값)보다 많은 달이 6개월 중 5개월 이상(4개월이면 주의)
  B3 단순 규칙 비교   모델 적중 합계가 단순 규칙 합계 이상
  B4 보정             건수(n)가 20 이상인 구간 중 |실제 − 예측|이 0.10 이하인 구간이 75% 이상
  B5 부분 집단        예방정비 행만 · 사후/긴급정비 행만 따로 봐도 ROC-AUC 0.60 이상
  B6 걸어가며 학습    각 달을 그 이전 데이터로만 학습해도 적중 합계가 단일 분할과 5 이내로 같고 오래된 순보다 많음
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drone_lib import ART, CARD_PATH, FEATURES, REPORTS, TARGET, make_model, prepare  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CAP = 20
TOL_AUC = 0.001
PASS, WARN, FAIL = "통과", "주의", "실패"


# ---------------------------------------------------------------- 별도 구현 (drone_lib · sklearn과 따로 계산해 대조한다)
def avg_precision(y, p):
    o = np.argsort(-p, kind="stable")
    y = y[o]
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def roc_auc(y, p):
    r = pd.Series(p).rank(method="average").to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def with_month(df):
    return df.assign(month=df["maintenance_date"].dt.to_period("M").astype(str))


def latest_candidates(df):
    """달마다 드론별 가장 최근 정비 기록 하나씩 (한 달에 20대를 고르는 업무 기준)."""
    d = with_month(df).sort_values(["maintenance_date", "equipment_id"])
    return d.drop_duplicates(["month", "equipment_id"], keep="last")


def top_picks(cand, score, ascending=False):
    out = []
    for m, g in cand.groupby("month"):
        out.append(g.assign(_s=score(g)).sort_values(["_s", "equipment_id"], ascending=[ascending, True]).head(CAP))
    return pd.concat(out)


def hits_by_month(picks):
    return picks.groupby("month")["failure"].sum().astype(int)


def main():
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    vp = pd.read_csv(ART / "validation_predictions.csv", parse_dates=["maintenance_date"])
    vp = vp.rename(columns={"failure": "failure"})
    y, p = vp["failure"].to_numpy(int), vp["prob"].to_numpy(float)
    checks, S = [], {"n_rows": len(vp)}

    # ------------------------------------------------------------ B1 보고서 숫자 재계산
    cand = latest_candidates(vp)
    pm = top_picks(cand, lambda g: g["prob"])                                     # 모델이 고른 20대
    po = top_picks(cand, lambda g: g["equipment_age_years"])                     # 경과연수 오래된 순
    rule = top_picks(cand, lambda g: (g["maintenance_type"].isin(["사후정비", "긴급정비"]).astype(int) * 10000
                                      + g["fail_count_90d"].fillna(0) * 100 + g["equipment_age_years"]))   # 단순 규칙
    hm, ho, hr = hits_by_month(pm), hits_by_month(po), hits_by_month(rule)
    n_c = cand.groupby("month").size()
    f_c = cand.groupby("month")["failure"].sum().astype(int)
    rnd = pd.Series({m: round(min(CAP, n_c[m]) * f_c[m] / n_c[m], 1) for m in n_c.index})
    ub = f_c.clip(upper=CAP)                                                      # 달성 가능한 최대 적중(그달 실제 재고장 수와 20 중 작은 값)
    monthly = pd.DataFrame({"month": n_c.index, "candidates": n_c.to_numpy(), "failures": f_c.to_numpy(), "upper_bound": ub.to_numpy(),
                            "model": hm.reindex(n_c.index).to_numpy(), "oldest": ho.reindex(n_c.index).to_numpy(),
                            "rule": hr.reindex(n_c.index).to_numpy(), "random": rnd.to_numpy()})
    monthly.to_csv(ART / "check_monthly_summary.csv", index=False, encoding="utf-8-sig")
    pm.assign(rank=pm.groupby("month").cumcount() + 1)[["month", "rank", "equipment_id", "maintenance_date", "maintenance_type", "prob", "failure"]] \
        .assign(maintenance_date=lambda d: d["maintenance_date"].dt.strftime("%Y-%m-%d")) \
        .to_csv(ART / "check_monthly_picks.csv", index=False, encoding="utf-8-sig")

    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    flagged = np.array([(p >= t).sum() for t in grid])
    missed = np.array([((p < t) & (y == 1)).sum() for t in grid])
    c300, c600 = missed * 300 + flagged * 60, missed * 600 + flagged * 60
    t300, t600 = float(grid[c300.argmin()]), float(grid[c600.argmin()])
    cv, cd = card["validation"], card["threshold_detail"]
    rows = [
        ("PR-AUC", cv["pr_auc"], round(avg_precision(y, p), 3), TOL_AUC),
        ("ROC-AUC", cv["roc_auc"], round(roc_auc(y, p), 3), TOL_AUC),
        ("월 20대 적중 — 모델(6개월 합)", cv["hits"]["model"], int(hm.sum()), 0),
        ("월 20대 적중 — 오래된 순", cv["hits"]["oldest"], int(ho.sum()), 0),
        ("월 20대 적중 — 무작위(기대값)", cv["hits"]["random"], round(float(rnd.sum()), 1), 0.15),
        ("위험 표시 기준(300만원)", card["threshold"], t300, 0),
        ("위험 표시 기준(600만원)", card["threshold_600"], t600, 0),
        ("총 비용 300만원 기준(만원)", cd["300"]["cost_300"], int(c300.min()), 0),
        ("총 비용 600만원 기준(만원)", cd["600"]["cost_600"], int(c600.min()), 0),
    ]
    b1 = pd.DataFrame([{"item": a, "report": b, "recomputed": c, "diff": round(abs(float(b) - float(c)), 4), "ok": abs(float(b) - float(c)) <= tol + 1e-9}
                       for a, b, c, tol in rows])
    S["b1"] = b1.to_dict("records")
    checks.append({"id": "B1", "name": "보고서 숫자 재계산 대조", "criterion": "model_card.json 값과 별도 코드로 다시 계산한 값이 일치",
                   "result": f"{int(b1['ok'].sum())}/{len(b1)}개 일치", "verdict": PASS if b1["ok"].all() else FAIL})

    # ------------------------------------------------------------ B2 월별 적중 · B3 단순 규칙
    better = int(((monthly["model"] > monthly["oldest"]) & (monthly["model"] > monthly["random"])).sum())
    checks.append({"id": "B2", "name": "월별 적중(오래된 순 · 무작위와 비교)", "criterion": "모델이 둘 다 이긴 달 6개월 중 5개월 이상",
                   "result": f"6개월 중 {better}개월 · 합계 모델 {int(monthly['model'].sum())} / 오래된 순 {int(monthly['oldest'].sum())} / 무작위 {monthly['random'].sum():.1f} / 최대치 {int(monthly['upper_bound'].sum())}",
                   "verdict": PASS if better >= 5 else (WARN if better == 4 else FAIL)})
    rule_all = (vp["maintenance_type"].isin(["사후정비", "긴급정비"]).astype(int) * 10000 + vp["fail_count_90d"].fillna(0) * 100
                + vp["equipment_age_years"]).to_numpy(float)
    S["rule_pr_auc"], S["rule_roc_auc"] = round(avg_precision(y, rule_all), 3), round(roc_auc(y, rule_all), 3)   # 참고: 행 단위 지표(합격 기준에는 쓰지 않음)
    checks.append({"id": "B3", "name": "단순 규칙과 비교", "criterion": "모델 적중 합계 ≥ 단순 규칙 합계(최근 정비가 사후 · 긴급인 드론, 최근 90일 고장 수 많은 순)",
                   "result": f"월 20대 적중 모델 {int(monthly['model'].sum())} / 단순 규칙 {int(monthly['rule'].sum())} · 참고: 행 단위 PR-AUC 모델 {card['validation']['pr_auc']} / 단순 규칙 {S['rule_pr_auc']}",
                   "verdict": PASS if monthly["model"].sum() >= monthly["rule"].sum() else WARN})

    # ------------------------------------------------------------ B4 보정
    edges = np.round(np.arange(0, 1.01, 0.1), 1)
    edges[-1] = 1.0001
    vp["bin"] = pd.cut(vp["prob"], edges, right=False)
    cal = vp.groupby("bin", observed=True).agg(n=("prob", "size"), predicted=("prob", "mean"), actual=("failure", "mean")).reset_index()
    cal["gap"] = cal["actual"] - cal["predicted"]
    cal["bin"] = cal["bin"].map(lambda i: f"{i.left:.1f}~{min(i.right, 1.0):.1f}")
    cal.round(3).to_csv(ART / "check_calibration.csv", index=False, encoding="utf-8-sig")
    big = cal[cal["n"] >= 20]
    share = float((big["gap"].abs() <= 0.10).mean()) if len(big) else 0.0
    checks.append({"id": "B4", "name": "보정(예측 확률 ↔ 실제 비율)", "criterion": "n 20 이상 구간의 75% 이상이 |실제 − 예측| ≤ 0.10",
                   "result": f"{int((big['gap'].abs() <= 0.10).sum())}/{len(big)}개 구간 · 최대 차이 {big['gap'].abs().max():.3f}",
                   "verdict": PASS if share >= 0.75 else WARN})

    # ------------------------------------------------------------ B5 부분 집단
    sub = {}
    for name, mask in (("예방정비 행만", vp["maintenance_type"] == "예방정비"), ("사후 · 긴급정비 행만", vp["maintenance_type"] != "예방정비")):
        g = vp[mask]
        yy, pp = g["failure"].to_numpy(int), g["prob"].to_numpy(float)
        sub[name] = {"rows": len(g), "base_rate": round(float(yy.mean()), 3), "roc_auc": round(roc_auc(yy, pp), 3), "pr_auc": round(avg_precision(yy, pp), 3)}
    S["b5"] = sub
    checks.append({"id": "B5", "name": "부분 집단 점검(누수 · 쉬운 구간만 맞히는지)", "criterion": "예방정비 행만 · 사후/긴급 행만 ROC-AUC ≥ 0.60",
                   "result": " · ".join(f"{k} ROC-AUC {v['roc_auc']}" for k, v in sub.items()),
                   "verdict": PASS if all(v["roc_auc"] >= 0.60 for v in sub.values()) else WARN})

    # ------------------------------------------------------------ B6 걸어가며 다시 학습 (각 달을 그 이전 데이터로만 학습)
    df = prepare()
    tr = df[df["_src"] == "train"].assign(**{TARGET: lambda x: x[TARGET].astype(int)})
    wf_rows, wf_pred = [], []
    for m in sorted(vp["maintenance_date"].dt.to_period("M").astype(str).unique()):
        start = pd.Period(m).to_timestamp()
        fit = tr[tr["maintenance_date"] <= start - pd.Timedelta(days=31)]        # 정답 확인 구간(30일)이 이 달에 걸치지 않게
        cur = tr[(tr["maintenance_date"] >= start) & (tr["maintenance_date"] < start + pd.offsets.MonthBegin(1))]
        pr = make_model().fit(fit[FEATURES], fit[TARGET]).predict_proba(cur[FEATURES])[:, 1]
        c = cur[["equipment_id", "maintenance_date", "equipment_age_years"]].assign(prob=pr, failure=cur[TARGET].to_numpy())
        wf_pred.append(c)
        cc = latest_candidates(c)
        pk = top_picks(cc, lambda g: g["prob"])
        wf_rows.append({"month": m, "train_rows": len(fit), "walk_forward_hits": int(pk["failure"].sum()),
                        "single_split_hits": int(hm[m]), "pr_auc": round(avg_precision(c["failure"].to_numpy(int), pr), 3)})
    wf = pd.DataFrame(wf_rows)
    wf.to_csv(ART / "check_walk_forward.csv", index=False, encoding="utf-8-sig")
    wfp = pd.concat(wf_pred)
    S["b6_pooled_pr_auc"] = round(avg_precision(wfp["failure"].to_numpy(int), wfp["prob"].to_numpy(float)), 3)
    tw, ts = int(wf["walk_forward_hits"].sum()), int(wf["single_split_hits"].sum())
    checks.append({"id": "B6", "name": "걸어가며 다시 학습(각 달을 그 이전 데이터로만)", "criterion": "적중 합계가 단일 분할과 5 이내로 같고 오래된 순보다 많음",
                   "result": f"걸어가며 {tw} / 단일 분할 {ts} / 오래된 순 {int(monthly['oldest'].sum())} · 전체 PR-AUC {S['b6_pooled_pr_auc']}",
                   "verdict": PASS if abs(tw - ts) <= 5 and tw > int(monthly["oldest"].sum()) else WARN})

    # ------------------------------------------------------------ 저장 · 출력
    S["checks"], S["monthly"] = checks, monthly.to_dict("records")
    S["calibration"], S["walk_forward"] = cal.round(3).to_dict("records"), wf.to_dict("records")
    S["totals"] = {k: (round(float(monthly[k].sum()), 1)) for k in ("candidates", "failures", "upper_bound", "model", "oldest", "rule", "random")}
    (ART / "check_summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== 모델 점검 B (검증 기간 2024-07 ~ 12) ==")
    for c in checks:
        print(f"[{c['verdict']}] {c['id']} {c['name']}\n        기준: {c['criterion']}\n        결과: {c['result']}")
    print("\n-- B1 재계산 대조 --\n" + b1.to_string(index=False))
    print("\n-- B2/B3 월별 --\n" + monthly.to_string(index=False))
    print("\n-- B4 보정 --\n" + cal.round(3).to_string(index=False))
    print("\n-- B5 부분 집단 --", json.dumps(sub, ensure_ascii=False))
    print("\n-- B6 걸어가며 --\n" + wf.to_string(index=False))
    build_report(checks, b1, monthly, cal, sub, wf, S)
    print("\n저장: artifacts/check_*.csv, artifacts/check_summary.json, reports/모델점검보고서.md")


def build_report(checks, b1, monthly, cal, sub, wf, S):
    def md(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                         + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])
    n_pass = sum(c["verdict"] == PASS for c in checks)
    T = S["totals"]
    rb, mb = int((monthly["rule"] > monthly["model"]).sum()), int((monthly["model"] > monthly["rule"]).sum())
    maxdiff = int((monthly["rule"] - monthly["model"]).abs().max())
    pm_ap = next(r["recomputed"] for r in S["b1"] if r["item"] == "PR-AUC")
    txt = f"""# 모델 점검 보고서 (B안: 과거로 되돌려 채점)

- 작성: AI 에이전트 초안 · 검토: 정비 담당자 (검토 전)
- 자료: 교육용 가상 데이터. 점검 대상은 검증 기간(2024-07 ~ 12)이며 시험(2025) 정답은 쓰지 않음
- 이 점검은 모델 · 피처 · 위험 표시 기준을 고치지 않음. 결과만 보고함

## 1. 종합
- 점검 {len(checks)}개 중 **{n_pass}개 통과**, 주의 {sum(c['verdict'] == WARN for c in checks)}개, 실패 {sum(c['verdict'] == FAIL for c in checks)}개
- 6개월 동안 매달 20대를 골랐을 때 실제 재고장 수: 모델 **{int(T['model'])}대**, 경과연수 오래된 순 {int(T['oldest'])}대, 단순 규칙 {int(T['rule'])}대, 무작위(기대값) {T['random']}대. 달성 가능한 최대치는 {int(T['upper_bound'])}대
- 출처: `python scripts/check_model.py` → `artifacts/check_summary.json`

## 2. 점검 기준과 판정
기준은 결과를 보기 전에 `scripts/check_model.py` 맨 위에 적어 둔 값임.

{md(['번호', '점검', '합격 기준', '결과', '판정'], [(c['id'], c['name'], c['criterion'], c['result'], c['verdict']) for c in checks])}

## 3. B1 보고서 숫자 재계산 대조
저장한 검증 예측값(`artifacts/validation_predictions.csv`)으로 PR-AUC · 적중 수 · 위험 표시 기준을 별도 코드로 다시 계산해 `model_card.json`의 값과 비교함.

{md(['항목', '보고서 값', '재계산 값', '차이', '일치'], [(r['item'], r['report'], r['recomputed'], r['diff'], '일치' if r['ok'] else '불일치') for r in b1.to_dict('records')])}

## 4. B2 · B3 월별 적중
후보는 달마다 드론별 가장 최근 정비 기록 하나씩이며, 이 가운데 확률이 높은 20대를 고름. '최대치'는 그달 후보 중 실제 재고장 수와 20 중 작은 값임. 단순 규칙은 '최근 정비가 사후 · 긴급정비인 드론부터, 최근 90일 고장 수가 많은 순'으로 고름.

{md(['월', '후보(드론)', '실제 재고장', '최대치', '모델', '오래된 순', '단순 규칙', '무작위(기대값)'], [(r['month'], r['candidates'], r['failures'], r['upper_bound'], r['model'], r['oldest'], r['rule'], r['random']) for r in monthly.to_dict('records')])}

- **B3 {checks[2]['verdict']}**: 월 20대 적중 합계가 모델 {int(T['model'])}대, 단순 규칙 {int(T['rule'])}대임. 달별로는 규칙이 앞선 달 {rb}개월, 모델이 앞선 달 {mb}개월, 같은 달 {6 - rb - mb}개월이며 달별 차이는 최대 {maxdiff}대임. 월 표본이 작아 우연일 수 있음
- 행 단위로 보면(검증 기간 {S['n_rows']}건 전체) 모델 PR-AUC {pm_ap}, 단순 규칙 PR-AUC {S['rule_pr_auc']}임. 모델은 ‘확률’을 주므로 위험 표시 기준(비용 계산)에는 모델이 필요하지만, 매달 상위 20대만 고르는 용도에서는 단순 규칙이 비슷한 수준임
- 달마다 고른 20대와 O/X는 앱의 ‘⑦ 모델 점검’ 탭에서 볼 수 있음

## 5. B4 보정 (예측 확률 ↔ 실제 비율)
검증 기간 정비 기록 643건을 예측 확률 구간별로 묶어 실제 재고장 비율과 비교함.

{md(['구간', '건수', '평균 예측', '실제 비율', '차이(실제 − 예측)'], [(r['bin'], r['n'], f"{r['predicted']:.3f}", f"{r['actual']:.3f}", f"{r['gap']:+.3f}") for r in cal.to_dict('records')])}

## 6. B5 부분 집단
{md(['집단', '건수', '재고장 비율', 'ROC-AUC', 'PR-AUC'], [(k, v['rows'], v['base_rate'], v['roc_auc'], v['pr_auc']) for k, v in sub.items()])}

## 7. B6 걸어가며 다시 학습
각 달을 그 이전(정답 확인 구간 30일 제외) 데이터로만 학습한 모델로 예측한 결과임.

{md(['월', '학습 건수', '걸어가며 적중', '단일 분할 적중', '그달 PR-AUC'], [(r['month'], f"{r['train_rows']:,}", r['walk_forward_hits'], r['single_split_hits'], r['pr_auc']) for r in wf.to_dict('records')])}

- 전체 PR-AUC(여섯 달을 합쳐서): {S['b6_pooled_pr_auc']}

## 8. 한계
- 위험 표시 기준과 피처 묶음은 같은 검증 기간에서 골랐으므로 검증 성능은 다소 낙관적일 수 있음. B6은 이 영향을 줄여 보기 위한 점검이지만 피처 묶음 선택은 여전히 같은 기간을 보았음
- 시험(2025) 정답이 없어 시험 성능은 점검하지 못함. 시험 성능은 강사의 채점 결과로 확인해야 함
- 후보가 달마다 약 80대(검증 기간 평균)이고 20대만 고르므로 월별 적중 수의 표본이 작음(한 달 4 ~ 10대 수준). 한두 대 차이는 우연일 수 있음
- 이 점검은 AI 에이전트가 작성한 초안이며 최종 판단은 정비 담당자가 함

## 9. 숫자의 출처
{md(['항목', '출처'], [('검증 예측 · 정답', '`artifacts/validation_predictions.csv` (`python scripts/train.py`)'), ('모든 점검 결과', '`python scripts/check_model.py` → `artifacts/check_summary.json` · `check_*.csv`'), ('보고서 비교 기준값', '`artifacts/model_card.json`')])}
"""
    (REPORTS / "모델점검보고서.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
