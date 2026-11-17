"""[3] 피처 · [4] 컬럼 · [5] 학습 · 검증 · [6] 위험 표시 기준 · [7] 시험 기간 예측.

실행 (drone_data_lab 폴더):  python scripts/train.py
출력: artifacts/ (모델 2개, model_card.json, feature_compare.csv, validation_by_month.csv, threshold_cost.csv)
      reports/test_predictions_2025.csv (제출물 1)
규칙은 AGENTS.md: 기간으로 나누고, 정답 확인 구간(30일)이 다음 기간에 걸치는 행은 학습에서 뺀다.
시험(2025) 정답은 없다. 시험 성능은 계산하지 않고 확률만 낸다.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drone_lib import (ART, CARD_PATH, FEATURE_SETS, FEATURES, MODEL_FINAL_PATH, MODEL_TEST_PATH, REFERENCE_ONLY,  # noqa: E402
                       REPORTS, TARGET, make_card, make_model, monthly_topk, prepare)

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
CAPACITY = 20                         # 한 달에 재점검할 수 있는 드론 수
COST_MISS, COST_MISS_HIGH, COST_CHECK = 300, 600, 60   # 놓친 재고장 1건 · (비교용) · 재점검 1대 비용 (만원, 교육용 가정)
FIT_END = "2024-05-31"                # 검증용 모델 학습 끝 (6월 행은 정답 구간이 검증 기간에 걸쳐 뺀다)
VALID_FROM, VALID_TO = "2024-07-01", "2024-12-31"
TEST_FIT_END = "2024-12-01"           # 시험 예측용 모델 학습 끝 (12월 2일 이후 행은 정답 구간이 2025년에 걸쳐 뺀다)
MIN_GAIN = 0.005                      # 피처를 더 넣으려면 검증 PR-AUC가 이만큼 좋아져야 한다


def threshold_table(y, prob):
    """임계값마다 기준 이상 건수 · 놓친 재고장 수 · 비용(300 · 600만원)."""
    ths = np.round(np.arange(0.05, 0.95, 0.01), 2)
    rows = []
    for t in ths:
        flagged, missed = int((prob >= t).sum()), int(((prob < t) & (y == 1)).sum())
        rows.append({"threshold": float(t), "flagged": flagged, "missed": missed,
                     "cost_300": missed * COST_MISS + flagged * COST_CHECK,
                     "cost_600": missed * COST_MISS_HIGH + flagged * COST_CHECK})
    return pd.DataFrame(rows)


def evaluate(valid, prob):
    """검증 기간 성능: PR-AUC · ROC-AUC · 매달 20대 적중 수(모델 · 오래된 순 · 무작위)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    v = valid.assign(prob=prob)
    back = monthly_topk(v, "prob", CAPACITY).rename(columns={"hits": "model"})
    back["oldest"] = monthly_topk(v, "equipment_age_years", CAPACITY)["hits"]
    back = back.rename(columns={"expected_random": "random"})
    return {"pr_auc": round(float(average_precision_score(v[TARGET], prob)), 3),
            "roc_auc": round(float(roc_auc_score(v[TARGET], prob)), 3),
            "base_rate": round(float(v[TARGET].mean()), 3),
            "hits": {"model": int(back["model"].sum()), "oldest": int(back["oldest"].sum()),
                     "random": round(float(back["random"].sum()), 1), "picked": int(back["picked"].sum())}}, back


def main():
    df = prepare()
    d = df["maintenance_date"]
    train = df[df["_src"] == "train"].assign(**{TARGET: lambda x: x[TARGET].astype(int)})
    test = df[df["_src"] == "test"]
    td = train["maintenance_date"]
    fit = train[td <= FIT_END]
    valid = train[(td >= VALID_FROM) & (td <= VALID_TO)]
    print(f"기간 분할: 학습 ~ {FIT_END} {len(fit):,}건 · 검증 {VALID_FROM} ~ {VALID_TO} {len(valid):,}건 · "
          f"(제외 6월 {int(((td > FIT_END) & (td < VALID_FROM)).sum())}건) · 재고장 비율 학습 {fit[TARGET].mean():.3f} / 검증 {valid[TARGET].mean():.3f}")

    # ---- [3][4] 피처 묶음 비교 (검증 기간으로만 비교한다) ----
    comp = []
    for name, feats in FEATURE_SETS.items():
        m = make_model(feats).fit(fit[feats], fit[TARGET])
        ev, _ = evaluate(valid, m.predict_proba(valid[feats])[:, 1])
        comp.append({"feature_set": name, "n_features": len(feats), "pr_auc": ev["pr_auc"], "roc_auc": ev["roc_auc"],
                     "top20_hits": ev["hits"]["model"], "selectable": name not in REFERENCE_ONLY})
    comp = pd.DataFrame(comp)
    comp.to_csv(ART / "feature_compare.csv", index=False, encoding="utf-8-sig")
    print("\n== 피처 묶음 비교 (검증 기간) ==  selectable=False는 비교용(시간이 지날수록 커지는 피처라 선택에서 뺀다)")
    print(comp.to_string(index=False))
    cand = comp[comp["selectable"]].sort_values("n_features", kind="stable")   # 가장 단순한 묶음부터 보고, 0.005 이상 좋아질 때만 옮긴다
    best = cand.iloc[0]
    for _, r in cand.iloc[1:].iterrows():
        if r["pr_auc"] - best["pr_auc"] >= MIN_GAIN:
            best = r
    chosen = next(k for k, v in FEATURE_SETS.items() if v == FEATURES)
    print(f"선택 규칙(PR-AUC {MIN_GAIN} 이상 좋아질 때만 복잡한 묶음): '{best['feature_set']}' / drone_lib.FEATURES: '{chosen}'"
          + ("" if best["feature_set"] == chosen else "  ← 불일치: FEATURES를 확인할 것"))

    # ---- [5] 검증 성능 + 기준선 ----
    mv = make_model().fit(fit[FEATURES], fit[TARGET])
    pv = mv.predict_proba(valid[FEATURES])[:, 1]
    ev, back = evaluate(valid, pv)
    back.to_csv(ART / "validation_by_month.csv", index=False, encoding="utf-8-sig")
    # 모델 점검(scripts/check_model.py)용: 검증 기간의 드론별 예측값과 실제 정답. 학습 데이터에서 나온 값이라 저장소에는 올리지 않는다(.gitignore).
    vp = valid[["equipment_id", "maintenance_date", "equipment_type", "maintenance_type", "equipment_age_years",
                "fail_count_90d", "days_since_last_failure"]].assign(prob=pv, failure=valid[TARGET].to_numpy())
    vp["maintenance_date"] = vp["maintenance_date"].dt.strftime("%Y-%m-%d")
    vp.to_csv(ART / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    print("\n== 검증 기간 성능 (2024-07 ~ 12) ==")
    print(f"PR-AUC {ev['pr_auc']} (재고장 비율 기준선 {ev['base_rate']}) · ROC-AUC {ev['roc_auc']}")
    print(f"매달 {CAPACITY}대 적중 합계 {ev['hits']['picked']}대 중: 모델 {ev['hits']['model']} / 오래된 순 {ev['hits']['oldest']} / 무작위(기대값) {ev['hits']['random']}")
    print(back.to_string(index=False))

    # ---- [6] 위험 표시 기준 (검증 기간에서 고른다) ----
    tab = threshold_table(valid[TARGET].to_numpy(), pv)
    tab.to_csv(ART / "threshold_cost.csv", index=False, encoding="utf-8-sig")
    t300 = tab.loc[tab["cost_300"].idxmin()]
    t600 = tab.loc[tab["cost_600"].idxmin()]
    n_pos, n_val = int(valid[TARGET].sum()), len(valid)
    policy = {"아무도 점검 안 함": {"cost_300": n_pos * COST_MISS, "cost_600": n_pos * COST_MISS_HIGH},
              "전부 점검": {"cost_300": n_val * COST_CHECK, "cost_600": n_val * COST_CHECK}}
    print("\n== 위험 표시 기준 (검증 기간 · 비용 최소) ==")
    for lab, r, c in (("놓친 재고장 300만원", t300, "cost_300"), ("놓친 재고장 600만원", t600, "cost_600")):
        print(f"{lab}: 기준 {r['threshold']:.2f} · 기준 이상 {int(r['flagged'])}건 · 놓침 {int(r['missed'])}건 · 총 비용 {int(r[c]):,}만원")
    print("비교 정책(검증 %d건, 재고장 %d건):" % (n_val, n_pos), policy)

    # ---- [7] 시험 기간 예측 (정답이 2024년 안에서 끝나는 학습 행만) ----
    mt = make_model().fit(train[td <= TEST_FIT_END][FEATURES], train.loc[td <= TEST_FIT_END, TARGET])
    out = test[["equipment_id", "maintenance_date"]].copy()
    out["prob"] = mt.predict_proba(test[FEATURES])[:, 1].round(6)
    out["maintenance_date"] = out["maintenance_date"].dt.strftime("%Y-%m-%d")
    raw_test = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "drone_test.csv", encoding="utf-8-sig")
    key = lambda x: set(zip(x["equipment_id"], x["maintenance_date"]))      # noqa: E731
    assert len(out) == 1241 == len(raw_test), (len(out), len(raw_test))
    assert key(out) == key(raw_test) and out.duplicated(["equipment_id", "maintenance_date"]).sum() == 0
    assert out["prob"].between(0, 1).all() and out["prob"].notna().all()
    REPORTS.mkdir(exist_ok=True)
    out.sort_values(["maintenance_date", "equipment_id"]).to_csv(REPORTS / "test_predictions_2025.csv", index=False, encoding="utf-8")
    print(f"\n== 시험 기간 예측 == drone_test.csv {len(raw_test):,}건 모두 예측 · 확률 평균 {out['prob'].mean():.3f} (학습 기간 재고장 비율 {train[TARGET].mean():.3f}) "
          f"· 학습에 쓴 행 {int((td <= TEST_FIT_END).sum()):,}건(~ {TEST_FIT_END})")
    print("저장: reports/test_predictions_2025.csv")

    # ---- 이번 달 예측용 최종 모델 (학습 데이터 전체) + 모델 정보 ----
    mf = make_model().fit(train[FEATURES], train[TARGET])
    ART.mkdir(exist_ok=True)
    joblib.dump(mt, MODEL_TEST_PATH)
    joblib.dump(mf, MODEL_FINAL_PATH)
    info = {
        "trained_on": f"{td.min():%Y-%m-%d} ~ {td.max():%Y-%m-%d} ({len(train):,}건) 전체 — 이번 달 예측용",
        "trained_on_test_model": f"{td.min():%Y-%m-%d} ~ {TEST_FIT_END} ({int((td <= TEST_FIT_END).sum()):,}건) — 시험 예측용",
        "split": {"fit": f"~ {FIT_END} ({len(fit):,}건)", "valid": f"{VALID_FROM} ~ {VALID_TO} ({len(valid):,}건)",
                  "dropped_between": f"{int(((td > FIT_END) & (td < VALID_FROM)).sum())}건 (2024-06, 정답 구간이 검증 기간에 걸침)"},
        "threshold": float(t300["threshold"]),
        "threshold_600": float(t600["threshold"]),
        "threshold_rule": f"검증 기간({VALID_FROM[:7]} ~ {VALID_TO[5:7]}) 비용 최소: 놓친 재고장 {COST_MISS}만원, 재점검 {COST_CHECK}만원",
        "threshold_detail": {"300": {k: (int(t300[k]) if k != "threshold" else float(t300[k])) for k in ("threshold", "flagged", "missed", "cost_300")},
                             "600": {k: (int(t600[k]) if k != "threshold" else float(t600[k])) for k in ("threshold", "flagged", "missed", "cost_600")},
                             "policy": policy, "valid_rows": n_val, "valid_failures": n_pos},
        "validation": ev,
        "feature_compare": comp.to_dict("records"),
        "feature_choice": f"{best['feature_set']} (검증 PR-AUC가 {MIN_GAIN} 이상 좋아질 때만 더 복잡한 묶음을 고르는 규칙)",
    }
    CARD_PATH.write_text(json.dumps(make_card(info), ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장: model_test.joblib, model_final.joblib, model_card.json, feature_compare.csv, validation_by_month.csv, threshold_cost.csv")


if __name__ == "__main__":
    main()
