"""[1] 데이터 살펴보기 + [2] 처리 기준 확인. 보고서에 쓰는 숫자를 만든다.

실행 (drone_data_lab 폴더):  python scripts/eda.py
출력: 화면 + artifacts/eda_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drone_lib import ART, FAIL_TYPES, TARGET, add_time_features, clean, load_all  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

raw = load_all()
S = {}                                                       # json으로 저장할 요약
tr, te, mo = (raw[raw["_src"] == s].copy() for s in ("train", "test", "month"))

# ---------------------------------------------------------------- 크기
S["size"] = {n: {"rows": len(d), "drones": int(d["equipment_id"].nunique()), "units": int(d["unit_code"].nunique()),
                 "from": f"{d['maintenance_date'].min():%Y-%m-%d}", "to": f"{d['maintenance_date'].max():%Y-%m-%d}"}
             for n, d in (("train", tr), ("test", te), ("month", mo))}
S["duplicate_drone_date"] = int(raw.duplicated(["equipment_id", "maintenance_date"]).sum())
print("== 크기 ==")
print(pd.DataFrame(S["size"]).T.to_string())
print("같은 드론 같은 날 중복 행:", S["duplicate_drone_date"])
S["month_rows_per_drone"] = {int(k): int(v) for k, v in mo.groupby("equipment_id").size().value_counts().sort_index().items()}
print("이번 달 드론당 정비 횟수 분포:", S["month_rows_per_drone"])

# ---------------------------------------------------------------- 결측
cols = [c for c in raw.columns if c not in ("_src", TARGET)]
miss = {n: {c: int(d[c].isna().sum()) for c in cols if d[c].isna().sum() > 0} for n, d in (("train", tr), ("test", te), ("month", mo))}
S["missing"] = miss
print("\n== 결측 건수 ==")
print(pd.DataFrame(miss).fillna(0).astype(int).to_string())
print("이번 달 파일에 없는 컬럼:", sorted(set(tr.columns) - set(mo.dropna(axis=1, how='all').columns)))

fc = tr.assign(m=tr["fault_category"].isna()).groupby("maintenance_type")["m"].agg(["sum", "size"])
S["fault_missing_by_type"] = {k: {"missing": int(r["sum"]), "rows": int(r["size"])} for k, r in fc.iterrows()}
first = tr.sort_values(["equipment_id", "maintenance_date"]).groupby("equipment_id").head(1)
S["days_since_missing"] = {"missing": int(tr["days_since_last_maintenance"].isna().sum()), "first_records": len(first),
                           "all_first": bool(tr.loc[first.index, "days_since_last_maintenance"].isna().all())}
print("\n[결측 이유] fault 결측은 정비 구분별로:\n", fc.to_string())
print("days_since_last_maintenance 결측 %d건 = 드론별 첫 기록 %d건? %s" % (
    S["days_since_missing"]["missing"], len(first), S["days_since_missing"]["all_first"]))

# ---------------------------------------------------------------- 이상치
dt = raw["downtime_hours"]
S["downtime_999"] = {n: int((d["downtime_hours"] >= 999).sum()) for n, d in (("train", tr), ("test", te), ("month", mo))}
S["downtime_next_max"] = float(dt[dt < 999].max())
big = raw[raw["maintenance_cost"] > 1000]
normal = raw[raw["maintenance_cost"] <= 1000]
S["cost_over_1000"] = {"rows": [{"equipment_id": r.equipment_id, "date": f"{r.maintenance_date:%Y-%m-%d}",
                                 "type": r.maintenance_type, "cost": int(r.maintenance_cost)} for r in big.itertuples()],
                       "by_file": {n: int((d["maintenance_cost"] > 1000).sum()) for n, d in (("train", tr), ("test", te), ("month", mo))},
                       "normal_max": float(normal["maintenance_cost"].max())}
post = normal[normal["maintenance_type"] == "사후정비"]["maintenance_cost"]
S["cost_post_normal"] = {"median": float(post.median()), "p99": float(post.quantile(0.99)), "max": float(post.max())}
fixed = big["maintenance_cost"] / 100
S["cost_fixed_range"] = [float(fixed.min()), float(fixed.max())]
print("\n== 이상치 ==")
print("downtime_hours 999:", S["downtime_999"], "| 999를 뺀 최댓값", S["downtime_next_max"])
print("maintenance_cost > 1000:", S["cost_over_1000"]["by_file"], "| 정상 행 최댓값", S["cost_over_1000"]["normal_max"])
print(big[["equipment_id", "maintenance_date", "maintenance_type", "maintenance_cost"]].to_string(index=False))
print("사후정비 정상 범위: 중앙값 %.0f, 99%% %.0f, 최대 %.0f → 100으로 나눈 값 %.0f ~ %.0f" % (
    S["cost_post_normal"]["median"], S["cost_post_normal"]["p99"], S["cost_post_normal"]["max"], *S["cost_fixed_range"]))
for c in ["repair_duration_hours", "parts_replaced_count", "cumulative_operating_hours", "equipment_age_years"]:
    print(f"  {c}: 최소 {raw[c].min()} · 99% {raw[c].quantile(.99):.1f} · 최대 {raw[c].max()}")

# ---------------------------------------------------------------- 기록 일관성
full = raw.sort_values(["equipment_id", "maintenance_date"]).reset_index(drop=True)
g = full.groupby("equipment_id")
gap = g["maintenance_date"].diff().dt.days
ok = (gap == full["days_since_last_maintenance"]) | gap.isna()
S["consistency"] = {
    "days_since_matches_date_gap": f"{int((gap == full['days_since_last_maintenance']).sum())}/{int(gap.notna().sum())}",
    "cum_hours_decrease": int((g["cumulative_operating_hours"].diff() < 0).sum()),
    "age_decrease": int((g["equipment_age_years"].diff() < 0).sum()),
    "test_drones_with_prior_records": float(te["equipment_id"].isin(tr["equipment_id"]).mean()),
    "month_drones_with_prior_records": float(mo["equipment_id"].isin(pd.concat([tr, te])["equipment_id"]).mean()),
}
print("\n== 기록 일관성 (학습 · 시험 · 이번 달을 이어서) ==")
print(S["consistency"])

# ---------------------------------------------------------------- 정답 정의 검증 (학습 데이터)
t = tr.sort_values(["equipment_id", "maintenance_date"]).reset_index(drop=True)
lab = np.zeros(len(t), int)
for _, gg in t.groupby("equipment_id", sort=False):
    fd = gg.loc[gg["maintenance_type"].isin(FAIL_TYPES), "maintenance_date"].to_numpy("datetime64[D]")
    for i, d in zip(gg.index, gg["maintenance_date"].to_numpy("datetime64[D]")):
        lab[i] = int(((fd > d) & (fd <= d + np.timedelta64(30, "D"))).any())
bad = t[lab != t[TARGET].astype(int)]
S["label_check"] = {"match_rate": round(float((lab == t[TARGET].astype(int)).mean()), 4), "mismatch": int(len(bad)),
                    "mismatch_months": {str(k): int(v) for k, v in bad.groupby(bad["maintenance_date"].dt.to_period("M")).size().items()},
                    "mismatch_label_values": {int(k): int(v) for k, v in bad[TARGET].value_counts().items()}}
print("\n== 정답 정의 검증 ==")
print("'같은 드론에 이 정비 뒤 30일 안 사후 · 긴급정비가 있으면 1'을 학습 기록만으로 다시 만들면 일치율",
      S["label_check"]["match_rate"], "| 불일치", S["label_check"]["mismatch"], S["label_check"]["mismatch_months"],
      "(전부 정답 1: 2025년 1월 기록에서 알 수 있는 재고장)")

# ---------------------------------------------------------------- 패턴 (학습 데이터)
tr2 = add_time_features(clean(tr)).assign(**{TARGET: lambda d: d[TARGET].astype(int)})
S["base_rate"] = round(float(tr2[TARGET].mean()), 4)
pat = {}
for c in ["equipment_type", "maintenance_type", "fault_category", "season", "maintenance_success", "severity"]:
    r = tr2.groupby(c)[TARGET].agg(["mean", "size"])
    pat[c] = {str(k): {"rate": round(float(v["mean"]), 3), "rows": int(v["size"])} for k, v in r.iterrows()}
age = tr2.groupby(pd.cut(tr2["equipment_age_years"], [0, 2, 4, 6, 8]), observed=True)[TARGET].agg(["mean", "size"])
pat["age_bin"] = {str(k): {"rate": round(float(v["mean"]), 3), "rows": int(v["size"])} for k, v in age.iterrows()}
pat["type_by_season"] = {a: {b: round(float(x), 3) for b, x in row.items()} for a, row in
                         tr2.pivot_table(index="equipment_type", columns="season", values=TARGET, aggfunc="mean").iterrows()}
pat["fail_count_90d"] = {str(int(k)): {"rate": round(float(v["mean"]), 3), "rows": int(v["size"])} for k, v in
                         tr2.groupby(tr2["fail_count_90d"].clip(upper=4))[TARGET].agg(["mean", "size"]).iterrows()}
pat["availability_rate_mean"] = {"재고장 1": round(float(tr2.loc[tr2[TARGET] == 1, "availability_rate"].mean()), 3),
                                "재고장 0": round(float(tr2.loc[tr2[TARGET] == 0, "availability_rate"].mean()), 3)}
S["patterns"] = pat
print("\n== 패턴 (학습 데이터, 전체 재고장률 %.3f) ==" % S["base_rate"])
for k in ["equipment_type", "maintenance_type", "fault_category", "season", "maintenance_success", "age_bin", "fail_count_90d"]:
    print(f"[{k}]", {a: f"{v['rate']:.3f} ({v['rows']}건)" for a, v in pat[k].items()})
print("[유형 × 계절]\n", pd.DataFrame(pat["type_by_season"]).T.round(2).to_string())
print("[availability_rate 평균]", pat["availability_rate_mean"], "→ 재고장 뒤 가동률이 낮아지는 '결과' 값이라 예측에 쓰면 안 된다")

# ---------------------------------------------------------------- 처리 전후 건수
cl = clean(raw)
S["clean_check"] = {
    "rows_before": len(raw), "rows_after": len(cl),
    "downtime_to_nan": int(cl["downtime_hours"].isna().sum() - raw["downtime_hours"].isna().sum()),
    "cost_divided": int(cl["cost_fixed"].sum()),
    "fault_category_filled": int((cl["fault_category"].eq("N/A") & raw["fault_category"].isna()).sum()),
    "fault_code_filled": int((cl["fault_code"].eq("N/A") & raw["fault_code"].isna()).sum()),
    "is_first_maintenance": int(cl["is_first_maintenance"].sum()),
    "cost_max_after": float(cl["maintenance_cost"].max()), "downtime_max_after": float(cl["downtime_hours"].max()),
}
exp = {"downtime_to_nan": 4, "cost_divided": 5, "fault_category_filled": 1707 + 876 + 62, "is_first_maintenance": 150}
S["clean_check"]["matches_agents_md"] = all(S["clean_check"][k] == v for k, v in exp.items()) and len(raw) == len(cl)
print("\n== 처리 전후 건수 (clean) ==")
print(pd.Series(S["clean_check"]).to_string())

ART.mkdir(exist_ok=True)
(ART / "eda_summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n저장: artifacts/eda_summary.json")
