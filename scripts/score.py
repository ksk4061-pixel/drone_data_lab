"""[8] 이번 달(2026-01) 재점검 20대 선정. 드론 위험도 · 순위 · 확률은 이 스크립트 출력만 보고서에 인용한다.

실행 (drone_data_lab 폴더):  python scripts/score.py            (상위 20대)
                             python scripts/score.py --top 15
출력: 화면 + reports/score_2026-01.json (전체 드론 순위)
한 드론이 이번 달에 여러 번 정비했으면 가장 최근 기록 하나로 판단한다.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drone_lib import FEATURES, REPORTS, TARGET, load_final_model, prepare, reasons  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=None, help="표로 보여 줄 대수 (기본: 월 재점검 가능 대수)")
    top = ap.parse_args().top
    try:
        model, card = load_final_model()
    except FileNotFoundError as e:
        sys.exit(str(e))
    cap, thr = card["capacity_per_month"], card["threshold"]
    top = top or cap

    df = prepare()
    month = df[df["_src"] == "month"].copy()
    n_rows = len(month)
    new = month.sort_values(["maintenance_date", "equipment_id"]).drop_duplicates("equipment_id", keep="last").copy()
    new["prob"] = model.predict_proba(new[FEATURES])[:, 1]
    new["reasons"] = reasons(model, new[FEATURES])
    new["multi"] = new["equipment_id"].map(month.groupby("equipment_id").size())          # 이번 달 정비 횟수
    new = new.sort_values(["prob", "equipment_id"], ascending=[False, True]).reset_index(drop=True)
    new["rank"] = new.index + 1
    new["risk_high"] = new["prob"] >= thr

    rows = [{"rank": int(r["rank"]), "equipment_id": r["equipment_id"], "equipment_type": r["equipment_type"],
             "unit_code": r["unit_code"], "maintenance_date": f"{r['maintenance_date']:%Y-%m-%d}",
             "maintenance_type": r["maintenance_type"], "prob": round(float(r["prob"]), 4), "reasons": r["reasons"],
             "risk_high": bool(r["risk_high"]), "month_maintenance_count": int(r["multi"]),
             "age_years": float(r["equipment_age_years"])} for _, r in new.iterrows()]
    out = {"period": "2026-01", "threshold": thr, "threshold_600": card["threshold_600"], "capacity": cap,
           "maintenance_rows": n_rows, "drones": len(new), "above_threshold": int(new["risk_high"].sum()),
           "above_threshold_over_capacity": int(max(0, new["risk_high"].sum() - cap)),
           "multi_maintenance_drones": int((new["multi"] > 1).sum()), "rows": rows}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "score_2026-01.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"## 2026-01 재점검 후보 상위 {top}대 (이번 달 정비 기록 {n_rows}건 · 드론 {len(new)}대 · 위험 표시 기준 {thr:.2f})\n")
    print("| 순위 | 드론 | 유형 | 부대 | 최근 정비일 | 정비 구분 | 확률 | 위험 | 위험을 올린 요인 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows[:top]:
        print(f"| {r['rank']} | {r['equipment_id']} | {r['equipment_type']} | {r['unit_code']} | {r['maintenance_date']} | "
              f"{r['maintenance_type']} | {r['prob'] * 100:.1f}% | {'높음' if r['risk_high'] else '-'} | {r['reasons']} |")
    print(f"\n- 위험 높음(기준 이상): {out['above_threshold']}대 · 월 재점검 가능 대수: {cap}대 · 20대를 넘는 위험 높음: {out['above_threshold_over_capacity']}대")
    print(f"- 이번 달 2회 이상 정비한 드론: {out['multi_maintenance_drones']}대 (가장 최근 기록으로 판단)")
    v = card["validation"]
    print(f"- 모델: 검증(2024-07 ~ 12) PR-AUC {v['pr_auc']} (기준선 {v['base_rate']}), 매달 20대 적중 모델 {v['hits']['model']} / 오래된 순 {v['hits']['oldest']} / 무작위 {v['hits']['random']}")
    print("- 전체 결과 저장: reports/score_2026-01.json")


if __name__ == "__main__":
    main()
