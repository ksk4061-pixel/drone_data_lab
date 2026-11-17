"""이번 달(2026-01) 재점검 20대를 '단순 규칙'으로도 뽑아 모델 목록과 비교한다. 다른 드론은 한 대씩 판단 근거를 붙인다.

실행 (drone_data_lab 폴더, data/ 필요. 먼저 python scripts/train.py · score.py):  python scripts/compare_rule.py
출력: 화면 + artifacts/rule_vs_model.json
단순 규칙(모델 점검 B3과 같은 규칙): 최근 정비가 사후 · 긴급정비인 드론을 먼저, 그다음 최근 90일 고장 수가 많은 순, 같으면 경과연수가 높은 순.
이 스크립트는 모델 · 순위를 바꾸지 않는다. 어느 드론을 점검할지는 정비 담당자가 정한다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drone_lib import ART, FEATURE_KO, FEATURES, REPORTS, contributions, load_final_model, prepare  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CAP = 20
RULE_TEXT = "최근 정비가 사후 · 긴급정비인 드론을 먼저, 그다음 최근 90일 고장 수가 많은 순, 같으면 경과연수가 높은 순"


def fmt(v):
    return "없음" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{v:g}" if isinstance(v, (int, float)) else str(v))


def main():
    model, card = load_final_model()
    score = json.loads((REPORTS / "score_2026-01.json").read_text(encoding="utf-8"))
    mrank = {r["equipment_id"]: r for r in score["rows"]}

    df = prepare()
    m = (df[df["_src"] == "month"].sort_values(["maintenance_date", "equipment_id"]).drop_duplicates("equipment_id", keep="last")
         .set_index("equipment_id"))
    m["is_fail"] = m["maintenance_type"].isin(["사후정비", "긴급정비"]).astype(int)
    m["rule_score"] = m["is_fail"] * 10000 + m["fail_count_90d"].fillna(0) * 100 + m["equipment_age_years"]
    rule = m.reset_index().sort_values(["rule_score", "equipment_id"], ascending=[False, True]).reset_index(drop=True)
    rule["rule_rank"] = rule.index + 1
    rrank = rule.set_index("equipment_id")["rule_rank"].to_dict()

    contrib = contributions(model, m[FEATURES])
    model_top = [r["equipment_id"] for r in score["rows"][:CAP]]
    rule_top = list(rule["equipment_id"][:CAP])
    both = [d for d in model_top if d in rule_top]

    def detail(did):
        r = m.loc[did]
        c = contrib.loc[did].sort_values()
        up = [f"{FEATURE_KO[f]}({fmt(m.at[did, f])})" for f in c.index[::-1][:3] if c[f] > 0]
        down = [f"{FEATURE_KO[f]}({fmt(m.at[did, f])})" for f in c.index[:3] if c[f] < 0]
        return {"equipment_id": did, "equipment_type": r["equipment_type"], "unit_code": r["unit_code"], "date": f"{r['maintenance_date']:%Y-%m-%d}",
                "maintenance_type": r["maintenance_type"], "model_rank": mrank[did]["rank"], "model_prob": mrank[did]["prob"],
                "rule_rank": int(rrank[did]), "fail_30d": fmt(r["fail_count_30d"]), "fail_90d": fmt(r["fail_count_90d"]), "fail_365d": fmt(r["fail_count_365d"]),
                "age": fmt(r["equipment_age_years"]), "success": fmt(r["maintenance_success"]), "up": up, "down": down}

    def basis_model_only(d):
        why_rule = ("최근 정비가 예방정비라 규칙에서는 뒤로 밀림" if d["maintenance_type"] == "예방정비"
                    else f"최근 90일 고장 {d['fail_90d']}건이라 규칙 {d['rule_rank']}위로 밀림")
        return (f"모델 {d['model_rank']}위(확률 {d['model_prob'] * 100:.1f}%), 규칙 {d['rule_rank']}위. {why_rule}. "
                f"모델은 {', '.join(d['up']) or '뚜렷한 상승 요인 없음'}을(를) 근거로 확률을 높게 봄.")

    def basis_rule_only(d):
        return (f"규칙 {d['rule_rank']}위(최근 정비 {d['maintenance_type']}, 최근 90일 고장 {d['fail_90d']}건), 모델 {d['model_rank']}위(확률 {d['model_prob'] * 100:.1f}%). "
                f"모델이 확률을 낮춘 요인: {', '.join(d['down']) or '뚜렷한 하락 요인 없음'}. 모델이 확률을 올린 요인: {', '.join(d['up']) or '없음'}.")

    model_only = [dict(detail(d), basis=None) for d in model_top if d not in rule_top]
    rule_only = [dict(detail(d), basis=None) for d in rule_top if d not in model_top]
    for d in model_only:
        d["basis"] = basis_model_only(d)
    for d in rule_only:
        d["basis"] = basis_rule_only(d)

    out = {"period": score["period"], "capacity": CAP, "rule": RULE_TEXT, "both": both, "n_both": len(both),
           "model_only": model_only, "rule_only": rule_only,
           "rule_top20": [{"rank": int(r.rule_rank), "equipment_id": r.equipment_id, "maintenance_type": r.maintenance_type,
                           "fail_90d": fmt(r.fail_count_90d), "model_rank": mrank[r.equipment_id]["rank"], "in_model_top": r.equipment_id in model_top}
                          for r in rule.head(CAP).itertuples()]}
    (ART / "rule_vs_model.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"규칙: {RULE_TEXT}\n겹치는 드론 {len(both)}대 / 모델만 {len(model_only)}대 / 규칙만 {len(rule_only)}대")
    print("\n[모델만 선정]"); [print(f"  {d['equipment_id']} — {d['basis']}") for d in model_only]
    print("\n[규칙만 선정]"); [print(f"  {d['equipment_id']} — {d['basis']}") for d in rule_only]


if __name__ == "__main__":
    main()
