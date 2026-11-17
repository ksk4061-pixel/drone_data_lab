"""[모델 점검 A] 시간 피처를 원본 기록에서 다시 세어 본다. 드론 한 대씩 근거를 따라가 볼 수 있는 자료도 만든다.

실행 (drone_data_lab 폴더, data/ 필요):  python scripts/check_features.py
출력: 화면 + artifacts/check_features_summary.json · trace_targets.json + reports/모델점검A보고서.md
점검은 pandas · drone_lib와 따로, 파이썬 기본(csv · datetime)만으로 모든 기록(3,693건)의 시간 피처를 다시 센다.
결과가 나빠도 코드를 이 스크립트가 고치지 않는다. 보고만 한다.

합격 기준(결과를 보기 전에 정한 값):
  A1 전수 재계산 대조   전체 기록의 시간 피처 6개가 별도 코드 재계산과 모두 일치(불일치 0건, 소수 피처는 0.05 이내)
  A2 경계 사례          정확히 30 · 90 · 365일 전 고장이 있는 기록(경계)과 하루 바깥(31 · 91 · 366일) 기록도 불일치 0건
  A3 파일 경계          시험 · 이번 달 기록(앞 파일의 기록과 이어서 계산하는 행)도 불일치 0건
  A4 미래 기록 비의존   어느 날짜 이후의 기록을 모두 지워도, 그 날짜까지의 기록의 피처가 변하지 않음
  A5 대상 20대 근거     이번 달 재점검 대상 20대의 피처가 재계산과 일치하고, 근거 기록을 직접 따라갈 수 있음
주의: 별도 코드지만 피처의 '정의'(이번 정비 포함, 30 · 90 · 365일은 이하)는 같은 해석으로 구현했다. 정의 자체가 맞는지는 사람이 판단한다.
"""
import csv
import datetime as dt
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drone_lib import (ART, FEATURE_KO, FEATURES, MONTH_CSV, REPORTS, TEST_CSV, TRAIN_CSV, add_time_features, clean,  # noqa: E402
                       contributions, load_all, load_final_model, prepare)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FAIL = ("사후정비", "긴급정비")
FEATS = ["fail_count_30d", "fail_count_90d", "fail_count_365d", "days_since_last_failure", "hours_since_last_pm", "hours_since_last_maintenance"]
PASS, FAILV = "통과", "실패"


def read_raw():
    """원본 CSV 3개를 파이썬 기본 모듈로 읽는다 (pandas 사용 안 함)."""
    rows = []
    for src, p in (("train", TRAIN_CSV), ("test", TEST_CSV), ("month", MONTH_CSV)):
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows.append({"src": src, "id": r["equipment_id"], "date": dt.date.fromisoformat(r["maintenance_date"]),
                             "type": r["maintenance_type"], "cum": float(r["cumulative_operating_hours"])})
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    for lst in by.values():
        lst.sort(key=lambda r: r["date"])
    return by


def recount(lst, i):
    """lst[i]의 시간 피처를 그 정비일까지의 기록(i번째 포함)만으로 센다."""
    today = lst[i]["date"]
    ages = [(today - x["date"]).days for x in lst[: i + 1] if x["type"] in FAIL]
    pm = [x for x in lst[: i + 1] if x["type"] == "예방정비"]
    return {"fail_count_30d": sum(a <= 30 for a in ages), "fail_count_90d": sum(a <= 90 for a in ages),
            "fail_count_365d": sum(a <= 365 for a in ages), "days_since_last_failure": min(ages) if ages else None,
            "hours_since_last_pm": round(lst[i]["cum"] - pm[-1]["cum"], 1) if pm else None,
            "hours_since_last_maintenance": round(lst[i]["cum"] - lst[i - 1]["cum"], 1) if i else None,
            "_boundary": any(a in (30, 90, 365) for a in ages), "_outside": any(a in (31, 91, 366) for a in ages)}


def same(a, b):
    """모델 값 a(NaN 가능)와 재계산 값 b(None 가능)가 같은가."""
    a_nan = a is None or (isinstance(a, float) and math.isnan(a))
    if b is None or a_nan:
        return a_nan and b is None
    return abs(float(a) - float(b)) <= 0.051


def jnum(v):
    if isinstance(v, str):                      # 범주형 피처(드론 유형 · 정비 구분 · 고장 유형)는 그대로
        return v
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else (int(v) if float(v).is_integer() else round(float(v), 4))


def main():
    by = read_raw()
    df = prepare()
    model_val = {(r.equipment_id, r.maintenance_date.date()): {f: getattr(r, f) for f in FEATS} for r in df.itertuples()}

    # ---------------------------------------------------------------- A1 ~ A3 전수 대조
    stats = {"all": [0, 0], "boundary": [0, 0], "outside": [0, 0], "nonTrain": [0, 0]}      # [기록 수, 불일치 수]
    per_feat = {f: 0 for f in FEATS}
    bad = []
    for did, lst in by.items():
        for i, r in enumerate(lst):
            rc = recount(lst, i)
            mv = model_val[(did, r["date"])]
            diffs = [f for f in FEATS if not same(mv[f], rc[f])]
            for key, cond in (("all", True), ("boundary", rc["_boundary"]), ("outside", rc["_outside"]), ("nonTrain", r["src"] != "train")):
                if cond:
                    stats[key][0] += 1
                    stats[key][1] += bool(diffs)
            for f in diffs:
                per_feat[f] += 1
            if diffs and len(bad) < 10:
                bad.append({"id": did, "date": str(r["date"]), "features": diffs, "model": {f: jnum(mv[f]) for f in diffs}, "recount": {f: rc[f] for f in diffs}})
    n_rows = stats["all"][0]

    # ---------------------------------------------------------------- A4 미래 기록 비의존
    base = clean(load_all())
    future = []
    for cut in ("2024-06-30", "2025-06-30"):
        c = pd.Timestamp(cut)
        a = add_time_features(base)
        a = a[a["maintenance_date"] <= c].set_index(["equipment_id", "maintenance_date"])[FEATS].sort_index()
        b = add_time_features(base[base["maintenance_date"] <= c]).set_index(["equipment_id", "maintenance_date"])[FEATS].sort_index()
        n_diff = int((a.fillna(-1e9) != b.fillna(-1e9)).any(axis=1).sum())
        future.append({"cut": cut, "rows": len(a), "changed": n_diff})

    # ---------------------------------------------------------------- A5 대상 20대 근거 + 표본
    model, card = load_final_model()
    score = json.loads((REPORTS / "score_2026-01.json").read_text(encoding="utf-8"))
    targets = score["rows"][: card["capacity_per_month"]]
    month = df[df["_src"] == "month"].sort_values(["maintenance_date", "equipment_id"]).drop_duplicates("equipment_id", keep="last").set_index("equipment_id")
    X = month.loc[[t["equipment_id"] for t in targets], FEATURES]
    contrib = contributions(model, X)
    drones, a5_bad = [], 0
    for t, (did, xr) in zip(targets, X.iterrows()):
        lst = by[did]
        i = next(k for k, r in enumerate(lst) if r["src"] == "month" and r["date"] == month.loc[did, "maintenance_date"].date())
        rc, today = recount(lst, i), lst[i]["date"]
        ok = all(same(xr[f], rc[f]) for f in FEATS)
        a5_bad += not ok
        events = [{"date": str(x["date"]), "type": x["type"], "age_days": (today - x["date"]).days,
                   "in30": (today - x["date"]).days <= 30, "in90": (today - x["date"]).days <= 90, "in365": (today - x["date"]).days <= 365}
                  for x in reversed(lst[: i + 1]) if x["type"] in FAIL and (today - x["date"]).days <= 400]
        pm = [x for x in lst[: i + 1] if x["type"] == "예방정비"]
        drones.append({"rank": t["rank"], "equipment_id": did, "equipment_type": t["equipment_type"], "unit_code": t["unit_code"],
                       "prob": t["prob"], "date": str(today), "type": lst[i]["type"],
                       "features": {f: jnum(xr[f]) for f in FEATURES}, "recount": {f: rc[f] for f in FEATS},
                       "match": ok, "events": events,
                       "last_pm": {"date": str(pm[-1]["date"]) if pm else None, "hours_since": rc["hours_since_last_pm"]},
                       "prev": {"date": str(lst[i - 1]["date"]) if i else None, "hours_since": rc["hours_since_last_maintenance"]},
                       "contrib": {f: round(float(contrib.loc[did, f]), 4) for f in FEATURES}})
    (ART / "trace_targets.json").write_text(json.dumps({"period": score["period"], "labels": {f: FEATURE_KO.get(f, f) for f in FEATURES},
                                                        "drones": drones}, ensure_ascii=False, indent=1), encoding="utf-8")

    rng = random.Random(42)                                                     # 표본: 대상 상위 5대 + 모든 기록에서 임의 5건
    pool = [(did, i) for did, lst in by.items() for i in range(len(lst))]
    sample = [(d["equipment_id"], next(k for k, r in enumerate(by[d["equipment_id"]]) if str(r["date"]) == d["date"])) for d in drones[:5]] + rng.sample(pool, 5)
    samples = []
    for did, i in sample:
        r, rc = by[did][i], recount(by[did], i)
        mv = model_val[(did, r["date"])]
        samples.append({"id": did, "date": str(r["date"]), "type": r["type"], "src": r["src"], "model": {f: jnum(mv[f]) for f in FEATS},
                        "recount": {f: rc[f] for f in FEATS}, "match": all(same(mv[f], rc[f]) for f in FEATS)})

    # ---------------------------------------------------------------- 판정 · 저장
    def ck(i, name, crit, res, ok):
        return {"id": i, "name": name, "criterion": crit, "result": res, "verdict": PASS if ok else FAILV}
    checks = [
        ck("A1", "전수 재계산 대조", "전체 기록의 시간 피처 6개가 모두 일치", f"{n_rows:,}건 중 불일치 {stats['all'][1]}건 (피처별 " + ", ".join(f"{FEATURE_KO[f]} {v}" for f, v in per_feat.items()) + ")", stats["all"][1] == 0),
        ck("A2", "경계 사례", "정확히 30 · 90 · 365일 전 고장 기록, 하루 바깥 기록 모두 일치",
           f"경계 {stats['boundary'][0]}건 중 불일치 {stats['boundary'][1]} · 하루 바깥 {stats['outside'][0]}건 중 불일치 {stats['outside'][1]}", stats["boundary"][1] + stats["outside"][1] == 0),
        ck("A3", "파일 경계(학습 → 시험 → 이번 달)", "앞 파일 기록과 이어서 계산한 행 모두 일치", f"시험 · 이번 달 {stats['nonTrain'][0]:,}건 중 불일치 {stats['nonTrain'][1]}건", stats["nonTrain"][1] == 0),
        ck("A4", "미래 기록 비의존", "날짜 이후 기록을 지워도 그 날짜까지의 피처가 변하지 않음",
           " · ".join(f"{x['cut']} 이후 삭제: {x['rows']:,}건 중 변한 것 {x['changed']}건" for x in future), all(x["changed"] == 0 for x in future)),
        ck("A5", "이번 달 대상 20대 근거", "20대의 피처가 재계산과 일치", f"{len(drones)}대 중 불일치 {a5_bad}대", a5_bad == 0),
    ]
    S = {"checks": checks, "rows": n_rows, "stats": stats, "per_feature_mismatch": per_feat, "mismatch_examples": bad, "future": future, "samples": samples}
    (ART / "check_features_summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== 모델 점검 A (시간 피처 전수 재계산) ==")
    for c in checks:
        print(f"[{c['verdict']}] {c['id']} {c['name']}\n        기준: {c['criterion']}\n        결과: {c['result']}")
    if bad:
        print("\n불일치 예:", json.dumps(bad[:5], ensure_ascii=False))
    print("\n-- 표본 10건 (모델 값 / 재계산) --")
    for s in samples:
        print(f"  {s['id']} {s['date']} {s['type']} [{s['src']}] 일치={s['match']} | 30/90/365일 {s['model']['fail_count_30d']}/{s['model']['fail_count_90d']}/{s['model']['fail_count_365d']} "
              f"· 직전 고장 {s['model']['days_since_last_failure']}일 · 예방정비 뒤 {s['model']['hours_since_last_pm']}h · 직전 정비 뒤 {s['model']['hours_since_last_maintenance']}h")
    build_report(checks, S, drones)
    print("\n저장: artifacts/check_features_summary.json, artifacts/trace_targets.json, reports/모델점검A보고서.md")


def build_report(checks, S, drones):
    def md(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                         + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])
    fmt = lambda v: "없음" if v is None else v                                     # noqa: E731
    line = lambda d: f"{d['fail_count_30d']} / {d['fail_count_90d']} / {d['fail_count_365d']} · {fmt(d['days_since_last_failure'])}일 · {fmt(d['hours_since_last_pm'])}h · {fmt(d['hours_since_last_maintenance'])}h"   # noqa: E731
    n_ok = sum(c["verdict"] == PASS for c in checks)
    txt = f"""# 모델 점검 A 보고서 (시간 피처 전수 재계산)

- 작성: AI 에이전트 초안 · 검토: 정비 담당자 (검토 전)
- 자료: 교육용 가상 데이터. 원본 기록 3개 파일(학습 · 시험 · 이번 달) 전체를 점검함
- 이 점검은 코드 · 모델을 고치지 않음. 결과만 보고함

## 1. 종합
- 점검 {len(checks)}개 중 **{n_ok}개 통과**. 전체 기록 {S['rows']:,}건의 시간 피처 6개(최근 30 · 90 · 365일 고장 수, 직전 고장 뒤 경과일, 예방정비 뒤 비행시간, 직전 정비 뒤 비행시간)를 원본 CSV에서 `drone_lib`와 따로 다시 세어 대조함
- 별도 코드는 pandas 없이 파이썬 기본(`csv` · `datetime`)만 사용함. 다만 **피처의 정의**(이번 정비 포함, 30 · 90 · 365일 ‘이하’)는 같은 해석으로 구현했으므로, 정의 자체가 업무에 맞는지는 사람이 판단해야 함
- 출처: `python scripts/check_features.py` → `artifacts/check_features_summary.json`

## 2. 점검 기준과 판정
기준은 결과를 보기 전에 `scripts/check_features.py` 맨 위에 적어 둔 값임.

{md(['번호', '점검', '합격 기준', '결과', '판정'], [(c['id'], c['name'], c['criterion'], c['result'], c['verdict']) for c in checks])}

## 3. 표본 10건 손검산용 표
대상 상위 5대(이번 달 최근 기록)와 전체 기록에서 임의로 뽑은 5건(`seed=42`)임. 값은 ‘최근 30일 / 90일 / 365일 고장 수 · 직전 고장 뒤 경과일 · 예방정비 뒤 비행시간 · 직전 정비 뒤 비행시간’ 순서임. 원본 기록에서 직접 세어 보려면 앱의 ‘⑧ 점검 A’ 탭에서 근거 기록을 볼 수 있음.

{md(['드론', '정비일', '정비 구분', '파일', '모델 값', '재계산 값', '일치'], [(s['id'], s['date'], s['type'], s['src'], line(s['model']), line(s['recount']), '일치' if s['match'] else '불일치') for s in S['samples']])}

## 4. 한계
- 같은 사람(AI)이 같은 정의로 구현한 것을 다시 센 것이므로 **구현 오류**는 잡지만 **정의 해석 오류**는 잡지 못함. 예를 들어 ‘이번 정비를 고장 수에 포함하는 것’이 옳은지는 별개의 판단임
- 점검은 피처 계산에 한정됨. 모델이 이 피처로 잘 맞히는지는 `모델점검보고서`(B안)가 다룸
- 원본 `data/`는 저장소에 올리지 않았으므로 이 점검을 다시 실행하려면 데이터 3개가 필요함

## 5. 숫자의 출처
{md(['항목', '출처'], [('모든 점검 결과', '`python scripts/check_features.py` → `artifacts/check_features_summary.json`'), ('대상 20대 근거 기록', '`artifacts/trace_targets.json`'), ('모델이 계산한 피처 값', '`drone_lib.prepare()`')])}
"""
    (REPORTS / "모델점검A보고서.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
