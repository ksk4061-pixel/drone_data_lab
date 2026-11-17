"""드론 예측정비 실습 공통 함수 — eda.py · train.py · score.py · make_reports.py가 함께 쓴다.

데이터는 교육용 가상 데이터다. 이 파일을 고치면 모든 스크립트의 결과가 함께 바뀐다.
규칙의 원본은 AGENTS.md다 (처리 기준 · 기간 분할 · 임계값 규칙).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"
TRAIN_CSV = DATA / "drone_train.csv"
TEST_CSV = DATA / "drone_test.csv"
MONTH_CSV = DATA / "drone_this_month.csv"
MODEL_TEST_PATH = ART / "model_test.joblib"        # 시험(2025) 예측용: 정답이 2024년 안에서 끝나는 학습 행만 사용
MODEL_FINAL_PATH = ART / "model_final.joblib"      # 이번 달(2026-01) 예측용: 학습 데이터 전체
CARD_PATH = ART / "model_card.json"

TARGET = "failure_within_30days"
FAIL_TYPES = ["사후정비", "긴급정비"]                 # 고장 때문에 한 정비 (예방정비가 아닌 것)
CAT_ALL = ["equipment_type", "maintenance_type", "fault_category", "season"]
BASE_CAT = ["equipment_type", "maintenance_type", "fault_category"]
BASE_NUM = ["equipment_age_years", "severity", "downtime_hours", "repair_duration_hours",
            "parts_replaced_count", "maintenance_cost", "maintenance_success",
            "cumulative_operating_hours", "days_since_last_maintenance", "is_first_maintenance"]
TIME_COLS = ["fail_count_30d", "fail_count_90d", "fail_count_365d", "days_since_last_failure",
             "hours_since_last_pm", "hours_since_last_maintenance"]
EXTRA_TIME = ["fail_count_total"]                     # 비교용: 시간이 지날수록 계속 커져 시험 · 이번 달에서 학습 범위를 벗어난다
LEAK_COLS = ["availability_rate"]                     # 그 달 전체 가동률 — 예측 시점에는 모른다 (이번 달 파일에도 없다)

# 후보 피처 묶음 (train.py가 검증 기간에서 비교한다)
FEATURES_BASE = BASE_CAT + BASE_NUM
FEATURES_TIME = FEATURES_BASE + TIME_COLS
FEATURES_NOCUM = [c for c in FEATURES_TIME if c != "cumulative_operating_hours"]   # 누적 비행시간(계속 커지는 raw 컬럼) 제외
FEATURES_SEASON = FEATURES_TIME + ["season"]
FEATURES_TOTAL = FEATURES_TIME + EXTRA_TIME
FEATURE_SETS = {"원본 컬럼": FEATURES_BASE, "원본 + 시간 피처": FEATURES_TIME,
                "원본 + 시간 피처 − 누적 비행시간": FEATURES_NOCUM, "원본 + 시간 피처 + 계절": FEATURES_SEASON,
                "원본 + 시간 피처 + 누적 고장 수": FEATURES_TOTAL}
REFERENCE_ONLY = {"원본 + 시간 피처 + 누적 고장 수"}   # 비교만 하고 선택 후보에서는 뺀다 (시간에 따라 커지는 피처)
FEATURES = FEATURES_NOCUM                             # 최종 선택 (train.py 검증 기간 비교 + 선택 규칙의 결과)

FEATURE_KO = {
    "equipment_type": "드론 유형", "maintenance_type": "정비 구분", "fault_category": "고장 유형", "season": "계절",
    "equipment_age_years": "경과연수", "severity": "고장 심각도", "downtime_hours": "운용중단 시간",
    "repair_duration_hours": "수리 시간", "parts_replaced_count": "교체 부품 수", "maintenance_cost": "정비 비용",
    "maintenance_success": "정비 성공 여부", "cumulative_operating_hours": "누적 비행시간",
    "days_since_last_maintenance": "직전 정비 뒤 경과일", "is_first_maintenance": "첫 정비 여부",
    "fail_count_30d": "최근 30일 고장 수", "fail_count_90d": "최근 90일 고장 수", "fail_count_365d": "최근 365일 고장 수",
    "fail_count_total": "누적 고장 수",
    "days_since_last_failure": "직전 고장 뒤 경과일", "hours_since_last_pm": "예방정비 뒤 비행시간",
    "hours_since_last_maintenance": "직전 정비 뒤 비행시간", "availability_rate": "그 달 가동률",
}


def _read(path):
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["maintenance_date"])
    return df


def load_all():
    """학습 → 시험 → 이번 달 기록을 시간순으로 한 표에 이어 붙인다. `_src` 열로 어느 파일인지 구분한다.
    시험 · 이번 달의 시간 피처가 앞 기간 기록과 이어서 계산되도록 하기 위해서다."""
    parts = [_read(TRAIN_CSV).assign(_src="train"), _read(TEST_CSV).assign(_src="test"),
             _read(MONTH_CSV).assign(_src="month")]
    return pd.concat(parts, ignore_index=True)


def clean(df):
    """이상치 · 결측을 AGENTS.md 기준대로 처리한다. 원본을 바꾸지 않고 행을 지우지 않는다."""
    df = df.copy()
    df["downtime_hours"] = df["downtime_hours"].astype(float)
    df.loc[df["downtime_hours"] >= 999, "downtime_hours"] = np.nan          # 999 = 미입력 표시값
    df["maintenance_cost"] = df["maintenance_cost"].astype(float)
    typo = df["maintenance_cost"] > 1000                                      # 자릿수 입력 오류 (정상 최댓값 478)
    df["cost_fixed"] = typo.astype(int)                                       # 나눈 행 표시
    df.loc[typo, "maintenance_cost"] = df.loc[typo, "maintenance_cost"] / 100
    df["fault_category"] = df["fault_category"].fillna("N/A")                 # 예방정비 = 고장 없음
    df["fault_code"] = df["fault_code"].fillna("N/A")
    df["is_first_maintenance"] = df["days_since_last_maintenance"].isna().astype(int)
    return df


def season_of(month):
    return {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름",
            9: "가을", 10: "가을", 11: "가을"}[month]


def add_time_features(df):
    """드론마다 지난 기록을 보고 시간 피처를 만든다. 이번 정비까지 포함해서 세고, 이후 기록은 쓰지 않는다.
    fail_count_30d / 90d / 365d: 최근 30 · 90 · 365일 고장(사후 · 긴급) 정비 수 · fail_count_total: 지금까지 고장 정비 수(비교용)
    days_since_last_failure: 직전 고장 정비 뒤 경과일 · hours_since_last_pm: 마지막 예방정비 뒤 비행시간
    hours_since_last_maintenance: 직전 정비 뒤 비행시간 · season: 정비일의 계절"""
    df = df.sort_values(["equipment_id", "maintenance_date"]).reset_index(drop=True)
    out = {k: np.full(len(df), np.nan) for k in TIME_COLS + EXTRA_TIME}
    pos = 0
    for _, g in df.groupby("equipment_id", sort=False):
        d = g["maintenance_date"].to_numpy("datetime64[D]")
        fail = g["maintenance_type"].isin(FAIL_TYPES).to_numpy()
        pm = (g["maintenance_type"] == "예방정비").to_numpy()
        cum = g["cumulative_operating_hours"].to_numpy(float)
        for i in range(len(g)):
            age = (d[i] - d[: i + 1][fail[: i + 1]]).astype(int)              # 지난 고장 정비로부터 경과일 (오늘 포함)
            j = pos + i
            out["fail_count_30d"][j] = (age <= 30).sum()
            out["fail_count_90d"][j] = (age <= 90).sum()
            out["fail_count_365d"][j] = (age <= 365).sum()
            out["fail_count_total"][j] = len(age)
            if len(age):
                out["days_since_last_failure"][j] = age.min()
            pm_idx = np.flatnonzero(pm[: i + 1])
            if len(pm_idx):
                out["hours_since_last_pm"][j] = cum[i] - cum[pm_idx[-1]]
            if i > 0:
                out["hours_since_last_maintenance"][j] = cum[i] - cum[i - 1]
        pos += len(g)
    for k, v in out.items():
        df[k] = np.round(v, 1)
    df["season"] = df["maintenance_date"].dt.month.map(season_of)
    return df


def prepare():
    """읽기 → 처리 → 시간 피처. 학습 · 시험 · 이번 달 기록이 한 표에 `_src`로 구분되어 나온다."""
    df = add_time_features(clean(load_all()))
    return df


def make_model(features=None):
    """범주형은 원-핫, 수치형은 그대로 LightGBM에 넣는 파이프라인 (결측은 LightGBM이 처리)."""
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    features = FEATURES if features is None else features
    cats = [c for c in CAT_ALL if c in features]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)],
                            remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=4, min_child_samples=40, reg_lambda=5.0,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
    return Pipeline([("pre", pre), ("model", lgbm)])


def latest_per_drone_month(df):
    """달마다 드론별 가장 최근 정비 기록 하나만 남긴다 (한 달에 20대를 고르는 업무 기준)."""
    d = df.sort_values(["maintenance_date", "equipment_id"])
    return d.assign(_m=d["maintenance_date"].dt.to_period("M")).drop_duplicates(["_m", "equipment_id"], keep="last")


def monthly_topk(df, score_col, k=20):
    """달마다 드론별 최근 기록 가운데 score_col이 높은 k대를 골랐을 때 실제 재고장 수.
    열: month, candidates, picked, hits, expected_random(무작위로 k대를 골랐을 때 기대 적중 수)."""
    rows = []
    for m, g in latest_per_drone_month(df).groupby("_m"):
        top = g.sort_values([score_col, "equipment_id"], ascending=[False, True]).head(k)
        rows.append({"month": str(m), "candidates": len(g), "picked": len(top), "hits": int(top[TARGET].sum()),
                     "expected_random": round(len(top) * float(g[TARGET].mean()), 1)})
    return pd.DataFrame(rows)


def contributions(model, X):
    """행마다 피처별 기여도(SHAP 값, 로그 오즈 단위). 원-핫으로 나뉜 열은 원래 컬럼으로 합친다."""
    Xt = model["pre"].transform(X)
    raw = model["model"].booster_.predict(Xt, pred_contrib=True)[:, :-1]     # 마지막 열 = 기준값
    contrib = pd.DataFrame(raw, columns=Xt.columns, index=X.index)
    groups = {c: next((k for k in CAT_ALL if c.startswith(k + "_")), c) for c in Xt.columns}
    return contrib.T.groupby(groups).sum().T[X.columns]


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "없음" if np.isnan(v) else f"{v:,.1f}".rstrip("0").rstrip(".")
    return str(v)


def reasons(model, X, top=3):
    """행마다 위험을 가장 많이 올린 피처 top개를 '이름(값)' 글로 돌려준다."""
    c = contributions(model, X)
    out = []
    for i in X.index:
        up = c.loc[i][c.loc[i] > 0].sort_values(ascending=False).head(top)
        out.append(", ".join(f"{FEATURE_KO.get(f, f)}({_fmt(X.at[i, f])})" for f in up.index))
    return out


def make_card(info):
    """모델 정보 json. info는 train.py가 실행해서 얻은 값 묶음이다. 한계(limits)는 보고서에 그대로 옮긴다."""
    return {
        "model": "LightGBM (drone_lib.make_model)",
        "target": "정비 뒤 30일 안 재고장(사후 · 긴급정비) 여부",
        "features": FEATURES,
        "excluded": {
            "availability_rate": "그 달 전체 가동률 — 예측 시점에는 알 수 없음(누수). 이번 달 파일에도 없음",
            "fault_code": "40종 — fault_category로 대신함",
            "equipment_id · unit_code": "식별자",
            "maintenance_date": "날짜 자체는 쓰지 않음(시간 피처만 사용)",
            "cumulative_operating_hours": "누적형 컬럼 — 시간이 지날수록 커져 시험 · 이번 달에서 학습 범위를 벗어남. 빼도 검증 성능 차이 없음(직전 정비 · 예방정비 뒤 비행시간으로 대신함)",
            "fail_count_total": "누적 고장 수 — 시간이 지날수록 계속 커짐(평균 학습 4.1, 시험 11.3, 이번 달 14.2). 최근 30 · 90 · 365일 고장 수로 대신함",
            "season": "계절 — 넣어도 검증 PR-AUC가 선택 기준(0.005)만큼 좋아지지 않아 뺌",
        },
        "capacity_per_month": 20,
        **info,
        "limits": [
            "교육용 가상 데이터로 학습했다. 실제 드론에 그대로 쓰지 않는다.",
            "확률은 '정비 뒤 30일' 기준이다. 정비 기록이 없는 드론은 판단하지 않는다.",
            "시험(2025) 정답이 없어 시험 성능은 계산하지 못했다. 성능은 검증 기간(2024-07 ~ 12) 기준이다.",
            "최종 판단은 정비 담당자가 한다. 모델은 점검 순서를 정하는 참고 자료다.",
        ],
    }


def load_final_model():
    if not MODEL_FINAL_PATH.exists():
        raise FileNotFoundError("artifacts/model_final.joblib 이 없습니다. python scripts/train.py 를 먼저 실행하세요.")
    return joblib.load(MODEL_FINAL_PATH), json.loads(CARD_PATH.read_text(encoding="utf-8"))
