# 드론 예측정비 (drone_data_lab)

정비를 마친 드론 가운데 **30일 안에 다시 고장 날 드론**을 예측하고, 2026-01에 재점검할 드론 20대를 고르는 분석 프로젝트다.
**교육용 가상 데이터**이며 실제 부대 · 기종 정보가 아니다. 최종 판단은 정비 담당자가 한다.

## 결과 보기
- 앱: `streamlit run app.py` (재점검 목록 · 검증 성능 · 위험 표시 기준 · 보고서 · 모델 정보)
- `reports/분석보고서.md` · `reports/2026-01_드론_재점검계획.md` (HTML 판: `.html`) · `reports/test_predictions_2025.csv`
- 숫자의 근거: `artifacts/` (model_card.json, validation_by_month.csv, threshold_cost.csv, feature_compare.csv, eda_summary.json)

## 다시 만들기 (Windows PowerShell)
강의 원본 데이터 3개(`drone_train.csv` · `drone_test.csv` · `drone_this_month.csv`)는 이 저장소에 올리지 않는다. `data/` 폴더에 넣고 실행한다.
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/eda.py            # 데이터 살펴보기
python scripts/train.py          # 학습 · 검증 · 시험 기간 예측
python scripts/score.py          # 2026-01 재점검 선정
python scripts/make_reports.py   # 보고서
python scripts/md_to_html.py reports/분석보고서.md
```
규칙의 원본은 `AGENTS.md`다.

## 배포 (Streamlit Community Cloud)
share.streamlit.io → **Create app** → **Deploy a public app from GitHub**

| 항목 | 값 |
|---|---|
| Repository | `<내 GitHub ID>/<이 저장소 이름>` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version | 3.12 (Advanced settings) |

앱은 `reports/` · `artifacts/`의 결과 파일만 읽는다. 비밀 값(Secrets)은 필요 없다.
