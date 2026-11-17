# 드론 예측정비 (drone_data_lab)

군 특화 AI 보수 교육 실습 2

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
python scripts/md_to_html.py reports/2026-01_드론_재점검계획.md   # 앱의 '⑤ 보고서' 탭이 이 HTML을 보여 준다
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

앱의 분석 결과는 `reports/` · `artifacts/`의 파일을 읽는다. Secrets는 아래 '점검 결과 저장'을 쓸 때만 필요하다.

## 점검 결과 저장 (Supabase)
앱의 **② 점검 결과** 탭은 재점검 대상 드론의 점검 결과를 Supabase(Postgres)에 저장한다. 표(`inspection_results`)는 처음 접속할 때 자동으로 만들어지고 행 수준 보안(RLS)이 켜진다.

1. supabase.com에서 프로젝트를 만든다(비밀번호는 영문 · 숫자 12자 이상). **Connect → Direct → Session pooler → URI**로 주소를 복사하고 `[YOUR-PASSWORD]`를 대괄호까지 지우고 비밀번호로 바꾼다. 포트는 5432여야 한다.
2. 내 PC: `.streamlit/secrets.toml`에 넣는다 (`.gitignore`로 커밋되지 않음).
3. 배포 앱: share.streamlit.io → 앱 **⋮ → Settings → Secrets**에 같은 내용을 붙여 넣고 **Save changes**를 누른다.
```toml
[connections.sql]
url = "postgresql://postgres.<프로젝트ID>:<비밀번호>@<지역>.pooler.supabase.com:5432/postgres"

# 선택: 공개 앱이면 꼭 넣는다. 넣으면 저장 · 삭제에 이 비밀번호가 필요하다.
write_password = "<쓰기 비밀번호>"
```
- 연결 주소와 비밀번호는 `config.toml` · 코드 · 커밋에 쓰지 않는다. Streamlit은 연결 정보를 `secrets.toml`에서만 읽는다.
- 무료 프로젝트는 1주일 접속이 없으면 멈춘다(데이터는 남는다). 이때 이 탭은 오류 안내를 보여 주고 나머지 탭은 정상 동작한다.
