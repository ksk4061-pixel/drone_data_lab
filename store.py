"""점검 결과 저장소 — Supabase(Postgres)에 재점검 결과를 저장하고 읽는다. app.py가 이 파일의 함수만 쓴다.

연결 주소는 코드에 쓰지 않는다. 내 PC는 .streamlit/secrets.toml, 배포 앱은 Streamlit Secrets에 둔다.
  [connections.sql]
  url = "postgresql://postgres.<프로젝트ID>:<비밀번호>@<지역>.pooler.supabase.com:5432/postgres"   # Session pooler(5432)
쓰기 비밀번호(선택): 같은 secrets에 write_password = "..." 를 넣으면 저장 · 삭제에 이 비밀번호를 요구한다.
SQL은 모두 매개변수로 넘긴다(값을 문자열에 이어 붙이지 않는다).
"""
import datetime as dt

import pandas as pd
import streamlit as st
from sqlalchemy import text

TABLE = "inspection_results"
RESULTS = ["이상 없음", "부품 교체", "수리 필요", "운용 중지"]
MAX_CREW, MAX_NOTE = 40, 300
COLUMNS = ["id", "period", "equipment_id", "inspected_on", "result", "crew", "note", "created_at"]

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id BIGSERIAL PRIMARY KEY,
    period TEXT NOT NULL,
    equipment_id TEXT NOT NULL,
    inspected_on DATE NOT NULL,
    result TEXT NOT NULL CHECK (result IN ('이상 없음', '부품 교체', '수리 필요', '운용 중지')),
    crew TEXT,
    note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _url():
    """secrets의 연결 주소. SQLAlchemy 2.1은 postgresql://의 기본 드라이버가 달라 psycopg2를 명시한다."""
    try:
        url = str(st.secrets["connections"]["sql"]["url"]).strip()
    except Exception:
        return None
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url or None


def configured():
    return _url() is not None


def write_password():
    try:
        pw = st.secrets.get("write_password")
    except Exception:
        return None
    return str(pw) if pw else None


def connect():
    return st.connection("sql", type="sql", url=_url())


@st.cache_resource(show_spinner=False)
def _init(_key):
    """표를 만든다(이미 있으면 그대로). 행 수준 보안(RLS)을 켜서 Supabase 공개 API 키로는 읽고 쓸 수 없게 한다.
    앱은 직접 연결(postgres 역할)로 접속하므로 영향이 없다."""
    with connect().session as s:
        s.execute(text(DDL))
        s.execute(text(f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY"))
        s.execute(text(f"CREATE INDEX IF NOT EXISTS {TABLE}_period_eq ON {TABLE} (period, equipment_id)"))
        s.commit()
    return True


def init_db():
    _init("v1")


def add_result(conn, period, equipment_id, inspected_on, result, crew="", note=""):
    """점검 결과 1건을 저장하고 새 번호를 돌려준다. 입력값이 잘못되면 ValueError."""
    equipment_id, crew, note = (equipment_id or "").strip(), (crew or "").strip(), (note or "").strip()
    if not equipment_id or not period:
        raise ValueError("드론 번호와 기간이 필요합니다.")
    if result not in RESULTS:
        raise ValueError("점검 결과가 올바르지 않습니다.")
    if not isinstance(inspected_on, dt.date):
        raise ValueError("점검일이 올바르지 않습니다.")
    if len(crew) > MAX_CREW or len(note) > MAX_NOTE:
        raise ValueError(f"점검반은 {MAX_CREW}자, 메모는 {MAX_NOTE}자까지입니다.")
    with conn.session as s:
        new_id = s.execute(
            text(f"INSERT INTO {TABLE} (period, equipment_id, inspected_on, result, crew, note) "
                 "VALUES (:p, :e, :d, :r, :c, :n) RETURNING id"),
            {"p": period, "e": equipment_id, "d": inspected_on, "r": result, "c": crew or None, "n": note or None},
        ).scalar_one()
        s.commit()
    return int(new_id)


def list_results(conn, period):
    df = conn.query(f"SELECT {', '.join(COLUMNS)} FROM {TABLE} WHERE period = :p ORDER BY inspected_on DESC, id DESC",
                    params={"p": period}, ttl=0)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M")
    df["inspected_on"] = pd.to_datetime(df["inspected_on"]).dt.strftime("%Y-%m-%d")
    return df


def delete_result(conn, result_id):
    with conn.session as s:
        n = s.execute(text(f"DELETE FROM {TABLE} WHERE id = :i"), {"i": int(result_id)}).rowcount
        s.commit()
    return n
