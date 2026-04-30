"""SQLite 기반 요청 로그.

사용:
    from request_log import log_request, log_feedback, get_recent

    log_id = log_request(query, top_k, retrieved_docs, answer, latency_ms,
                         llm_model="gpt-4o-mini", embedding_model="text-embedding-3-small")
    # later
    log_feedback(log_id, "up")

    # 조회
    rows = get_recent(limit=20)
"""

import json
import os
import sqlite3
import time
from pathlib import Path

LOG_DB_PATH = Path(os.getenv("REQUEST_LOG_DB", "data/logs/requests.sqlite"))


def _connect():
    LOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LOG_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            query TEXT NOT NULL,
            top_k INTEGER,
            retrieved_json TEXT,
            answer TEXT,
            latency_ms INTEGER,
            llm_model TEXT,
            embedding_model TEXT,
            feedback TEXT,
            feedback_at TEXT
        )
    """)
    conn.commit()
    return conn


def _serialize_retrieved(retrieved_docs):
    """검색 결과를 가벼운 JSON으로 직렬화 (전체 본문은 저장 안 함)."""
    items = []
    for d in retrieved_docs:
        if d.get("doc_type") == "regulation":
            items.append({
                "type": "regulation",
                "citation": d.get("citation"),
                "distance": round(d.get("distance", 0), 4),
            })
        else:
            items.append({
                "type": "qna",
                "id": d.get("id"),
                "a_date": d.get("a_date"),
                "distance": round(d.get("distance", 0), 4),
            })
    return json.dumps(items, ensure_ascii=False)


def log_request(query, top_k, retrieved_docs, answer, latency_ms,
                llm_model="", embedding_model=""):
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO requests (ts, query, top_k, retrieved_json, answer, latency_ms,"
        " llm_model, embedding_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            query,
            top_k,
            _serialize_retrieved(retrieved_docs),
            answer,
            int(latency_ms),
            llm_model,
            embedding_model,
        ),
    )
    log_id = cur.lastrowid
    conn.commit()
    conn.close()
    return log_id


def log_feedback(log_id: int, feedback: str):
    """feedback: 'up' | 'down' (또는 빈 문자열)."""
    conn = _connect()
    conn.execute(
        "UPDATE requests SET feedback = ?, feedback_at = ? WHERE id = ?",
        (feedback, time.strftime("%Y-%m-%dT%H:%M:%S"), log_id),
    )
    conn.commit()
    conn.close()


def get_recent(limit: int = 20):
    conn = _connect()
    rows = conn.execute(
        "SELECT id, ts, query, top_k, latency_ms, feedback FROM requests"
        " ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def stats():
    """총 요청 수, 피드백 비율 요약."""
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    up = conn.execute("SELECT COUNT(*) FROM requests WHERE feedback='up'").fetchone()[0]
    down = conn.execute("SELECT COUNT(*) FROM requests WHERE feedback='down'").fetchone()[0]
    avg_latency = conn.execute("SELECT AVG(latency_ms) FROM requests").fetchone()[0] or 0
    conn.close()
    return {
        "total": total,
        "feedback_up": up,
        "feedback_down": down,
        "feedback_none": total - up - down,
        "avg_latency_ms": int(avg_latency),
    }


if __name__ == "__main__":
    import sys, io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    s = stats()
    print(f"총 {s['total']}건, 👍 {s['feedback_up']} / 👎 {s['feedback_down']} / 무 {s['feedback_none']}")
    print(f"평균 latency: {s['avg_latency_ms']}ms")
    print()
    print("최근 10건:")
    for r in get_recent(10):
        fb = {"up": "👍", "down": "👎"}.get(r["feedback"], "  ")
        print(f"  [{r['id']:4d}] {r['ts']} {fb} ({r['latency_ms']}ms) {r['query'][:60]}")
