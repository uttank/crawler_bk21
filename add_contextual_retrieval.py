"""Contextual Retrieval (Anthropic 기법) — 규정·매뉴얼 청크에 맥락 prefix 추가.

각 청크에 대해 LLM이 50자 이내 맥락 요약을 생성하고, document 텍스트 앞에 prefix로 붙여
재임베딩한다. 같은 chroma_id로 upsert하므로 metadata와 ID는 보존됨.

효과: 짧은 청크(예: 매뉴얼의 단편)도 검색 키워드가 풍부해져 retrieval 적중률↑.

비용 추정 (현 데이터 기준): ~$0.10 (LLM 요약 + 재임베딩)
시간: ~10분 (직렬 LLM 호출)

사용:
    python add_contextual_retrieval.py             # 규정·매뉴얼 전체
    python add_contextual_retrieval.py --doc-key manual  # 특정 doc_key만
    python add_contextual_retrieval.py --redo      # 이미 적용된 것도 다시
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
COLLECTION_NAME = "bk21_qna"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"
PREFIX_MARKER = "[맥락]"
BATCH_SIZE = 50

CONTEXT_PROMPT = """다음은 BK21 사업 규정/매뉴얼 청크입니다.
이 청크가 검색에서 잘 매칭되도록 청크의 핵심 주제와 키워드를 한 줄로 요약하세요.

규칙:
- 50자 이내, 한 줄
- 명사 위주, 검색 키워드가 풍부하게 (예: "단기연수", "참여대학원생", "국제학술대회 인정 기준")
- "이 청크는..." 같은 군더더기 없이 직접 표현
- 도메인 용어 그대로 사용
- 문서·섹션 메타는 따로 이미 인덱싱되므로 본문 내용 위주

문서명: {doc_name}
섹션: {citation}
본문:
{body}

한 줄 요약:"""


def generate_context(client, doc_name: str, citation: str, body: str) -> str:
    """LLM이 만든 한 줄 맥락 요약."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": CONTEXT_PROMPT.format(
                doc_name=doc_name,
                citation=citation,
                body=body[:1500],  # 토큰 절감
            ),
        }],
        temperature=0.0,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip().split("\n")[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-key", default=None,
                        help="특정 doc_key만 처리 (mgmt/budget/internal/manual)")
    parser.add_argument("--redo", action="store_true",
                        help="이미 적용된 청크도 다시 처리")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[!] OPENAI_API_KEY 미설정")
        return

    client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBEDDING_MODEL)
    collection = chroma_client.get_collection(COLLECTION_NAME, embedding_function=ef)

    # 규정·매뉴얼 청크 로드
    where = {"doc_type": "regulation"}
    if args.doc_key:
        where = {"$and": [{"doc_type": "regulation"}, {"doc_key": args.doc_key}]}

    print(f"[*] 규정·매뉴얼 청크 로드 (where={where})...")
    data = collection.get(where=where, include=["documents", "metadatas"])
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]
    print(f"[*] 전체: {len(ids)}건")

    # prefix 미적용 청크만 (또는 --redo면 전체)
    pending = []
    for cid, doc, meta in zip(ids, docs, metas):
        if not args.redo and doc.lstrip().startswith(PREFIX_MARKER):
            continue
        pending.append((cid, doc, meta))
    print(f"[*] 처리 대상: {len(pending)}건"
          + (" (이미 적용됨 skip)" if not args.redo else ""))

    if not pending:
        print("[v] 처리할 청크 없음.")
        return

    # LLM 호출 + 새 document 생성
    new_ids, new_docs, new_metas = [], [], []
    failures = 0
    start = time.time()

    for i, (cid, doc, meta) in enumerate(pending, 1):
        # 기존 prefix 있는 경우(redo) 제거 후 재생성
        body = doc
        if body.lstrip().startswith(PREFIX_MARKER):
            # "[맥락] ...\n\n<original>" → original만
            parts = body.split("\n\n", 1)
            body = parts[1] if len(parts) > 1 else body

        try:
            ctx = generate_context(
                client,
                meta.get("doc_name", ""),
                meta.get("citation", ""),
                body,
            )
        except Exception as e:
            print(f"[!] {cid} 요약 실패: {type(e).__name__}: {str(e)[:100]}")
            failures += 1
            continue

        new_doc = f"{PREFIX_MARKER} {ctx}\n\n{body}"
        new_meta = {**meta, "contextual_summary": ctx}
        new_ids.append(cid)
        new_docs.append(new_doc)
        new_metas.append(new_meta)

        if i % 50 == 0 or i == len(pending):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed else 0
            eta = (len(pending) - i) / rate if rate else 0
            print(f"  [{i:4d}/{len(pending)}] {elapsed:.0f}s 경과, ETA {eta:.0f}s — 예: {ctx[:50]}")

    if failures:
        print(f"[!] {failures}건 LLM 호출 실패 (skip)")

    # 배치 upsert (재임베딩 발생)
    print(f"\n[*] upsert 중 ({len(new_ids)}건, 재임베딩 발생)...")
    for i in range(0, len(new_ids), BATCH_SIZE):
        collection.upsert(
            ids=new_ids[i:i+BATCH_SIZE],
            documents=new_docs[i:i+BATCH_SIZE],
            metadatas=new_metas[i:i+BATCH_SIZE],
        )
        print(f"  {min(i+BATCH_SIZE, len(new_ids))}/{len(new_ids)}")

    elapsed = time.time() - start
    print(f"\n[v] 완료: {len(new_ids)}건, 총 {elapsed:.0f}s")
    print(f"    BM25 인덱스도 다시 빌드해 주세요: python build_bm25.py")


if __name__ == "__main__":
    main()
