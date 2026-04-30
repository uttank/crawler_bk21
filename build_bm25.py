"""ChromaDB 컬렉션과 평행한 BM25 인덱스 구축.

사용:
    python build_bm25.py

생성물:
    data/bm25/index/...      # bm25s 인덱스 파일
    data/bm25/mapping.json   # bm25 internal idx → chroma id, metadata, document
"""

import io
import json
import os
import sys
from pathlib import Path

import bm25s
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
COLLECTION_NAME = "bk21_qna"
EMBEDDING_MODEL = "text-embedding-3-small"
BM25_DIR = Path("data/bm25")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"[*] ChromaDB 로드: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # 임베딩 함수는 query 시점 외엔 불필요하지만, 일관성 유지
    if api_key:
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=EMBEDDING_MODEL
        )
        collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    else:
        collection = client.get_collection(COLLECTION_NAME)

    print(f"[*] 컬렉션 문서 수: {collection.count()}")

    # 한 번에 모두 로드
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    docs = all_data["documents"]
    metas = all_data["metadatas"]
    print(f"[*] 문서 로드: {len(ids)}건")

    # 토큰화 — bm25s 기본 토크나이저는 비단어 문자로 분리.
    # 한국어 어절(공백 단위)이 그대로 토큰이 되어 키워드 매칭에 적합.
    print("[*] 토큰화 중...")
    tokens = bm25s.tokenize(docs, stopwords=None, stemmer=None, show_progress=True)

    # BM25 인덱스 빌드
    print("[*] BM25 인덱스 빌드 중...")
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=True)

    # 저장
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    index_path = BM25_DIR / "index"
    retriever.save(str(index_path))

    # bm25 internal index → 메타·문서 매핑
    mapping = {
        "ids": ids,
        "metadatas": metas,
        "documents": docs,
    }
    with open(BM25_DIR / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)

    # 크기 통계
    total_size = sum(p.stat().st_size for p in BM25_DIR.rglob("*") if p.is_file())
    print(f"[v] 저장 완료: {BM25_DIR} ({total_size / 1024 / 1024:.1f}MB)")
    print(f"    인덱스: {index_path}")
    print(f"    매핑:   {BM25_DIR / 'mapping.json'}")


if __name__ == "__main__":
    main()
