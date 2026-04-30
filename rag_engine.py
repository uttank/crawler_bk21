import json
import os
import re
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

try:
    import bm25s
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

try:
    from query_rewrite import rewrite_query
    _HAS_REWRITE = True
except ImportError:
    _HAS_REWRITE = False

load_dotenv()

# 상수 설정
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
COLLECTION_NAME = "bk21_qna"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

# Date re-ranking: 후보를 RERANK_FACTOR배 가져온 뒤 답변일 가중치를 더해 top_k로 좁힌다.
# DATE_PENALTY는 "1년 오래된 문서당 더해질 가상 거리". 0.02는 1년에 약 4% 페널티 수준.
RERANK_FACTOR = 2
DATE_PENALTY_PER_YEAR = 0.02
_YEAR_RE = re.compile(r'(\d{4})')
NTTID_RE = re.compile(r'nttId[\s:]*(\d+)|\bnttId\s+(\d+)|\(\s*(\d{4,6})\s*,')

# Hybrid retrieval 설정
BM25_DIR = Path("data/bm25")
RRF_K = 60                   # RRF 표준 상수 — 작은 차이를 부드럽게 함
HYBRID_DENSE_TOP_N = 30      # dense 채널에서 가져올 후보 수 (각 doc_type별)
HYBRID_BM25_TOP_N = 50       # BM25 채널 (단일 인덱스, 사후 분리)
DATE_RRF_BOOST = 0.0005      # 1년에 RRF score 감소량 (RRF 자체가 0~0.05 범위라 약하게)

# Query rewriting 설정
QUERY_REWRITE_VARIANTS = 2   # 원 질문 + 변형 N개 → 총 N+1 query로 검색

class BKRAGEngine:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key or self.api_key == "sk-your-openai-api-key-here":
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            
        # OpenAI 클라이언트 초기화
        self.openai_client = OpenAI(api_key=self.api_key)
        
        # ChromaDB 클라이언트 연결
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name=EMBEDDING_MODEL
        )
        
        try:
            self.collection = self.chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.openai_ef
            )
        except Exception as e:
            print("[!] ChromaDB 컬렉션을 찾을 수 없습니다. 벡터 DB를 먼저 구축해 주세요.")
            self.collection = None

        # BM25 인덱스 로드 (있으면 hybrid, 없으면 dense-only)
        self.bm25 = None
        self.bm25_mapping = None
        if _HAS_BM25 and (BM25_DIR / "mapping.json").exists():
            try:
                self.bm25 = bm25s.BM25.load(str(BM25_DIR / "index"), load_corpus=False)
                with open(BM25_DIR / "mapping.json", encoding="utf-8") as f:
                    self.bm25_mapping = json.load(f)
                print(f"[v] BM25 인덱스 로드: {len(self.bm25_mapping['ids'])}건")
            except Exception as e:
                print(f"[!] BM25 인덱스 로드 실패: {e}. dense-only 모드.")
                self.bm25 = None
                self.bm25_mapping = None

    def _parse_regulation(self, meta, doc_text, distance):
        return {
            "doc_type": "regulation",
            "id": meta.get("citation", "규정"),
            "citation": meta.get("citation", ""),
            "doc_name": meta.get("doc_name", ""),
            "version": meta.get("version", ""),
            "chapter": meta.get("chapter", ""),
            "article_no": meta.get("article_no", ""),
            "article_title": meta.get("article_title", ""),
            "sub_no": meta.get("sub_no", ""),
            "sub_title": meta.get("sub_title", ""),
            "full_text": doc_text,
            "distance": distance,
            "_score": distance,
        }

    def _parse_qna(self, meta, doc_text, distance):
        current_year = datetime.now().year
        a_date = meta.get("Answer_Date", "")
        q_date = meta.get("Question_Date", "")
        year_match = _YEAR_RE.search(a_date or q_date or '')
        years_old = max(0, current_year - int(year_match.group(1))) if year_match else 5
        return {
            "doc_type": "qna",
            "id": meta.get("nttId", "Unknown"),
            "q_date": q_date,
            "a_date": a_date,
            "question": meta.get("Question", ""),
            "answer": meta.get("Answer", ""),
            "full_text": doc_text,
            "distance": distance,
            "_score": distance + years_old * DATE_PENALTY_PER_YEAR,
        }

    def _dense_search(self, query: str, n: int, where: dict):
        """Chroma dense 검색 — (chroma_id, meta, doc_text, distance) 리스트 반환."""
        results = self.collection.query(
            query_texts=[query], n_results=n, where=where,
        )
        out = []
        if results and results.get("metadatas") and results["metadatas"][0]:
            for i, meta in enumerate(results["metadatas"][0]):
                out.append((
                    results["ids"][0][i],
                    meta,
                    results["documents"][0][i],
                    results["distances"][0][i] if "distances" in results else 0.0,
                ))
        return out

    def _bm25_search(self, query: str, n: int):
        """BM25 검색 — (chroma_id, meta, doc_text, bm25_score) 리스트 반환.

        단일 인덱스이므로 doc_type 분리는 호출자가 처리.
        """
        if not self.bm25 or not self.bm25_mapping:
            return []
        query_tokens = bm25s.tokenize([query], stopwords=None, stemmer=None)
        results, scores = self.bm25.retrieve(query_tokens, k=n, show_progress=False)
        out = []
        for idx, score in zip(results[0], scores[0]):
            i = int(idx)
            out.append((
                self.bm25_mapping["ids"][i],
                self.bm25_mapping["metadatas"][i],
                self.bm25_mapping["documents"][i],
                float(score),
            ))
        return out

    def _rrf_multi(self, channels, k=RRF_K):
        """여러 ranked list를 한 번에 RRF로 결합.

        channels: [(kind, ranked_list), ...] where kind in {'dense', 'bm25'}.
                  ranked_list 항목은 (cid, meta, doc, value).
                  같은 cid가 여러 list에 등장하면 rank 점수 가산.

        반환: rrf_score 내림차순 정렬된
              [(cid, meta, doc, dense_dist, bm25_score, rrf_score), ...].
        """
        scores = {}
        info = {}  # cid -> {meta, doc, dense_dist, bm25_score}
        for kind, ranked in channels:
            for rank, hit in enumerate(ranked):
                cid, meta, doc, val = hit
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                if cid not in info:
                    info[cid] = {"meta": meta, "doc": doc, "dense_dist": None, "bm25_score": None}
                if kind == "dense" and info[cid]["dense_dist"] is None:
                    info[cid]["dense_dist"] = val
                elif kind == "bm25" and info[cid]["bm25_score"] is None:
                    info[cid]["bm25_score"] = val
        ordered = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        return [
            (c, info[c]["meta"], info[c]["doc"], info[c]["dense_dist"], info[c]["bm25_score"], scores[c])
            for c in ordered
        ]

    def _build_channels(self, query: str, top_k: int):
        """단일 query에 대해 (dense_qna, bm25_qna, dense_reg, bm25_reg) 채널을 생성."""
        n_dense_qna = max(top_k * RERANK_FACTOR, top_k * 3)
        n_dense_reg = max(top_k, HYBRID_DENSE_TOP_N // 2)
        n_bm25 = HYBRID_BM25_TOP_N

        dense_qna = self._dense_search(
            query, n_dense_qna, where={"doc_type": {"$ne": "regulation"}},
        )
        dense_reg = self._dense_search(
            query, n_dense_reg, where={"doc_type": "regulation"},
        )
        bm25_all = self._bm25_search(query, n_bm25)
        bm25_qna = [h for h in bm25_all if h[1].get("doc_type") != "regulation"]
        bm25_reg = [h for h in bm25_all if h[1].get("doc_type") == "regulation"]
        return dense_qna, bm25_qna, dense_reg, bm25_reg

    def retrieve(self, query: str, top_k: int = 5, use_rewrite: bool = True):
        """하이브리드 + (선택) Multi-Query 검색.

        - dense + BM25 두 채널 / 규정·Q&A 두 도메인 → 4채널을 RRF로 결합.
        - use_rewrite=True면 LLM이 N개 변형 query 생성, 변형 각각의 4채널까지 모두 RRF.
        """
        if not self.collection:
            return []

        # 1) 사용할 query 목록 결정
        queries = [query]
        if use_rewrite and _HAS_REWRITE and QUERY_REWRITE_VARIANTS > 0:
            try:
                queries = rewrite_query(self.openai_client, query, n_variants=QUERY_REWRITE_VARIANTS)
            except Exception as e:
                print(f"[!] query rewrite 실패, 원 질문만 사용: {e}")
                queries = [query]

        # 2) 각 query마다 4채널 생성
        qna_channels, reg_channels = [], []
        for q in queries:
            dense_qna, bm25_qna, dense_reg, bm25_reg = self._build_channels(q, top_k)
            qna_channels.append(("dense", dense_qna))
            qna_channels.append(("bm25", bm25_qna))
            reg_channels.append(("dense", dense_reg))
            reg_channels.append(("bm25", bm25_reg))

        # 3) Multi-channel RRF
        fused_qna = self._rrf_multi(qna_channels)
        fused_reg = self._rrf_multi(reg_channels)

        # 4) 파싱 + 정렬용 _score
        current_year = datetime.now().year

        qna_docs = []
        for cid, meta, doc, dist, bm25_score, rrf in fused_qna:
            d = self._parse_qna(meta, doc, dist if dist is not None else 1.0)
            d["bm25_score"] = bm25_score
            d["rrf_score"] = rrf
            year_match = _YEAR_RE.search(d.get("a_date") or d.get("q_date") or "")
            years_old = max(0, current_year - int(year_match.group(1))) if year_match else 5
            d["_score"] = -rrf + years_old * DATE_RRF_BOOST
            qna_docs.append(d)

        reg_docs = []
        for cid, meta, doc, dist, bm25_score, rrf in fused_reg:
            d = self._parse_regulation(meta, doc, dist if dist is not None else 1.0)
            d["bm25_score"] = bm25_score
            d["rrf_score"] = rrf
            d["_score"] = -rrf
            reg_docs.append(d)

        qna_docs.sort(key=lambda x: x["_score"])
        reg_docs.sort(key=lambda x: x["_score"])

        # 5) 균형 top_k (기존 로직 유지)
        max_regs = max(1, top_k // 2)
        n_regs = min(max_regs, len(reg_docs))
        n_qnas = top_k - n_regs
        if n_qnas > len(qna_docs):
            n_regs = min(top_k - len(qna_docs), len(reg_docs))
            n_qnas = len(qna_docs)
        if n_regs > len(reg_docs):
            n_regs = len(reg_docs)
            n_qnas = min(top_k - n_regs, len(qna_docs))

        return reg_docs[:n_regs] + qna_docs[:n_qnas]

    def generate_answer(self, user_query: str, retrieved_docs: list):
        """검색된 문서를 바탕으로 프롬프트를 구성하고 LLM 응답을 스트리밍으로 생성합니다."""

        if not retrieved_docs:
            yield "죄송합니다. 관련된 참고 문서를 찾지 못했습니다."
            return

        # 출처 종류별로 컨텍스트 분리
        reg_docs = [d for d in retrieved_docs if d.get("doc_type") == "regulation"]
        qna_docs = [d for d in retrieved_docs if d.get("doc_type") == "qna"]

        sections = []

        if reg_docs:
            reg_parts = []
            for i, d in enumerate(reg_docs, 1):
                version_info = f" (버전: {d['version']})" if d.get('version') else ""
                reg_parts.append(
                    f"[규정 {i}] {d['citation']}{version_info}\n"
                    f"{d['full_text']}"
                )
            sections.append("[규정 출처 — 권위 있는 1차 자료]\n" + "\n\n".join(reg_parts))

        if qna_docs:
            qna_parts = []
            for i, d in enumerate(qna_docs, 1):
                date_info = d.get('a_date') or d.get('q_date') or ''
                qna_parts.append(
                    f"[Q&A {i}] 게시물 ID: {d['id']} (날짜: {date_info})\n"
                    f"질문: {d['question']}\n"
                    f"답변: {d['answer']}"
                )
            sections.append("[과거 Q&A — 운영팀 적용 사례]\n" + "\n\n".join(qna_parts))

        context_str = "\n\n".join(sections)

        # 시스템 프롬프트
        system_prompt = f"""당신은 BK21 FOUR 사업의 친절하고 전문적인 상담원입니다.
아래 [참고 자료]는 두 종류로 구성되어 있습니다:
- **[규정 출처]**: BK21 사업의 공식 규정집(관리운영지침, 예산편성·집행기준, 사업단 자체규정). 권위 있는 1차 자료입니다.
- **[과거 Q&A]**: 운영팀이 실제 문의에 답한 사례. 규정의 해석·적용 예시입니다.

{context_str}

지시사항:
1. **규정 우선 인용**: 답변할 때 규정에 명시된 내용이 있으면 그것을 가장 먼저 근거로 들고, 정확한 조문(citation)을 인용하세요. Q&A는 보조적인 적용 사례로 사용합니다.
2. **인용 형식**:
   - 규정: "(관리운영지침 제12조)", "(예산편성·집행기준 제6조)", "(사업단 자체규정 Ⅲ.2.)" 등 [규정 N] 헤더에 표시된 citation 문자열 그대로 사용. **헤더에 없는 조문 번호는 절대 만들지 마세요.**
   - Q&A: "(nttId 37833, 2026-04-27)"
3. **충돌 처리**: 규정과 Q&A가 다르거나 오래된 Q&A가 현재 규정과 모순될 수 있는 경우, 규정을 우선시하고 두 출처를 모두 언급하며 차이를 설명하세요.
4. **종합 추론**: 여러 자료를 연결해 한 답변으로 정리하세요. 질문이 자료와 정확히 일치하지 않더라도 같은 조건이 적용되는 유사 사례라면 "유사 사례를 참고할 때..." 형태로 추론해도 됩니다.
5. **근거 부족 시**: 어떤 자료에도 관련 근거가 없으면 "제공된 자료에서는 직접적인 답변을 찾기 어렵습니다"라고 솔직히 안내하고, 가장 가까운 주제의 자료가 있다면 함께 소개하세요. 외부 지식·추측은 금지입니다.
6. **포맷**: 정중한 존댓말, 글머리 기호·번호로 가독성 있게."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # OpenAI API 스트리밍 호출
        response = self.openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,  # 사실 기반 응답을 위해 낮은 온도 설정
            stream=True
        )
        
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

if __name__ == "__main__":
    # 간단한 로컬 테스트용
    print("=== RAG 엔진 테스트 ===")
    try:
        engine = BKRAGEngine()
        test_q = "장기연수 체재비 지원이 가능한가요?"
        print(f"질문: {test_q}\n")
        
        docs = engine.retrieve(test_q, top_k=3)
        print(f"[-] {len(docs)}개의 관련 문서 검색됨\n")
        
        print("[-] 생성된 답변:\n")
        for chunk in engine.generate_answer(test_q, docs):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"오류 발생: {e}")
