import os
import re
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 상수 설정
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
COLLECTION_NAME = "bk21_qna"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Date re-ranking: 후보를 RERANK_FACTOR배 가져온 뒤 답변일 가중치를 더해 top_k로 좁힌다.
# DATE_PENALTY는 "1년 오래된 문서당 더해질 가상 거리". 0.02는 1년에 약 4% 페널티 수준.
RERANK_FACTOR = 2
DATE_PENALTY_PER_YEAR = 0.02
_YEAR_RE = re.compile(r'(\d{4})')

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

    def retrieve(self, query: str, top_k: int = 5):
        """질문과 유사한 문서를 검색합니다.

        규정과 Q&A를 별도 쿼리로 가져와 후보 풀이 한쪽으로 쏠리는 것을 방지.
        규정은 168건 / Q&A는 10,801건이라 단일 쿼리만으로는 규정이 묻힘.
        """
        if not self.collection:
            return []

        n_qna_candidates = max(top_k * RERANK_FACTOR, top_k * 3)
        n_reg_candidates = max(top_k, 6)

        # 1) Q&A 검색 — doc_type 누락된 기존 Q&A는 where=$ne 안 걸리므로 nttId 존재로 식별
        qna_results = self.collection.query(
            query_texts=[query],
            n_results=n_qna_candidates,
            where={"doc_type": {"$ne": "regulation"}},
        )
        # 2) 규정만 별도 검색
        reg_results = self.collection.query(
            query_texts=[query],
            n_results=n_reg_candidates,
            where={"doc_type": "regulation"},
        )

        regs, qnas = [], []
        if reg_results and reg_results['metadatas'] and reg_results['metadatas'][0]:
            for i, meta in enumerate(reg_results['metadatas'][0]):
                regs.append(self._parse_regulation(
                    meta,
                    reg_results['documents'][0][i],
                    reg_results['distances'][0][i] if 'distances' in reg_results else 0,
                ))
        if qna_results and qna_results['metadatas'] and qna_results['metadatas'][0]:
            for i, meta in enumerate(qna_results['metadatas'][0]):
                qnas.append(self._parse_qna(
                    meta,
                    qna_results['documents'][0][i],
                    qna_results['distances'][0][i] if 'distances' in qna_results else 0,
                ))

        regs.sort(key=lambda d: d['_score'])
        qnas.sort(key=lambda d: d['_score'])

        # 균형 있게 top_k 채움. 규정은 최대 top_k의 절반(최소 1개), 나머지는 Q&A.
        max_regs = max(1, top_k // 2)
        n_regs = min(max_regs, len(regs))
        n_qnas = top_k - n_regs
        if n_qnas > len(qnas):
            n_regs = min(top_k - len(qnas), len(regs))
            n_qnas = len(qnas)
        if n_regs > len(regs):
            n_regs = len(regs)
            n_qnas = min(top_k - n_regs, len(qnas))

        return regs[:n_regs] + qnas[:n_qnas]

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
