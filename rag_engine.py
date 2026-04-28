import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 상수 설정
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "bk21_qna"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

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

    def retrieve(self, query: str, top_k: int = 5):
        """질문과 유사한 기존 Q&A 문서를 검색합니다."""
        if not self.collection:
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved_docs = []
        # ChromaDB 결과 파싱
        if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
            for i in range(len(results['metadatas'][0])):
                meta = results['metadatas'][0][i]
                doc_text = results['documents'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0
                
                retrieved_docs.append({
                    "id": meta.get("nttId", "Unknown"),
                    "q_date": meta.get("Question_Date", ""),
                    "a_date": meta.get("Answer_Date", ""),
                    "question": meta.get("Question", ""),
                    "answer": meta.get("Answer", ""),
                    "full_text": doc_text,
                    "distance": distance
                })
                
        return retrieved_docs

    def generate_answer(self, user_query: str, retrieved_docs: list):
        """검색된 문서를 바탕으로 프롬프트를 구성하고 LLM 응답을 스트리밍으로 생성합니다."""
        
        if not retrieved_docs:
            yield "죄송합니다. 관련된 기존 Q&A 데이터를 찾지 못했습니다."
            return

        # 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            date_info = doc['a_date'] if doc['a_date'] else doc['q_date']
            context_parts.append(
                f"[문서 {i}] 게시물 ID: {doc['id']} (날짜: {date_info})\n"
                f"질문: {doc['question']}\n"
                f"답변: {doc['answer']}"
            )
            
        context_str = "\n\n".join(context_parts)
        
        # 시스템 프롬프트
        system_prompt = f"""당신은 BK21 FOUR 사업 Q&A 게시판의 친절하고 전문적인 상담원입니다.
아래에 제공된 [참고 문서]만을 기반으로 사용자의 질문에 답변하세요.

[참고 문서]
{context_str}

지시사항:
1. 반드시 제공된 [참고 문서]의 내용에만 근거하여 답변하세요. 
2. 외부 지식이나 환각(Hallucination)을 섞지 마세요.
3. 제공된 문서에 질문에 대한 명확한 답이 없다면, "제공된 기존 Q&A 데이터에서는 해당 내용을 찾을 수 없습니다"라고 솔직하게 안내하세요.
4. 답변 시 어떤 문서(게시물 ID)를 참고했는지 간략히 명시해 주세요.
5. 정중하고 가독성 좋은 포맷(글머리 기호 등)을 사용하세요."""

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
