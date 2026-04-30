import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# 상수 설정
INPUT_FILE = "data/bk21_qna_cleaned.csv"
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
COLLECTION_NAME = "bk21_qna"
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100  # 한 번에 임베딩할 텍스트 수 (API 속도 제한 고려)

def build_vector_db():
    print("=== 벡터 DB 구축 시작 ===")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-your-openai-api-key-here":
        print("[!] OPENAI_API_KEY가 설정되지 않았습니다.")
        print("[!] .env 파일을 생성하고 API 키를 입력해 주세요.")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"[!] 전처리된 파일을 찾을 수 없습니다: {INPUT_FILE}")
        return

    # 1. 데이터 로드
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"[-] 데이터 로드 완료: 총 {len(df):,}건")

    # 2. ChromaDB 클라이언트 및 임베딩 함수 초기화
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL
    )
    
    # 컬렉션 생성 (이미 존재하면 가져오기)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"description": "BK21 FOUR Q&A Dataset"}
    )

    # 3. 기존 데이터 확인 (증분 업데이트용)
    existing_count = collection.count()
    print(f"[-] 기존 DB 문서 수: {existing_count:,}건")
    
    if existing_count >= len(df):
        print("[v] 모든 데이터가 이미 벡터 DB에 저장되어 있습니다.")
        return

    # 4. 임베딩 및 저장 처리
    print(f"[-] 임베딩 모델: {EMBEDDING_MODEL}")
    print(f"[-] 남은 데이터 임베딩 및 DB 저장 진행 중... (배치 크기: {BATCH_SIZE})")

    documents = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():
        # nttId를 고유 ID로 사용 (문자열)
        doc_id = str(row['nttId'])
        text = str(row['Combined_Text'])
        
        # 메타데이터 구성 (LLM 컨텍스트로도 사용되므로 자르지 않음)
        meta = {
            "nttId": doc_id,
            "Question_Date": str(row['Date']),
            "Answer_Date": str(row['Answer_Date']) if pd.notna(row['Answer_Date']) else "",
            "Question": str(row['Question']),
            "Answer": str(row['Answer'])
        }
        
        documents.append(text)
        metadatas.append(meta)
        ids.append(doc_id)

    # 배치로 나누어서 DB에 추가
    total_added = 0
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_metas = metadatas[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        
        collection.upsert(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        total_added += len(batch_docs)
        print(f"    - 저장 진행률: {total_added:,} / {len(documents):,} 완료")

    print("\n=== 벡터 DB 구축 완료 ===")
    print(f"[v] DB 저장 경로: {CHROMA_DB_DIR}")
    print(f"[v] 현재 DB 총 문서 수: {collection.count():,}건")

if __name__ == "__main__":
    build_vector_db()
