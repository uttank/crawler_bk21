import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rag_engine import BKRAGEngine
engine = BKRAGEngine()
query = "두개의 학회에 연이어 참석하는 경우 하나의 단기연수로 처리할 수 있는지요"
print(f"질문: {query}")
docs = engine.retrieve(query, top_k=5)
for i, d in enumerate(docs):
    print(f"\n--- [문서 {i+1}] ID: {d['id']} (거리: {d['distance']:.4f}) ---")
    print(f"Q: {d['question'][:200]}")
    print(f"A: {d['answer'][:200]}")
