# BK21 RAG 시스템 고도화 계획 (Advanced Roadmap)

본 문서는 현재 구축된 BK21 FOUR Q&A 및 규정 기반 RAG 시스템의 성능, 안정성, 확장성을 높이기 위한 기술적 개선 방안을 담고 있습니다.

---

## 1. 데이터 파이프라인 및 적재 (Ingestion)

### 1.1 데이터 정제 및 무결성 강화 (`preprocess_data.py`)
*   **중복 제거 로직 강화**: `nttId`를 기준으로 한 물리적 중복 체크 외에도, 질문 내용의 유사도를 기반으로 한 중복 데이터(Near-duplicate) 필터링 로직 추가.
*   **정제 품질 검증**: 전처리 전후의 데이터 통계(문자 수 변화, 제거된 보일러플레이트 비율 등)를 리포트로 출력하여 정제 로직의 부작용 모니터링.

### 1.2 PDF 파싱 고도화 (`ingest_regulations.py`)
*   **구조적 파싱 도입**: 단순 텍스트 추출에서 벗어나 `pdfplumber` 또는 `Unstructured` 라이브러리를 활용하여 표(Table), 리스트, 계층 구조를 보존한 채 청킹(Chunking) 수행.
*   **지능적 청킹 전략**: 문서의 목차(TOC) 구조를 명시적으로 파싱하여, 각 청크가 속한 상위 섹션의 맥락을 메타데이터뿐만 아니라 본문 내에 삽입(Context Enrichment).

---

## 2. 벡터 데이터베이스 및 임베딩 (`build_vectordb.py`)

### 2.1 인프라 안정성
*   **지수적 백오프 재시도 (Exponential Backoff)**: `tenacity` 라이브러리를 도입하여 OpenAI API Rate Limit 및 네트워크 일시 오류에 대한 자동 복구 메커니즘 구축.
*   **임베딩 캐싱**: 동일한 텍스트에 대한 중복 임베딩 호출을 방지하기 위해 로컬 캐시 레이어 도입하여 비용 절감.

### 2.2 메타데이터 관리
*   **버전 관리 시스템**: 사용된 임베딩 모델 명칭, 차원 수, 인덱싱 날짜 등을 담은 `manifest.json`을 생성하여 DB와 모델 간의 버전 일치 보장.

---

## 3. 검색 및 응답 생성 엔진 (RAG Engine)

### 3.1 검색 품질 고도화 (`rag_engine.py`)
*   **질문 확장 및 재구성 (Query Transformation)**:
    *   **Multi-Query**: 사용자의 질문을 다양한 관점에서 3~5개로 확장하여 검색 범위 확대.
    *   **HyDE (Hypothetical Document Embeddings)**: LLM이 가상의 답변을 먼저 생성하고, 그 답변과 유사한 문서를 검색하여 검색 정밀도 향상.
*   **Cross-Encoder 기반 리랭킹 (Re-ranking)**:
    *   ChromaDB에서 가져온 Top-N 후보군에 대해 `BGE-Reranker` 또는 `Cohere Rerank` 모델을 사용하여 질문과의 실제 의미적 연관성을 재계산.

### 3.2 대화 맥락 및 추론
*   **대화 이력 관리 (Conversation Summary Buffer)**: 단순 1회성 응답이 아닌, 이전 대화 맥락을 요약하여 다음 검색 쿼리에 반영하는 세션 관리 기능 추가.
*   **출처 가독성 개선**: 답변 내 인용구에 하이퍼링크 스타일이나 각주 형태를 도입하여 사용자가 원문 위치를 더 쉽게 식별하도록 개선.

---

## 4. 아키텍처 및 운영 환경

### 4.1 성능 및 확장성
*   **비동기 엔진 전환**: `BKRAGEngine`의 모든 외부 API 호출을 `asyncio` 기반 비동기로 처리하여 동시 요청 처리량(Throughput) 개선.
*   **구조적 로깅**: `logging` 모듈을 도입하여 실시간 검색어, 소요 시간, 토큰 사용량, 검색 결과(Distances) 등을 로그 파일로 기록.

### 4.2 품질 평가 (Evaluation)
*   **RAGAS 기반 평가**: 생성된 답변의 충실도(Faithfulness), 관련성(Answer Relevance), 맥락 정밀도(Context Precision)를 정량적으로 평가하는 파이프라인 구축.

---

## 5. 실행 우선순위 (Roadmap)

1.  **Phase 1 (안정성)**: 예외 처리 강화, 비동기 처리 도입, 로깅 시스템 구축.
2.  **Phase 2 (정확도)**: 리랭커(Reranker) 도입, 질문 확장(Query Expansion) 적용.
3.  **Phase 3 (사용성)**: 대화 맥락(Memory) 추가, 출처 표시 UI/UX 개선.
