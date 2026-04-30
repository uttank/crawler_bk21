# BK21 FOUR Q&A 챗봇

BK21 FOUR 사업 Q&A 게시판(`bk21four.nrf.re.kr`) 약 10,800건 + 공식 규정집(관리운영지침·예산편성집행기준·실무자 매뉴얼·사업단 자체규정)을 기반으로 한 한국어 RAG 챗봇.

- 검색: dense(`text-embedding-3-large`) + BM25 하이브리드 + LLM Query Rewriting
- 답변: `gpt-4o-mini` 스트리밍, 시스템 프롬프트 환각 방지, 인용 검증
- UI: Streamlit, NRF 게시판 원본 링크, 답변 평가 버튼 (👍/👎)

현재 측정값(golden v2, 25문항, top-k=5): **retrieval@k 87.5% · faithfulness 100%**.

---

## 다른 PC 셋업

### 0. 사전 요구사항

- **Python 3.10+** (현재 3.10에서 검증됨)
- **OpenAI API 키** (결제 활성 상태 필요)
- **OS**: Windows 검증됨. macOS/Linux는 자동 호환되나 미검증.
- **디스크**: 약 350MB (의존성 ~120MB + 데이터 ~230MB)
- **OpenAI 비용**: 데이터 재구축 시 약 $0.50 일회성. 이후 사용은 쿼리당 ~$0.0002.

### 1. 저장소 클론

```bash
git clone https://github.com/uttank/crawler_bk21.git
cd crawler_bk21
```

### 2. 가상환경 + 의존성

**Windows (PowerShell)**:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

설치 시간: 약 2~5분. ChromaDB 빌드 의존성 때문에 첫 설치는 다소 느림.

### 3. `.env` 작성

저장소 루트에 `.env` 파일 생성:

```env
OPENAI_API_KEY=sk-proj-...
# 선택 (기본값 ./data/chroma_db)
# CHROMA_DB_DIR=./data/chroma_db
```

### 4. 데이터 준비 — 두 가지 경로 중 택1

데이터(`data/bk21_qna_*.csv`, `data/chroma_db/`, `data/bm25/`)는 git에 들어있지 않습니다. 규정 원본(`data/regulations/`)만 commit됨.

#### 옵션 A — 처음부터 재생성 (권장)

크롤 + 임베딩이 약 1.5시간 + $0.50.

```bash
# 1. Q&A 게시판 크롤 (1~2시간, 무료. 안전을 위해 random delay 0.5~1.5s 적용)
python llm_crawler.py

# 2. Q&A 정제 (1분, 무료)
python preprocess_data.py

# 3. Q&A 임베딩 (5~10분, ~$0.40)
python build_vectordb.py

# 4. 규정·매뉴얼 적재 (1~2분, ~$0.05)
python ingest_regulations.py

# 5. BM25 인덱스 빌드 (1분, 무료)
python build_bm25.py

# 6. (선택) Contextual Retrieval prefix — 효과 marginal하지만 적용 가능 (8분, ~$0.10)
# python add_contextual_retrieval.py
```

#### 옵션 B — 기존 데이터 백업 복원

이전 PC의 `data/` 디렉토리(약 230MB)를 USB·클라우드로 옮겨 그대로 배치하면 4번 단계는 건너뜀. 백업할 디렉토리:

- `data/bk21_qna_dataset.csv`
- `data/bk21_qna_cleaned.csv`
- `data/chroma_db/` (~180MB)
- `data/bm25/` (~31MB)

규정(`data/regulations/`)은 git에 있어서 별도 복사 불필요.

### 5. 실행

```bash
streamlit run app.py
```

기본 포트 `http://localhost:8501`. 브라우저 자동 열림.

---

## 동작 검증

### 평가 스크립트 (셋업 확인용)

```bash
# 검색만 (LLM 호출 없음, ~30초, 무료)
python eval/evaluate.py --no-generate

# 전체 (LLM 답변 생성 포함, ~3분, ~$0.05)
python eval/evaluate.py
```

기대 출력:
```
=== 요약 ===
질문 25개
retrieval@k: ~87%
recall(평균): ~74%
kw recall(평균): ~80%
faithfulness(평균): 100%
```

리포트는 `eval/reports/<날짜시각>_eval.md`에 저장됨.

### 디버그 (단일 쿼리 검증)

```bash
python debug.py     # 미리 정해진 한 쿼리 검색 결과 확인
python rag_engine.py # retrieve + generate 한 사이클 smoke test
```

---

## 디렉토리 구조

```
crawler_bk21/
├── app.py                          # Streamlit UI
├── rag_engine.py                   # RAG 코어 (hybrid retrieve + LLM)
├── llm_crawler.py                  # Q&A 게시판 크롤러
├── preprocess_data.py              # CSV 정제
├── build_vectordb.py               # Q&A 임베딩
├── ingest_regulations.py           # 규정·매뉴얼 적재
├── build_bm25.py                   # BM25 인덱스 빌더
├── query_rewrite.py                # Multi-Query LLM rewriter
├── add_contextual_retrieval.py     # (선택) 청크 맥락 prefix
├── request_log.py                  # SQLite 요청 로거
├── eval/
│   ├── evaluate.py                 # 자동 평가 스크립트
│   ├── golden_set.yaml             # 평가 데이터셋 v2 (25문항)
│   ├── METRICS.md                  # 지표 의미 + 측정 추적
│   └── reports/                    # 자동 생성 리포트
├── data/                           # gitignored (regulations 제외)
│   ├── bk21_qna_*.csv              # 크롤·정제 결과
│   ├── chroma_db/                  # 벡터 DB
│   ├── bm25/                       # BM25 인덱스
│   ├── regulations/                # 원본 DOCX·PDF (commit됨)
│   └── logs/requests.sqlite        # 요청·피드백 로그
├── old/                            # legacy 보관
├── CLAUDE.md                       # 코드 가이드 (Claude Code 용)
├── claude_plan.md                  # 개선 로드맵
└── README.md                       # 이 파일
```

---

## 자주 마주칠 문제

### `OPENAI_API_KEY가 설정되지 않았습니다.`
`.env` 파일 위치 확인 (저장소 루트). 키 값 끝에 공백·주석이 같은 줄에 붙어 있으면 인식 안 됨 — 키 끝과 다음 줄 사이를 줄바꿈으로 분리.

### `billing_not_active`
OpenAI 결제 수단 등록 안 됨 또는 무료 크레딧 소진. https://platform.openai.com/settings/organization/billing

### `ChromaDB 컬렉션을 찾을 수 없습니다`
4번 데이터 준비 단계 미실행. `python build_vectordb.py` 후 `python ingest_regulations.py`.

### `BM25 인덱스 로드 실패`
무해 — dense-only 모드로 자동 폴백. BM25 재구축: `python build_bm25.py`.

### Windows에서 PDF 파싱 실패
NRF 공식 PDF는 CID 폰트 문제로 `pdfplumber`로 안 됨 (이미 우회 적용). 매뉴얼은 PyMuPDF + `sort=True` 사용. 다른 PDF 추가 시 동일 문제 가능성 있음.

---

## 다음 단계

개선 로드맵: `claude_plan.md`
지표 의미 + 측정 추적: `eval/METRICS.md`
