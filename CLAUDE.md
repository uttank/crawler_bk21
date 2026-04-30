# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Korean-language RAG chatbot for the BK21 FOUR program. Two evidence sources flow into one Chroma collection:
- **Q&A** — ~10,800 entries scraped from the BK21 FOUR Q&A board (`bk21four.nrf.re.kr`).
- **Regulations** — three NRF regulation documents + the practitioner manual (PDF/DOCX).

Pipeline: scrape → preprocess → embed Q&A → embed regulations → Streamlit chat UI that mixes both sources in every answer.

## Layout

```
data/
  bk21_qna_dataset.csv      # raw scrape (llm_crawler.py output)
  bk21_qna_cleaned.csv      # preprocessed (preprocess_data.py output)
  chroma_db/                # vector DB (collection: bk21_qna; ~11,200 docs)
  regulations/              # regulation/manual source files
    mgmt_*.docx, budget_*.docx, internal_*.docx
    실무자를 위한 ... 매뉴얼_*.pdf
old/                        # legacy artifacts (early CSVs, sample HTML, plan doc, crawl log)
venv/                       # Python virtual env (already set up)
```

Path constants live at the top of each module — `data/` is hardcoded, not configurable. `CHROMA_DB_DIR` is the only path env-var (defaults to `./data/chroma_db`).

## Commands

`.env` at repo root must contain `OPENAI_API_KEY`. `CHROMA_DB_DIR` is optional.

```bash
# Install deps
pip install -r requirements.txt

# 1. Crawl Q&A board → data/bk21_qna_dataset.csv
python llm_crawler.py                  # full re-crawl (DELETES existing CSV)
python llm_crawler.py --update         # incremental: stop at first known nttId
python llm_crawler.py --months 3       # past 3 months only
python llm_crawler.py --limit 100      # cap at 100 posts

# 2. Preprocess Q&A → data/bk21_qna_cleaned.csv
python preprocess_data.py

# 3. Build/refresh ChromaDB with Q&A (uses OpenAI embeddings, ~$0.10 for full rebuild)
python build_vectordb.py

# 4. Ingest regulations + manual into the same collection
python ingest_regulations.py           # parses every file in data/regulations/

# 5. Run the chat UI
streamlit run app.py

# Diagnostics
python debug.py                        # retrieval sanity check
python rag_engine.py                   # smoke-test retrieve + generate end-to-end
python estimate_tokens.py --sample 500 # token/cost estimate on the cleaned CSV
```

No test suite, lint config, or CI.

## Architecture

```
llm_crawler.py        →  data/bk21_qna_dataset.csv     (raw: nttId, Date, Question, Answer)
preprocess_data.py    →  data/bk21_qna_cleaned.csv     (adds Answer_Date, Combined_Text; strips greetings)
build_vectordb.py     →  data/chroma_db/  (Q&A docs in collection bk21_qna)
ingest_regulations.py →  data/chroma_db/  (regulation docs in same collection, doc_type=regulation)
rag_engine.py         ←  data/chroma_db/  (retrieve + GPT-4o-mini answer streaming)
app.py                ←  rag_engine.BKRAGEngine        (Streamlit UI)
```

Each stage reads the previous stage's output by hardcoded path. Renaming a file requires editing both producer and consumer.

### Crawler (`llm_crawler.py`)

Two-phase scrape:
1. Walk `selectBoardList.do` pages, extract `nttId` from `onclick="fn_egov_inqire_notice(...)"`. Skip rows marked `공지`.
2. Per `nttId`, fetch `selectBoardArticle.do`, parse with rule-based BeautifulSoup (`h5.bbs_title`, `div.bbs_text`, `table.listTable`). Despite the filename, **no LLM is called** — name is historical.

`--update` reads existing `nttId`s from CSV and stops at the first known one (relies on chronological listing). Without `--update`, the existing CSV is **deleted** first. Rows are appended after each fetch (resumable). 0.5–1.5s random delay between requests.

### Preprocessing (`preprocess_data.py`)

Three steps that downstream depends on:
1. Drop rows whose `Answer` contains `답변이 등록되지`.
2. Extract answer timestamp via regex `BKS관리자\n\d{4}-…` into `Answer_Date` **before** stripping that header from the body. Extraction must precede greeting removal.
3. Build `Combined_Text = "[질문]\n…\n\n[답변]\n…"` — this is what gets embedded.

`remove_greetings` is **line-based, not substring-based** — it drops greeting-only lines but never edits sentence-internal tokens. Earlier substring-replacement broke sentence boundaries throughout the corpus; do not regress to that.

### Q&A vector DB (`build_vectordb.py`)

- Collection: `bk21_qna`. Embedding: `text-embedding-3-small` (1536-dim). `nttId` is the primary key; `upsert` is idempotent.
- **Incremental skip is naive**: returns early if `collection.count() >= len(df)`. Delete `data/chroma_db/` to force a rebuild.
- Metadata `Question`/`Answer` are stored full-length (no truncation) — the LLM context relies on these.
- Batches of 100.

### Regulations + manual (`ingest_regulations.py`)

Parses files in `data/regulations/` into the same Chroma collection with `doc_type=regulation` metadata. Filename prefix → parser style:
- `mgmt_*.docx`, `budget_*.docx` — DOCX, "article style" (제N조 unit). Detects 부칙 boundaries to disambiguate `제2조` collisions, recognises `별표`/`【별표】`.
- `internal_*.docx` — DOCX, "outline style" (Ⅰ./1./(1) hierarchy).
- `실무자…매뉴얼*.pdf` — PDF, "manual style". Maps via the keyword `매뉴얼`/`실무자` (no prefix). Uses `PyMuPDF` with `sort=True`.

The manual PDF has CID-keyed font issues that drop digits inside `제N조` markers — manual style abandons fine-grained section detection and chunks by `[큰섹션]` (4 top-level brackets) plus length-based splits. Citations therefore name only the top section, not sub-articles.

`split_oversized` cascades: `①②③` → `❚` → `➊➋` → length fallback. Chunks under 40 chars are dropped, those over 2000 chars are split.

Regulation chunk IDs are `reg_<doc>_<chapter>_<art>_<sub>_<title-slug>`. Identical IDs get a `_dupN` suffix automatically.

### RAG engine (`rag_engine.py`)

- `retrieve` makes **two separate Chroma queries** with `where` filters: one for `doc_type=regulation`, one for everything else (`$ne: regulation`). This is essential — regulations (~440 docs) are dwarfed by Q&A (~10,800) and would never surface from a single mixed query. The function then balances results: at most `top_k // 2` regulations, rest filled with Q&A, with bidirectional fallback if either side is short.
- Q&A results carry an age penalty (`distance + years_old * 0.02`); regulations do not.
- `generate_answer` builds a context with two clearly separated sections (`[규정 출처]` / `[과거 Q&A]`) and a system prompt that:
  - Treats regulations as the authoritative primary source, Q&A as application examples.
  - **Forbids inventing article numbers** — must use the citation string from the chunk header verbatim. Without this guard the LLM hallucinates plausible-looking 제N조 references.
  - Prefers regulations on conflict, mentions both when they disagree.
- `__init__` is non-fatal when the collection is missing — prints a warning and `retrieve` returns `[]`.
- Streamed tokens via `stream=True`, surfaced through a single placeholder in `app.py`.

### UI (`app.py`)

Sources expander is split into two: `📖 참고한 규정 N건` and `💬 참고한 과거 Q&A N건`. Top-K slider defaults to 5. Engine is created once per session.

## Conventions worth knowing

- **Encoding**: CSVs are `utf-8-sig` (BOM) for Excel on Windows. Read with the same.
- **Dates**: `Date`, `Answer_Date` are opaque strings — never parsed. Year is regex-extracted only for the retrieval age penalty.
- **Path constants are API**: each module has hardcoded `data/...` paths. Search for the literal string before renaming.
- **All user-facing strings are Korean.** Match the polite ~습니다 form when adding UI text or system prompts.
- **Log prefix conventions**: `[*]` info, `[-]` progress, `[!]` warning, `[v]`/`[✓]` success, `[X]` error.
- **PDF text extraction** is unreliable for the official NRF regulation PDFs (digits drop out). Always prefer DOCX. The practitioner manual is the only PDF in the pipeline because its DOCX conversion damaged the original; PyMuPDF + `sort=True` is the workaround.
