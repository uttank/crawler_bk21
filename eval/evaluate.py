"""BK21 RAG 평가 스크립트.

사용:
    python eval/evaluate.py                     # default golden_set.yaml + top_k=5
    python eval/evaluate.py --top_k 3
    python eval/evaluate.py --golden eval/golden_set_v2.yaml
    python eval/evaluate.py --no-generate       # 검색만 (LLM 비용 절감)

지표:
    - retrieval@k         : expected_sources 중 적어도 하나가 top-k에 들어왔는가 (1/0)
    - retrieval_recall    : expected_sources 중 들어온 비율 (0..1)
    - kw_recall           : expected_keywords 중 답변에 등장한 비율 (0..1)
    - faithfulness        : 답변이 인용한 nttId 중 retrieved에 있는 비율 (0..1)
"""

import argparse
import datetime
import io
import os
import re
import sys
import time
from pathlib import Path

import yaml

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# repo root을 path에 추가 — `python eval/evaluate.py` 실행 호환
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_engine import BKRAGEngine


NTTID_RE = re.compile(r'nttId[\s:]*(\d+)|\bnttId\s+(\d+)|\(\s*(\d{4,6})\s*,')


def matches_source(src: dict, retrieved_docs: list) -> bool:
    """expected_source 하나가 retrieved 안에 있는가."""
    if src["type"] == "qna":
        target = str(src["id"])
        return any(d["doc_type"] == "qna" and str(d["id"]) == target for d in retrieved_docs)
    if src["type"] == "regulation":
        needle = src["citation_match"]
        return any(
            d["doc_type"] == "regulation" and needle in d.get("citation", "")
            for d in retrieved_docs
        )
    return False


def extract_cited_nttids(answer: str) -> set:
    """답변에 인용된 nttId 추출. 'nttId 12345' / '(12345, 2026-...)' 패턴."""
    ids = set()
    for m in NTTID_RE.finditer(answer):
        for g in m.groups():
            if g:
                ids.add(g)
    return ids


def evaluate_question(engine: BKRAGEngine, q: dict, top_k: int, do_generate: bool):
    t0 = time.time()
    docs = engine.retrieve(q["query"], top_k=top_k)
    t_retrieve = time.time() - t0

    answer = ""
    t_generate = 0
    if do_generate:
        t1 = time.time()
        for chunk in engine.generate_answer(q["query"], docs):
            answer += chunk
        t_generate = time.time() - t1

    # 1. retrieval
    expected_sources = q.get("expected_sources", []) or []
    if expected_sources:
        hits = [matches_source(s, docs) for s in expected_sources]
        retrieval_at_k = any(hits)
        retrieval_recall = sum(hits) / len(hits)
    else:
        # edge case: 기대 출처 없음 (무관 질문) — 답변이 회피했는지 키워드로 판단
        retrieval_at_k = None
        retrieval_recall = None

    # 2. keyword recall
    kws = q.get("expected_keywords", []) or []
    if kws and answer:
        kw_hit = [kw in answer for kw in kws]
        kw_recall = sum(kw_hit) / len(kw_hit)
    else:
        kw_recall = None

    # 3. faithfulness
    if answer:
        cited = extract_cited_nttids(answer)
        retrieved_qna_ids = {str(d["id"]) for d in docs if d["doc_type"] == "qna"}
        if cited:
            faithful = cited & retrieved_qna_ids
            faithfulness = len(faithful) / len(cited)
            unfaithful = sorted(cited - retrieved_qna_ids)
        else:
            faithfulness = None
            unfaithful = []
    else:
        faithfulness = None
        unfaithful = []

    return {
        "id": q["id"],
        "category": q.get("category", ""),
        "query": q["query"],
        "retrieved": [
            {
                "type": d["doc_type"],
                "ref": d.get("citation") if d["doc_type"] == "regulation" else d.get("id"),
                "dist": round(d.get("distance", 0), 3),
            }
            for d in docs
        ],
        "answer": answer,
        "retrieval_at_k": retrieval_at_k,
        "retrieval_recall": retrieval_recall,
        "kw_recall": kw_recall,
        "faithfulness": faithfulness,
        "unfaithful_nttids": unfaithful,
        "expected_sources": expected_sources,
        "t_retrieve_s": round(t_retrieve, 3),
        "t_generate_s": round(t_generate, 3),
    }


def aggregate(results: list) -> dict:
    def avg(values):
        vs = [v for v in values if v is not None]
        return sum(vs) / len(vs) if vs else None

    n = len(results)
    n_with_sources = sum(1 for r in results if r["retrieval_at_k"] is not None)
    n_hit = sum(1 for r in results if r["retrieval_at_k"])

    return {
        "n_questions": n,
        "n_with_expected_sources": n_with_sources,
        "retrieval_at_k_rate": n_hit / n_with_sources if n_with_sources else None,
        "avg_retrieval_recall": avg(r["retrieval_recall"] for r in results),
        "avg_kw_recall": avg(r["kw_recall"] for r in results),
        "avg_faithfulness": avg(r["faithfulness"] for r in results),
        "n_with_unfaithful": sum(1 for r in results if r["unfaithful_nttids"]),
        "avg_t_retrieve": avg(r["t_retrieve_s"] for r in results),
        "avg_t_generate": avg(r["t_generate_s"] for r in results),
    }


def fmt_pct(v):
    if v is None:
        return "-"
    return f"{v:.1%}"


def write_report(results: list, summary: dict, top_k: int, out_path: Path):
    lines = []
    lines.append(f"# BK21 RAG 평가 리포트")
    lines.append(f"- 생성: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- top_k: {top_k}")
    lines.append(f"- 질문 수: {summary['n_questions']}")
    lines.append("")
    lines.append("## 요약")
    lines.append("")
    lines.append(f"- **retrieval@k 적중률**: {fmt_pct(summary['retrieval_at_k_rate'])} "
                 f"({summary.get('n_with_expected_sources', 0)}개 중)")
    lines.append(f"- **평균 retrieval recall**: {fmt_pct(summary['avg_retrieval_recall'])}")
    lines.append(f"- **평균 키워드 recall**: {fmt_pct(summary['avg_kw_recall'])}")
    lines.append(f"- **평균 인용 신뢰도(faithfulness)**: {fmt_pct(summary['avg_faithfulness'])}")
    lines.append(f"- **인용 환각 발생 질문 수**: {summary['n_with_unfaithful']}")
    lines.append(f"- **평균 검색시간**: {summary['avg_t_retrieve']:.2f}s "
                 f"/ 평균 생성시간: {summary['avg_t_generate']:.2f}s" if summary['avg_t_generate'] else
                 f"- **평균 검색시간**: {summary['avg_t_retrieve']:.2f}s")
    lines.append("")
    lines.append("## 카테고리별 retrieval@k")
    lines.append("")
    cats = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, []).append(r)
    for c, rs in sorted(cats.items()):
        with_src = [r for r in rs if r["retrieval_at_k"] is not None]
        if not with_src:
            continue
        rate = sum(1 for r in with_src if r["retrieval_at_k"]) / len(with_src)
        lines.append(f"- {c}: {fmt_pct(rate)} ({len(with_src)}개)")
    lines.append("")
    lines.append("## 질문별 결과")
    lines.append("")
    for r in results:
        marker = "✅" if r["retrieval_at_k"] else ("❌" if r["retrieval_at_k"] is False else "—")
        lines.append(f"### {r['id']} {marker} `[{r['category']}]` {r['query']}")
        lines.append("")
        lines.append(
            f"- retrieval@k: {marker} | recall: {fmt_pct(r['retrieval_recall'])} "
            f"| kw: {fmt_pct(r['kw_recall'])} | faithful: {fmt_pct(r['faithfulness'])}"
        )
        if r["unfaithful_nttids"]:
            lines.append(f"- ⚠️ 인용 환각 nttId: `{r['unfaithful_nttids']}`")
        if r["expected_sources"]:
            esrc = []
            for s in r["expected_sources"]:
                if s["type"] == "qna":
                    esrc.append(f"qna#{s['id']}")
                else:
                    esrc.append(f"reg~{s['citation_match']}")
            lines.append(f"- expected: {', '.join(esrc)}")
        lines.append(f"- retrieved (top-{top_k}):")
        for d in r["retrieved"]:
            tag = "📖" if d["type"] == "regulation" else "💬"
            lines.append(f"  - {tag} `{d['ref']}` (dist={d['dist']})")
        if r["answer"]:
            preview = r["answer"][:280].replace("\n", " ")
            lines.append(f"- answer: {preview}{'…' if len(r['answer']) > 280 else ''}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden_set.yaml")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--no-generate", action="store_true",
                        help="LLM 답변 생성 생략 (검색 지표만)")
    parser.add_argument("--out-dir", default="eval/reports")
    args = parser.parse_args()

    with open(args.golden, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    questions = data["questions"]

    print(f"[*] 골든셋: {args.golden} ({len(questions)}개)")
    print(f"[*] top_k={args.top_k}, generate={not args.no_generate}")
    print()

    engine = BKRAGEngine()
    if not engine.collection:
        print("[!] ChromaDB 컬렉션이 없습니다.")
        sys.exit(1)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i:2d}/{len(questions)}] {q['id']} — {q['query'][:50]}...", end=" ", flush=True)
        r = evaluate_question(engine, q, args.top_k, do_generate=not args.no_generate)
        marker = "✅" if r["retrieval_at_k"] else ("❌" if r["retrieval_at_k"] is False else "—")
        print(f"{marker} kw={fmt_pct(r['kw_recall'])} faithful={fmt_pct(r['faithfulness'])}")
        results.append(r)

    summary = aggregate(results)

    print()
    print("=== 요약 ===")
    print(f"질문 {summary['n_questions']}개")
    print(f"retrieval@k: {fmt_pct(summary['retrieval_at_k_rate'])}")
    print(f"recall(평균): {fmt_pct(summary['avg_retrieval_recall'])}")
    print(f"kw recall(평균): {fmt_pct(summary['avg_kw_recall'])}")
    print(f"faithfulness(평균): {fmt_pct(summary['avg_faithfulness'])}")
    print(f"인용 환각 발생: {summary['n_with_unfaithful']}건")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}_eval.md"
    out_path = out_dir / fname
    write_report(results, summary, args.top_k, out_path)
    print(f"\n[v] 리포트: {out_path}")


if __name__ == "__main__":
    main()
