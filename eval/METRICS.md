# BK21 RAG 평가 지표

`eval/evaluate.py`가 측정하는 지표와 그 해석. 골든셋(`eval/golden_set.yaml`)을 입력으로 받아 매 질문마다 retrieve + 답변 생성 후 4가지 지표를 계산하고 평균을 낸다.

---

## 측정 지표

### 1. retrieval@k

**정의**: 골든셋의 `expected_sources` 중 적어도 하나가 `top-k` 검색 결과에 등장하는 질문의 비율.

**계산**:
```
질문별: 1 if (any expected_source in retrieved[:k]) else 0
지표:  1로 표시된 질문 수 ÷ expected_sources가 명시된 질문 수
```

**해석**:
- 100%: 모든 질문에서 기대 출처가 top-k에 들어옴 (이상).
- 50%: 절반은 검색 자체에서 정답 문서를 놓침 → 답변 생성 단계가 아무리 좋아도 못 답함.
- **이 지표는 검색 시스템의 recall 성능을 가장 직접적으로 보여준다**.

**영향 받는 변수**: 임베딩 모델, BM25 가중치, RRF 후보 풀 크기, query rewriting, 청킹 단위.

---

### 2. retrieval_recall

**정의**: 한 질문의 `expected_sources` 중 몇 %가 top-k에 들어왔는가의 평균.

**계산**:
```
질문별: hit_count(expected_sources in retrieved[:k]) ÷ len(expected_sources)
지표:  모든 질문의 평균
```

**retrieval@k와의 차이**:
- retrieval@k는 "**적어도 하나** 들어왔는가" (이진).
- retrieval_recall은 "**얼마나 많이** 들어왔는가" (비율).
- 질문 q가 expected = [규정A, Q&A1, Q&A2]를 가지고 검색이 [규정A, X, Y]면 → retrieval@k = 1, recall = 33%.

**해석**: 정답이 여러 개인 케이스에서 다양성을 측정. retrieval@k는 좋은데 recall이 낮으면 "한 가지 출처만 잡고 다른 출처는 놓침" 의미.

---

### 3. kw_recall (키워드 recall)

**정의**: 골든셋의 `expected_keywords`가 LLM이 생성한 답변 본문에 등장하는 비율의 평균.

**계산**:
```
질문별: present_count(expected_keywords in answer) ÷ len(expected_keywords)
지표:  모든 질문의 평균
```

**해석**:
- 검색이 잘 됐어도 답변에 핵심 키워드가 안 들어가면 사용자가 못 찾음.
- 답변 길이·스타일·LLM의 추론 품질에 영향 받음.
- 100%: 모든 핵심 키워드가 답변에 등장.
- **검색과 답변 생성의 통합 품질을 측정**.

**주의**: 키워드 매칭은 부분 문자열 (대소문자·공백 그대로). 의미 매칭이 아님. golden set의 `expected_keywords`를 너무 좁게 잡으면 인위적으로 낮아짐.

---

### 4. faithfulness (인용 신뢰도)

**정의**: LLM 답변 본문에 인용된 `nttId` 중 실제 retrieved 문서에 존재하는 비율.

**계산**:
```
질문별: |cited_nttIds ∩ retrieved_nttIds| ÷ |cited_nttIds|
지표:  cited_nttIds가 비어있지 않은 질문들의 평균
```

**해석**:
- 100%: LLM이 인용한 nttId가 모두 실제 검색 결과에 존재 → 환각 없음.
- 100% 미만: LLM이 컨텍스트에 없는 nttId를 만들어냄 (환각). 사용자가 클릭해도 그 게시물은 없음.
- **현재 시스템 100% 유지** — 시스템 프롬프트의 "헤더에 없는 인용 만들지 말 것" 가드와 인용 검증 후처리의 조합 효과.

**한계**: 규정 인용("관리운영지침 제12조")은 자동 검증이 어려워 현재 nttId 인용만 측정. 규정 인용 환각은 별도 점검 필요.

---

## 부가 지표

리포트에는 다음도 출력됨:

- **n_with_unfaithful**: 인용 환각이 발생한 질문 수 (faithfulness < 100%인 질문 개수). 0건이 이상적.
- **avg_t_retrieve / avg_t_generate**: 평균 검색·생성 시간 (초). 모델·옵션 변경 시 latency 모니터링 용도.
- **카테고리별 retrieval@k**: 골든셋의 `category` 필드 기준 — 어떤 도메인에서 검색이 약한지 식별.

---

## 측정 추적 — 누적 개선 기록

평가 리포트는 `eval/reports/<날짜시각>_eval.md`에 자동 저장됨. 다음 표는 주요 마일스톤만 정리:

| 단계 | 변경 | retrieval@k | recall | kw recall | faithfulness |
|---|---|---|---|---|---|
| **Baseline** | small + dense only | 52.6% | 50.0% | 74.2% | 100% |
| **B2 Hybrid** | + BM25 (RRF 1:1) | 57.9% | 55.3% | 71.7% | 100% |
| **B5 Large** | small → large 임베딩 | 63.2% | 60.5% | 79.2% | 100% |
| **B4 Rewrite** | + Multi-Query (n=2) | **68.4%** | **65.8%** | **82.5%** | 100% |

**누적 +15.8%p retrieval@k**, faithfulness 100% 유지.

---

## 평가 실행

```bash
# 기본 (top_k=5, LLM 답변 생성 포함)
python eval/evaluate.py

# 검색만 측정 (LLM 비용 절감, ~30초)
python eval/evaluate.py --no-generate

# top_k 변경
python eval/evaluate.py --top_k 3

# 다른 골든셋 사용 (예: 카테고리별 v2)
python eval/evaluate.py --golden eval/golden_set_v2.yaml
```

**1회 실행 비용 (top_k=5, 20질문, generate=True)**:
- 임베딩: 골든셋 질문 수만큼 (~$0.001)
- LLM 답변: 20× gpt-4o-mini (~$0.05)
- Query rewriter (B4부터): 20× gpt-4o-mini (~$0.002)
- **총 ~$0.05/회**. 매 변경마다 돌릴 만한 수준.

---

## 골든셋(`eval/golden_set.yaml`) 갱신 가이드

### 추가할 질문의 조건

1. **검증 가능**: expected_sources에 명시한 nttId/citation이 실제 데이터에 존재.
2. **다양성**: 카테고리·도메인 어휘를 넓게.
3. **실패 케이스 우선**: baseline에서 잘 안 잡히는 질문이 평가 가치가 큼 (이미 잘 되는 질문은 회귀 감지용).
4. **edge case**: 무관 질문(점심 메뉴 등) → expected_sources 비워두면 회피 동작 검증.

### 부정확한 expected_sources

기대 출처가 정답 풀에 없거나 오해의 소지가 있으면 **퇴보가 아니라 골든셋 부정확** 가능. 이 경우:
- expected_sources를 더 넓게 (`citation_match`를 짧게).
- expected_keywords로 보완.
- notes에 의도 기록.

현재 의심 케이스: q13(인건비), q16(중간평가). 향후 정밀화 후보.
