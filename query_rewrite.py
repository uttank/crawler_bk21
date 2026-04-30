"""사용자 질문을 BK21 도메인 키워드를 활용한 변형 질문 N개로 확장.

검색 다양성을 높여 모호한 질문에서도 정답 문서가 candidate 풀에 들어오도록.

사용:
    from query_rewrite import rewrite_query
    queries = rewrite_query(openai_client, "장학금 인계", n_variants=2)
    # queries = [원본, 변형1, 변형2]
"""

import re
from typing import List

REWRITE_MODEL = "gpt-4o-mini"
REWRITE_TEMPERATURE = 0.3

REWRITE_SYSTEM = """당신은 BK21 FOUR 사업(한국연구재단 두뇌한국21) 도메인의 검색 전문가입니다.
사용자 질문을 의미는 같지만 다른 표현·동의어·관련 도메인 용어를 사용한 변형 질문으로 다시 작성합니다.

목적: 한국어 RAG 검색에서 다양한 키워드 표현이 매칭되도록 검색 다양성 확보.

도메인 어휘 예시 (필요 시 활용):
- 사업 구조: 4단계 BK21, 미래인재양성사업, 혁신인재양성사업, 교육연구단, 교육연구팀, 사업단
- 인력: 참여대학원생, 지원대학원생, 신진연구인력, 산학협력 전담인력, 참여교수
- 비목: 사업비, 대학원생 연구장학금, 인건비, 국제화경비, 학술활동지원비, 단기연수, 중기연수, 장기연수, 체재비, 항공료, 등록비
- 절차: 신청, 선발, 선정평가, 중간평가, 종합평가, 자체점검, 정산, 종합정보관리시스템, 협약
- 기준: 인정 기준, 자격 요건, 중복 수혜, 1개학기, 4대보험, 전일제

규칙:
1. 변형은 의미를 바꾸지 마세요. 동의어·풀어쓰기·관련 키워드 추가만.
2. 변형마다 한 줄, 번호·기호 없이.
3. 원 질문은 출력하지 마세요 — 변형만 출력.
4. 원 질문이 매우 모호하면 BK21 맥락에 맞춰 좁히되, 다른 해석을 추가로 만들지는 마세요.
"""


def rewrite_query(client, query: str, n_variants: int = 2) -> List[str]:
    """원 질문 + N개 변형을 반환. LLM 실패 시 원 질문만 반환.

    Args:
        client: OpenAI 클라이언트.
        query: 사용자 원 질문.
        n_variants: 생성할 변형 개수 (default 2 → 총 3개 query).

    Returns:
        [원 질문, 변형1, 변형2, ...] (실패 시 [원 질문]).
    """
    if n_variants <= 0:
        return [query]

    user_msg = (
        f"원 질문: {query}\n\n"
        f"위 질문의 변형 {n_variants}개를 생성하세요. 변형마다 한 줄."
    )
    try:
        resp = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=REWRITE_TEMPERATURE,
            max_tokens=300,
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[!] query rewrite 실패: {e}")
        return [query]

    # 줄 단위 split, 번호·기호·공백 정리
    variants = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # "1. ", "1) ", "- ", "• " 같은 prefix 제거
        line = re.sub(r"^\s*(?:[-•\*]|\d+[.\)])\s*", "", line)
        line = line.strip().strip('"').strip("'").strip()
        if line and line != query:
            variants.append(line)

    if not variants:
        return [query]
    return [query] + variants[:n_variants]


if __name__ == "__main__":
    import os
    import sys
    import io
    from openai import OpenAI
    from dotenv import load_dotenv

    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    test_queries = [
        "장학금 중복 수혜가 가능한가요?",
        "외국인 대학원생이 비자 만료 후 공백기간이 생긴 경우 어떻게 처리해야 하나요?",
        "두 개의 학회에 연이어 참석하는 경우 하나의 단기연수로 처리할 수 있나요?",
        "신진연구인력 채용 절차는 어떻게 되나요?",
    ]
    for q in test_queries:
        print(f"\n=== 원: {q}")
        for v in rewrite_query(client, q, n_variants=2):
            print(f"  - {v}")
