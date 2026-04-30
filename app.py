import re
import time

import streamlit as st

from rag_engine import BKRAGEngine, EMBEDDING_MODEL, LLM_MODEL
from request_log import log_feedback, log_request

# NRF BK21 Q&A 게시판 원문 URL 패턴
NRF_QNA_URL = "https://bk21four.nrf.re.kr/qnaBbs/selectBoardArticle.do?nttId={ntt_id}&bbsId=BBSMSTR_000000000022"

# 답변 본문 내 nttId 인용 패턴 — "nttId 12345" 형식만 안전하게 처리
# (괄호 안의 단순 숫자는 다른 것과 모호하므로 처리 X)
# 한글 조사("을", "와", "에서")가 뒤에 붙어도 매칭하도록 `\b` 대신 명시적 경계 사용
_NTTID_CITATION_RE = re.compile(r'(?<![A-Za-z0-9])nttId\s+(\d{4,6})(?!\d)')


def _nrf_url(ntt_id: str) -> str:
    return NRF_QNA_URL.format(ntt_id=ntt_id)


def linkify_nttids(text: str) -> str:
    """답변 본문의 'nttId 12345' 표기를 NRF 게시판 markdown 링크로 변환."""
    return _NTTID_CITATION_RE.sub(
        lambda m: f"[nttId {m.group(1)}]({_nrf_url(m.group(1))})",
        text,
    )

# 페이지 기본 설정
st.set_page_config(
    page_title="BK21 FOUR Q&A 챗봇",
    page_icon="🤖",
    layout="centered"
)

# 커스텀 CSS (UI 다듬기)
st.markdown("""
<style>
.stChatMessage {
    padding: 1rem;
    border-radius: 0.5rem;
}
.source-box {
    font-size: 0.85rem;
    padding: 10px;
    background-color: #1E1E2E;
    border-radius: 5px;
    border-left: 3px solid #FF4B4B;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine" not in st.session_state:
    try:
        st.session_state.engine = BKRAGEngine()
    except Exception as e:
        st.error(f"엔진 초기화 실패: {e}")
        st.session_state.engine = None


def _render_sources(docs):
    """검색된 문서를 규정/Q&A로 분리해 표시."""
    regs = [d for d in docs if d.get("doc_type") == "regulation"]
    qnas = [d for d in docs if d.get("doc_type") == "qna"]

    if regs:
        with st.expander(f"📖 참고한 규정 {len(regs)}건"):
            for idx, d in enumerate(regs, 1):
                version = f" · 버전 {d['version']}" if d.get('version') else ""
                st.markdown(f"**[{idx}] {d['citation']}**{version}")
                st.markdown(f"> {d['full_text']}")
                st.divider()

    if qnas:
        with st.expander(f"💬 참고한 과거 Q&A {len(qnas)}건"):
            for idx, d in enumerate(qnas, 1):
                date_str = f"({d['a_date']})" if d.get('a_date') else ""
                ntt_id = str(d['id'])
                link = _nrf_url(ntt_id)
                st.markdown(f"**[{idx}] [nttId {ntt_id}]({link})** {date_str}  ↗ NRF 게시판")
                st.markdown(f"> **Q:** {d['question']}")
                st.markdown(f"> **A:** {d['answer']}")
                st.divider()


def _submit_feedback(message_idx: int, feedback: str):
    """피드백 콜백 — 세션 메시지 갱신 + DB 기록."""
    msg = st.session_state.messages[message_idx]
    msg["feedback"] = feedback
    log_id = msg.get("log_id")
    if log_id is not None:
        log_feedback(log_id, feedback)


def _render_feedback(message_idx: int, msg: dict):
    """답변 옆 👍/👎 버튼. 이미 피드백된 경우 결과만 표시."""
    if msg.get("log_id") is None:
        return  # 로깅 실패한 응답에는 표시 안 함
    fb = msg.get("feedback")
    if fb:
        st.caption(f"피드백 기록: {'👍 도움됨' if fb == 'up' else '👎 부족함'}")
        return
    cols = st.columns([1, 1, 8])
    cols[0].button("👍", key=f"fb_up_{message_idx}",
                   on_click=_submit_feedback, args=(message_idx, "up"))
    cols[1].button("👎", key=f"fb_down_{message_idx}",
                   on_click=_submit_feedback, args=(message_idx, "down"))


def main():
    st.title("🤖 BK21 FOUR Q&A 챗봇")
    st.caption("사업 관련 문의사항을 자연어로 질문해 보세요. 공식 규정집 + 과거 Q&A 10,800여 건을 함께 검색하여 답변합니다.")

    # 사이드바 (설정 메뉴)
    with st.sidebar:
        st.header("⚙️ 검색 설정")
        top_k = st.slider("참고할 문서 수 (Top-K)", min_value=1, max_value=10, value=5)
        st.divider()
        st.info("""
        **💡 이용 팁**
        - 구체적인 상황을 포함하여 질문하면 더 정확한 답변을 얻을 수 있습니다.
        - "장학금 지급", "국제화경비", "외국인 대학원생" 등 키워드 위주로도 질문이 가능합니다.
        - 답변 아래 👍/👎 버튼으로 품질을 평가해 주세요. (개선에 활용됩니다)
        """)

    # 대화 기록 출력
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if msg["role"] == "assistant":
                content = linkify_nttids(content)
            st.markdown(content)
            if msg["role"] == "assistant":
                if "sources" in msg and msg["sources"]:
                    _render_sources(msg["sources"])
                _render_feedback(idx, msg)

    # 채팅 입력창
    if prompt := st.chat_input("질문을 입력하세요 (예: 장학금 중복 수혜 가능한가요?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.engine:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                t0 = time.time()

                with st.spinner("관련 문서를 검색 중입니다..."):
                    retrieved_docs = st.session_state.engine.retrieve(prompt, top_k=top_k)

                if not retrieved_docs:
                    st.warning("관련된 참고 문서를 찾지 못했습니다.")
                else:
                    for chunk in st.session_state.engine.generate_answer(prompt, retrieved_docs):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    # 스트리밍 완료 후 nttId → NRF 링크로 변환해 재렌더
                    message_placeholder.markdown(linkify_nttids(full_response))

                    _render_sources(retrieved_docs)

                    latency_ms = int((time.time() - t0) * 1000)
                    log_id = None
                    try:
                        log_id = log_request(
                            query=prompt,
                            top_k=top_k,
                            retrieved_docs=retrieved_docs,
                            answer=full_response,
                            latency_ms=latency_ms,
                            llm_model=LLM_MODEL,
                            embedding_model=EMBEDDING_MODEL,
                        )
                    except Exception as e:
                        st.warning(f"요청 로그 기록 실패: {e}")

                    new_msg = {
                        "role": "assistant",
                        "content": full_response,
                        "sources": retrieved_docs,
                        "log_id": log_id,
                        "feedback": None,
                    }
                    st.session_state.messages.append(new_msg)
                    _render_feedback(len(st.session_state.messages) - 1, new_msg)
        else:
            st.error("API 키가 없거나 DB가 구성되지 않아 응답할 수 없습니다.")


if __name__ == "__main__":
    main()
