import streamlit as st
import time
from rag_engine import BKRAGEngine

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

def main():
    st.title("🤖 BK21 FOUR Q&A 챗봇")
    st.caption("사업 관련 문의사항을 자연어로 질문해 보세요. (기존 10,800여 건의 Q&A 데이터를 검색하여 답변합니다)")

    # 사이드바 (설정 메뉴)
    with st.sidebar:
        st.header("⚙️ 검색 설정")
        top_k = st.slider("참고할 문서 수 (Top-K)", min_value=1, max_value=10, value=3)
        st.divider()
        st.info("""
        **💡 이용 팁**
        - 구체적인 상황을 포함하여 질문하면 더 정확한 답변을 얻을 수 있습니다.
        - "장학금 지급", "국제화경비", "외국인 대학원생" 등 키워드 위주로도 질문이 가능합니다.
        """)

    # 대화 기록 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 참고 문헌이 있으면 함께 출력
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 참고한 원본 Q&A 보기"):
                    for idx, doc in enumerate(msg["sources"], 1):
                        date_str = f"({doc['a_date']})" if doc['a_date'] else ""
                        st.markdown(f"**[{idx}] {doc['id']} {date_str}**")
                        st.markdown(f"> **Q:** {doc['question']}")
                        st.markdown(f"> **A:** {doc['answer']}")
                        st.divider()

    # 채팅 입력창
    if prompt := st.chat_input("질문을 입력하세요 (예: 장학금 중복 수혜 가능한가요?)"):
        # 사용자 메시지 화면에 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 챗봇 응답 처리
        if st.session_state.engine:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                with st.spinner("관련 문서를 검색 중입니다..."):
                    # 1. 유사 문서 검색
                    retrieved_docs = st.session_state.engine.retrieve(prompt, top_k=top_k)
                
                if not retrieved_docs:
                    st.warning("관련된 기존 Q&A를 찾지 못했습니다.")
                else:
                    # 2. 스트리밍 응답 생성
                    for chunk in st.session_state.engine.generate_answer(prompt, retrieved_docs):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # 3. 출처 정보 표시
                    with st.expander("📚 참고한 원본 Q&A 보기"):
                        for idx, doc in enumerate(retrieved_docs, 1):
                            date_str = f"({doc['a_date']})" if doc['a_date'] else ""
                            st.markdown(f"**[{idx}] {doc['id']} {date_str}**")
                            st.markdown(f"> **Q:** {doc['question']}")
                            st.markdown(f"> **A:** {doc['answer']}")
                            st.divider()
                    
                    # 세션에 저장
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": retrieved_docs
                    })
        else:
            st.error("API 키가 없거나 DB가 구성되지 않아 응답할 수 없습니다.")

if __name__ == "__main__":
    main()
