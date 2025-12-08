"""
SKN20-3rd-4TEAM 고급 창업 가이드 챗봇 UI (멀티쿼리 + LLM 관련성 검증)
- Streamlit 기반 프론트엔드
- 백엔드: multi_copy_llm추가.py의 고급 RAG 로직 활용
- 특징: 멀티쿼리, 유사도 필터링, LLM 관련성 검증, Fallback LLM
- 디자인: 데모 시나리오 스타일
"""
import streamlit as st
import base64
from pathlib import Path
import os
import sys

# =========================
# 경로 설정 및 고급 RAG 임포트
# =========================
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# multi_copy_llm추가.py에서 필요한 함수들을 임포트
try:
    from rag_base_multi import (
        multi_query_rag_with_qt,
    )
except ImportError:
    try:
        # 파일명이 다를 수 있으니 다시 시도
        import importlib.util
        spec = importlib.util.spec_from_file_location("multi_llm", "multi_copy_llm추가.py")
        multi_llm = importlib.util.module_from_spec(spec)
        sys.modules["multi_llm"] = multi_llm
        spec.loader.exec_module(multi_llm)
        multi_query_rag_with_qt = multi_llm.multi_query_rag_with_qt
    except Exception as e:
        st.error(f"❌ 고급 RAG 모듈 로드 실패: {e}")
        st.stop()


# =========================
# 페이지 기본 설정
# =========================
st.set_page_config(
    page_title="Startup-Guide-Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================
# CSS 스타일링 (데모 시나리오 스타일)
# =========================
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background-color: #f5f5f7;
    }
    
    /* 헤더 */
    .header-container {
        background: white;
        padding: 15px 30px;
        border-bottom: 1px solid #e5e5e5;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .logo {
        font-size: 20px;
        font-weight: 700;
        color: #5b5bff;
    }
    
    /* 메인 컨테이너 */
    .main-container {
        display: grid;
        grid-template-columns: 350px 1fr;
        gap: 20px;
        padding: 20px;
        height: calc(100vh - 80px);
    }
    
    /* 왼쪽 사이드바 */
    .sidebar-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e5e5e5;
        overflow-y: auto;
    }
    
    .sidebar-title {
        font-size: 16px;
        font-weight: 600;
        color: #333;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 2px solid #5b5bff;
    }
    
    .menu-item {
        background: #f5f5f7;
        border: 2px solid #e5e5e5;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .menu-item:hover {
        border-color: #5b5bff;
        background: #f0f0ff;
    }
    
    .menu-title {
        font-weight: 600;
        color: #333;
        margin-bottom: 5px;
    }
    
    .menu-desc {
        font-size: 12px;
        color: #999;
        line-height: 1.4;
    }
    
    /* 채팅 컨테이너 */
    .chat-container {
        background: white;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .chat-header {
        background: white;
        padding: 20px;
        border-bottom: 1px solid #e5e5e5;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .chat-header-status {
        width: 10px;
        height: 10px;
        background: #31a24c;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .chat-header-title {
        font-weight: 600;
        color: #333;
    }
    
    .chat-header-subtitle {
        font-size: 12px;
        color: #999;
        margin-left: auto;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
    }
    
    .message {
        margin-bottom: 16px;
    }
    
    .message-bot {
        display: flex;
        gap: 12px;
    }
    
    .message-bot .avatar {
        width: 32px;
        height: 32px;
        background: #5b5bff;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .message-content {
        background: #f0f0f5;
        padding: 12px 16px;
        border-radius: 8px;
        max-width: 80%;
    }
    
    .message-user {
        display: flex;
        justify-content: flex-end;
        gap: 12px;
    }
    
    .message-user .message-content {
        background: #5b5bff;
        color: white;
    }
    
    /* 참조 문서 */
    .references {
        background: #f9f9fb;
        border-top: 1px solid #e5e5e5;
        padding: 12px 16px;
        border-radius: 0 0 8px 8px;
        font-size: 12px;
        color: #666;
    }
    
    .reference-item {
        padding: 4px 0;
    }
    
    .reference-label {
        display: inline-block;
        background: #e8e8f5;
        color: #5b5bff;
        padding: 2px 6px;
        border-radius: 3px;
        margin-right: 6px;
        font-weight: 600;
        font-size: 11px;
    }
    
    /* 입력창 */
    .chat-input-container {
        padding: 20px;
        border-top: 1px solid #e5e5e5;
        background: white;
        display: flex;
        gap: 8px;
    }
    
    .chat-input {
        flex: 1;
    }
    
    /* 버튼 */
    .send-button {
        width: 40px;
        height: 40px;
        background: #5b5bff;
        border: none;
        border-radius: 50%;
        color: white;
        font-size: 20px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
    }
    
    .send-button:hover {
        background: #4a4ae5;
        transform: scale(1.05);
    }
    
    /* 스크롤 */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #999;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================
# 세션 상태 초기화
# =========================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None

# 통합 히스토리
if "all_history" not in st.session_state:
    st.session_state["all_history"] = []


# =========================
# 헤더
# =========================
st.markdown("""
    <div class="header-container">
        <div class="logo">💬 Startup-Guide-Bot</div>
        <div style="text-align: right; font-size: 12px; color: #999;">
            <span style="color: #31a24c;">●</span> Online | Powered by GPT-4o-mini
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# 메인 레이아웃
# =========================
col_left, col_main = st.columns([0.25, 0.75], gap="medium")

# 왼쪽 사이드바
with col_left:
    # 추천 질문 버튼들
    st.markdown("<div style='font-size: 12px; color: #999; margin-bottom: 8px;'>💡 추천 질문</div>", unsafe_allow_html=True)
    
    col_cat1, col_cat2 = st.columns(2, gap="small")
    with col_cat1:
        if st.button("👥 창업자\n요건", key="btn_startup_req", use_container_width=True):
            st.session_state["user_input"] = "창업자의 정의가 뭐예요?"
            st.rerun()
    with col_cat2:
        if st.button("📘 IP\n가이드", key="btn_ip_guide", use_container_width=True):
            st.session_state["user_input"] = "2025년 IP 전략의 주요 내용이 뭐예요?"
            st.rerun()
    
    if st.button("💰 지원사업\n비교", key="btn_support_program", use_container_width=True):
        st.session_state["user_input"] = "지원사업별 차이가 뭐예요?"
        st.rerun()
    
    st.divider()
    
    # 통합 히스토리 섹션
    st.markdown("""
        <div style="font-size: 14px; font-weight: 600; color: #333; margin-bottom: 10px;">
        📌 전체 질문 히스토리
        </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state["all_history"]:
        st.info("아직 질문이 없습니다")
    else:
        for i, item in enumerate(reversed(st.session_state["all_history"]), 1):
            with st.expander(f"Q{len(st.session_state['all_history'])-i+1}: {item['question'][:25]}...", expanded=False):
                st.write(item["question"])
                if item.get("sources"):
                    st.caption("참고자료: " + ", ".join(item["sources"]))

# 오른쪽 채팅
with col_main:
    # 채팅 헤더
    st.markdown("""
        <div class="chat-header">
            <div class="chat-header-status"></div>
            <div class="chat-header-title">Startup-Guide-Bot</div>
            <div class="chat-header-subtitle">Powered by GPT-4o-mini</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 채팅 메시지
    chat_container = st.container(border=False)
    with chat_container:
        if st.session_state["messages"]:
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    st.markdown(f"""
                        <div class="message message-user">
                            <div class="message-content">{msg['content']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    content = msg['content']
                    sources = msg.get('sources', [])
                    
                    st.markdown(f"""
                        <div class="message message-bot">
                            <div class="avatar">⚙️</div>
                            <div>
                                <div class="message-content">{content}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # 참고자료를 expander로 표시
                    if sources:
                        with st.expander("📎 참고 자료 보기"):
                            for source in sources:
                                st.caption(f"• {source}")
        else:
            st.markdown("""
                <div style="text-align: center; padding: 40px; color: #ccc;">
                    <div style="font-size: 48px; margin-bottom: 10px;">💬</div>
                    <div>메시지를 입력하세요...</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # 입력창
    cols = st.columns([1, 0.08], gap="small")
    with cols[0]:
        user_input = st.text_input("메시지를 입력하세요...", label_visibility="collapsed", key="user_input")
    
    with cols[1]:
        if st.button("➤", key="send_btn", use_container_width=True):
            if user_input:
                # 유저 메시지 추가
                st.session_state["messages"].append({"role": "user", "content": user_input})
                
                # RAG 호출
                with st.spinner(""):
                    try:
                        answer_text, sources = multi_query_rag_with_qt(user_input, top_k=10, similarity_threshold=0.2)
                        
                    except Exception as e:
                        answer_text = f"오류 발생: {str(e)}"
                        sources = []
                
                # 어시스턴트 메시지 추가 (참고자료 제외)
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": answer_text,
                    "sources": sources if sources else []
                })
                
                # 통합 히스토리에 저장
                qa_pair = {
                    "question": user_input,
                    "answer": answer_text,
                    "sources": sources if sources else []
                }
                st.session_state["all_history"].append(qa_pair)
                
                st.rerun()

