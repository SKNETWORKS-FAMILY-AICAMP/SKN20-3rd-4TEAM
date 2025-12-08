"""
SKN20-3rd-4TEAM 고급 창업 가이드 챗봇 UI (멀티쿼리 + LLM 관련성 검증)
- Streamlit 기반 프론트엔드
- 백엔드: multi copy_llm추가.py의 고급 RAG 로직 활용
- 특징: 멀티쿼리, 유사도 필터링, LLM 관련성 검증, Fallback LLM
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
    from multi_copy_llm추가 import (
        multi_query_rag_with_qt,
    )
except ImportError:
    try:
        # 파일명이 다를 수 있으니 다시 시도
        import importlib.util
        spec = importlib.util.spec_from_file_location("multi_llm", "multi copy_llm추가.py")
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
    page_title="고급 창업 가이드 챗봇",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 배경 이미지 + CSS 설정
# =========================
def set_background(image_path: str):
    """배경 이미지 설정 (파일이 없으면 그래디언트 적용)"""
    if not os.path.exists(image_path):
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            }
            .stMarkdown, .stText, h1, h2, h3, h4, h5, h6 {
                color: #ffffff !important;
            }
            div.stButton > button {
                background-color: #667eea !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 10px 16px !important;
            }
            div.stButton > button:hover {
                background-color: #5568d3 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: 
                    linear-gradient(rgba(30, 60, 114, 0.8), rgba(42, 82, 152, 0.8)),
                    url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                color: #ffffff;
            }}
            .stMarkdown, .stText, h1, h2, h3, h4, h5, h6 {{
                color: #ffffff !important;
            }}
            div.stButton > button {{
                background-color: #667eea !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 10px 16px !important;
            }}
            div.stButton > button:hover {{
                background-color: #5568d3 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"배경 이미지 로드 실패: {e}")


set_background("img/Start-up-post.jpg")


# =========================
# 세션 상태 초기화
# =========================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "show_details" not in st.session_state:
    st.session_state["show_details"] = False


# =========================
# 상단 타이틀
# =========================
st.markdown(
    """
    <div style='text-align:center; margin-bottom:30px;'>
        <h1 style='margin-bottom:10px;'>🚀 고급 창업 가이드 챗봇</h1>
        <p style='font-size:16px;'>멀티쿼리 + 관련성 검증으로 더 정확한 답변을</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# 예시 질문
# =========================
st.markdown("### 🔥 추천 질문")
example_questions = [
    "서울에서 AI 기반 스타트업을 할 때 받을 수 있는 지원사업은?",
    "창업 초기 자금이 부족할 때 어떻게 해야 할까?",
    "기술 창업자를 위한 멘토링 프로그램이 있나요?",
]

cols = st.columns(3)
for i, q in enumerate(example_questions):
    with cols[i]:
        if st.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()

st.markdown("---")

# =========================
# 사용 가이드
# =========================
with st.expander("💡 질문하는 방법"):
    st.markdown("""
    **이 앱의 장점:**
    - 🔄 같은 질문을 다양한 관점으로 검색
    - ✅ 검색된 문서가 실제 질문과 관련있는지 검증
    - 🔀 관련 문서가 없으면 LLM 자체 지식으로 답변
    
    **질문 팁:**
    - 구체적인 상황을 설명하면 더 정확한 답변
    - 예: "25세 AI 개발자인데 서울에서 창업하려면?"
    - 기본 앱에서 만족스럽지 못한 답변을 이 앱에서 다시 물어보기
    """)

# =========================
# 채팅 영역
# =========================
st.markdown("### 💬 대화")

# 사이드바 - 대화 내역 (최신 10개만 표시)
with st.sidebar:
    st.markdown("### 📋 최근 질문 내역")
    
    if st.session_state["messages"]:
        # 모든 질문을 찾고, 최신 10개만 필터링
        user_questions = [msg for msg in st.session_state["messages"] if msg["role"] == "user"]
        total_questions = len(user_questions)
        recent_questions = user_questions[-10:]  # 최신 10개만
        
        if user_questions:
            st.markdown(f"**전체: {total_questions}개 | 표시: {len(recent_questions)}개**")
            st.divider()
            
            # 역순으로 표시 (최신 질문이 위에)
            for idx, msg in enumerate(reversed(recent_questions), 1):
                display_text = f"{msg['content'][:30]}..." if len(msg['content']) > 30 else msg['content']
                st.markdown(f"<span style='color: #e0e0e0; font-size: 11px;'>**Q{total_questions-idx+1}:** {display_text}</span>", unsafe_allow_html=True)
                st.write("")  # 간격
            
            if total_questions > 10:
                st.caption(f"⬇️ 이전 {total_questions - 10}개 질문은 보관 중...")
        
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    else:
        st.markdown("<span style='color: #999999;'>아직 질문이 없습니다.</span>", unsafe_allow_html=True)

# 최신 메시지 표시
if st.session_state["messages"]:
    last_user_msg_idx = None
    for i in range(len(st.session_state["messages"]) - 1, -1, -1):
        if st.session_state["messages"][i]["role"] == "user":
            last_user_msg_idx = i
            break
    
    if last_user_msg_idx is not None:
        for i in range(last_user_msg_idx, len(st.session_state["messages"])):
            msg = st.session_state["messages"][i]
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])


# =========================
# 입력창 + 고급 RAG 연동
# =========================
user_input = st.chat_input("질문을 입력해주세요...", key="chat_input")

if "pending_question" in st.session_state and st.session_state["pending_question"]:
    user_input = st.session_state["pending_question"]
    st.session_state["pending_question"] = None

if user_input:
    # 유저 메시지 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # 고급 RAG 백엔드 호출
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 멀티쿼리 검색 + 관련성 검증 중..."):
            try:
                answer_text, sources = multi_query_rag_with_qt(user_input, top_k=10, similarity_threshold=0.3)
                
                # 메인 답변
                st.markdown(answer_text)
                
                # 참고 문서
                if sources:
                    with st.expander("📚 참고 자료"):
                        for j, source in enumerate(sources, 1):
                            st.markdown(f"- {source}")
                else:
                    st.info("💡 **LLM 자체 지식으로 답변했습니다.** (관련 문서가 충분하지 않았습니다)")
                
            except Exception as e:
                error_msg = f"❌ 오류 발생: {str(e)}"
                st.error(error_msg)
                answer_text = error_msg

    # 세션에 저장
    st.session_state["messages"].append({"role": "assistant", "content": answer_text})


# =========================
# 하단 정보
# =========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #cccccc; font-size: 12px;'>
    <p>고급 창업 가이드 챗봇 | 멀티쿼리 + LLM 검증 기반</p>
    <p>더 정확한 답변이 필요하면 이 앱을 사용하세요!</p>
</div>
""", unsafe_allow_html=True)
