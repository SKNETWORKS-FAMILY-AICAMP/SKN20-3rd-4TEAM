"""
SKN20-3rd-4TEAM 초기 창업 가이드 챗봇 UI
- Streamlit 기반 프론트엔드
- 백엔드 RAG와 연동 (rag_base.py 활용)
"""
import streamlit as st
import base64
from pathlib import Path
import os
import sys

# =========================
# 경로 설정 및 RAG 임포트
# =========================
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# rag_base.py에서 필요한 함수와 객체들을 임포트
from rag_base import (
    retriever,
    llm,
    choose_prompt,
    format_docs,
    qt_chain,
)
from langchain_core.output_parsers import StrOutputParser


# =========================
# RAG 백엔드 연동 함수
# =========================
def answer_question_with_rag(question: str) -> dict:
    """
    rag_base.py의 로직을 활용하여 질문에 답변하는 함수
    
    Args:
        question: 사용자의 질문
        
    Returns:
        dict: {
            "answer": 답변 텍스트,
            "sources": 참고 문서 정보
        }
    """
    try:
        # Query Transformation 적용 (rag_base.py의 qt_chain 사용)
        transformed_q = qt_chain.invoke({"question": question})
        
        # 적절한 프롬프트 선택 (원본 질문 기준)
        prompt = choose_prompt(question)
        
        # 변환된 질문으로 문서 검색
        docs = retriever.invoke(transformed_q)
        context = format_docs(docs)
        
        # 선택된 프롬프트로 최종 답변 생성
        answer_chain = prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({
            "context": context,
            "question": question
        })
        
        # 참고 문서 정보 수집
        sources = []
        data_types = set()
        
        for doc in docs:
            data_type = doc.metadata.get("data_type", "unknown")
            data_types.add(data_type)
            
            # 출처 정보 포맷
            source_info = f"[{data_type}]"
            if "source" in doc.metadata:
                source_info += f" {doc.metadata['source']}"
            sources.append(source_info)
        
        # 중복 제거
        sources = list(set(sources))
        
        return {
            "answer": answer,
            "sources": sources,
        }
        
    except Exception as e:
        # 에러 발생 시 안내 메시지
        error_msg = f"⚠️ 답변 생성 중 오류가 발생했습니다.\n\n오류: {str(e)}"
        return {
            "answer": error_msg,
            "sources": ["오류 발생 - 다시 시도해주세요"],
        }


# =========================
# 페이지 기본 설정
# =========================
st.set_page_config(
    page_title="초기 창업 가이드 챗봇",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 배경 이미지 + CSS 설정
# =========================
def set_background(image_path: str):
    """배경 이미지 설정 (파일이 없으면 스킵)"""
    if not os.path.exists(image_path):
        # 배경 이미지 없으면 기본 스타일만 적용
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .chat-container {
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
            }
            .chat-message {
                border-radius: 12px;
                padding: 12px 16px;
                margin-bottom: 10px;
            }
            .user-msg {
                background: #e3f2fd;
                margin-left: 60px;
                text-align: right;
            }
            .assistant-msg {
                background: #f5f5f5;
                margin-right: 60px;
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
                    linear-gradient(rgba(255, 255, 255, 0.75), rgba(255, 255, 255, 0.75)),
                    url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                color: #1f2937;
            }}
            .stMarkdown, .stText, h1, h2, h3, h4, h5, h6 {{
                color: #1f2937 !important;
            }}
            /* 버튼 스타일 */
            div.stButton > button {{
                background-color: #ffffff !important;
                color: #1f2937 !important;
                border: 1px solid #d1d5db !important;
                border-radius: 8px !important;
                padding: 10px 16px !important;
                font-size: 14px !important;
                font-weight: 500 !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }}
            div.stButton > button:hover {{
                background-color: #f3f4f6 !important;
                border-color: #9ca3af !important;
            }}
            .chat-container {{
                max-width: 900px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"배경 이미지 로드 실패: {e}")


# 👉 배경 이미지 파일 이름 (img 폴더에 저장됨)
set_background("img/Start-up-post.jpg")


# =========================
# 세션 상태 초기화 (채팅 내역)
# =========================
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # {"role": "user"|"assistant", "content": str}


# =========================
# 상단 타이틀 / 설명 영역
# =========================
st.markdown(
    """
    <div style='text-align:center; margin-bottom:30px;'>
        <h1 style='margin-bottom:10px;'>💡 초기 창업 가이드 챗봇</h1>
        <p style='color:#666; font-size:16px;'>창업에 필요한 모든 정보를 한 곳에서</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# 예시 질문 섹션
# =========================
st.markdown("### 🔥 자주 묻는 질문")
example_questions = [
    "지원사업 신청하려면 어떤 조건이 필요한가요?",
    "재도전성공패키지는 어떤 사람이 대상인가요?",
    "1인 창업 초기에 가장 조심해야 할 점은?",
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
    **다음과 같은 주제로 질문할 수 있습니다:**
    - 🎯 창업 준비 단계별 가이드
    - 💰 지원사업 신청 조건 및 자격
    - 🚀 재도전 전략 및 성공 사례
    - 📋 필요한 서류 및 준비물
    - ❓ 창업 관련 일반적인 질문
    
    **질문 팁:**
    - 구체적인 상황을 설명하면 더 정확한 답변을 받을 수 있습니다
    - 예: "1인 창업인데 어디서 자금 지원을 받을 수 있나요?"
    - 여러 번 묻는 것도 좋습니다. 다양한 관점에서 답변해드립니다
    """)


# =========================
# 채팅 영역
# =========================
st.markdown("### 💬 대화")

# 사이드바에 채팅 내역 표시
with st.sidebar:
    st.markdown("### 📋 대화 내역")
    
    if st.session_state["messages"]:
        for i, msg in enumerate(st.session_state["messages"]):
            if msg["role"] == "user":
                display_text = f"👤 **{msg['content'][:50]}...**" if len(msg['content']) > 50 else f"👤 **{msg['content']}**"
                st.markdown(f"<span style='color: white;'>{display_text}</span>", unsafe_allow_html=True)
        
        # 대화 초기화 버튼
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    else:
        st.markdown("<span style='color: white;'>아직 질문이 없습니다.</span>", unsafe_allow_html=True)

# 최신 메시지만 표시 (마지막 한 쌍의 질문-답변)
if st.session_state["messages"]:
    # 마지막 사용자 질문 찾기
    last_user_msg_idx = None
    for i in range(len(st.session_state["messages"]) - 1, -1, -1):
        if st.session_state["messages"][i]["role"] == "user":
            last_user_msg_idx = i
            break
    
    if last_user_msg_idx is not None:
        # 마지막 질문-답변 쌍 표시
        for i in range(last_user_msg_idx, len(st.session_state["messages"])):
            msg = st.session_state["messages"][i]
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])


# =========================
# 입력창 + RAG 연동
# =========================
user_input = st.chat_input("질문을 입력해주세요...", key="chat_input")

# pending_question이 있으면 그것을 사용
if "pending_question" in st.session_state and st.session_state["pending_question"]:
    user_input = st.session_state["pending_question"]
    st.session_state["pending_question"] = None

if user_input:
    # 1) 유저 메시지 세션/화면에 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # 2) rag_base.py의 로직을 활용한 RAG 백엔드 호출
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("답변 준비 중..."):
            result = answer_question_with_rag(user_input)
            answer_text = result["answer"]
            sources = result.get("sources", [])

            # 메인 답변
            st.markdown(answer_text)

            # 참고 문서/출처 영역
            if sources:
                with st.expander("📚 참고 자료"):
                    for s in sources:
                        st.markdown(f"- {s}")

    # 3) 어시스턴트 메시지도 세션에 저장
    st.session_state["messages"].append(
        {"role": "assistant", "content": answer_text}
    )