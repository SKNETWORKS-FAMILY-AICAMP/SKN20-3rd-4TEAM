# app.py
"""
SKN20-3rd-4TEAM 초기 창업 가이드 챗봇 UI
- Streamlit 기반 프론트엔드
- 백엔드 RAG와 연동 예정
"""
import streamlit as st
import base64
from pathlib import Path
import os
import sys

# =========================
# 경로 설정 및 RAG 임포트
# =========================
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
load_dotenv()

# =========================
# RAG 초기화
# =========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 없음! .env 파일을 확인해주세요")

# 벡터DB 로드
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_startup_all",
    collection_name="startup_all_rag",
    embedding_function=embedding_model,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 기본 Base RAG 프롬프트
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 예비·초기 창업자를 도와주는 '창업 지원 통합 AI 어시스턴트'입니다.

[사용 가능한 정보 유형]
- 지원사업 공고 (announcement)
- 실패/재도전 사례 (cases)
- 창업 공간 정보 (space)
- 법령: 중소기업창업 지원법 등 (law)
- 통계, 매뉴얼 등 참고 자료

[답변 원칙]
1. 반드시 제공된 문맥(Context) 안의 정보만 사용하세요.
2. 문맥에 없는 내용은 추측하지 말고 솔직하게 말하세요.
3. 질문 성격에 따라 다음 정보 유형을 우선 활용하세요.
   - 지원사업·신청 가능 여부 → announcement
   - 법적 정의·자격 요건 → law
   - 조언·주의점 → cases
   - 공간·입주 → space
4. 핵심 답변 후 필요하면 bullet로 정리하세요.
5. 마지막에 참고 근거 유형을 요약하세요.

[출력 예]
- 핵심 답변
- 세부 정리
- [참고: 지원사업 공고, 실패 사례, 중소기업창업 지원법]
"""),
    ("human", """
[문맥]
{context}

[질문]
{question}

[답변]
""")
])

# 법령 전용 프롬프트
law_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 중소기업창업 지원법을 바탕으로 창업 제도와 요건을 설명하는 AI입니다.

[규칙]
1. 반드시 문맥에 있는 법령 내용만 사용하세요.
2. 가능하면 조문 번호(제○조)를 함께 제시하세요.
3. 문맥에 없는 내용은 "제공된 법령 문서에서 해당 내용은 확인되지 않습니다."라고 답하세요.
4. 답변 끝에 다음 문장을 반드시 포함하세요.
   "※ 본 답변은 일반 정보 제공이며, 구체적인 법률 자문은 아닙니다."
"""),
    ("human", """
[법령 문맥]
{context}

[질문]
{question}

[설명]
""")
])

# 지원사업 추천 전용 프롬프트
recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 예비·초기 창업자에게 가장 적합한 '지원사업을 추천하는 전문가 AI'입니다.

[목표]
사용자의 조건(나이, 지역, 업종, 창업 단계 등)을 기준으로
'실질적인 도움이 되는 사업(자금·공간·R&D·시제품·교육)'을 우선적으로 추천합니다.

[추천 우선순위]
1. 현금성 지원(사업화 자금, 시제품 제작비, R&D)
2. 입주 공간, 장비 지원
3. 엑셀러레이팅, 멘토링
4. 단순 교육/특강은 **마지막 순위**로 고려합니다.

[추천 규칙]
1. 반드시 announcement 문서만 사용하세요.
2. 사용자 조건과 '지역·연령·단계·업종'이 명확히 맞는 것만 추천하세요.
3. 최대 2개까지만 추천하세요. (너무 많으면 혼란을 줌)
4. 조건이 정확히 맞는 사업이 없으면 솔직하게:
   "현재 조건에 정확히 부합하는 자금 지원 사업은 제한적입니다."라고 말하세요.
5. IT·서비스업이면 반드시 '기술·콘텐츠·플랫폼' 키워드가 포함된 사업을 우선 고려하세요.

[출력 형식]
■ ✅ 추천 사업명
■ 🎯 왜 이 사용자에게 적합한지
■ 💰 지원 내용(자금/공간/교육 중 무엇인지 명확히)
■ 📝 신청 대상 요약
■ ⏳ 접수 기간
■ ⚠️ 주의사항
■ 🔗 추가 확인 필요 여부

마지막 줄에 반드시:
[참고: 지원사업 공고]
"""),
    ("human", """
[지원사업 문맥]
{context}

[사용자 조건]
{question}

위 형식에 맞춰 추천해 주세요.
""")
])

# 프롬프트 선택 함수
def choose_prompt(question: str):
    """질문 내용에 따라 적절한 프롬프트 선택"""
    recommend_keywords = ["추천", "맞는", "신청할 수 있는", "지원해주는", "사업 알려줘"]
    law_keywords = ["정의", "자격", "요건", "지원법", "법에서", "법상", "제도"]

    if any(k in question for k in recommend_keywords):
        return recommend_prompt
    if any(k in question for k in law_keywords):
        return law_prompt
    return rag_prompt

# 문서 포맷 함수
def format_docs(docs):
    """검색된 문서들을 문자열로 포맷"""
    return "\n\n---\n\n".join(d.page_content for d in docs)

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
            }}/'
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"배경 이미지 로드 실패: {e}")


# 👉 배경 이미지 파일 이름 (img 폴더에 저장됨)
set_background("img/Start-up-post.jpg")


# =========================
# RAG 백엔드 연동
# =========================
def answer_question_with_rag(question: str) -> dict:
    """
    RAG 체인을 사용하여 질문에 답변하는 함수
    
    Args:
        question: 사용자의 질문
        
    Returns:
        dict: {
            "answer": 답변 텍스트,
            "sources": 참고 문서 정보
        }
    """
    try:
        # 적절한 프롬프트 선택
        prompt = choose_prompt(question)
        
        # RAG 체인 구성
        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 체인 실행
        answer = chain.invoke(question)
        
        # 참고 문서 정보 수집
        docs = retriever.invoke(question)
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

    # 2) RAG 백엔드 호출
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
