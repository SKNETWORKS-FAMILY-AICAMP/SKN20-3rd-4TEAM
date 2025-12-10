import os
import warnings
import sys
from dotenv import load_dotenv

# LangChain 및 OpenAI 임포트
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document

# Streamlit UI 임포트
import streamlit as st

warnings.filterwarnings("ignore")
load_dotenv()

# ========================================
# 1. 환경 설정 및 초기화
# ========================================

# 페이지 설정 (가장 먼저 선언)
st.set_page_config(
    page_title="Startup-Guide-Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY가 없습니다. .env 파일을 확인해주세요.")
    st.stop()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") 

# 벡터DB 로드 (캐싱 적용)
@st.cache_resource
def load_vectorstore():
    print("🔄 벡터DB 로딩 중...")
    try:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory="./chroma_startup_all",
            collection_name="startup_all_rag",
            embedding_function=embedding_model,
        )
        return vectorstore
    except Exception as e:
        print(f"⚠ 벡터DB 로드 실패: {e}")
        return None

vectorstore = load_vectorstore()

if vectorstore:
    try:
        # DB 연결 테스트
        all_data = vectorstore.get()
        ids = all_data.get("ids", [])
        print(f"✅ 벡터DB 로드 완료 / 총 문서 청크: {len(ids)}")
    except Exception as e:
        st.warning(f"⚠ 벡터DB 상태 불안정: {e}")

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ========================================
# 2. 프롬프트 정의
# ========================================

# 관련성 검증
relevance_check_prompt = ChatPromptTemplate.from_template("""
당신은 문서와 질문의 관련성을 엄격하게 판단하는 전문가입니다.

[질문]
{question}

[검색된 문서 샘플]
{documents}

[판단 기준]
1. 질문의 핵심 주제와 문서의 내용이 직접적으로 관련되는가?
2. 문서가 질문에 대한 구체적인 정보를 제공하는가?
3. 단순히 유사한 단어가 있는 것이 아니라, 실제 답변 가능한 내용인가?

다음 중 하나로만 답변: "관련있음" 또는 "관련없음"
답변:""")

# Query Transformation
qt_prompt = ChatPromptTemplate.from_template("""
다음 사용자 질문을 벡터 검색에 적합한 '핵심 키워드 중심 문장'으로 바꾸세요.
불필요한 말은 제거하고, 핵심 조건만 남기세요.

원본 질문: {question}

변환된 검색용 문장:""")

# Multi Query
multi_query_prompt = ChatPromptTemplate.from_template("""
다음 질문에 대해 3가지 다른 관점의 검색 쿼리를 생성하세요.
번호나 설명 없이 쿼리만 한 줄씩 출력하세요.

원본질문: {question}""")

# 기본 RAG 프롬프트
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 예비·초기 창업자를 도와주는 '창업 지원 통합 AI 어시스턴트'입니다.

[답변 원칙]
1. 반드시 제공된 문맥(Context) 안의 정보만 사용하세요.
2. 문맥에 없는 내용은 추측하지 말고 솔직하게 모른다고 하세요.
3. 표(Table)를 출력해야 할 경우 반드시 Markdown 문법을 사용하세요.
4. 핵심 답변 후 필요하면 bullet point로 정리하세요.
"""),
    ("human", "[문맥]\n{context}\n\n[질문]\n{question}\n\n[답변]")
])

# 법령 프롬프트
law_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 중소기업창업 지원법을 바탕으로 창업 제도와 요건을 설명하는 AI입니다.
반드시 문맥에 있는 법령 내용만 사용하고, 조문 번호를 함께 제시하세요.
표(Table) 작성 시 Markdown 형식을 준수하세요.
"""),
    ("human", "[법령 문맥]\n{context}\n\n[질문]\n{question}\n\n[설명]")
])

# 추천 프롬프트
recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 예비·초기 창업자에게 가장 적합한 '지원사업을 추천하는 전문가 AI'입니다.
사용자 조건에 맞는 사업을 추천하고, 비교가 필요한 경우 Markdown 표를 사용하여 정리하세요.

[출력 형식]
■ 추천 사업명
■ 적합한 이유
■ 지원 내용
■ 접수 기간
■ 주의사항
"""),
    ("human", "[지원사업 문맥]\n{context}\n\n[사용자 조건]\n{question}\n\n추천해주세요.")
])

# Fallback 프롬프트
fallback_prompt = ChatPromptTemplate.from_template("""
질문: {question}

내부 문서에서 관련 정보를 찾지 못했습니다.
일반적인 지식을 바탕으로 답변해주세요.
표가 필요한 경우 Markdown을 사용하세요.

답변:""")

# 체인 생성
qt_chain = qt_prompt | llm | StrOutputParser()
multi_query_chain = multi_query_prompt | llm | StrOutputParser()
relevance_chain = relevance_check_prompt | llm | StrOutputParser()
fallback_chain = fallback_prompt | llm | StrOutputParser()


# ========================================
# 3. 헬퍼 함수
# ========================================

def choose_prompt(question: str):
    """질문 의도에 따른 프롬프트 선택"""
    recommend_keywords = ["추천", "맞는", "신청할 수 있는", "지원해주는", "사업 알려줘", "혜택"]
    law_keywords = ["정의", "자격", "요건", "지원법", "법에서", "법상", "제도", "규정"]
    
    if any(k in question for k in recommend_keywords):
        return recommend_prompt, "recommend_prompt"
    if any(k in question for k in law_keywords):
        return law_prompt, "law_prompt"
    return rag_prompt, "rag_prompt"

def format_docs_as_context(docs):
    """문서를 텍스트 컨텍스트로 변환"""
    if not docs:
        return ""
    parts = []
    for i, d in enumerate(docs, 1):
        if isinstance(d, Document):
            parts.append(f"[문서 {i}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def extract_web_sources(docs):
    """
    [수정됨] 문서 리스트에서 '웹 검색' 결과만 URL 추출
    내부 문서는 source가 'web'이 아니거나 URL 형태가 아니므로 제외됨
    """
    sources = set()
    for d in docs:
        if isinstance(d, Document):
            src_meta = d.metadata.get("source", "")
            url_meta = d.metadata.get("url", "")
            
            # 조건: source가 'web'이거나, url 메타데이터가 http로 시작할 때
            if src_meta == "web" or str(url_meta).startswith("http"):
                # 유효한 링크가 있으면 추가
                link = url_meta if url_meta else src_meta
                if str(link).startswith("http"):
                    sources.add(link)
    return list(sources)

def search_documents(queries, k_per_query=5):
    """멀티쿼리 벡터 검색"""
    if not vectorstore:
        return []
        
    all_docs_with_scores = []
    seen_contents = set()

    for q in queries:
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(q, k=k_per_query)
        except Exception as e:
            print(f"⚠ 검색 에러: {e}")
            docs_with_scores = []

        for doc, distance in docs_with_scores:
            if doc.page_content in seen_contents:
                continue
            seen_contents.add(doc.page_content)
            # 거리 -> 유사도 변환
            similarity = max(0.0, 1.0 - (distance / 2.0))
            all_docs_with_scores.append((doc, similarity))
            
    return all_docs_with_scores

def filter_by_similarity(docs_with_scores, threshold=0.3):
    return [(doc, sim) for doc, sim in docs_with_scores if sim >= threshold]

def check_relevance(question, docs_with_scores):
    """LLM을 이용한 문서 관련성 검증"""
    top_docs = docs_with_scores[:3]
    if not top_docs:
        return False
    
    docs_text = "\n".join([d[0].page_content[:400] for d in top_docs])
    
    try:
        res = relevance_chain.invoke({"question": question, "documents": docs_text})
        return "관련있음" in res
    except:
        return False

def web_search(query: str, k=3):
    """Tavily 웹 검색"""
    try:
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY API KEY 없음")
            
        retriever = TavilySearchAPIRetriever(k=k)
        results = retriever.invoke(query)
        
        # 결과를 Document 객체로 변환 및 메타데이터 정규화
        docs = []
        for r in results:
            content = r.page_content if hasattr(r, 'page_content') else str(r)
            meta = r.metadata if hasattr(r, 'metadata') else {}
            
            # 메타데이터에 'web' 표시
            meta['source'] = 'web'
            docs.append(Document(page_content=content, metadata=meta))
            
        return docs
    except Exception as e:
        print(f"❌ 웹검색 실패: {e}")
        return []

def rag_answer_from_docs(question: str, documents):
    """문서를 기반으로 답변 생성 및 웹 출처 추출"""
    # 튜플에서 문서 객체만 추출
    docs = [item[0] if isinstance(item, tuple) else item for item in documents]
    
    context = format_docs_as_context(docs)
    
    # [수정] 웹 출처만 추출
    web_sources = extract_web_sources(docs)

    if not context.strip():
        return fallback_chain.invoke({"question": question}), []

    prompt, pname = choose_prompt(question)
    print(f"👉 프롬프트: {pname}")
    
    try:
        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question
        })
        return answer, web_sources
    except Exception as e:
        print(f"⚠ 생성 오류: {e}")
        return fallback_chain.invoke({"question": question}), []

# ========================================
# 4. 메인 파이프라인 (RAG Logic)
# ========================================

def multi_query_rag_with_qt(question: str):
    """최적화된 RAG 파이프라인"""
    
    # 1. 쿼리 변환
    try:
        qt_query = qt_chain.invoke({"question": question})
        queries = multi_query_chain.invoke({"question": qt_query}).splitlines()
        queries = [q.strip() for q in queries if q.strip()][:3] # 최대 3개
    except:
        queries = [question]

    # 2. 내부 문서 검색
    all_docs = search_documents(queries)
    filtered_docs = filter_by_similarity(all_docs, threshold=0.3)
    
    is_relevant = False
    if filtered_docs:
        is_relevant = check_relevance(question, filtered_docs)
    
    # 3. 분기 처리
    if is_relevant:
        # 내부 문서 사용 (출처 리스트는 extract_web_sources에 의해 빈 리스트가 됨)
        return rag_answer_from_docs(question, filtered_docs[:7])
    else:
        # 웹 검색 전환
        web_docs = web_search(question)
        if web_docs:
            return rag_answer_from_docs(question, web_docs)
        else:
            return fallback_chain.invoke({"question": question}), []


# ========================================
# 5. Streamlit UI
# ========================================

# CSS 스타일
st.markdown("""
<style>
    .message-user { background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: right; }
    .message-bot { background-color: #e8f5e9; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
    /* 링크 스타일 */
    a { text-decoration: none; color: #0068c9; font-weight: bold; }
    a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

st.title("💬 Startup-Guide-Bot")
st.caption("예비/초기 창업자를 위한 AI 어시스턴트")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 사이드바
with st.sidebar:
    st.header("💡 추천 질문")
    if st.button("창업자의 정의가 뭐예요?"):
        st.session_state["pending_input"] = "창업자의 정의가 뭐예요?"
    if st.button("최근 AI 기술 트렌드 알려줘"):
        st.session_state["pending_input"] = "최근 AI 기술 트렌드 알려줘"
    
    st.divider()
    st.info("내부 문서에 정보가 없으면 웹 검색을 수행합니다.")

# 채팅 기록 출력
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 저장된 출처가 있으면 표시
        if msg.get("sources"):
            with st.expander("🌐 웹 검색 참고 자료"):
                for src in msg["sources"]:
                    st.markdown(f"- [{src}]({src})")

# 입력 처리
user_input = st.chat_input("질문을 입력하세요...")

# 사이드바 버튼 클릭 시 처리
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input:
    # 1. 사용자 메시지 표시 및 저장
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # 2. AI 답변 생성 (st.status 사용으로 화면 비활성화 방지)
    with st.chat_message("assistant"):
        # [핵심] spinner 대신 status 사용
        with st.status("🚀 답변을 생성하고 있습니다...", expanded=True) as status:
            try:
                st.write("🔍 질문 분석 및 문서 검색 중...")
                
                # RAG 파이프라인 실행
                answer_text, sources = multi_query_rag_with_qt(user_input)
                
                st.write("📝 답변 작성 중...")
                
                # 상태 완료 처리 (접힘)
                status.update(label="답변 생성 완료!", state="complete", expanded=False)
                
                # 답변 출력
                st.markdown(answer_text)
                
                # 웹 소스가 있을 경우에만 출처 표시
                if sources:
                    with st.expander("🌐 웹 검색 참고 자료"):
                        for src in sources:
                            st.markdown(f"- [{src}]({src})")
                
                # 세션에 저장
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": answer_text,
                    "sources": sources
                })
                
            except Exception as e:
                status.update(label="오류 발생", state="error")
                st.error(f"에러가 발생했습니다: {e}")