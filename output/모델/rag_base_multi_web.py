import os
import warnings
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document

warnings.filterwarnings("ignore")
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 없음! .env 확인해줘")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") 

# 벡터DB 로드
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_startup_all",
    collection_name="startup_all_rag",
    embedding_function=embedding_model,
)

try:
    all_data = vectorstore.get()
    ids = all_data.get("ids", [])
    print(f"✅ 벡터DB 로드 완료 / 총 벡터 개수: {len(ids)}")
except Exception as e:
    print("⚠ 벡터DB 상태 확인 중 에러:", e)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ========================================
# 프롬프트 정의
# ========================================

# 관련성 검증 프롬프트
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

[예시]
- "서울 동물병원" vs "서울 창업 공간" → 관련없음 (서울만 공통)
- "AI 교육" vs "창업 교육" → 관련없음 (교육만 공통, 주제 다름)
- "창업 지원사업" vs "창업 자금 지원" → 관련있음 (직접 연관)

다음 중 하나로만 답변: "관련있음" 또는 "관련없음"

답변:""")

# Query Transformation
qt_prompt = ChatPromptTemplate.from_template("""
다음 사용자 질문을 벡터 검색에 적합한 '핵심 키워드 중심 문장'으로 바꾸세요.
불필요한 말은 제거하고, 핵심 조건만 남기세요.

원본 질문: {question}

변환된 검색용 문장:""")

# 멀티쿼리 생성
multi_query_prompt = ChatPromptTemplate.from_template("""
다음질문에 대해 3가지 다른 관점의 검색 쿼리를 생성하세요.
각쿼리는 세 줄로 구분하여 출력하세요        
번호나 설명 없이 쿼리만 출력하세요

원본질문: {question}""")

# 기본 RAG 프롬프트
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
"""),
    ("human", "[문맥]\n{context}\n\n[질문]\n{question}\n\n[답변]")
])

# 법령 전용 프롬프트
law_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 중소기업창업 지원법을 바탕으로 창업 제도와 요건을 설명하는 AI입니다.

[규칙]
1. 반드시 문맥에 있는 법령 내용만 사용하세요.
2. 가능하면 조문 번호(제○조)를 함께 제시하세요.
3. 문맥에 없는 내용은 "제공된 법령 문서에서 해당 내용은 확인되지 않습니다."라고 답하세요.
4. 답변 끝에 "※ 본 답변은 일반 정보 제공이며, 구체적인 법률 자문은 아닙니다."를 포함하세요.
"""),
    ("human", "[법령 문맥]\n{context}\n\n[질문]\n{question}\n\n[설명]")
])

# 지원사업 추천 프롬프트
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
4. 단순 교육/특강은 마지막 순위

[추천 규칙]
1. 반드시 announcement 문서만 사용
2. 사용자 조건과 '지역·연령·단계·업종'이 명확히 맞는 것만 추천
3. 최대 2개까지만 추천
4. 조건이 맞는 사업이 없으면 솔직하게 말하기
5. IT·서비스업이면 '기술·콘텐츠·플랫폼' 키워드 포함 사업 우선

[출력 형식]
■ 추천 사업명
■ 왜 이 사용자에게 적합한지
■ 지원 내용(자금/공간/교육 중 무엇인지 명확히)
■ 신청 대상 요약
■ 접수 기간
■ 주의사항

마지막 줄: [참고: 지원사업 공고]
"""),
    ("human", "[지원사업 문맥]\n{context}\n\n[사용자 조건]\n{question}\n\n위 형식에 맞춰 추천해 주세요.")
])

# Fallback 프롬프트
fallback_prompt = ChatPromptTemplate.from_template("""
질문: {question}

내부 문서에서 관련 정보를 찾지 못했습니다.
일반적인 지식을 바탕으로 답변해주세요.

답변:""")

# 체인 생성
qt_chain = qt_prompt | llm | StrOutputParser()
multi_query_chain = multi_query_prompt | llm | StrOutputParser()
relevance_chain = relevance_check_prompt | llm | StrOutputParser()
fallback_chain = fallback_prompt | llm | StrOutputParser()


# ========================================
# 헬퍼 함수
# ========================================

def choose_prompt(question: str):
    """질문 유형에 따라 적절한 프롬프트 선택"""
    recommend_keywords = ["추천", "맞는", "신청할 수 있는", "지원해주는", 
                         "사업 알려줘", "혜택", "지원금", "지원사업"]
    law_keywords = ["정의", "자격", "요건", "지원법", "법에서", "법상", "제도", "시행","규정"]
    
    if any(k in question for k in recommend_keywords):
        return recommend_prompt, "recommend_prompt"
    if any(k in question for k in law_keywords):
        return law_prompt, "law_prompt"
    return rag_prompt, "rag_prompt"

def format_docs_as_context(docs):
    """RAG 프롬프트에 넣기 좋은 컨텍스트로 변환 (출처 메타데이터 포함)"""
    if not docs:
        return ""
    parts = []
    for i, d in enumerate(docs, 1):
        if isinstance(d, Document):
            src = d.metadata.get("source", d.metadata.get("url", "unknown"))
            parts.append(f"[문서 {i}] (출처: {src})\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def search_documents(queries, k_per_query=10):
    """vectorstore에서 멀티쿼리 검색 후 중복제거 → (Document, similarity) 리스트 반환
    내부 문서는 이미 Document 형식이므로 변환 불필요"""
    all_docs_with_scores = []
    seen_contents = set()

    for q in queries:
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(q, k=k_per_query)
        except Exception as e:
            print("⚠ vectorstore.similarity_search_with_score 오류:", e)
            docs_with_scores = []

        for doc, distance in docs_with_scores:
            # 중복 체크
            if doc.page_content in seen_contents:
                continue
            seen_contents.add(doc.page_content)
            
            # Chroma distance -> 유사도(0-1) 변환
            similarity = max(0.0, 1.0 - (distance / 2.0))
            all_docs_with_scores.append((doc, similarity))
            
    return all_docs_with_scores

def filter_by_similarity(docs_with_scores, threshold=0.3):
    return [(doc, sim) for doc, sim in docs_with_scores if sim >= threshold]

def check_relevance(question, docs_with_scores):
    """LLM에게 상위 N개 문서로 관련성 물어보기 -> True/False 반환"""
    # docs_with_scores는 (doc, sim)
    top_docs = docs_with_scores[:5]
    if not top_docs:
        return False
    # Document 객체에서 직접 접근
    docs_text = "\n\n---\n\n".join(
        f"[문서 {i+1}] (출처: {getattr(doc.metadata,'get',lambda k, d=None: 'unknown')('source','unknown')})\n{getattr(doc,'page_content',str(doc))[:600]}"
        for i, (doc, _) in enumerate(top_docs)
    )
    try:
        res = relevance_chain.invoke({"question": question, "documents": docs_text})
        return "관련있음" in res
    except Exception as e:
        print(f"⚠ 관련성 검증 오류: {e}")
        return False

def web_search(query: str, k=3):
    """Tavily API 기반 웹검색
    외부 소스만 Document로 변환 필요"""
    try:
        retriever = TavilySearchAPIRetriever(k=k)
        results = retriever.invoke(query) 

        # Tavily 결과를 Document로 정규화
        docs = []
        for r in results:
            if isinstance(r, Document):
                docs.append(r)
            elif isinstance(r, dict):
                content = r.get("content") or r.get("snippet") or r.get("title") or str(r)
                docs.append(Document(
                    page_content=content,
                    metadata={"source": "web", "url": r.get("url", "unknown")}
                ))
            else:
                docs.append(Document(
                    page_content=str(r),
                    metadata={"source": "web"}
                ))
        return docs
        
    except Exception as e:
        raise RuntimeError(f"Tavily 웹검색 오류: {e}")

# -----------------------
# RAG 응답 생성 함수
# -----------------------
def rag_answer_from_docs(question: str, documents):
    """문서 리스트(Document 또는 (doc,score) 혼합)을 받아 RAG 응답 생성
    이미 Document 형식이므로 tuple만 풀어주면 됨"""
    # tuple 형식이면 Document만 추출
    docs = []
    for item in documents:
        if isinstance(item, tuple):
            docs.append(item[0])  # (doc, score) → doc
        else:
            docs.append(item)     # 이미 Document
    
    context = format_docs_as_context(docs)
    if not context.strip():
        print("⚠ context 비어있음 -> fallback LLM로 이동")
        return fallback_chain.invoke({"question": question})

    prompt, pname = choose_prompt(question)
    print(f"[프롬프트 사용] {pname} (문서 기반)")
    
    try:
        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question
        })
        return answer
    except Exception as e:
        print(f"⚠ RAG 생성 오류: {e}")
        return fallback_chain.invoke({"question": question})
    
# ========================================
# 메인 RAG 함수
# ========================================
def multi_query_rag_with_qt(question: str, top_k=10, similarity_threshold=0.3):
    """
    전체 흐름:
      1) 쿼리 트랜스폼 (QT)
      2) 멀티쿼리 생성 (MQ)
      3) 벡터검색 (멀티쿼리)
      4) 유사도 필터링
        - 문서 있음 -> LLM 관련성 검증
        - 문서 없음 / 관련 없음 -> 웹검색 또는 Fallback (단일 분기)
    """
    
    # 1) Query Transform
    try:
        qt_query = qt_chain.invoke({"question": question})
    except Exception as e:
        qt_query = question 
    print(f"[QT] 변환: {qt_query}")

    # 2) Multi Query
    try:
        mq_text = multi_query_chain.invoke({"question": qt_query})
        queries = [line.strip() for line in mq_text.splitlines() if line.strip()]
    except Exception as e:
        queries = [qt_query]
    print(f"[멀티쿼리] {len(queries)}개: {queries}")

    # 3) Vector search (멀티쿼리)
    all_docs = search_documents(queries)
    print(f"[검색] 총 {len(all_docs)}개 문서 후보 확보")

    # 4) 유사도 필터링
    filtered_docs = filter_by_similarity(all_docs, similarity_threshold)
    print(f"[1차 필터링] 유사도 >={similarity_threshold}: {len(filtered_docs)}개")

    # ----------------------------------------------------
    # 조건 검사 및 웹 검색 통합
    # ----------------------------------------------------
    
    is_relevant = False
    
    if filtered_docs:
        # 문서가 있는 경우에만 LLM으로 관련성 검사 (기존 로직 유지)
        print("[2차 필터링] LLM 관련성 검증 중...")
        is_relevant = check_relevance(question, filtered_docs)


    # 5) 최종 분기: 내부 RAG vs. 웹 검색/Fallback
    if is_relevant:
        #  내부 RAG 실행: 문서가 있고, 관련성도 있음
        print(" 내부 문서가 질문과 관련있음 → 내부 RAG 실행")
        useful = filtered_docs[:top_k]
        return rag_answer_from_docs(question, useful)
    else:
        print(" 내부 문서 (없거나/무관) → 웹검색으로 전환")
        
        try:
            # 웹 검색 시도
            web_docs = web_search(question)
        except Exception as e:
            # 웹 검색 실패 시 최종 Fallback
            return fallback_chain.invoke({"question": question})

        if not web_docs:
            print(" 웹검색 결과 없음 → LLM 자체지식으로 응답")
            # 웹 검색 결과가 없을 시 최종 Fallback
            return fallback_chain.invoke({"question": question})
        
        # 웹 검색 결과가 있을 시 웹 RAG 실행
        return rag_answer_from_docs(question, web_docs)

# -----------------------
# 실행 테스트
# -----------------------
if __name__ == "__main__":
    test_cases = ["헤어악세사리 사업했는데 판매가 매우 저조해. 어떻게 하면 될까?"
        # "서울에서 AI 챗봇 창업을 하려고 하는데 받을 수 있는 지원사업 있나요?",
        # "서울에 있는 동물병원 알려주세요",
        # "오늘 서울 날씨 알려줘"
    ]

    for q in test_cases:
        print("\n" + "="*80)
        print("[질문]", q)
        print("="*80)
        try:
            ans = multi_query_rag_with_qt(q)
            print("\n[답변]\n", ans)
        except Exception as e:
            print("전체 파이프라인 오류:", e)
