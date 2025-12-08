import os
import warnings
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 없음! .env 확인해줘")

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

# Fallback 프롬프트 (LLM 자체 지식으로 답변)
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
    law_keywords = ["정의", "자격", "요건", "지원법", "법에서", "법상", "제도"]

    if any(k in question for k in recommend_keywords):
        return recommend_prompt, "recommend_prompt"
    if any(k in question for k in law_keywords):
        return law_prompt, "law_prompt"
    return rag_prompt, "rag_prompt"


def format_docs(docs):
    """문서 리스트를 문자열로 포맷팅"""
    return "\n\n---\n\n".join(d.page_content for d in docs)


def search_documents(queries):
    """멀티쿼리로 문서 검색 및 중복 제거"""
    all_docs_with_scores = []
    seen_contents = set()
    
    for query in queries:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=10)
        for doc, distance in docs_with_scores:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                similarity = max(0, 1 - (distance / 2))
                all_docs_with_scores.append((doc, similarity))
    
    return all_docs_with_scores


def filter_by_similarity(docs_with_scores, threshold):
    """유사도 기준으로 필터링"""
    return [(doc, sim) for doc, sim in docs_with_scores if sim >= threshold]


def check_relevance(question, docs_with_scores):
    """LLM으로 문서 관련성 검증"""
    # 상위 5개 문서로 검증
    top_docs = docs_with_scores[:5]
    docs_text = "\n\n---\n\n".join([
        f"[문서 {i+1}] (출처: {os.path.basename(doc.metadata.get('source', 'unknown'))})\n"
        f"{doc.page_content[:400]}"
        for i, (doc, _) in enumerate(top_docs)
    ])
    
    result = relevance_chain.invoke({
        "question": question,
        "documents": docs_text
    })
    
    return "관련있음" in result


# ========================================
# 메인 RAG 함수
# ========================================

def multi_query_rag_with_qt(question, top_k=10, similarity_threshold=0.3):
    """
    하이브리드 필터링 RAG:
    1단계: 유사도 필터링 (후보 확보)
    2단계: LLM 관련성 검증 (최종 판단)
    """
    
    # 1. Query Transformation
    qt_query = qt_chain.invoke({"question": question})
    print(f"[QT] 변환: {qt_query}")

    # 2. 멀티쿼리 생성
    queries_text = multi_query_chain.invoke({"question": qt_query})
    queries = [q.strip() for q in queries_text.strip().split("\n") if q.strip()]
    print(f"[멀티쿼리] {len(queries)}개: {queries}")

    # 3. 문서 검색
    all_docs = search_documents(queries)
    print(f"[검색] 총 {len(all_docs)}개 문서")

    # 4. 유사도 필터링
    filtered_docs = filter_by_similarity(all_docs, similarity_threshold)
    print(f"[1차 필터링] 유사도 >={similarity_threshold}: {len(filtered_docs)}개")
    
    if not filtered_docs:
        print("⚠️ 유사도 필터링 통과 문서 없음 → LLM 자체 지식으로 답변")
        answer = fallback_chain.invoke({"question": question})
        return answer, []
    
    # 유사도 TOP 5 출력
    #print("[유사도 TOP 5]")
    for i, (doc, sim) in enumerate(filtered_docs[:5], 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        #print(f"  {i}. {source}: {sim:.3f}")

    # 5. LLM 관련성 검증
    print("[2차 필터링] LLM 관련성 검증 중...")
    is_relevant = check_relevance(question, filtered_docs)
    
    if not is_relevant:
        print("❌ 문서가 질문과 관련없음 → LLM 자체 지식으로 답변")
        answer = fallback_chain.invoke({"question": question})
        return answer, []
    
    print("✅ 문서가 질문과 관련있음")

    # 6. 답변 생성
    useful_docs = [doc for doc, _ in filtered_docs[:top_k]]
    context = format_docs(useful_docs)
    
    prompt, prompt_name = choose_prompt(question)
    print(f"[프롬프트] {prompt_name}")
    
    answer_chain = prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({"context": context, "question": question})

    # 7. 참조 문서
    #sources = [os.path.basename(d.metadata.get("source", "unknown")) for d in useful_docs]
    
    return answer #, sources


# ========================================
# 테스트
# ========================================

if __name__ == "__main__":
    test_cases = [
        "서울에서 AI 챗봇 창업을 하려고 하는데 받을 수 있는 지원사업 있나요?",
        "서울에 있는 동물병원 알려주세요",
        "오늘 날씨 알려주세요",
        "창업 자금 지원 제도가 궁금합니다",
    ]
    
    for i, question in enumerate(test_cases, 1):
        print("\n" + "="*70)
        print(f"[테스트 {i}] {question}")
        print("="*70)
        
        #answer, sources = multi_query_rag_with_qt(question)
        answer = multi_query_rag_with_qt(question)
        
        print("\n" + "-"*70)
        print("[답변]")
        print("-"*70)
        print(answer)
        
        # print("\n[참조 문서]")
        # if sources:
        #     for j, source in enumerate(sources, 1):
        #         print(f"  {j}. {source}")
        # else:
        #     print("  없음 (LLM 직접 답변)")
        # print()