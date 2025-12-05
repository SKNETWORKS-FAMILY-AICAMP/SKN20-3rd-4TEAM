import os
import warnings
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

## 실행 전 반드시 main_chunking, build_vector_db.py를 실행하여 벡터 DB를 만들어 주세요!

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
# 리트리버 값을 5 -> 10으로 늘려 다양한 문서 가져오게 함 (필요시 줄여도 됨)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 기본 base RAG 프롬프트 (대부분 질문이 이쪽으로 감)
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
   “현재 조건에 정확히 부합하는 자금 지원 사업은 제한적입니다.”라고 말하세요.
5. IT·서비스업이면 반드시 '기술·콘텐츠·플랫폼' 키워드가 포함된 사업을 우선 고려하세요.

[출력 형식]
■ ✅ 추천 사업명
■ 🎯 추천 이유
■ 💰 지원 내용
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

# Query Transformation (최소 버전)
#  → 검색 정확도 올리기 위한 전처리 단계
qt_prompt = ChatPromptTemplate.from_template("""
다음 사용자 질문을 벡터 검색에 적합한 '핵심 키워드 중심 문장'으로 바꾸세요.
불필요한 말은 제거하고, 핵심 조건만 남기세요.

원본 질문: {question}

변환된 검색용 문장:
""")

qt_chain = qt_prompt | llm | StrOutputParser()

# 프롬프트 선택 함수
def choose_prompt(question: str):
    # 추천 관련 키워드
    recommend_keywords = ["추천", "맞는", "신청할 수 있는", "지원해주는", "사업 알려줘", "혜택", "지원금", "지원사업"]
    # 법령 관련 키워드
    law_keywords = ["정의", "자격", "요건", "지원법", "법에서", "법상", "제도"]

    if any(k in question for k in recommend_keywords):
        return recommend_prompt
    if any(k in question for k in law_keywords):
        return law_prompt
    return rag_prompt

# 문서 포맷 함수
def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

#테스트 (test_questions 안에 있는 질문만 수정)
if __name__ == "__main__":

    print("✅ rag(base).py 실행 시작")

    test_questions = [
        "AI 기술을 활용해서 창업을 준비중인데 정보를 얻을 수 있는 교육이 있을까요? 그리고 창업 준비를 할 수 있는 공간대여가 서비스를 이용할 수 있는지 궁금합니다."
        "AI 챗봇을 활용하여 창업을 준비 중인데, 지원 가능한 사업이 있는지 알려주세요. 그리고 창업 준비를 위한 절차를 알려주세요",
    ]

    for i, q in enumerate(test_questions, 1):

        print("=" * 80)
        print(f"[테스트 {i}] 질문: {q}")

        # Query Transformation 적용
        transformed_q = qt_chain.invoke({"question": q})
        print(f"[QT 변환된 검색용 질문] {transformed_q}")

        # 프롬프트 선택 (원본 질문 기준)
        prompt = choose_prompt(q)

        # 변환된 질문으로 문서 검색
        docs = retriever.invoke(transformed_q)
        context = format_docs(docs)

        # 선택된 프롬프트로 최종 답변 생성
        answer_chain = prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({
            "context": context,
            "question": q
        })

        print("\n[답변]")
        print(answer)

        # 디버그: 어떤 타입 문서를 참조했는지
        types = {d.metadata.get("data_type", "unknown") for d in docs}
        print(f"\n[참조된 data_type들] {types}")