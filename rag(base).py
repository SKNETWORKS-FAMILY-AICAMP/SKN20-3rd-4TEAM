import os
import warnings
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

## 실행 전 반드시 vector.py를 실행하여 벡터 DB를 만들어 주세요!

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
# 리브리버 값을 일단 5->10으로 늘려서 다양한 문서 가져오게 했습니다.(변경해도 됩니다.)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 기본 base Rag 프롬포트(대부분 질문이 일로감)
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
■ 🎯 왜 이 사용자에게 적합한지
■ 💰 지원 내용(자금/공간/교육 중 무엇인지 명확히)
■ 📝 신청 대상 요약
■ ⏳ 접수 기간
■ ⚠️ 주의사항
■ 🔗 추가 확인 필요 여부
'이 사업이 자금/공간/기술/교육 중 무엇을 중심으로 지원하는지도 한 줄로 정리해 주세요.'

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
    recommend_keywords = ["추천", "맞는", "신청할 수 있는", "지원해주는", "사업 알려줘"]
    law_keywords = ["정의", "자격", "요건", "지원법", "법에서", "법상", "제도"]

    if any(k in question for k in recommend_keywords):
        return recommend_prompt
    if any(k in question for k in law_keywords):
        return law_prompt
    return rag_prompt

# 문서 포맷 함수
def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

# 테스트 (test_qeuestions 안에 있는 질문만 수정하며)
if __name__ == "__main__":

    test_questions = [
        "제가 25살이고 창업을 하려는데 나라에서 받을 수 있는 혜택이 무엇이 있나요?",
    ]

    for i, q in enumerate(test_questions, 1):

        print("=" * 80)
        print(f"[테스트 {i}] 질문: {q}")

        prompt = choose_prompt(q)
        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(q)

        print("\n[답변]")
        print(answer)

        docs = retriever.invoke(q)
        types = {d.metadata.get("data_type", "unknown") for d in docs}
        print(f"\n[참조된 data_type들] {types}")
