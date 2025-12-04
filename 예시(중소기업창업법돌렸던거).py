import os
import time
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 벡터DB 로드 
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="./chroma_startup_law",
    collection_name="startup_law_rag",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# 문서 포맷터 
def format_docs_with_source(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(
            f"문서 {i} (source: {os.path.basename(source)}):\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


# RAG 프롬프트 
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 '중소기업창업 지원법'과 관련 규정을 바탕으로
창업 준비 중인 사람들에게 정보를 제공하는 AI입니다.

[역할]
- 제공된 문맥(Context)에 포함된 조문 내용을 우선적으로 인용·설명합니다.
- 문맥에 직접적인 표현이 없더라도, 조문의 취지와 구조를 근거로
  합리적인 범위 내에서 추론하여 설명할 수 있습니다.

[답변 규칙]
1. 가능한 경우, 관련 조문(조·항·호 등)을 함께 언급하며 설명하세요.
2. 문맥에 명시된 내용을 그대로 옮기기보다는,
   사용자가 이해하기 쉽게 풀어서 설명하세요.
3. 문맥에 직접적으로 규정되어 있지 않은 부분을 추론할 때에는,
   예를 들어 다음과 같이 단서를 달아주세요.
   - "법령에 직접적으로 명시되어 있지는 않지만, ○○ 조문의 취지에 비추어 보면 …로 해석될 수 있습니다."
   - "정확한 적용 여부는 개별 사업 공고나 담당 기관의 안내를 통해 확인이 필요합니다."
4. 문맥과 전혀 관련 없는 내용을 만들어내지는 마세요.
   반드시 제공된 문맥과 논리적으로 연결되는 범위 안에서만 추론하세요.
5. 답변은 한국어로, 가능한 한 구체적이고 이해하기 쉽게 작성하세요.
6. 이 답변은 법률 자문이 아닌 일반적인 정보 제공이라는 점을
   답변 마지막에 한 줄로 명시하세요.
   예: "※ 본 답변은 일반적인 정보 제공으로, 구체적인 사안에 대한 법률 자문은 아닙니다."
""",
        ),
        (
            "human",
            """[문맥]
{context}

[질문]
{question}

[답변]""",
        ),
    ]
)

# 5. Query Transformation (검색용 질문 변환)
query_transform_prompt = ChatPromptTemplate.from_template(
    """
다음 질문을 '중소기업창업 지원법' 검색에 가장 적합한 핵심 키워드 형태로 변환하세요.
불필요한 말은 제거하고,
법령·지원·대상·요건 중심 키워드로 만들어 주세요.
한 줄로만 출력하세요.

원본 질문: {question}
변환된 검색어:
"""
)

query_transform_chain = query_transform_prompt | llm | StrOutputParser()

# 6. Multi-Query (다른 관점 검색 질문 3개 생성)
multi_query_prompt = ChatPromptTemplate.from_template(
    """
다음 질문을 기준으로
서로 다른 관점의 '법령 검색용 질문' 3개를 생성하세요.

- 각 줄마다 하나씩 출력
- 번호, 설명 없이 질문만 출력
- 모두 중소기업창업 지원법 검색에 적합해야 함

원본 질문: {question}

다중 검색 질문들:
"""
)

multi_query_chain = multi_query_prompt | llm | StrOutputParser()

# 고급 RAG: Query Transform + Multi-Query + RAG
def answer_question(question: str):
    # 1) Query Transformation
    print("\n[1] Query Transformation 진행 중...")
    transformed_query = query_transform_chain.invoke({"question": question})
    print(f"검색용 변환 질문: {transformed_query}")

    # 2) Multi-Query 생성
    print("\n[2] Multi-Query 생성 중...")
    multi_queries_text = multi_query_chain.invoke(
        {"question": transformed_query}
    )
    multi_queries = [
        q.strip() for q in multi_queries_text.split("\n") if q.strip()
    ]

    print("다중 검색 질문:")
    for q in multi_queries:
        print(" -", q)

    # 각 쿼리로 검색, 문서 통합
    all_docs = []
    seen_contents = set()

    for mq in multi_queries:
        docs = retriever.invoke(mq)
        for d in docs:
            if d.page_content not in seen_contents:
                seen_contents.add(d.page_content)
                all_docs.append(d)

    print(f"\n[3] 통합된 검색 문서 수: {len(all_docs)}")

    # 상위 몇 개만 사용
    context = format_docs_with_source(all_docs[:4])

    # 프롬프트에 직접 context 주입해서 답변 생성
    start = time.time()

    local_rag_chain = rag_prompt | llm | StrOutputParser()
    answer = local_rag_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    elapsed = time.time() - start

    sources = [
        os.path.basename(d.metadata.get("source", "unknown"))
        for d in all_docs
    ]

    return answer, sources, elapsed


# 챗봇
if __name__ == "__main__":
    print("\n=== 중소기업창업 지원법 RAG 챗봇 (고급 버전) ===")
    print("질문을 입력하세요. (종료: exit)\n")

    while True:
        q = input("질문: ").strip()
        if not q:
            continue
        if q.lower() in ["exit", "quit", "종료"]:
            print("챗봇을 종료합니다.")
            break

        answer, sources, t = answer_question(q)

        print("\n[답변]")
        print(answer)
        print("\n[참조 문서]")
        print(sources)
        print(f"[소요 시간] {t:.2f}초")
        print("=" * 80)
