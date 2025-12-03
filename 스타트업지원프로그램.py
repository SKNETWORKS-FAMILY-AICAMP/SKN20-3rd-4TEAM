import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 경로 설정
CHUNK_DIR = "processed_strategy"
DB_DIR = "chroma_strategy_db"


# ==============================
# Chunk 파일을 읽어 리스트로 로드
# ==============================
def load_chunks_from_folder(folder_path):
    chunks = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                chunks.append(f.read())
    print(f"[INFO] {len(chunks)}개의 chunk 로드 완료")
    return chunks


# ==============================
# VectorDB 생성
# ==============================
def build_vector_db(chunks):
    print("[INFO] 전략 문서 전용 VectorDB 생성 중...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    vectordb.persist()
    print("[INFO] VectorDB 저장 완료 →", DB_DIR)
    return vectordb


# ==============================
# RAG 체인 생성
# ==============================
def create_rag_chain(vectordb):

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
당신은 '2025 지식재산경영 전략 가이드' 전문입니다.
주어진 문서 내용만 기반으로 답변하세요.
문서에 없는 내용은 절대로 지어내지 마세요.

[참고 문서]
{context}

[사용자 질문]
{question}

위 내용을 바탕으로 한국어로 정확하게 답변하십시오.
""")

    def format_docs(docs):
        return "\n\n---\n\n".join([d.page_content for d in docs])

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough()
        )
        | prompt
        | llm
    )

    return rag_chain


# ==============================
# 메인 실행
# ==============================
def main():

    # 1) chunk 로드
    chunks = load_chunks_from_folder(CHUNK_DIR)

    # 2) DB 생성 또는 로드
    if not os.path.exists(DB_DIR):
        vectordb = build_vector_db(chunks)
    else:
        print("[INFO] 기존 VectorDB 로드")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=DB_DIR
        )

    rag = create_rag_chain(vectordb)

    print("\n====== 2025 지식재산경영 전략 RAG 챗봇 ======")
    print("문서를 기반으로 답변합니다. 종료하려면 quit 입력.\n")

    while True:
        q = input("질문 > ").strip()
        if q.lower() == "quit":
            break

        res = rag.invoke(q)
        print("\n[답변]\n")
        print(res.content)
        print("\n--------------------------------------\n")


if __name__ == "__main__":
    main()
