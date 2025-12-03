import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("ERROR: OPENAI_API_KEY가 .env에 없습니다.")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

PDF_PATH = "data/manual.pdf"
DB_DIR = "chroma_db"

def load_and_split_pdf():
    print("\n[1] PDF 로드 중...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    print(f"    - 총 {len(pages)} 페이지")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    print("[2] 청킹(Chunking) 중...")
    chunks = splitter.split_documents(pages)
    print(f"    - 분할된 청크 개수: {len(chunks)}")

    return chunks

def create_vector_db(chunks):
    print("\n[3] 벡터 DB 생성 중...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    vectordb.persist()
    print("    - Chroma DB 생성 완료")
    return vectordb


def load_vector_db():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    print("\n[0] 기존 Vector DB 로드 완료")
    return vectordb

def create_rag_chain(vectordb):

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
당신은 '스타트업 지식재산(IP) 관리 매뉴얼' 전문입니다.
아래 참고 문서 내용을 기반으로 질문에 답하세요.
관련 없는 부분은 제외하고, 관련 있는 내용만 그대로 출력하세요.
관련 내용이 없으면 "관련 없음"이라고 출력하세요.
문서에 없는 내용을 지어내지 말고, 반드시 실제 근거에 기반하여 답하십시오.

[참고 문서]
{context}

[사용자 질문]
{question}

위 정보를 바탕으로 한국어로 설명하세요.
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

def main():
    # Vector DB에 없으면 PDF 읽어서 만들기
    if not os.path.exists(DB_DIR):
        chunks = load_and_split_pdf()
        vectordb = create_vector_db(chunks)
    else:
        vectordb = load_vector_db()

    rag = create_rag_chain(vectordb)

    print("\n====== 스타트업 지식재산 RAG 챗봇 ======")
    print("PDF 기반으로 답변합니다. 종료하려면 quit 입력.\n")

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
