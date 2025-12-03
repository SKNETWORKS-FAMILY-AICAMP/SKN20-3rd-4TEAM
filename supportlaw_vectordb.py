from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

LAW_FILE = "중소기업창업_지원법.txt"

# txt로더입니다.
loader = TextLoader(LAW_FILE, encoding="utf-8")
documents = loader.load()
print(f"원본 문서 수: {len(documents)}")

# 텍스트청킹입니다.(조절 할 수 있음)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", "제", ".", " "]
)

doc_splits = text_splitter.split_documents(documents)
print(f"청킹된 문서 수: {len(doc_splits)}")

# 벡터화입니다.(임베딩)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Chroma벡터DB 저장하기입니다.
persist_dir = "./chroma_startup_law"

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embedding_model,
    collection_name="startup_law_rag",
    persist_directory=persist_dir
)

print("생성완료 디버깅")
