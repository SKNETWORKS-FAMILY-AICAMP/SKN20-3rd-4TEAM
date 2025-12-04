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
import pickle

with open("chunked_documents.pkl", "rb") as f:
    final_docs = pickle.load(f)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=final_docs,
    embedding=embedding_model,
    collection_name="startup_support_rag",
    persist_directory="./chroma_startup_support",
)
