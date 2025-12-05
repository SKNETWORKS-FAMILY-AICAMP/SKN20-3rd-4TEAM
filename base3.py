from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from typing import List, Any
import numpy as np
import pickle
import os
from konlpy.tag import Okt
from dotenv import load_dotenv

load_dotenv()
# ============================================
#   1) 임베딩 모델
# ============================================
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ============================================
#   2) chunked_documents 로드
# ============================================
with open("chunked_documents.pkl", "rb") as f:
    chunked_documents = pickle.load(f)

# metadata['id'] 부여
for idx, doc in enumerate(chunked_documents):
    if "id" not in doc.metadata:
        doc.metadata["id"] = f"doc_{idx}"
# ============================================
#   3) Chroma 벡터스토어 생성
# ============================================

# CHROMA_PATH = './chroma_db'
# COLLECTION_NAME = 'basic_rag_collection'

# 저장된 DB가 있는지 확인 후 로드, 없으면 새로 생성
# try:
#     vectorstore = Chroma(
#         persist_directory=CHROMA_PATH,
#         embedding_function=embedding_model,
#         collection_name=COLLECTION_NAME
#     )
    
#     # 로드 성공 후 문서 개수 확인. 비어있으면 새로 생성
#     if vectorstore._collection.count() == 0:
#         vectorstore = Chroma.from_documents(
#             documents=chunked_documents,
#             embedding=embedding_model,
#             collection_name=COLLECTION_NAME,
#             persist_directory=CHROMA_PATH
#         )
#         print(f" Chroma DB '{COLLECTION_NAME}'를 새로 생성했습니다.")
#     else:
#         print(f" Chroma DB '{COLLECTION_NAME}' ({vectorstore._collection.count()}개 문서)를 로드했습니다.")

# except Exception:
#     # 로드 실패 시 (DB 파일이 없는 경우) 새로 생성
#     vectorstore = Chroma.from_documents(
#         documents=chunked_documents,
#         embedding=embedding_model,
#         collection_name=COLLECTION_NAME,
#         persist_directory=CHROMA_PATH
#     )
#     print(f" Chroma DB '{COLLECTION_NAME}'를 새로 생성했습니다.")

vectorstore = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embedding_model,
    collection_name='basic_rag_collection',
    persist_directory='./chroma_db',
    #ids=doc_ids # 처음 한 번만 시행
)

# ============================================
#   4) Dense Retriever
# ============================================
dense_retriever = vectorstore.as_retriever(search_kwargs = {'k':3})

# ============================================
#   5) BM25 Sparse 구성
# ============================================

okt = Okt() # 형태소 분석기 객체 생성

def konlpy_tokenize(text):
    """
    KoNLPy를 사용하여 한국어에 최적화된 토큰화 (명사, 동사 어간 등 추출)
    """
    # KoNLPy의 pos(품사 태깅) 기능을 사용하여 키워드 추출
    tokens = [
        word for word, tag in okt.pos(text, norm=True, stem=True) 
        if tag in ['Noun', 'Verb', 'Adjective'] # 명사, 동사, 형용사 등 핵심 키워드만 사용
    ]
    # 필요하다면 추가적으로 okt.phrases(text)를 넣어 구문 정보를 추가할 수 있습니다.
    return tokens

# BM25 인덱스 생성 시 토큰화 함수 변경
tokenized_docs = [konlpy_tokenize(doc.page_content) for doc in chunked_documents]
bm25 = BM25Okapi(tokenized_docs)

# ============================================
#   6) HybridRetriever
# ============================================
class HybridRetriever(BaseRetriever):
    dense_retriever: Any # Dense Retriever 객체(벡터 유사도)
    documents: List[Document] # 전체 문서 리스트 (Sparse 검색용)
    bm25: BM25Okapi # BM25 인덱스 객체
    k: int = 3  #반환할 문서 개수
    dense_weight: float = 0.7    # Dense 가중치

    def __init__(self, dense_retriever, documents, bm25, k=3, dense_weight=0.7,**kwargs: Any):
        super().__init__(
                    dense_retriever=dense_retriever, 
                    documents=documents, 
                    bm25=bm25, 
                    k=k, 
                    dense_weight=dense_weight,
                    **kwargs)
        
    def _get_relevant_documents(self,query: str, *, run_manager: Any,) -> List[Document]:
        sparse_weight = 1 - self.dense_weight

        # Dense 검색
        dense_results = self.dense_retriever.invoke(query)

        # Sparse 검색
        tokenized_query = konlpy_tokenize(query) # 쿼리(질문) 토큰화
        scores = self.bm25.get_scores(tokenized_query) # BM25점수계산(키워드매칭)
        top_idx = np.argsort(scores)[::-1][:self.k] # 점수 높은 순으로 정렬해서 K개
        sparse_results = [self.documents[i] for i in top_idx]

        # ---- RRF 결합 ----
        merged = {}
        # Dense 결과 처리
        for rank, doc in enumerate(dense_results):
            doc_id = doc.metadata.get("id", id(doc))
            merged.setdefault(doc_id, {"doc": doc, "score": 0})
            merged[doc_id]["score"] += self.dense_weight + 1/(rank+1)
        # Sparse 결과 처리
        for rank, doc in enumerate(sparse_results):
            doc_id = doc.metadata.get("id", id(doc))
            merged.setdefault(doc_id, {"doc": doc, "score": 0})
            merged[doc_id]["score"] += sparse_weight + 1/(rank+1)

        # 점수 높은 순 반환
        return [v["doc"] for v in sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:self.k]]
        
    async def _aget_relevant_documents(self, query,*, run_manager: Any) -> List[Document] : # 비동기 메서드
        raise NotImplementedError
    
# 하이브리드 리트리버 생성

hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    documents=chunked_documents,
    bm25=bm25,
    k=3,
    dense_weight=0.7
)

# ============================================
#   7) LLM 모델
# ============================================
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# ============================================
#   8) 프롬프트
# ============================================

# 문서 포맷팅
def format_docs(docs):
    return "\n\n---\n\n".join([d.page_content for d in docs])

# 1) 다중 쿼리 생성용 프롬프트
multi_query_prompt = ChatPromptTemplate.from_template("""
다음 질문에 대해 서로 다른 관점의 검색 쿼리 3개를 생성하세요.
번호 없이, 각 쿼리는 한 줄씩 출력하세요.

원본 질문: {question}

다른 관점의 쿼리들:
""")

# RAG 최종 프롬프트
rag_prompt = PromptTemplate.from_template("""
아래 문서들을 참고하여 사용자의 질문에 답변하세요.
가능한 한 문서에 근거하여 간결하게 답변하세요.

[문서]
{context}

[질문]
{question}
""")

# **CRAG 추가 1: 문서 관련성 판단용 프롬프트**
# 검색된 context가 질문에 답변하기에 충분히 관련성이 있는지 확인
verification_prompt = ChatPromptTemplate.from_template("""
주어진 질문과 검색된 문서를 검토하세요.
검색된 문서들이 질문에 직접적이고 충분히 답변할 수 있다고 판단되면 'Relevant'를 출력하세요.
그렇지 않다면 'Irrelevant'를 출력하세요. 다른 어떤 설명도 붙이지 마세요.

[질문]
{question}

[문서]
{context}
""")

# LCEL MultiQuery 생성 체인
multi_query_chain = multi_query_prompt | llm | StrOutputParser()

# **CRAG 추가 2: 검증 체인**
verification_chain = verification_prompt | llm | StrOutputParser()

# 2) MultiQuery + HybridRetriever 조합 함수 -> CRAG 함수로 수정
def crag_rag(question): # 함수 이름 변경
    K_CONTEXT_FINAL = 5 # 최종 답변에 최대 5개의 문서를 사용
    
    # 1. 다중 쿼리 생성
    queries_text = multi_query_chain.invoke({"question": question})
    queries = [q.strip() for q in queries_text.split("\n") if q.strip()]

    print(f"생성된 3개의 검색 쿼리:\n{queries_text}")

    # 2. 각 쿼리로 HybridRetriever 검색 (기존 로직 유지)
    all_docs = []
    seen_contents = set()

    for q in queries:
        docs = hybrid_retriever.invoke(q)
        for d in docs:
            if d.page_content not in seen_contents:
                seen_contents.add(d.page_content)
                all_docs.append(d) 
    
    print(f"검색된 총 문서 개수: {len(all_docs)}")

    # 3. 유니크 문서들 중 최상위 K_CONTEXT_FINAL개만 선택 및 포맷팅
    context_docs = all_docs[:K_CONTEXT_FINAL]
    context = format_docs(context_docs)

    # ----------------------------------------
    # **4. CRAG: 검색 문서 관련성 검증**
    # ----------------------------------------
    
    # LLM이 'Relevant' 또는 'Irrelevant'를 판단
    verification_result = verification_chain.invoke({
        "question": question,
        "context": context
    }).strip().lower()
    
    print(f"**문서 관련성 검증 결과: {verification_result.capitalize()}**")

    # ----------------------------------------
    # **5. CRAG: 수정(Correction) 및 최종 답변**
    # ----------------------------------------
    if "irrelevant" in verification_result:
        # **관련성이 낮다고 판단되면, 답변을 생성하지 않고 사용자에게 알림**
        print("→ 관련성 낮음: CRAG 답변 생성 포기 (재검색 로직은 추가 구현 필요)")
        return f"검색된 문서가 질문에 충분히 관련되지 않아 답변을 생성할 수 없습니다. (검증 결과: {verification_result.capitalize()})"
    
    else: # 'relevant' 또는 그 외 (일단 답변 시도)
        # **관련성이 높다고 판단되면, 최종 답변 생성**
        answer_chain = rag_prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({
            "context": context,
            "question": question
        })
        return answer

# ============================================
#   8) 결과
# ============================================

# 테스트 질문 설정
test_question = "창업을 시작하려는 20대 청년이 지원할수 있는 지원 사업과 사용할 수 있는 공간, 그리고 카페 창업에 필요한 법적인 규제와 절차를 알려주세요"

# 함수 호출 및 결과 저장 (crag_rag으로 변경)
final_answer = crag_rag(test_question)

# 결과 출력
print("\n" + "="*50)
print(f"원본 질문: {test_question}")
print("="*50)
print(f"최종 답변:\n{final_answer}")
print("\n" + "-"*50)