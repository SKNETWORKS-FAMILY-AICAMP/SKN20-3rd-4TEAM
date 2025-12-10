import os
import warnings
from dotenv import load_dotenv
from typing import List, Tuple
import hashlib
import time

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import TavilySearchAPIRetriever, BM25Retriever
from langchain_core.documents import Document

warnings.filterwarnings("ignore")
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 없음! .env 확인해줘")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") 

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터DB 로드
vectorstore = Chroma(
    persist_directory="./chroma_startup_all",
    collection_name="startup_all_rag",
    embedding_function=embedding_model,
)

cache_vectorstore = Chroma(
    persist_directory="./chroma_cache",
    collection_name="semantic_cache",
    embedding_function=embedding_model,
)

# 벡터스토어 상태 확인 함수
def check_vectorstore(name, store):
    try:
        data = store.get()
        ids = data.get("ids", [])
        print(f"✅ {name} 로드 완료 / 총 개수: {len(ids)}")
        return len(ids) >= 0
    except Exception as e:
        print(f"⚠ {name} 상태 확인 중 에러:", e)
        return False

main_db_ok = check_vectorstore("메인 벡터DB", vectorstore)
cache_db_ok = check_vectorstore("캐시 벡터DB", cache_vectorstore)

# ========================================
# 하이브리드 리트리버 구현
# ========================================

class HybridRetriever:
    """Dense(의미) + BM25(키워드) 하이브리드 리트리버"""
    
    def __init__(self,
        vectorstore: Chroma,
        documents: List[Document],
        k: int = 10,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4):
        
        self.vectorstore = vectorstore
        self.k = k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._internal_k = k * 2
        
        self.dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self._internal_k}
        )
        
        print(f"[하이브리드] BM25 인덱스 생성 중... (문서 수: {len(documents)})")
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = self._internal_k
        print(f"[하이브리드] BM25 인덱스 생성 완료")
    
    def invoke(self, query: str) -> List[Tuple[Document, float]]:
        dense_docs = self._search_dense(query)
        bm25_docs = self._search_bm25(query)
        fused_docs = self._fuse_results(dense_docs, bm25_docs)
        return fused_docs[:self.k]
    
    def _search_dense(self, query: str) -> List[Document]:
        try:
            return self.dense_retriever.invoke(query)
        except Exception as e:
            print(f"⚠ Dense 검색 오류: {e}")
            return []
    
    def _search_bm25(self, query: str) -> List[Document]:
        try:
            return self.bm25_retriever.invoke(query)
        except Exception as e:
            print(f"⚠ BM25 검색 오류: {e}")
            return []
    
    def _fuse_results(self, dense_docs: List[Document], bm25_docs: List[Document]) -> List[Tuple[Document, float]]:
        scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(dense_docs):
            key = self._get_doc_key(doc)
            scores[key] = scores.get(key, 0.0) + self.dense_weight / (rank + 1)
            doc_map[key] = doc
        
        for rank, doc in enumerate(bm25_docs):
            key = self._get_doc_key(doc)
            scores[key] = scores.get(key, 0.0) + self.bm25_weight / (rank + 1)
            doc_map[key] = doc
        
        ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [(doc_map[k], scores[k]) for k in ranked_keys]
    
    @staticmethod
    def _get_doc_key(doc: Document) -> str:
        content_hash = hashlib.md5(doc.page_content[:500].encode()).hexdigest()
        return content_hash

# 하이브리드 리트리버 초기화
print("\n[초기화] 하이브리드 리트리버 생성 중...")
hybrid_retriever = None

if main_db_ok:
    try:
        all_docs_data = vectorstore.get(include=['documents', 'metadatas'])
        all_documents = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(
                all_docs_data.get('documents', []),
                all_docs_data.get('metadatas', [])
            )
        ]
        
        if len(all_documents) > 0:
            hybrid_retriever = HybridRetriever(
                vectorstore=vectorstore,
                documents=all_documents,
                k=10,
                dense_weight=0.6,
                bm25_weight=0.4
            )
            print("[초기화] 하이브리드 리트리버 준비 완료!\n")
        else:
            print("⚠ 문서가 없어 하이브리드 리트리버를 생성할 수 없습니다.")
    except Exception as e:
        print(f"⚠ 하이브리드 리트리버 생성 실패: {e}")
        print("⚠ Dense 리트리버만 사용합니다.")

# ========================================
# 캐시 시스템
# ========================================

class SimpleCache:
    """L1 캐시 - 완전 일치 기반"""
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def get(self, key):
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            print(f"   [L1 캐시 히트]")
            return self.cache[key]
        return None

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            if self.access_count:
                least_used = min(self.access_count, key=self.access_count.get)
                del self.cache[least_used]
                del self.access_count[least_used]
        
        self.cache[key] = value
        self.access_count[key] = 0
        print(f"   [L1 저장 완료] 총 {len(self.cache)}개 캐시")

    def stats(self):
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_accesses": sum(self.access_count.values())
        }


class SemanticCache:
    """L2 캐시 - 의미적 유사도 기반"""
    
    def __init__(self, cache_vectorstore: Chroma, similarity_threshold=0.85):
        self.cache_vectorstore = cache_vectorstore
        self.similarity_threshold = similarity_threshold
        self.cache_data = {}
        self.cache_metadata = {}
        self._load_existing_cache()
    
    def _load_existing_cache(self):
        try:
            existing = self.cache_vectorstore.get(include=['metadatas'])
            for meta in existing.get('metadatas', []):
                cache_id = meta.get('cache_id')
                if cache_id:
                    self.cache_metadata[cache_id] = meta
            print(f"[L2 캐시] 기존 캐시 {len(self.cache_metadata)}개 복구")
        except Exception as e:
            print(f"⚠ 캐시 복구 중 오류: {e}")
    
    def get(self, query: str):
        try:
            similar_docs = self.cache_vectorstore.similarity_search_with_score(query, k=1)
            
            if similar_docs:
                doc, distance = similar_docs[0]
                similarity = max(0.0, 1.0 - (distance / 2.0))
                
                if similarity >= self.similarity_threshold:
                    cache_id = doc.metadata.get('cache_id')
                    
                    if cache_id in self.cache_data:
                        full_response = self.cache_data[cache_id]
                        
                        if cache_id in self.cache_metadata:
                            self.cache_metadata[cache_id]['hits'] = \
                                self.cache_metadata[cache_id].get('hits', 0) + 1
                        
                        print(f"  ✅ [L2 캐시 히트] 유사도: {similarity:.3f}")
                        return full_response
            
            return None
        except Exception as e:
            print(f"⚠ L2 캐시 검색 오류: {e}")
            return None
    
    def set(self, query: str, response: str):
        try:
            cache_id = f"cache_{len(self.cache_data)}_{int(time.time())}"
            
            self.cache_data[cache_id] = response
            
            metadata = {
                'cache_id': cache_id,
                'response_preview': response[:200] + "..." if len(response) > 200 else response,
                'timestamp': time.time(),
                'hits': 0
            }
            self.cache_metadata[cache_id] = metadata
            
            self.cache_vectorstore.add_texts(
                texts=[query],
                metadatas=[metadata],
                ids=[cache_id]
            )
            
            print(f"  💾 [L2 저장 완료] cache_id: {cache_id}")
        except Exception as e:
            print(f"⚠ L2 캐시 저장 오류: {e}")
    
    def stats(self):
        return {
            "size": len(self.cache_data),
            "total_hits": sum(m.get('hits', 0) for m in self.cache_metadata.values()),
        }


class MultiLevelCache:
    """멀티레벨 캐시"""
    
    def __init__(self, cache_vectorstore: Chroma):
        self.l1_cache = SimpleCache(max_size=100)
        self.l2_cache = SemanticCache(
            cache_vectorstore=cache_vectorstore,
            similarity_threshold=0.85
        )
        self.stats_data = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0
        }
    
    def get(self, key: str, execute_rag_callable):
        print(f"\n[캐시 조회] 질문: {key[:80]}...")
        
        # L1 확인
        cached = self.l1_cache.get(key)
        if cached:
            self.stats_data['l1_hits'] += 1
            return cached
    
        # L2 확인
        cached = self.l2_cache.get(key)
        if cached:
            self.stats_data['l2_hits'] += 1
            self.l1_cache.set(key, cached)
            return cached

        # 캐시 미스
        self.stats_data['misses'] += 1
        print('❌ [캐시 미스] RAG 실행\n')
        
        response = execute_rag_callable(key)
        
        # 캐시 저장
        print(f"\n [캐시 저장 시작] 응답 길이: {len(response)}자")
        self.l1_cache.set(key, response)
        self.l2_cache.set(key, response)
        print(" [캐시 저장 완료]\n")

        return response
        
    def stats(self):
        total = sum(self.stats_data.values())
        hit_rate = (self.stats_data['l1_hits'] + self.stats_data['l2_hits']) / max(total, 1) * 100
        
        print("\n" + "="*60)
        print(" 캐시 통계")
        print("="*60)
        print(f"L1 히트: {self.stats_data['l1_hits']}")
        print(f"L2 히트: {self.stats_data['l2_hits']}")
        print(f"캐시 미스: {self.stats_data['misses']}")
        print(f"총 요청: {total}")
        print(f"캐시 히트율: {hit_rate:.1f}%")
        print(f"\nL1 상세: {self.l1_cache.stats()}")
        print(f"L2 상세: {self.l2_cache.stats()}")
        print("="*60 + "\n")

# ========================================
# 프롬프트 정의
# ========================================

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

다음 중 하나로만 답변: "관련있음" 또는 "관련없음"

답변:""")

qt_prompt = ChatPromptTemplate.from_template("""
다음 사용자 질문을 벡터 검색에 적합한 '핵심 키워드 중심 문장'으로 바꾸세요.
불필요한 말은 제거하고, 핵심 조건만 남기세요.

원본 질문: {question}

변환된 검색용 문장:""")

multi_query_prompt = ChatPromptTemplate.from_template("""
다음 질문에 대해 3가지 다른 관점의 검색 쿼리를 생성하세요.
각 쿼리는 한 줄씩 구분하여 출력하세요.
번호나 설명 없이 쿼리만 출력하세요.

원본 질문: {question}""")

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 예비·초기 창업자를 도와주는 '창업 지원 통합 AI 어시스턴트'입니다.

[답변 원칙]
1. 반드시 제공된 문맥(Context) 안의 정보만 사용하세요.
2. 문맥에 없는 내용은 추측하지 말고 솔직하게 말하세요.
3. 핵심 답변 후 필요하면 bullet로 정리하세요.
"""),
    ("human", "[문맥]\n{context}\n\n[질문]\n{question}\n\n[답변]")
])

recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 예비·초기 창업자에게 가장 적합한 '지원사업을 추천하는 전문가 AI'입니다.

[추천 규칙]
1. 반드시 announcement 문서만 사용
2. 사용자 조건과 '지역·연령·단계·업종'이 명확히 맞는 것만 추천
3. 최대 2개까지만 추천
"""),
    ("human", "[지원사업 문맥]\n{context}\n\n[사용자 조건]\n{question}\n\n추천해 주세요.")
])

law_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 중소기업창업 지원법을 바탕으로 창업 제도와 요건을 설명하는 AI입니다.

[규칙]
1. 반드시 문맥에 있는 법령 내용만 사용하세요.
2. 가능하면 조문 번호(제○조)를 함께 제시하세요.
"""),
    ("human", "[법령 문맥]\n{context}\n\n[질문]\n{question}\n\n[설명]")
])

fallback_prompt = ChatPromptTemplate.from_template("""
질문: {question}

내부 문서에서 관련 정보를 찾지 못했습니다.
일반적인 지식을 바탕으로 답변해주세요.

답변:""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qt_chain = qt_prompt | llm | StrOutputParser()
multi_query_chain = multi_query_prompt | llm | StrOutputParser()
relevance_chain = relevance_check_prompt | llm | StrOutputParser()
fallback_chain = fallback_prompt | llm | StrOutputParser()

# ========================================
# 헬퍼 함수
# ========================================

def choose_prompt(question: str):
    recommend_keywords = ["추천", "맞는", "신청할 수 있는", "지원해주는", 
                         "사업 알려줘", "혜택", "지원금", "지원사업"]
    law_keywords = ["정의", "자격", "요건", "지원법", "법에서", "법상", "제도", "시행", "규정"]
    
    if any(k in question for k in recommend_keywords):
        return recommend_prompt, "recommend_prompt"
    if any(k in question for k in law_keywords):
        return law_prompt, "law_prompt"
    return rag_prompt, "rag_prompt"

def format_docs_as_context(docs):
    if not docs:
        return ""
    parts = []
    for i, d in enumerate(docs, 1):
        if isinstance(d, Document):
            src = d.metadata.get("source", d.metadata.get("url", "unknown"))
            parts.append(f"[문서 {i}] (출처: {src})\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def search_documents_hybrid(queries: List[str], k_per_query=10) -> List[Tuple[Document, float]]:
    all_docs_with_scores = []
    seen_hashes = set()
    
    if hybrid_retriever is None:
        print("[검색] 하이브리드 리트리버 없음, Dense만 사용")
        for q in queries:
            try:
                docs = vectorstore.similarity_search_with_score(q, k=k_per_query)
                for doc, distance in docs:
                    doc_hash = hashlib.md5(doc.page_content[:500].encode()).hexdigest()
                    if doc_hash in seen_hashes:
                        continue
                    seen_hashes.add(doc_hash)
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                    all_docs_with_scores.append((doc, similarity))
            except Exception as e:
                print(f"⚠ 검색 오류: {e}")
        return all_docs_with_scores
    
    print(f"[하이브리드 검색] {len(queries)}개 쿼리로 검색 중...")
    for q in queries:
        try:
            docs_with_scores = hybrid_retriever.invoke(q)
            
            for doc, score in docs_with_scores:
                doc_hash = hashlib.md5(doc.page_content[:500].encode()).hexdigest()
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)
                all_docs_with_scores.append((doc, score))
        except Exception as e:
            print(f"⚠ 하이브리드 검색 오류: {e}")
    
    all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    return all_docs_with_scores

def check_relevance(question, docs_with_scores):
    top_docs = docs_with_scores[:5]
    if not top_docs:
        return False
    
    docs_text = "\n\n---\n\n".join(
        f"[문서 {i+1}]\n{doc.page_content[:600]}"
        for i, (doc, _) in enumerate(top_docs)
    )
    
    try:
        res = relevance_chain.invoke({"question": question, "documents": docs_text})
        return "관련있음" in res
    except Exception as e:
        print(f"⚠ 관련성 검증 오류: {e}")
        return False

def web_search(query: str, k=3):
    try:
        retriever = TavilySearchAPIRetriever(k=k)
        results = retriever.invoke(query)

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

def rag_answer_from_docs(question: str, documents):
    docs = []
    for item in documents:
        if isinstance(item, tuple):
            docs.append(item[0])
        else:
            docs.append(item)
    
    context = format_docs_as_context(docs)
    if not context.strip():
        print("⚠ context 비어있음 -> fallback")
        return fallback_chain.invoke({"question": question})

    prompt, pname = choose_prompt(question)
    print(f"[프롬프트 사용] {pname}")

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

if cache_db_ok:
    multi_level_cache = MultiLevelCache(cache_vectorstore=cache_vectorstore)
    print(" 멀티레벨 캐시 초기화 완료\n")
else:
    multi_level_cache = None
    print("⚠ 캐시 DB 없음 - 캐시 비활성화\n")

def multi_query_rag_with_cache(question: str, top_k=10):
    
    def execute_rag(question):
        print("\n" + "="*80)
        print(" RAG 파이프라인 시작")
        print("="*80)
        
        # 1) Query Transform
        try:
            qt_query = qt_chain.invoke({"question": question})
            print(f"[1단계] QT: {qt_query}")
        except Exception as e:
            print(f"⚠ QT 실패: {e}")
            qt_query = question

        # 2) Multi Query
        try:
            mq_text = multi_query_chain.invoke({"question": qt_query})
            queries = [line.strip() for line in mq_text.splitlines() if line.strip()]
            print(f"[2단계] 멀티쿼리 {len(queries)}개 생성")
        except Exception as e:
            print(f"⚠ 멀티쿼리 실패: {e}")
            queries = [qt_query]

        # 3) 하이브리드 검색
        all_docs_with_scores = search_documents_hybrid(queries, k_per_query=10)
        top_docs = all_docs_with_scores[:top_k]

        # 4) 내부 RAG / 웹 RAG 분기
        if top_docs:
            print(f"[3단계] {len(top_docs)}개 문서 확보")
            is_relevant = check_relevance(question, top_docs)
            
            if is_relevant:
                print(" [4단계] 내부 RAG 실행")
                return rag_answer_from_docs(question, top_docs)
            else:
                print("⚠ [4단계] 관련성 낮음 → 웹검색")
        else:
            print("⚠ [3단계] 문서 없음 → 웹검색")

        # 5) 웹 검색
        try:
            web_docs = web_search(question)
            if web_docs:
                print(f" [5단계] 웹 RAG 실행")
                return rag_answer_from_docs(question, web_docs)
            else:
                print("⚠ [5단계] 웹 결과 없음 → Fallback")
                return fallback_chain.invoke({"question": question})
        except Exception as e:
            print(f"⚠ 웹검색 실패: {e}")
            return fallback_chain.invoke({"question": question})

    if multi_level_cache:
        return multi_level_cache.get(question, execute_rag)
    else:
        print("⚠ 캐시 없음 → 직접 RAG 실행")
        return execute_rag(question)

# ========================================
# 테스트 실행
# ========================================

if __name__ == "__main__":
    test_cases = [
        "서울에서 AI 챗봇 창업을 하려고 하는데 받을 수 있는 지원사업 있나요?",
        "서울에서 AI 챗봇 창업을 하려고 하는데 받을 수 있는 지원사업 있나요?",  # 동일 질문
        "헤어악세사리 사업했는데 판매가 매우 저조해. 어떻게 하면 될까?",
    ]
    
    for i, q in enumerate(test_cases, 1):
        print("\n" + "-"*40)
        print(f"테스트 {i}/{len(test_cases)}")
        print("-"*40)
        print(f"질문: {q}")
        
        try:
            ans = multi_query_rag_with_cache(q)
            
            print("\n" + "-"*40)
            print("최종 답변")
            print("-"*40)
            print(ans)
            print("\n")
        except Exception as e:
            print(f" 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    if multi_level_cache:
        multi_level_cache.stats()
    
    print("\n✅ 테스트 완료!")