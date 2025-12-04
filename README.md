![팀명 소개](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=창업%20IP%20전략%20가이드%20RAG%20챗봇&fontSize=60)

# 🚀 SKN20 3rd Project — 창업 IP 전략 가이드 RAG 챗봇  
**Team: SKN20-3rd-4TEAM (김태빈 · 정래원 · 최소영 · 최유정)**

스타트업·중소기업 창업자에게 필요한  
**지식재산(IP)·지원정책·창업지원·재창업 사례** 정보를  
PDF 기반 RAG(Retrieval-Augmented Generation) 기술로 제공하는  
지능형 정책 안내 챗봇 시스템입니다.

---

# 📌 1. 프로젝트 개요

이 프로젝트는 창업자와 중소기업이 실제 현장에서 자주 마주치는 다음 질문에 대한  
정확한 답변을 제공하기 위해 개발되었습니다:

- “국가에서 제공하는 창업 IP 지원 프로그램은 무엇인가?"
- “해외 특허 비용은 지원 받을 수 있나요?”
- “재창업·실패 사례를 참고하고 싶은데 어디서 찾을 수 있나요?”
- “중소기업창업지원법의 핵심 내용을 알고 싶다.”
- “지식재산 관리 매뉴얼을 따라 무엇을 해야 하지?”

이를 위해 총 **4개의 문서군(文書群)** 을 수집·정제 후  
각 문서별로 독립된 **RAG 엔진(VectorDB + LLM)** 을 구축하여  
정확도 높은 문서기반 지식검색 시스템을 설계했습니다.

---

## **1.1 개발 동기 및 목적**

최근 LLM이 급속도로 발전하면서, 사용자들은 자연어 기반의 간편한 정보 탐색을 기대합니다.  
그러나 LLM 단독 모델은 여전히 **환각(Hallucination)** 을 생성하여  
정책, 법령, 지식재산 등 **정확한 사실 기반 정보를 요구하는 분야에서 위험 요소**가 존재합니다.

창업자 및 중소기업은 다음과 같은 문제를 겪습니다:
- 정책·법령 문서가 방대하고 해석이 어려움  
- 필요한 지원제도·IP 전략을 정확히 찾기 어려움  
- 잘못된 정보 기반 의사결정은 재정·법률적 리스크로 이어짐  

따라서 본 프로젝트의 목적은  
**정확하고 신뢰할 수 있는 문서 기반 질의응답 시스템**을 구축하여  
창업자가 가장 빠르게 필요한 정책 및 지식재산 정보를 얻을 수 있도록 하는 것입니다.

---

## **1.2 필요성**

- **LLM 환각 최소화 필요**  
  공공문서·법령 기반 안내에서는 사실 정확도가 절대적으로 중요합니다.
- **정책 문서의 난해함**  
  100~300페이지 이상의 문서를 직접 해석하기 어려움.
- **창업자 지원간 정보 비대칭 해소**  
  지원사업·법령·IP 전략을 하나의 플랫폼에서 검색할 수 있는 도구의 부재.
- **문서 기반 RAG의 실사용성 검증**  
  공공 서비스 분야에서도 RAG 필요성이 증가하고 있어 그 실증적 가치가 큼.

---

## **1.3 개발 목표**

- PDF 문서를 기반으로 신뢰성 높은 RAG QA 시스템 구현  
- 문서별 독립 VectorDB 구축을 통한 정확한 검색 제공  
- 조건 기반 chunking + 섹션 태그로 검색 품질 극대화  
- 근거 기반 답변을 제공하여 환각 최소화  
- 향후 Streamlit UI로 확장 가능한 모듈형 구조 설계  

---

# 📌 2. 요구사항 정의 (Requirements)

---

## **2.1 데이터 수집 및 전처리**

- 원본 문서 수집 (PDF, TXT, JSON 등)
- PDF 텍스트 추출(pypdf, docling)
- 문단 기반 전처리 및 노이즈 제거
- 섹션 태그 자동 부착
- Chunking(chunk_size 400~700, overlap 50)
- 전처리 결과 저장: `.txt`, `.pkl`, `.json`

---

## **2.2 설계 및 구축 요구사항**

- 문서별 독립 RAG 엔진 구축  
- VectorDB(ChromaDB) 기반 탐색 시스템  
- GPT-4o-mini 기반 사실 기반 응답 생성  
- 검색된 문단(Context) 기반으로만 답변 생성  
- Streamlit UI 확장 가능 구조  

---

## **2.3 ETL 요구사항**

### **Extract**
- PDF/ZIP에서 텍스트 수집
- TXT/JSON Parsing

### **Transform**
- 텍스트 정제  
- 문단 단위 분리  
- 섹션 태그 추가  
- Chunking  
- Lemmatization(선택)  

### **Load**
- Embedding 생성(OpenAIEmbeddings)  
- 문서군별 VectorDB 저장  
- 검색 파이프라인 구성  

---

## **2.4 모델링 결과 및 시각화 요구사항**

- 문서별 RAG 챗봇 구현
- 근거 문단 + 응답 생성
- 검색 정확도 테스트
- chunk 분포도 시각화
- 임베딩 구조 2D/3D 차원축소 분석(선택)

---

# 📌 3. 수집데이터 목록 (Data Collection Summary)

## **3.1 원본 데이터 Raw Data**

| 파일명 | 설명 |
|--------|------|
| manual.pdf | 스타트업 지식재산 관리 매뉴얼 |
| ip_strategy_2025.pdf | 2025 지식재산 경영 전략 가이드 |
| 실패와 재창업 사례.pdf | 재창업 사례집(1) |
| 실패와 재창업 사례2.pdf | 재창업 사례집(2) |
| 지식재산관리매뉴얼txt.zip | 매뉴얼 텍스트 자료 |
| 스타트업지원프로그램txt.zip | 지원 프로그램 텍스트 |
| 지원사업,정보,창업공간.zip | 창업공간/정보 데이터 |
| 중소기업창업_지원법.txt | 법령 텍스트 |

---

## **3.2 전처리 데이터 Processed Data**

| 파일명 | 설명 |
|--------|------|
| manual_chunks/ | manual.pdf chunk |
| strategy_chunks/ | ip_strategy chunk |
| 지식재산관리매뉴얼txt/ | 전처리 매뉴얼 txt |
| 스타트업지원프로그램txt/ | 지원 프로그램 txt |
| chunked_documents.pkl | 통합 chunk 저장 |
| dataset.json | 지원사업·정보 dataset |

---

## **3.3 VectorDB 데이터**

- `chroma_db/`  
- `strategy_db/`  
- `supportlaw_vectordb/`  
- `supportprogram_vectordb/`  
- `failure_case_vectordb/`

---

## **3.4 처리 및 모델링 코드**

- preprocess_two_files.py  
- mydocling.py  
- supportlaw_vectordb.py  
- 지식재산관리매뉴얼.py  
- 스타트업지원프로그램.py  
- 지원사업_정보_창업공간.py  
- main.py  
- main_strategy.py  

---

# 🧠 시스템 특징 (Features)

### 📄 1. 독립 문서 기반 RAG 시스템
각 문서별로 **전용 vectorDB + 전용 챗봇 엔진**을 구성  
→ 문서별 응답 정확도 ↑  
→ 주제 혼합으로 인한 환각 ↓

### 🔎 2. 정책 기반 "근거 제시형" 답변
전체 답변은 반드시 **문서 내 실제 텍스트(Context)** 를 기반으로 함  
→ 허위 정보 생성 제로 목표

### ⚙️ 3. 전처리 자동화 파이프라인
- PDF → 문서 텍스트 추출
- 문단 기반 chunking (700 tokens)
- 섹션 태그 자동 추가

### 💾 4. 확장성 높은 VectorDB 구조
문서군별 VectorDB:
- chroma_db (manual)
- strategy_db (ip_strategy_2025)
- supportlaw_vectordb (중소기업창업지원법)
- supportprogram_vectordb (지원사업)
- failure_case_vectordb (실패 사례)

---

# 🧩 프로젝트 전체 폴더 구조

```txt
📁 SKN20-3rd-4TEAM/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── vectordb/
│
├── src/
│   ├── preprocess/
│   ├── rag/
│   └── ui/
│
├── images/
├── notebooks/
├── requirements.txt
├── .env
└── README.md
