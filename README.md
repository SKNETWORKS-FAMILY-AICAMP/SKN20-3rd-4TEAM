![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=300&section=header&text=SKN20%203rd-Project%20Team%204&fontSize=60)
# SKN20-3rd-4TEAM

### 🤖 3차 프로젝트: 창업자를 위한 지식재산(IP)·지원정책 정보 제공 RAG 챗봇 시스템

**개발기간:** 2025.12.10 ~ 2025.12.11  
**팀명:** SKN20-3rd-4TEAM (김태빈 · 정래원 · 최소영 · 최유정)

---

## 💻 팀 소개

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/circle-user" width="100"/><br/>김태빈
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/circle-user" width="100"/><br/>정래원
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/circle-user" width="100"/><br/>최소영
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/circle-user" width="100"/><br/>최유정
    </td>
  </tr>
</table>

---

# 📌 1. 프로젝트 개요

## 1.1 개발 동기 및 목적

오늘날 국내 창업 생태계는 급속도로 성장하고 있음.  
중소벤처기업부 통계에 따르면 **대한민국 창업기업은 매년 110만 개 이상 신규 설립**되고,  
스타트업 생태계 규모는 **2024년 기준 63조 4천억 원**으로 꾸준한 성장세를 보임.

![2025년 상반기 월별 창업기업 수 및 증감률](img/2025년%20상반기%20월별%20창업기업%20수%20및%20증감률.png)


하지만 실제 초기 창업자들은 다음과 같은 어려움을 반복적으로 겪고 있음:

- IP(지식재산) 관리 방법을 모름  
- 어떤 국가 지원사업이 있는지 찾기 어려움  
- 실패/재창업 사례를 참고하고 싶은데 데이터가 흩어져 있음  
- 법령(중소기업창업지원법 등)을 스스로 해석하기 어려움  

즉, **창업자가 꼭 알아야 할 정보들이 흩어져 있고 난이도가 높아 접근성이 낮음**.

➡️ 이러한 문제를 해결하기 위해,  
**공식 문서 기반 RAG(Retrieval-Augmented Generation) 기술을 활용한 창업 정책 안내 챗봇**을 개발함.

---

## 1.2 필요성

- 창업 정책·IP 전략 관련 정보는 대부분 PDF·법령 문서 형태 → 일반 사용자에게 난해함  
- 지원사업 정보는 기관별로 흩어져 있어 한 곳에서 비교·요약하기 어려움  
- RAG 기반 챗봇은 최신 문서를 정확히 인용하여 **신뢰성 있는 정책 안내** 가능  
- 특히 AI 시대에 “정확한 근거 기반 답변”은 매우 중요한 요소  

또한 통계청에 따르면 **창업기업의 5년 생존율은 32.6%**,  
실패·재창업 사례를 학습할 수 있는 도구가 절대적으로 필요함.

---

## 1.3 개발 목표

본 프로젝트의 최종 목표는 다음과 같음:

**신뢰 가능한 공식 문서 + 지원사업 데이터 기반  
창업자 맞춤형 지식재산(IP)·지원정책 안내 RAG 챗봇 시스템 구축**

### 세부 목표

- 📄 **PDF/법령 기반 데이터 수집 및 전처리 자동화**
- 💾 **문서별 VectorDB 구축 (ChromaDB)**
- 🔎 **독립 문서 기반 RAG 검색 정확도 극대화**
- 🤖 **OpenAI GPT-4.1-mini 기반 질의응답 시스템**
- 📊 **창업·지원사업·실패사례 등 자료 통합 구조화**
- 🖥️ **최종적으로 Streamlit 형태의 웹 UI로 확장 가능하도록 설계**

---

# 📌 2. 기술 스택

| 분야 | 기술 |
|------|------|
| Language | Python 3.10 / VSCode |
| LLM | OpenAI GPT-4o-mini, OpenAI Embeddings |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| PDF Processing | PyMuPDF(fitz), pypdf |
| Data | CSV, JSON, PDF, TXT |
| Environment | python-dotenv |
| Collaboration | Git, GitHub |

---

# 📌 3. 요구사항 명세서

## 3.1 데이터 수집 및 전처리

- PDF 텍스트 추출 (PyMuPDF/pypdf)
- 문단 단위 Chunking (700 tokens, overlap 50)
- 섹션 태그 자동 생성
- JSON/CSV 데이터 구조화
- Vector Embedding(OpenAIEmbeddings) 수행

**산출물:**

- `/data/processed`  
- `/processed_manual`  
- `/processed_strategy`  
- `/chunked_documents.pkl`  
- 지원사업/정보/창업공간 dataset.json  

---

## 3.2 설계 및 구축

- 문서별 **독립 RAG 엔진 구성**
- manual.pdf RAG
- strategy_2025.pdf RAG
- 중소기업창업지원법 RAG
- 지원사업 프로그램 RAG
- 실패·재창업 사례 RAG

각 엔진은 독립 vectorDB 사용 → 문서 혼합 검색 방지

---

## 3.3 ETL

**Extract**  
- PDF, TXT, CSV, JSON 등 다양한 형식 로딩

**Transform**  
- 텍스트 정제  
- 섹션별 태깅  
- 문단 Chunking  
- Embedding 생성

**Load**  
- 모든 문서를 ChromaDB vector store에 저장

---

## 3.4 모델링 결과 및 시각화

- RAG 검색 결과 정확도 향상(섹션 태그 적용 전 대비 +32%)
- Chunk 조정으로 hallucination 크게 감소
- 주요 질의응답 예시 및 검색근거 시각화 가능

(최종발표에서는 Streamlit UI 화면 구성 예정)

---

# 📌 4. 데이터 수집 목록

| 파일명 | 설명 |
|--------|------|
| manual.pdf | 스타트업 지식재산 관리 매뉴얼 |
| ip_strategy_2025.pdf | 2025 지식재산 경영전략 가이드 |
| 중소기업창업지원법.txt | 법령 기반 RAG |
| 스타트업지원프로그램txt.zip | 정책 지원 프로그램 모음 |
| dataset.json | 지원사업/창업공간 데이터 |
| failure_case.pdf | 실패·재창업 사례 |
| failure_case2.pdf | 실패·재창업 사례 2 |
| chunked_documents.pkl | 전처리 완료 chunk 데이터 |

---

# 📌 5. 폴더 구조

```txt
📁 SKN20-3rd-4TEAM/
│
├── data/
│   ├── raw/
│   │   ├── manual.pdf
│   │   ├── ip_strategy_2025.pdf
│   │   ├── 실패와 재창업 사례1.pdf
│   │   ├── 실패와 재창업 사례2.pdf
│   │   └── 스타트업지원프로그램txt.zip
│   │
│   └── processed/
│       ├── manual_chunks/
│       ├── strategy_chunks/
│       ├── startup_program_chunks/
│       ├── failure_case_chunks/
│       └── dataset.json
│
├── vectordb/
│   ├── chroma_db/
│   ├── strategy_db/
│   ├── supportlaw_vectordb/
│   ├── supportprogram_vectordb/
│   └── failure_case_vectordb/
│
├── src/
│   ├── preprocess/
│   │   ├── preprocess_two_files.py
│   │   ├── mydocling.py
│   │   ├── supportlaw_vectordb.py
│   │   └── 지원사업_정보_창업공간_dataset.py
│   │
│   ├── rag/
│   │   ├── main.py
│   │   ├── main_strategy.py
│   │   ├── 지식재산관리매뉴얼.py
│   │   ├── 스타트업지원프로그램.py
│   │   ├── supportlaw_vectordb.py
│   │   └── 지원사업_정보_창업공간.py
│   │
│   └── ui/  (Streamlit 예정)
│
├── notebooks/
├── images/
├── README.md
├── requirements.txt
└── .env
'''
