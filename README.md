![팀명 소개](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=창업%20IP%20전략%20가이드%20RAG%20챗봇&fontSize=60)

# 🚀 SKN20 3rd Project — 창업 IP 전략 가이드 RAG 챗봇  
**Team: SKN20-3rd-4TEAM (김태빈 · 정래원 · 최소영 · 최유정)**

스타트업·중소기업 창업자에게 필요한  
**지식재산(IP)·지원정책·창업지원·재창업 사례** 정보를  
PDF 기반 RAG(Retrieval-Augmented Generation) 기술로 제공하는  
지능형 정책 안내 챗봇 시스템입니다.

---

# 📌 프로젝트 개요

이 프로젝트는 창업자와 중소기업이 실제 현장에서 자주 마주하는 다음 질문에 답하는 것을 목표로 합니다:

- “국가에서 제공하는 창업 IP 지원 프로그램이 뭐가 있지?”
- “해외 특허 비용은 지원 받을 수 있나요?”
- “실패·재창업 사례는 어디서 볼 수 있나요?”
- “중소기업창업지원법의 요약을 알려줘.”
- “지식재산 관리 매뉴얼을 따라 무엇을 해야 하지?”

이를 위해 **총 4개의 문서군(文書群)** 을 수집 및 구조화하였고,  
각 문서를 **독립 RAG 엔진**으로 구축했습니다.

---

# 📚 사용된 문서(데이터셋)

## 1) 공식 정책 문서
- 「스타트업을 위한 지식재산 관리 매뉴얼」
- 「2025 지식재산경영 전략 가이드」
- 「중소기업창업지원법」 (법령 요약/텍스트)

## 2) 창업 지원 프로그램
- 각종 정부·지자체 지원사업 텍스트 파일
- 창업공간/지원정보 데이터셋(dataset.json)

## 3) 실패와 재창업 사례
- 실패와 재창업 사례.pdf  
- 실패와 재창업 사례2.pdf  
→ PDF를 텍스트화 + chunk 후 벡터DB 구축

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
- 저장/누락 방지 자동화

### 💾 4. 확장성 높은 VectorDB 구조
문서군별 VectorDB:
- chroma_db (manual)
- strategy_db (ip_strategy_2025)
- supportlaw_vectordb (중소기업창업지원법)
- supportprogram_vectordb (지원사업)
- failure_case_vectordb (실패와 재창업 사례)

### 🌐 5. 향후 Streamlit UI 확장 가능
src/ui/ 아래에 Streamlit 기반 서비스 추가 예정

---

# 🧩 프로젝트 전체 폴더 구조

```txt
📁 SKN20-3rd-4TEAM/
│
├── data/
│   ├── raw/                         # 원본 PDF, ZIP
│   │   ├── manual.pdf
│   │   ├── ip_strategy_2025.pdf
│   │   ├── 실패와 재창업 사례.pdf
│   │   ├── 실패와 재창업 사례2.pdf
│   │   └── 스타트업지원프로그램txt.zip
│   │
│   └── processed/                   # 전처리 완료 텍스트/PKL
│       ├── manual_chunks/
│       ├── strategy_chunks/
│       ├── 지식재산관리매뉴얼txt/
│       ├── 스타트업지원프로그램txt/
│       ├── 지원사업_정보_창업공간_dataset.py
│       ├── chunked_documents.pkl
│       └── dataset.json
│
├── vectordb/                        # VectorDB 저장소
│   ├── chroma_db/                   # manual RAG
│   ├── strategy_db/                 # strategy RAG
│   ├── supportlaw_vectordb/         # 지원법 RAG
│   ├── supportprogram_vectordb/     # 지원 프로그램 RAG
│   └── failure_case_vectordb/       # 실패/재창업 사례 RAG
│
├── src/
│   ├── preprocess/
│   │   ├── preprocess_two_files.py
│   │   ├── mydocling.py
│   │   ├── supportlaw_vectordb.py
│   │   └── 지원사업_정보_창업공간_dataset.py
│   │
│   ├── rag/
│   │   ├── main.py                  # manual.pdf 챗봇
│   │   ├── main_strategy.py         # strategy.pdf 챗봇
│   │   ├── 지식재산관리매뉴얼.py
│   │   ├── 스타트업지원프로그램.py
│   │   ├── supportlaw_vectordb.py
│   │   └── 지원사업_정보_창업공간.py
│   │
│   └── ui/                          # Streamlit UI (추가 예정)
│
├── images/                          # README·발표용 이미지
│
├── notebooks/                       # 분석·실험용 Jupyter
│
├── requirements.txt
├── .env                             # OpenAI API KEY
└── README.md
