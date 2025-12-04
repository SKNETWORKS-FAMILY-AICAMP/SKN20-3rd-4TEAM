# 🚀 SKN20 3rd Project — 창업 IP 전략 가이드 RAG 챗봇  
**Team: SKN20-3rd-4TEAM (김태빈 · 정래원 · 최소영 · 최유정)**

스타트업·중소기업 창업자에게 필요한  
**지식재산(IP)·지원정책·창업지원·재창업 사례** 정보를  
PDF 기반 RAG(Retrieval-Augmented Generation) 기술로 제공하는  
지능형 정책 안내 챗봇 시스템입니다.

---

# 📌 1. 프로젝트 개요

이 프로젝트는 창업자와 중소기업이 실제 현장에서 자주 마주하는 다음 질문에 답하는 것을 목표로 합니다:

- “국가에서 제공하는 창업 IP 지원 프로그램이 뭐가 있지?”
- “해외 특허 비용은 지원 받을 수 있나요?”
- “실패·재창업 사례는 어디서 볼 수 있나요?”
- “중소기업창업지원법의 요약을 알려줘.”
- “지식재산 관리 매뉴얼을 따라 무엇을 해야 하지?”

이를 위해 **총 4개의 문서군(文書群)** 을 수집 및 구조화하였고,  
각 문서를 **독립 RAG 엔진**으로 구축했습니다.

---

# 📘 1.1 개발 동기 및 목적

최근 생성형 AI는 높은 언어 생성 능력을 갖추고 있으나,  
정확하지 않은 답변을 만들어내는 **환각(Hallucination)** 문제가 존재합니다.  

정책·법령·지원사업·지식재산과 같은 분야는  
잘못된 정보 제공 시 실제 법적·재정적 피해가 발생할 수 있어  
**신뢰할 수 있는 문서 기반 AI 시스템의 필요성**이 커지고 있습니다.

따라서 본 프로젝트는  
**정확한 정책 문서 기반의 RAG 챗봇 시스템을 구축해 창업자와 중소기업이 실제로 활용 가능한 지식 기반 서비스**를 만드는 것을 목표로 합니다.

---

# 📗 1.2 필요성

1. **정책·법령 문서의 복잡성**  
   - 길고 복잡한 문서를 직접 이해하기 어려움  
   - 필요한 정보를 빠르게 찾기 어려움

2. **LLM 환각 문제 최소화 필요**
   - 정책 정보는 정확한 근거가 필수  
   - 검색 기반 RAG 구조가 효과적

3. **창업자 및 중소기업 대상 정보 접근성 부족**
   - 상황에 맞는 정책·지원 정보 탐색 어려움  
   - 실패 사례 및 성공 전략 학습 기회 부족

4. **문서 기반 AI 서비스의 산업적 수요 증가**
   - 여러 기관·기업이 내부 문서 RAG 시스템 도입 중  
   - 본 프로젝트는 실제 서비스 가능성을 검증하는 의의 있음

---

# 🎯 1.3 개발 목표

본 프로젝트의 최종 목표는 **정확하고 신뢰할 수 있는 문서 기반 질의응답 시스템**을 구현하는 것입니다.

### 세부 개발 목표

- PDF 문서 → 텍스트 전처리 자동화
- 문서 단위 Chunking + 섹션 태깅
- 문서별 VectorDB 구축
- GPT 기반 RAG 챗봇 5종 구현 (문서별 독립)
- 검색 근거 기반의 신뢰성 있는 답변 생성
- 향후 Streamlit UI 기반 서비스 확장 가능성 확보

---

# 📚 2. 사용된 문서(데이터셋)

### 1) 공식 정책 문서
- 「스타트업을 위한 지식재산 관리 매뉴얼」
- 「2025 지식재산경영 전략 가이드」
- 「중소기업창업지원법」 텍스트 요약본

### 2) 창업 지원 프로그램
- 정부/지자체 지원사업 파일
- 창업공간 및 지원정보 dataset.json

### 3) 실패와 재창업 사례
- 실패와 재창업 사례.pdf  
- 실패와 재창업 사례2.pdf  
→ 데이터 전처리 후 RAG 구조 구축

---

# 📝 3. 요구사항 정의 (Requirements)

## 3.1 데이터 수집 및 전처리
- PDF 텍스트 추출 (pypdf / docling)
- 문단 단위 분리 및 정제
- Chunking (400~700 tokens + overlap)
- 텍스트 누락 방지 및 구조 보존
- 섹션 태그 자동 부착

## 3.2 설계 및 구축
- 문서별 독립 VectorDB 구축
- LLM + Retriever 기반 RAG 시스템 구현
- 문서 선택형 챗봇 구조 설계
- “근거 기반 답변” 강제 시스템 프롬프트 구성

## 3.3 ETL Pipeline
- **Extract:** PDF/ZIP/JSON 데이터 수집  
- **Transform:** 정제 → 태깅 → 청킹  
- **Load:** VectorDB 저장(chroma_db, strategy_db 등)

## 3.4 모델링 결과 및 시각화
- 문서별 RAG 챗봇 구현  
- Retrieval 기반 답변 정확도 평가  
- Chunk 분포 시각화, 벡터 임베딩 맵(2D/3D)  
- 환각 발생률 분석 및 개선  

---

# 📦 4. 수집 데이터 목록

## (1) Raw Data
| 파일명 | 설명 |
|--------|------|
| manual.pdf | 지식재산 관리 매뉴얼 |
| ip_strategy_2025.pdf | 2025 지식재산 전략 가이드 |
| 실패와 재창업 사례.pdf | 사례집 PDF 1 |
| 실패와 재창업 사례2.pdf | 사례집 PDF 2 |
| 중소기업창업_지원법.txt | 지원법 원문 텍스트 |
| 스타트업지원프로그램txt.zip | 지원 프로그램 원문 |
| 지식재산관리매뉴얼txt.zip | 매뉴얼 텍스트 |
| 지원사업,정보,창업공간.zip | 창업공간/지원사업 데이터 |

## (2) Processed Data
- manual_chunk_###.txt  
- strategy_2025_chunk_###.txt  
- dataset.json  
- chunked_documents.pkl  
- 스타트업지원프로그램txt/  
- 지식재산관리매뉴얼txt/

## (3) VectorDB
- chroma_db/  
- strategy_db/  
- supportlaw_vectordb/  
- supportprogram_vectordb/  
- failure_case_vectordb/

---

# 🧠 5. 시스템 특징 (Features)

### 📄 1. 문서 기반 독립 RAG 시스템  
문서별 VectorDB & 문서별 챗봇 구조로 환각 최소화

### 🔍 2. 근거 기반 답변  
검색된 chunk를 함께 반환하여 답변 신뢰성 보장

### ⚙️ 3. 자동 전처리 파이프라인  
PDF → 텍스트 → 태깅 → Chunking → VectorDB 자동 저장

### 💾 4. 확장성 높은 구조  
문서가 추가되도 동일한 구조로 RAG 시스템 확장 가능

---

# 🧩 6. 프로젝트 전체 폴더 구조

(아래 내용은 최신 반영된 상태)

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
├── images/
├── notebooks/
├── requirements.txt
├── .env
└── README.md
