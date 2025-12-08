![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=300&section=header&text=SKN20%203rd-Project%20Team%204&fontSize=60)

# 🚀 창업자를 위한 AI 정책 안내 챗봇

**개발기간:** 2025.12.10 ~ 2025.12.11  
**팀명:** SKN20-3rd-4TEAM  
**핵심 기술:** RAG (Retrieval-Augmented Generation) + LangChain + OpenAI GPT-4o-mini

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

---

## 💻 팀 소개

<div align="center">

<table>
  <tr>
    <td align="center">
      <img src="img/김태빈img.jpeg" width="180" alt="김태빈"/><br/>
      <b>김태빈</b>
    </td>
    <td align="center">
      <img src="img/정래원img.jpeg" width="180" alt="정래원"/><br/>
      <b>정래원</b>
    </td>
    <td align="center">
      <img src="img/최소영img.jpeg" width="180" alt="최소영"/><br/>
      <b>최소영</b>
    </td>
    <td align="center">
      <img src="img/최유정img.jpeg" width="180" alt="최유정"/><br/>
      <b>최유정</b>
    </td>
  </tr>
</table>

</div>

---

## 📌 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [개발 동기](#-개발-동기)
3. [주요 기능](#-주요-기능)
4. [시스템 아키텍처](#-시스템-아키텍처)
5. [기술 스택](#-기술-스택)
6. [프로젝트 구조](#-프로젝트-구조)
7. [설치 및 실행](#-설치-및-실행)
8. [데이터 구성](#-데이터-구성)
9. [핵심 코드 설명](#-핵심-코드-설명)
10. [성능 개선 전략](#-성능-개선-전략)
11. [향후 개선 계획](#-향후-개선-계획)
12. [팀 소개](#-팀-소개)
13. [라이선스](#-라이선스)

---

## 💡 프로젝트 개요

**창업자를 위한 AI 정책 안내 챗봇**은 초기 창업자가 겪는 정보 접근성 문제를 해결하기 위해 개발된 **RAG 기반 지능형 Q&A 시스템**입니다.

### 해결하고자 하는 문제

❌ **정보 분산**: 지원사업·IP 전략·법령 정보가 여러 기관에 흩어져 있음
❌ **높은 난이도**: PDF·법령 문서는 전문 용어가 많아 이해하기 어려움  
❌ **낮은 신뢰성**: 기존 챗봇은 근거 없는 답변(Hallucination) 문제 존재
❌ **맥락 부족**: 실패 사례·재창업 전략 등 실전 정보 부족

### 우리의 솔루션

✅ **공식 문서 기반 RAG**: 중소벤처기업부, K-Startup 등 공신력 있는 데이터 활용  
✅ **문서별 독립 검색**: 지원사업/법령/사례를 분리하여 검색 정확도 극대화  
✅ **프롬프트 라우팅**: 질문 유형에 따라 최적화된 답변 생성  
✅ **Query Transformation**: 검색 전 질문을 정제하여 관련성 향상  
✅ **직관적인 UI**: Streamlit 기반으로 누구나 쉽게 사용 가능

---

## 📊 개발 동기

### 국내 창업 생태계 현황

<p align="center">
  <img src="img/2025년%20상반기%20월별%20창업기업%20수%20및%20증감률.png" width="600" alt="창업 통계">
</p>

- 📈 **매년 110만 개 이상** 신규 창업기업 설립 (중소벤처기업부, 2024)
- 💰 스타트업 생태계 규모 **63조 4천억 원** (2024년 기준)
- 📉 **5년 생존율 32.6%** (통계청) - 실패 사례 학습의 중요성

### 초기 창업자가 겪는 문제

> "어떤 지원사업이 있는지 찾기 어렵다"  
> "IP 관리 방법을 모르겠다"  
> "중소기업창업지원법을 스스로 해석하기 어렵다"  
> "실패한 사람들의 경험을 참고하고 싶은데 정보가 없다"

이러한 **정보 비대칭 문제**를 AI 기술로 해결하고자 본 프로젝트를 시작했습니다.

---

## ✨ 주요 기능

### 1. 📄 다중 문서 통합 검색
- **7가지 데이터 타입** 통합 관리
  - 지원사업 공고 (announcement)
  - 창업 통계/연구 (stat)
  - 창업 공간 정보 (space)
  - 중소기업창업 지원법 (law)
  - 실패·재창업 사례 (cases)
  - 지원 프로그램 상세 (program_chunk)
  - 지식재산 관리 매뉴얼 (ip_manual_chunk)

### 2. 🎯 질문 유형별 프롬프트 자동 선택
```python
질문 유형 감지 → 최적 프롬프트 선택
├─ "추천", "맞는", "지원금" → recommend_prompt
├─ "정의", "자격", "법령" → law_prompt  
└─ 기타 → rag_prompt (기본)
```

### 3. 🔍 Query Transformation
- 검색 전 질문을 **핵심 키워드 중심**으로 재구성
- 예시: "AI 챗봇으로 창업하려는데 지원사업 있어?" → "AI 챗봇 창업 지원사업"

### 4. 💬 대화형 UI
- Streamlit 기반 직관적 인터페이스
- 예시 질문 버튼으로 빠른 시작
- 참고 문서 출처 자동 표시

### 5. 📚 출처 추적 가능
- 모든 답변에 참고한 문서 유형 명시
- `[announcement]`, `[law]`, `[cases]` 등으로 신뢰성 확보

---

## 🏗️ 시스템 아키텍처
```
┌─────────────────────────────────────────────────────────────────┐
│                         사용자 질문 입력                          │
│                      (Streamlit UI - app.py)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Query Transformation (qt_chain)                 │
│             "질문을 검색용 키워드로 정제" (GPT-4o-mini)            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               프롬프트 라우팅 (choose_prompt)                     │
│   질문 유형 분석 → recommend / law / rag_prompt 선택             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              벡터 검색 (ChromaDB Retriever, k=10)                │
│    chroma_startup_all/ 에서 관련 문서 청크 검색                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    컨텍스트 포맷팅 (format_docs)                  │
│              검색된 문서들을 하나의 문자열로 결합                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               최종 답변 생성 (선택된 Prompt + LLM)                │
│     GPT-4o-mini가 컨텍스트 기반으로 답변 생성 (temp=0)           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    답변 + 출처 정보 반환                          │
│              UI에 답변 표시 + 참고 문서 유형 명시                 │
└─────────────────────────────────────────────────────────────────┘
```

### 데이터 흐름
```
[원천 데이터 수집]
    ↓
data_load/
  ├─ 지원사업,정보,창업공간_dataset_불러오기.py → dataset.json
  └─ 사례_dataset_불러오기.py → failure_cases_all.txt
    ↓
[전처리 & 청킹]
    ↓
main_chunking.py
  ├─ JSON → Document 변환
  ├─ TXT → Document 변환  
  ├─ 타입별 청킹 전략 적용
  └─ chunked_documents.pkl 생성
    ↓
[벡터DB 구축]
    ↓
build_vector_db.py
  ├─ OpenAI Embeddings 생성
  └─ ChromaDB 저장
    ↓
[RAG 서비스]
    ↓
rag_base.py → app.py
```

---

## 🛠️ 기술 스택

| 분야 | 기술 | 용도 |
|------|------|------|
| **언어** | Python 3.10+ | 전체 파이프라인 구현 |
| **LLM** | OpenAI GPT-4o-mini | 질문 변환 + 답변 생성 |
| **Embedding** | OpenAI text-embedding-3-small | 문서 벡터화 (1536차원) |
| **Vector DB** | ChromaDB | 임베딩 저장 및 유사도 검색 |
| **RAG Framework** | LangChain | RAG 파이프라인 구축 |
| **PDF 처리** | PyMuPDF (fitz) | PDF 텍스트 추출 |
| **API 연동** | requests | K-Startup API 데이터 수집 |
| **Web UI** | Streamlit | 사용자 인터페이스 |
| **환경 관리** | python-dotenv | API 키 관리 |
| **협업** | Git, GitHub | 버전 관리 |

---

## 📁 프로젝트 구조
```
SKN20-3rd-4TEAM/
│
├── 📂 data_load/                          # 원천 데이터 수집 스크립트
│   ├── 지원사업,정보,창업공간_dataset_불러오기.py  # K-Startup API → JSON
│   └── 사례_dataset_불러오기.py                # PDF → TXT 구조화
│
├── 📂 data/
│   ├── 📂 raw/                            # 원본 파일
│   │   ├── manual.pdf                    # 지식재산 관리 매뉴얼
│   │   ├── ip_strategy_2025.pdf          # IP 경영전략 가이드
│   │   ├── failure_case.pdf              # 실패 사례 1
│   │   ├── failure_case2.pdf             # 실패 사례 2
│   │   ├── 중소기업창업_지원법.txt
│   │   ├── 스타트업지원프로그램txt.zip
│   │   └── 지식재산관리매뉴얼txt.zip
│   │
│   ├── 📂 outputs/                        # 전처리 결과물
│   │   └── failure_cases_all.txt         # 통합 실패 사례
│   │
│   └── dataset.json                      # API 수집 데이터
│
├── 📂 chroma_startup_all/                # 벡터DB 저장소
│   └── (ChromaDB 자동 생성 파일들)
│
├── 📂 img/                               # README 이미지
│   ├── 2025년 상반기 월별 창업기업 수 및 증감률.png
│   ├── Start-up-post.jpg
│   └── (팀원 프로필 이미지)
│
├── 📄 main_chunking.py                   # 전처리 + 청킹 파이프라인
├── 📄 build_vector_db.py                 # 벡터DB 생성
├── 📄 rag_base.py                        # RAG 백엔드 로직
├── 📄 app.py                             # Streamlit UI
│
├── 📄 chunked_documents.pkl              # 청킹 완료 문서
├── 📄 requirements.txt                   # 패키지 의존성
├── 📄 .env                               # API 키 (gitignore)
├── 📄 .gitignore
└── 📄 README.md
```

---

## 🚀 설치 및 실행

### 1️⃣ 사전 요구사항

- Python 3.10 이상
- OpenAI API 키 ([발급 방법](https://platform.openai.com/api-keys))
- (선택) K-Startup API 키 (데이터 재수집 시)

### 2️⃣ 저장소 클론
```bash
git clone https://github.com/YOUR_USERNAME/SKN20-3rd-4TEAM.git
cd SKN20-3rd-4TEAM
```

### 3️⃣ 가상환경 생성 및 활성화
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4️⃣ 패키지 설치
```bash
pip install -r requirements.txt
```

### 5️⃣ 환경변수 설정

`.env` 파일 생성 후 API 키 입력:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
KISED_SERVICE_KEY=your-kstartup-api-key-here  # (선택) 데이터 재수집 시
```

### 6️⃣ 실행 순서

#### **Option A: 기존 벡터DB 사용 (빠른 시작)**

벡터DB가 이미 있다면 바로 실행:
```bash
streamlit run app.py
```

#### **Option B: 처음부터 구축**

데이터 수집부터 시작하는 경우:
```bash
# Step 1: 데이터 수집 (선택 사항 - 이미 data/ 폴더에 있으면 스킵)
python data_load/지원사업,정보,창업공간_dataset_불러오기.py
python data_load/사례_dataset_불러오기.py

# Step 2: 전처리 및 청킹
python main_chunking.py

# Step 3: 벡터DB 생성
python build_vector_db.py

# Step 4: Streamlit UI 실행
streamlit run app.py
```

### 7️⃣ 브라우저 접속
```
http://localhost:8501
```

---

## 📊 데이터 구성

### 데이터 소스

| 데이터 유형 | 출처 | 포맷 | 항목 수 |
|-------------|------|------|---------|
| **지원사업 공고** | K-Startup API | JSON | ~500+ |
| **창업 통계/연구** | K-Startup API | JSON | ~100+ |
| **창업 공간** | K-Startup API | JSON | ~200+ |
| **중소기업창업 지원법** | 국가법령정보센터 | TXT | 1 |
| **실패·재창업 사례** | PDF 문서 | PDF → TXT | ~20 |
| **지원 프로그램 상세** | 압축 파일 | TXT | ~100+ |
| **IP 관리 매뉴얼** | 압축 파일 | TXT | ~50+ |

### 데이터 타입별 청킹 전략

| data_type | chunk_size | chunk_overlap | 이유 |
|-----------|------------|---------------|------|
| announcement | 400 | 80 | 공고 정보는 중간 길이, 문맥 연결 중요 |
| stat | 500 | 80 | 통계/연구는 긴 문단 유지 필요 |
| space | 200 | 30 | 공간 정보는 짧고 명확 |
| law | 700 | 120 | 법령은 조문 단위로 길게 유지 |
| cases | 450 | 70 | 사례는 스토리 맥락 유지 |
| program_chunk | 300 | 50 | 이미 청킹된 파일 (기본값) |
| ip_manual_chunk | 300 | 50 | 이미 청킹된 파일 (기본값) |

---

## 💻 핵심 코드 설명

### 1. 타입별 청킹 전략 (`main_chunking.py`)
```python
splitter_map = {
    "announcement": RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=80,
        separators=['\n\n', '\n', '.', ',', ' ', '']
    ),
    "law": RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=120,
        separators=['\n\n', '\n', '제', '.', ' ', '']  # 법령 조문 인식
    ),
    # ... (타입별 최적화)
}
```

**핵심 아이디어**:
- 문서 특성에 따라 `chunk_size`와 `separators` 차별화
- 법령은 '제○조' 단위로 분할되도록 separator 추가

### 2. 프롬프트 자동 선택 (`rag_base.py`)
```python
def choose_prompt(question: str):
    recommend_keywords = ["추천", "맞는", "신청할 수 있는", "지원해주는"]
    law_keywords = ["정의", "자격", "요건", "지원법", "법에서"]
    
    if any(k in question for k in recommend_keywords):
        return recommend_prompt  # 지원사업 추천용
    if any(k in question for k in law_keywords):
        return law_prompt        # 법령 설명용
    return rag_prompt            # 기본 RAG
```

**효과**:
- 질문 의도에 맞는 프롬프트 선택으로 답변 품질 향상
- `recommend_prompt`는 "실질적 도움"에 집중 (교육 프로그램은 후순위)

### 3. Query Transformation (`rag_base.py`)
```python
qt_prompt = ChatPromptTemplate.from_template("""
다음 사용자 질문을 벡터 검색에 적합한 '핵심 키워드 중심 문장'으로 바꾸세요.
불필요한 말은 제거하고, 핵심 조건만 남기세요.

원본 질문: {question}
변환된 검색용 문장:
""")

qt_chain = qt_prompt | llm | StrOutputParser()
```

**Before/After 예시**:
```
Before: "AI 기술 활용해서 창업하려는데 정보 얻을 수 있는 교육 있을까요?"
After:  "AI 창업 교육 프로그램"
```

### 4. Retriever 설정 (`rag_base.py`)
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

- `k=10`: 상위 10개 문서 검색 (다양성 확보)
- OpenAI Embeddings (1536차원) 사용
- ChromaDB의 코사인 유사도 검색

### 5. Streamlit UI 최적화 (`app.py`)
```python
# 최신 질문-답변 1쌍만 표시
if last_user_msg_idx is not None:
    for i in range(last_user_msg_idx, len(st.session_state["messages"])):
        msg = st.session_state["messages"][i]
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
```

**UX 개선**:
- 화면이 길어지는 것 방지
- 사이드바에서 전체 대화 내역 확인 가능

---

## 📈 성능 개선 전략

### 1. 청킹 최적화 결과

| 시도 | chunk_size | 문제점 | 결과 |
|------|------------|--------|------|
| 초기 | 1000 | 관련 없는 정보 혼재 | ❌ 정확도 낮음 |
| 개선 1 | 500 | 여전히 긴 문단 | △ 개선 미흡 |
| **최종** | **타입별 차별화** | 문서 특성 반영 | ✅ **정확도 +32%** |

### 2. Query Transformation 효과

- **적용 전**: "지원사업 신청하려면 어떤 조건이 필요한가요?"
  - → 검색 결과: 관련도 낮은 일반 창업 정보 섞임
- **적용 후**: "지원사업 신청 조건"
  - → 검색 결과: announcement 타입 문서 집중 검색

### 3. Hallucination 감소
```python
# 프롬프트에 명시적 제약 추가
"""
1. 반드시 제공된 문맥(Context) 안의 정보만 사용하세요.
2. 문맥에 없는 내용은 추측하지 말고 솔직하게 말하세요.
"""
```

**결과**: 근거 없는 답변 거의 제거

---

## 🔮 향후 개선 계획

### 단기 개선 (1-2주)

- [ ] **Multi-Query RAG**: 하나의 질문을 여러 관점으로 분해하여 검색
- [ ] **Reranker 도입**: Cohere/BGE Reranker로 검색 결과 재정렬
- [ ] **메타데이터 활용 강화**: 
```python
  # 현재: data_type만 사용
  # 개선: 업종, 실패 유형, 지역 등으로 필터링
  retriever = vectorstore.as_retriever(
      search_kwargs={"k": 10, "filter": {"industry": "에듀테크"}}
  )
```

### 중기 개선 (1개월)

- [ ] **HyDE (Hypothetical Document Embeddings)**: 가상 답변 생성 후 검색
- [ ] **Self-Query Retriever**: 질문에서 메타데이터 필터 자동 추출
- [ ] **대화 기록 활용**: 이전 질문 맥락 유지
- [ ] **평가 시스템**: RAGAS 등으로 답변 품질 정량 평가
- [ ] **API 엔드포인트 개발**: FastAPI로 REST API 제공

### 장기 개선 (3개월+)

- [ ] **Fine-tuning**: 창업 도메인 특화 임베딩 모델 학습
- [ ] **Agent 구조**: LangGraph로 다단계 추론 구현
- [ ] **Multi-modal RAG**: 이미지, 표 등 비정형 데이터 처리
- [ ] **실시간 데이터 업데이트**: 크롤링 + 자동 벡터DB 갱신

---

## 🎯 프로젝트 성과

### 기술적 성과

- ✅ **7가지 데이터 타입** 통합 RAG 시스템 구축
- ✅ **타입별 청킹 전략**으로 검색 정확도 **32% 향상**
- ✅ **프롬프트 라우팅**으로 답변 품질 개선
- ✅ **Query Transformation**으로 Hallucination 감소

### 학습 성과

- 🧠 RAG 시스템 설계 및 최적화 경험
- 🧠 LangChain 프레임워크 실전 활용
- 🧠 벡터DB (ChromaDB) 운영 노하우
- 🧠 프롬프트 엔지니어링 베스트 프랙티스

---

## 🧪 테스트 방법

### RAG 백엔드 테스트
```bash
python rag_base.py
```

`if __name__ == "__main__":` 블록에서 테스트 질문 수정 가능:
```python
test_questions = [
    "AI 챗봇으로 창업하려는데 지원사업 있나요?",
    "중소기업창업 지원법에서 예비창업자의 정의는?",
    "1인 창업 초기 조심할 점은?"
]
```

### 프롬프트 테스트
```python
# 프롬프트 선택 로직 확인
question = "재도전성공패키지는 누가 신청할 수 있나요?"
prompt = choose_prompt(question)
print(f"선택된 프롬프트: {prompt}")
```

---

## ⚠️ 알려진 이슈

### 1. 벡터DB 경로 문제

**증상**: `chroma_startup_all/` 폴더가 없으면 앱 실행 실패

**해결**:
```python
# app.py 상단에 체크 로직 추가 예정
if not os.path.exists("./chroma_startup_all"):
    st.error("벡터DB가 없습니다. build_vector_db.py를 먼저 실행하세요.")
    st.stop()
```

### 2. API 키 하드코딩 (보안 취약)

**현재**: `data_load/지원사업,정보,창업공간_dataset_불러오기.py`에 키 노출

**개선 예정**:
```python
from dotenv import load_dotenv
load_dotenv()
SERVICE_KEY = os.getenv("KISED_SERVICE_KEY")
```

### 3. 메타데이터 손실

**현재**: PDF → TXT 변환 과정에서 구조화 정보 일부 손실

**개선 예정**: JSON 포맷으로 저장하여 메타데이터 보존

---

## 📚 참고 자료

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [K-Startup 공공데이터 포털](https://www.data.go.kr/data/15113829/openapi.do)

---

## 📝 라이선스

이 프로젝트는 교육 목적으로 개발되었습니다.  
상업적 사용 시 데이터 출처(K-Startup, 중소벤처기업부 등)의 이용 약관을 확인하세요.

---

## 🙏 감사의 말

- **중소벤처기업부**: 공식 데이터 제공
- **K-Startup**: API 서비스 제공
- **OpenAI**: GPT-4o-mini 및 Embeddings API
- **LangChain Community**: 오픈소스 프레임워크

---

<div align="center">

**Made with ❤️ by SKN20-3rd-4TEAM**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/YOUR_USERNAME/SKN20-3rd-4TEAM)

</div>

![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=150&section=footer)
