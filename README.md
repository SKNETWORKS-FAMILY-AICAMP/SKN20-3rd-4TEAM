![팀명 소개](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=창업%20IP%20전략%20가이드%20RAG%20챗봇&fontSize=60)
# 🌐 SKN20 3rd Project – 창업 IP 전략 가이드 RAG 챗봇  
**Team: SKN20-3rd-4TEAM (김태빈 · 정래원 · 최소영 · 최유정)**  
지식재산(IP) 기반 스타트업/중소기업 창업자를 위한 RAG 기반 정책 안내 챗봇 프로젝트

---

## 📌 프로젝트 개요
본 프로젝트는 **스타트업 및 중소기업이 실제로 활용할 수 있는 지식재산(IP) 전략·지원정책**을  
PDF 문서 기반 RAG(Retrieval-Augmented Generation) 기술을 활용해 안내하는 **LLM 챗봇 서비스**입니다.

특히 다음 두 문서를 기반으로 구축되었습니다:

1. **「스타트업을 위한 지식재산 관리 매뉴얼」**
2. **「2025 지식재산경영 전략 가이드」**

각 문서별로 **독립된 RAG 시스템**을 구성하여  
정책·지원사업 정보 제공 정확도를 극대화했습니다.

---

## 👥 팀원 소개 (SKN20-3rd-4TEAM)
| 이름 | 역할 |
|------|------|
| **김태빈** | 시스템 설계, RAG 구조 설계 |
| **정래원** | 엔지니어링 리서치, 테스트 시나리오 |
| **최소영** | 정책 분석, 문서 구조 정의 |
| **최유정** | 데이터 전처리, Chunking, RAG 구현 |

---

## 🧠 주요 기능 (Features)

- **PDF 기반 문서 이해 및 질의응답**
- **Vector Search 기반 지식 검색**
- **문서별 독립 RAG 시스템 (manual / ip_strategy_2025 분리)**
- **Chunk 기반 정교한 검색 최적화**
- **OpenAI GPT-4o-mini 기반 자연어 응답 생성**
- **문단·섹션 태그 자동 부착을 통한 검색 품질 향상**

---

## 📁 폴더 구조

```txt
SKN20-3rd-4TEAM/
│
├── data/
│   ├── manual.pdf
│   └── ip_strategy_2025.pdf
│
├── processed_manual/
│   ├── manual_chunk_001.txt
│   └── ...
│
├── processed_strategy/
│   ├── strategy_2025_chunk_001.txt
│   └── ...
│
├── chroma_db/
├── strategy_db/
│
├── main.py
├── main_strategy.py
├── preprocess_two_files.py
│
├── .env
└── requirements.txt
