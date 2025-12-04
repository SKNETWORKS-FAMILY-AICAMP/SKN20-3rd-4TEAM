import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# =============================
# PDF 별 개별 처리 설정
# =============================
PDF_JOBS = [
    {
        "name": "manual",
        "pdf_path": "data/manual.pdf",
        "output_dir": "processed_manual"
    },
    {
        "name": "strategy_2025",
        "pdf_path": "data/ip_strategy_2025.pdf",
        "output_dir": "processed_strategy"
    }
]

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# =============================
# 섹션 태그 함수 (공용)
# =============================
def add_section_tags(text: str):
    if "특허" in text[:150]:
        text = "[섹션: 특허]\n" + text
    if "상표" in text[:150] or "디자인" in text[:150]:
        text = "[섹션: 상표·디자인]\n" + text
    if "지식재산" in text or "IP" in text:
        text = "[섹션: 지식재산 전략]\n" + text
    return text


# =============================
# 단일 PDF 전처리 함수
# =============================
def preprocess_single_pdf(pdf_path, output_dir, name):

    print(f"\n==============================")
    print(f"📌 [{name}] 전처리 시작")
    print(f"==============================")

    # PDF 로드
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[1] PDF 페이지 수: {len(pages)}")

    # 태그 추가
    for page in pages:
        page.page_content = add_section_tags(page.page_content)

    # 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"[2] 생성된 chunk 수: {len(chunks)}")

    # 저장 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # chunk 파일 저장
    for i, chunk in enumerate(chunks, start=1):
        filename = f"{name}_chunk_{i:03}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chunk.page_content)

    print(f"[3] 저장 완료 → {output_dir}/")
    print(f"==============================\n")


# =============================
# 전체 실행
# =============================
if __name__ == "__main__":
    for job in PDF_JOBS:
        preprocess_single_pdf(
            pdf_path=job["pdf_path"],
            output_dir=job["output_dir"],
            name=job["name"]
        )
