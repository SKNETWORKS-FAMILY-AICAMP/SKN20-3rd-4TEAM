import os
import warnings
import pickle
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 경고메세지 삭제
warnings.filterwarnings('ignore')
load_dotenv()

# openapi key 확인
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('.env확인, key없음')

# ============================================
# 유틸리티 함수
# ============================================
def format_date(date_str):
    """날짜 포맷 변환 (20251201 → 2025-12-01)"""
    if date_str and len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str

def clean_html(text):
    """간단한 HTML 태그 제거"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# documents 
def create_announcement_document(item):
    """announcement 항목을 Document로 변환"""
    page_content = f"""
【사업 공고】 {item.get('biz_pbanc_nm', '제목 없음')}

■ 사업 개요
{item.get('pbanc_ctnt', '내용 없음')}

■ 신청 대상
- 대상 유형: {item.get('aply_trgt', '')}
- 상세 조건:
{item.get('aply_trgt_ctnt', '')}

■ 제외 대상
{item.get('aply_excl_trgt_ctnt', '')}

■ 지원 정보
- 분류: {item.get('supt_biz_clsfc', '')}
- 지역: {item.get('supt_regin', '')}
- 창업 연수: {item.get('biz_enyy', '')}
- 대상 연령: {item.get('biz_trgt_age', '')}
- 감독 기관: {item.get('sprv_inst', '')}

■ 신청 정보
- 접수 기간: {format_date(item.get('pbanc_rcpt_bgng_dt', ''))} ~ {format_date(item.get('pbanc_rcpt_end_dt', ''))}
- 온라인 신청: {item.get('aply_mthd_onli_rcpt_istc', '해당 없음')}
- 상세 페이지: {item.get('detl_pg_url', '')}
- 안내 페이지: {item.get('biz_gdnc_url', '')}

■ 주관 기관
- 기관명: {item.get('pbanc_ntrp_nm', '')}
- 담당 부서: {item.get('biz_prch_dprt_nm', '')}
- 연락처: {item.get('prch_cnpl_no', '')}
    """.strip()
    
    metadata = {
        "source": "k-startup",
        "data_type": "announcement",
        "biz_id": item.get('pbanc_sn'),
        "biz_name": item.get('biz_pbanc_nm', ''),
        "category": item.get('supt_biz_clsfc', ''),
        "region": item.get('supt_regin', ''),
        "start_date": item.get('pbanc_rcpt_bgng_dt', ''),
        "end_date": item.get('pbanc_rcpt_end_dt', ''),
        "is_recruiting": item.get('rcrt_prgs_yn', ''),
        "supervisor": item.get('sprv_inst', ''),
        "url": item.get('detl_pg_url', '')
    }
    
    return Document(page_content=page_content, metadata=metadata)

def create_stat_document(item):
    """stat 항목을 Document로 변환"""
    page_content = f"""
【창업 통계/연구 자료】 {item.get('titl_nm', '제목 없음')}

■ 내용
{clean_html(item.get('ctnt', ''))}

■ 파일 정보
- 파일명: {item.get('file_nm', '')}
- 상세 페이지: {item.get('detl_pg_url', '')}
    """.strip()
    
    metadata = {
        "source": "k-startup",
        "data_type": "stat",
        "title": item.get('titl_nm', ''),
        "file_name": item.get('file_nm', ''),
        "register_date": item.get('fstm_reg_dt', ''),
        "modified_date": item.get('last_mdfcn_dt', ''),
        "url": item.get('detl_pg_url', '')
    }
    
    return Document(page_content=page_content, metadata=metadata)

def create_space_document(item):
    """space 항목을 Document로 변환"""
    page_content = f"""
【창업 공간】 {item.get('spce_nm', '공간명 없음')}

■ 센터 정보
- 센터명: {item.get('cntr_nm', '')}
- 센터 유형: {item.get('cntr_type_nm', '')}

■ 위치 정보
- 주소: {item.get('addr', '')}
    """.strip()
    
    metadata = {
        "source": "k-startup",
        "data_type": "space",
        "space_id": item.get('spce_id'),
        "center_id": item.get('cntr_id'),
        "center_name": item.get('cntr_nm', ''),
        "center_type": item.get('cntr_type_nm', ''),
        "space_name": item.get('spce_nm', ''),
        "address": item.get('addr', ''),
        "space_type": item.get('spce_type_nm', ''),
        "space_count": item.get('spce_cnt', 0),
        "area": item.get('excuse_ar', ''),
        "rent": item.get('rent', 0),
        "management_fee": item.get('guam', 0),
        "reservation": item.get('rsvt_psbl_clss', ''),
        "homepage": item.get('hmpg', ''),
        "building_name": item.get('buld_nm', ''),
        "postal_code": item.get('pstno', ''),
        "latitude": item.get('latde', ''),
        "longitude": item.get('lgtde', ''),
        "current_tenant": item.get('seat_co', ''),
        "center_intro": clean_html(item.get('cntr_intrd_type_nm', ''))
    }
    
    return Document(page_content=page_content, metadata=metadata)



# 1. JSON → Document 변환
print("=" * 60)
print("1단계: JSON → Document 변환")
print("=" * 60)

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)
documents = []

# announcement 처리
print("\nannouncement 문서 생성 중...")
for item in data.get("announcement", []):
    doc = create_announcement_document(item)
    documents.append(doc)
print(f"  → {len([d for d in documents if d.metadata['data_type'] == 'announcement'])}개 완료")

# stat 처리
print("stat 문서 생성 중...")
for item in data.get("stat", []):
    doc = create_stat_document(item)
    documents.append(doc)
print(f"  → {len([d for d in documents if d.metadata['data_type'] == 'stat'])}개 완료")

# space 처리
print("space 문서 생성 중...")
for item in data.get("space", []):
    doc = create_space_document(item)
    documents.append(doc)
print(f"  → {len([d for d in documents if d.metadata['data_type'] == 'space'])}개 완료")

print(f"\n총 {len(documents)}개의 Document 생성 완료!")

# # documents.pkl 저장 (백업)
# with open("documents.pkl", "wb") as f:
#     pickle.dump(documents, f)
# print(" documents.pkl 저장 완료 (백업)")


# 2. 청킹
print("\n" + "=" * 60)
print("2단계: 문서 청킹")
print("=" * 60)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=['\n\n', '\n', '.', ',', ' ', '']
)

print(f"\n청킹 중... (원본: {len(documents)}개)")
chunked_documents = text_splitter.split_documents(documents)
print(f" 청킹 완료: {len(chunked_documents)}개로 분할됨")

# 청킹된 문서 저장
with open("chunked_documents.pkl", "wb") as f:
    pickle.dump(chunked_documents, f)
print(" chunked_documents.pkl 저장 완료")