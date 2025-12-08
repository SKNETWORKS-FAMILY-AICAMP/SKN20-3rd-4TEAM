"""
eval_compare.py
RAG 시스템 vs 일반 LLM 성능 비교 평가

목적:
1. RAG 시스템의 정확도와 신뢰도 측정
2. 일반 LLM과의 성능 차이 비교
3. 정량적 지표로 RAG의 효과 입증
"""

import os
import warnings
from dotenv import load_dotenv
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RAG 시스템 임포트
from rag_base import (
    retriever,
    llm,
    choose_prompt,
    format_docs,
    qt_chain,
)

warnings.filterwarnings("ignore")
load_dotenv()

# =========================
# 평가용 테스트 질문 세트
# =========================
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "제 회사가 폐업한 지 3년이 되었는데, 재도전성공패키지 신청이 가능한가요? 어떤 조건을 충족해야 하나요?",
        "category": "지원사업",
        "expected_keywords": ["폐업", "7년", "재창업"],
    },
    {
        "id": 2,
        "question": "중소기업창업 지원법 제2조에서 규정한 창업자의 정의를 정확하게 말씀해주세요. 중소기업 범위와 함께요.",
        "category": "법령",
        "expected_keywords": ["7년", "중소기업", "사업을 개시"],
    },
    {
        "id": 3,
        "question": "저는 1인 창업을 준비 중인데, 창업 초기에 가장 주의해야 할 점이 무엇인지 알려주세요. 특히 자금관리와 시장조사에 대해서요.",
        "category": "실패사례",
        "expected_keywords": ["자금관리", "시장조사", "번아웃"],
    },
    {
        "id": 4,
        "question": "저는 35세 청년인데, 창업을 준비하고 있습니다. 청년 창업자를 위한 지원사업이 있는지, 연령 제한은 어디까지인지 알려주세요.",
        "category": "지원사업",
        "expected_keywords": ["39세", "청년", "창업"],
    },
    {
        "id": 5,
        "question": "서울에서 저렴한 가격으로 창업 초기에 사용할 수 있는 오피스 공간이 있나요? 창업보육센터나 공유오피스 같은 곳 정보를 주세요.",
        "category": "창업공간",
        "expected_keywords": ["창업보육센터", "공유오피스", "입주", "센터"],
    },
    {
        "id": 6,
        "question": "저는 AI 챗봇을 활용한 소프트웨어를 개발해서 창업하려고 합니다. 기술 창업자를 위한 지원사업이나 투자 프로그램이 있나요?",
        "category": "지원사업",
        "expected_keywords": ["기술", "AI", "지원"],
    },
    {
        "id": 7,
        "question": "제가 예전에 창업에 실패한 경험이 있습니다. 이번에 다시 창업하려고 할 때, 이전 실패 경험을 어떻게 활용해야 하고 주의할 점은 무엇인가요?",
        "category": "실패사례",
        "expected_keywords": ["실패", "재창업", "경험"],
    },
    {
        "id": 8,
        "question": "저는 아직 사업자등록을 하지 않은 예비창업자입니다. 사업 아이디어만 있는데도 지원받을 수 있는 교육이나 프로그램이 있나요?",
        "category": "지원사업",
        "expected_keywords": ["예비창업자", "교육", "지원"],
    },
]

# =========================
# 각 질문에 대한 기준 정답 (Reference Answers)
# =========================
REFERENCE_ANSWERS = {
    1: """
재도전성공패키지는 폐업 후 7년 이내의 (예비)재도전 창업자를 지원하는 사업입니다.
따라서 폐업 후 3년이 지났다면 신청이 가능합니다.
다만, 이전 사업에 대한 폐업 신고가 완료되어 있어야 하고, 재창업을 위한 사업계획서 등 필수 서류를 준비해야 합니다.
""",
    2: """
중소기업창업 지원법 제2조에서 "창업"은 대통령령으로 정하는 바에 따라 중소기업을 새로 설립하는 것을 의미합니다.
여기서 "중소기업"은 중소기업기본법 제2조에 따른 중소기업을 말하며,
일반적으로 업종별로 상시근로자 수, 매출액, 자산 기준 등을 충족해야 합니다.
즉, 창업자는 중소기업을 새로 설립하는 자를 의미합니다.
""",
    3: """
1인 창업 초기에 가장 중요한 것은 자금관리와 시장조사입니다.
자금관리 측면에서는 초기 비용과 고정비를 명확히 파악하고, 비상자금을 확보하며 불필요한 지출을 줄여야 합니다.
시장조사 측면에서는 목표 고객, 경쟁사, 시장 규모를 조사해 수요가 있는지 확인해야 하고,
이를 통해 과도한 기대나 번아웃을 피하는 것이 중요합니다.
""",
    4: """
청년 창업 지원사업은 일반적으로 만 39세 이하를 청년으로 보는 경우가 많습니다.
따라서 35세라면 대부분의 청년 창업 지원사업 신청이 가능합니다.
다만 사업별로 연령 기준과 창업 단계(예비창업자, 창업 후 몇 년 이내 등)가 다를 수 있으므로
각 사업의 공고문에서 신청 대상과 연령 제한을 반드시 확인해야 합니다.
""",
    5: """
서울에는 창업 초기에 사용할 수 있는 저렴한 오피스 공간으로 창업보육센터와 공유오피스가 다양하게 있습니다.
예를 들어 서울창업허브, 서울산업진흥원(SBA)에서 운영하는 센터, 대학 창업보육센터 등이 있으며,
일정 심사를 거쳐 입주하면 임대료를 낮게 책정받거나 보조를 받을 수 있습니다.
또한 민간 공유오피스(위워크, 패스트파이브 등)도 단기·소규모로 사용할 수 있는 상품을 제공합니다.
""",
    6: """
AI 챗봇 기반 소프트웨어를 개발하는 기술 창업자는 기술 창업 지원사업과 투자 프로그램을 함께 검토할 수 있습니다.
정부 차원에서는 중소벤처기업부와 창업진흥원에서 운영하는 기술 기반 창업 지원사업,
TIPS와 같은 민간투자 연계 프로그램이 대표적입니다.
또한 액셀러레이터, VC, 인큐베이터 등에서 AI·소프트웨어 분야 초기 스타트업을 대상으로
투자와 보육 프로그램을 운영하고 있습니다.
""",
    7: """
이전 창업 실패 경험은 재창업에서 중요한 자산이 될 수 있습니다.
먼저 당시 실패 원인(시장검증 부족, 자금관리 실패, 팀 문제 등)을 구체적으로 분석하고,
이를 바탕으로 다시 같은 실수를 반복하지 않는 전략을 세워야 합니다.
또한 재창업 과정에서 보수적인 자금 계획과 현실적인 매출 목표를 세우고,
멘토링이나 지원 프로그램을 통해 외부 시각을 적극적으로 활용하는 것이 좋습니다.
""",
    8: """
사업자등록을 하지 않은 예비창업자도 지원받을 수 있는 교육 및 프로그램이 많이 있습니다.
정부와 지자체, 창업지원기관에서는 예비창업자를 대상으로 아이디어 구체화, 시장조사, 사업계획서 작성 등을 돕는
창업교육, 멘토링, 창업캠프 등을 운영합니다.
또한 일부 사업화 지원사업에서도 예비창업자를 대상으로 사업계획서 평가 후 자금을 지원하기도 합니다.
"""
}


# =========================
# 일반 LLM (RAG 없이)
# =========================
def get_normal_llm_answer(question: str) -> str:
    """RAG 없이 일반 LLM만 사용한 답변"""
    
    normal_prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 창업 관련 질문에 답변하는 AI 어시스턴트입니다.
일반적인 지식을 바탕으로 답변해주세요.
"""),
        ("human", "{question}")
    ])
    
    chain = normal_prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question})
    
    return answer


# =========================
# RAG 시스템 답변
# =========================
def get_rag_answer(question: str) -> dict:
    """RAG 시스템을 사용한 답변"""
    
    try:
        # Query Transformation
        transformed_q = qt_chain.invoke({"question": question})
        
        # 프롬프트 선택
        prompt = choose_prompt(question)
        
        # 문서 검색
        docs = retriever.invoke(transformed_q)
        context = format_docs(docs)
        
        # 답변 생성
        answer_chain = prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({
            "context": context,
            "question": question
        })
        
        # 참고 문서 정보
        sources = []
        for doc in docs:
            data_type = doc.metadata.get("data_type", "unknown")
            sources.append(data_type)
        
        return {
            "answer": answer,
            "sources": list(set(sources)),
            "num_docs": len(docs)
        }
        
    except Exception as e:
        return {
            "answer": f"오류 발생: {str(e)}",
            "sources": [],
            "num_docs": 0
        }


# =========================
# LLM Judge를 활용한 사실성 평가
# =========================
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 두 개의 답변을 비교하여 사실성·정확성을 평가하는 채점관입니다.
- 기준 답변(reference)과 실제 답변(answer)을 비교하세요.
- 법령, 연도, 기간, 금액, 조건 등 핵심 정보가 기준 답변과 얼마나 일치하는지를 보세요.
- 0.0에서 1.0 사이의 점수를 숫자만 출력하세요.
- 1.0은 거의 완전히 정확, 0.0은 대부분 틀린 답변입니다.
"""),
    ("human", """
[기준 답변]
{reference}

[실제 답변]
{answer}

위 두 답변을 비교했을 때, 사실성과 정확성 측면에서 몇 점을 줄 수 있을지
0.0에서 1.0 사이 숫자만 한 줄로 출력하세요.
""")
])

def get_fact_score(answer: str, reference_answer: str) -> float:
    """LLM Judge를 이용해 사실성 점수(0~1)를 계산"""
    try:
        chain = judge_prompt | llm | StrOutputParser()
        raw = chain.invoke({
            "reference": reference_answer,
            "answer": answer
        }).strip()

        # "0.85", "0.85점", "0.85 / 1.0" 같이 나올 수 있으니 숫자만 뽑기
        import re
        m = re.search(r"0\.\d+|1\.0|1|0", raw)
        if m:
            score = float(m.group())
            # 안전하게 0~1 클램핑
            score = max(0.0, min(1.0, score))
            return score
        return 0.0
    except Exception:
        return 0.0



# =========================
# 평가 지표 계산
# =========================
def evaluate_answer(answer: str, expected_keywords: list, fact_score: float = 0.0) -> dict:
    """
    답변 품질 평가
    
    지표:
    1. 키워드 매칭률 (Keyword Match Rate) - 40%
    2. 구체성 점수 (Specificity Score) - 20%
    3. 사실성 점수 (Fact Accuracy Score, LLM Judge 기반) - 40%
    """
    
    answer_lower = answer.lower()
    
    # 1. 키워드 매칭률
    matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    keyword_match_rate = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0
    
    # 2. 답변 길이
    answer_length = len(answer)
    
    # 3. 구체성 점수 (숫자, 날짜, 고유명사 등이 포함되었는지)
    specificity_indicators = [
        "년", "월", "일", "만",  # 날짜/기간
        "원", "억", "%",  # 금액/비율
        "법", "조", "항",  # 법령
        "센터", "공단", "진흥원",  # 기관명
    ]
    specificity_count = sum(1 for ind in specificity_indicators if ind in answer)
    specificity_score = min(specificity_count / 5, 1.0)  # 최대 1.0
    
    # 4. 사실성 점수 (없으면 0으로 처리)
    if fact_score is None:
        fact_score = 0.0
    
    # 가중치 설정 (맘에 안 들면 숫자만 바꿔도 됨)
    w_keyword = 0.4
    w_specific = 0.2
    w_fact = 0.4
    
    total_score = (
        keyword_match_rate * w_keyword +
        specificity_score * w_specific +
        fact_score * w_fact
    )

    return {
        "keyword_match_rate": keyword_match_rate,
        "matched_keywords": matched_keywords,
        "answer_length": answer_length,
        "specificity_score": specificity_score,
        "fact_score": fact_score,
        "total_score": total_score
    }


# =========================
# 비교 평가 실행
# =========================
def run_comparison():
    """RAG vs 일반 LLM 비교 평가 실행"""
    
    print("=" * 80)
    print("🎯 RAG 시스템 vs 일반 LLM 비교 평가")
    print("=" * 80)
    print(f"\n📅 평가 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 평가 질문 수: {len(TEST_QUESTIONS)}개\n")
    
    results = []
    
    for idx, test_case in enumerate(TEST_QUESTIONS, 1):
        question = test_case["question"]
        category = test_case["category"]
        expected_keywords = test_case["expected_keywords"]
        
        print(f"\n{'='*80}")
        print(f"[질문 {idx}/{len(TEST_QUESTIONS)}] {question}")
        print(f"카테고리: {category}")
        print(f"기대 키워드: {', '.join(expected_keywords)}")
        print(f"{'='*80}")
        

        #1. 기준 정답 가져오기
        ref_answer = REFERENCE_ANSWERS.get(test_case["id"], "")
        # 1. 일반 LLM 답변
        print("\n일반 LLM 답변 중...")
        normal_answer = get_normal_llm_answer(question)
        normal_fact_score = get_fact_score(normal_answer, ref_answer) if ref_answer else 0.0
        normal_eval = evaluate_answer(normal_answer, expected_keywords, fact_score=normal_fact_score)
        
        print(f"\n[일반 LLM 답변]")
        print(normal_answer[:300] + "..." if len(normal_answer) > 300 else normal_answer)
        print(f"\n평가:")
        print(f"  - 키워드 매칭률: {normal_eval['keyword_match_rate']:.1%}")
        print(f"  - 매칭된 키워드: {', '.join(normal_eval['matched_keywords']) if normal_eval['matched_keywords'] else '없음'}")
        print(f"  - 구체성 점수: {normal_eval['specificity_score']:.2f}")
        print(f"  - 사실성 점수: {normal_eval['fact_score']:.2f}")
        print(f"  - 총점: {normal_eval['total_score']:.2f}")
        # 2. RAG 시스템 답변
        print("\nRAG 시스템 답변 중...")
        rag_result = get_rag_answer(question)
        rag_answer = rag_result["answer"]
        rag_fact_score = get_fact_score(rag_answer, ref_answer) if ref_answer else 0.0
        rag_eval = evaluate_answer(rag_answer, expected_keywords, fact_score=rag_fact_score)
        
        print(f"\n[RAG 시스템 답변]")
        print(rag_answer[:300] + "..." if len(rag_answer) > 300 else rag_answer)
        print(f"\n평가:")
        print(f"  - 키워드 매칭률: {rag_eval['keyword_match_rate']:.1%}")
        print(f"  - 매칭된 키워드: {', '.join(rag_eval['matched_keywords']) if rag_eval['matched_keywords'] else '없음'}")
        print(f"  - 구체성 점수: {rag_eval['specificity_score']:.2f}")
        print(f"  - 사실성 점수: {rag_eval['fact_score']:.2f}")
        print(f"  - 총점: {rag_eval['total_score']:.2f}")
        print(f"  - 참조 문서 수: {rag_result['num_docs']}개")
        print(f"  - 참조 문서 유형: {', '.join(rag_result['sources'])}")
        
        
        # 3. 비교
        score_diff = rag_eval['total_score'] - normal_eval['total_score']
        winner = "RAG 🏆" if score_diff > 0 else "일반 LLM 🏆" if score_diff < 0 else "동점"
        
        print(f"\n🎯 승자: {winner} (점수 차: {score_diff:+.2f})")
        
        # 결과 저장
        results.append({
            "question_id": test_case["id"],
            "question": question,
            "category": category,
            "expected_keywords": expected_keywords,
            "normal_llm": {
                "answer": normal_answer,
                "evaluation": normal_eval
            },
            "rag_system": {
                "answer": rag_answer,
                "evaluation": rag_eval,
                "sources": rag_result["sources"],
                "num_docs": rag_result["num_docs"]
            },
            "score_difference": score_diff,
            "winner": winner
        })
    
    # =========================
    # 최종 통계
    # =========================
    print("\n\n" + "=" * 80)
    print("📊 최종 통계")
    print("=" * 80)
    
    normal_scores = [r["normal_llm"]["evaluation"]["total_score"] for r in results]
    rag_scores = [r["rag_system"]["evaluation"]["total_score"] for r in results]
    
    normal_avg = sum(normal_scores) / len(normal_scores)
    rag_avg = sum(rag_scores) / len(rag_scores)
    
    rag_wins = sum(1 for r in results if "RAG" in r["winner"])
    normal_wins = sum(1 for r in results if "일반 LLM" in r["winner"])
    ties = sum(1 for r in results if "동점" in r["winner"])
    
    print(f"\n🤖 일반 LLM")
    print(f"  - 평균 점수: {normal_avg:.3f}")
    print(f"  - 승리 횟수: {normal_wins}회")
    
    print(f"\n🔍 RAG 시스템")
    print(f"  - 평균 점수: {rag_avg:.3f}")
    print(f"  - 승리 횟수: {rag_wins}회")
    
    print(f"\n🏆 최종 결과")
    print(f"  - RAG 승률: {rag_wins}/{len(TEST_QUESTIONS)} ({rag_wins/len(TEST_QUESTIONS)*100:.1f}%)")
    print(f"  - 평균 점수 향상: {(rag_avg - normal_avg):+.3f} ({(rag_avg - normal_avg)/normal_avg*100:+.1f}%)")
    print(f"  - 동점: {ties}회")
    
    # 카테고리별 분석
    print(f"\n📂 카테고리별 성능")
    categories = set(r["category"] for r in results)
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_rag_wins = sum(1 for r in cat_results if "RAG" in r["winner"])
        print(f"  - {cat}: RAG {cat_rag_wins}/{len(cat_results)}승")
    
    # =========================
    # 결과 저장
    # =========================
    output_file = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    summary = {
        "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_questions": len(TEST_QUESTIONS),
        "statistics": {
            "normal_llm": {
                "average_score": normal_avg,
                "wins": normal_wins
            },
            "rag_system": {
                "average_score": rag_avg,
                "wins": rag_wins
            },
            "ties": ties,
            "improvement": {
                "absolute": rag_avg - normal_avg,
                "percentage": (rag_avg - normal_avg) / normal_avg * 100
            }
        },
        "detailed_results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # HTML 리포트 생성
    html_file = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    generate_html_report(html_file, summary, results)
    
    print(f"\n💾 JSON 결과 저장: {output_file}")
    print(f"📊 HTML 리포트 저장: {html_file}")
    print(f"\n✅ 평가 완료!")
    
    return summary


# =========================
# HTML 리포트 생성
# =========================
def generate_html_report(filename: str, summary: dict, results: list):
    """평가 결과를 HTML 리포트로 생성"""
    
    stats = summary["statistics"]
    normal_avg = stats["normal_llm"]["average_score"]
    rag_avg = stats["rag_system"]["average_score"]
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG 성능 평가 리포트</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                color: #333;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 10px; 
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                padding: 40px;
            }}
            h1 {{ 
                color: #667eea; 
                margin-bottom: 10px; 
                text-align: center;
            }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .eval-time {{ color: #666; font-size: 14px; }}
            
            .stats-container {{ 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 20px; 
                margin-bottom: 40px;
            }}
            .stat-card {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            .stat-title {{ font-size: 14px; opacity: 0.9; margin-bottom: 10px; }}
            .stat-value {{ font-size: 32px; font-weight: bold; }}
            .stat-desc {{ font-size: 12px; opacity: 0.8; margin-top: 8px; }}
            
            .winner {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
            .comparison {{ 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 40px;
            }}
            .comparison h3 {{ color: #667eea; margin-bottom: 15px; }}
            
            table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin-top: 20px;
            }}
            th {{ 
                background: #667eea; 
                color: white; 
                padding: 12px; 
                text-align: left; 
                font-weight: 600;
            }}
            td {{ 
                padding: 12px; 
                border-bottom: 1px solid #e0e0e0;
            }}
            tr:hover {{ background: #f5f5f5; }}
            
            .question-section {{ 
                margin-bottom: 30px; 
                padding: 20px; 
                background: #f9f9f9; 
                border-left: 4px solid #667eea;
                border-radius: 5px;
            }}
            .question-text {{ 
                font-size: 16px; 
                font-weight: 600; 
                color: #333; 
                margin-bottom: 15px;
            }}
            .answer-box {{ 
                background: white; 
                padding: 15px; 
                border-radius: 5px; 
                margin-bottom: 10px;
            }}
            .answer-title {{ 
                font-weight: 600; 
                color: #667eea; 
                margin-bottom: 8px; 
                font-size: 14px;
            }}
            .answer-text {{ 
                font-size: 13px; 
                line-height: 1.6; 
                color: #555;
                max-height: 150px;
                overflow: hidden;
            }}
            
            .score-bar {{ 
                width: 100%; 
                height: 20px; 
                background: #e0e0e0; 
                border-radius: 10px; 
                overflow: hidden;
                margin-top: 5px;
            }}
            .score-fill {{ 
                height: 100%; 
                background: linear-gradient(90deg, #667eea, #764ba2); 
                transition: width 0.3s ease;
            }}
            .score-text {{ 
                font-size: 12px; 
                font-weight: 600; 
                color: #333; 
                margin-top: 5px;
            }}
            
            .footer {{ 
                text-align: center; 
                margin-top: 40px; 
                padding-top: 20px; 
                border-top: 1px solid #e0e0e0; 
                color: #666; 
                font-size: 12px;
            }}
            
            .winner-badge {{ 
                display: inline-block; 
                padding: 4px 8px; 
                background: #4CAF50; 
                color: white; 
                border-radius: 4px; 
                font-size: 12px; 
                font-weight: 600;
            }}
            .loser-badge {{ 
                display: inline-block; 
                padding: 4px 8px; 
                background: #ff6b6b; 
                color: white; 
                border-radius: 4px; 
                font-size: 12px; 
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 RAG 시스템 성능 평가 리포트</h1>
                <p class="eval-time">평가 시간: {summary['evaluation_time']}</p>
            </div>
            
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-title">일반 LLM 평균 점수</div>
                    <div class="stat-value">{normal_avg:.1%}</div>
                    <div class="stat-desc">승리: {stats['normal_llm']['wins']}/{summary['total_questions']}회</div>
                </div>
                <div class="stat-card winner">
                    <div class="stat-title">RAG 시스템 평균 점수</div>
                    <div class="stat-value">{rag_avg:.1%}</div>
                    <div class="stat-desc">승리: {stats['rag_system']['wins']}/{summary['total_questions']}회</div>
                </div>
            </div>
            
            <div class="comparison">
                <h3>성능 비교</h3>
                <table>
                    <tr>
                        <th>지표</th>
                        <th>일반 LLM</th>
                        <th>RAG 시스템</th>
                        <th>차이</th>
                    </tr>
                    <tr>
                        <td>평균 점수</td>
                        <td><strong>{normal_avg:.3f}</strong></td>
                        <td><strong>{rag_avg:.3f}</strong></td>
                        <td><span style="color: {'green' if stats['improvement']['absolute'] > 0 else 'red'}">{stats['improvement']['absolute']:+.3f} ({stats['improvement']['percentage']:+.1f}%)</span></td>
                    </tr>
                    <tr>
                        <td>승리 횟수</td>
                        <td><strong>{stats['normal_llm']['wins']}</strong></td>
                        <td><strong>{stats['rag_system']['wins']}</strong></td>
                        <td>동점: {stats['ties']}</td>
                    </tr>
                </table>
            </div>
            
            <h2 style="color: #667eea; margin-bottom: 20px;">질문별 상세 분석</h2>
    """
    
    for idx, result in enumerate(results, 1):
        winner_class = "winner-badge" if "RAG" in result["winner"] else "loser-badge"
        winner_text = "RAG 승리" if "RAG" in result["winner"] else "LLM 승리"
        
        normal_score = result["normal_llm"]["evaluation"]["total_score"]
        rag_score = result["rag_system"]["evaluation"]["total_score"]
        
        html_content += f"""
            <div class="question-section">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div>
                        <div class="question-text">[질문 {idx}] {result['question'][:80]}...</div>
                        <div style="font-size: 12px; color: #666;">카테고리: {result['category']}</div>
                    </div>
                    <span class="{winner_class}">{winner_text}</span>
                </div>
                
                <table style="font-size: 13px;">
                    <tr>
                        <th style="background: #667eea;">항목</th>
                        <th style="background: #667eea;">일반 LLM</th>
                        <th style="background: #667eea;">RAG 시스템</th>
                    </tr>
                    <tr>
                        <td><strong>총점</strong></td>
                        <td>
                            <div class="score-bar">
                                <div class="score-fill" style="width: {normal_score*100}%"></div>
                            </div>
                            <div class="score-text">{normal_score:.2f}</div>
                        </td>
                        <td>
                            <div class="score-bar">
                                <div class="score-fill" style="width: {rag_score*100}%"></div>
                            </div>
                            <div class="score-text">{rag_score:.2f}</div>
                        </td>
                    </tr>
                    <tr>
                        <td>키워드 매칭률</td>
                        <td>{result['normal_llm']['evaluation']['keyword_match_rate']:.1%}</td>
                        <td>{result['rag_system']['evaluation']['keyword_match_rate']:.1%}</td>
                    </tr>
                    <tr>
                        <td>구체성 점수</td>
                        <td>{result['normal_llm']['evaluation']['specificity_score']:.2f}</td>
                        <td>{result['rag_system']['evaluation']['specificity_score']:.2f}</td>
                    </tr>
                    <tr>
                        <td>사실성 점수</td>
                        <td>{result['normal_llm']['evaluation']['fact_score']:.2f}</td>
                        <td>{result['rag_system']['evaluation']['fact_score']:.2f}</td>
                    </tr>
                </table>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <div class="answer-box">
                        <div class="answer-title">일반 LLM 답변</div>
                        <div class="answer-text">{result['normal_llm']['answer'][:200]}...</div>
                    </div>
                    <div class="answer-box">
                        <div class="answer-title">RAG 답변</div>
                        <div class="answer-text">{result['rag_system']['answer'][:200]}...</div>
                    </div>
                </div>
            </div>
        """
    
    html_content += """
            <div class="footer">
                <p>이 리포트는 자동으로 생성되었습니다.</p>
                <p>JSON 파일을 통해 더 자세한 정보를 확인할 수 있습니다.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)


# =========================
# 메인 실행
# =========================
if __name__ == "__main__":
    try:
        results = run_comparison()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()