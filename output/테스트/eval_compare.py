"""
🎯 RAG 시스템 통합 평가 스크립트 (All-in-One)

사용법:
    python rag_evaluator.py

결과:
    - rag_evaluation_report.html 생성
    - 브라우저에서 바로 볼 수 있는 리포트
"""

import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

warnings.filterwarnings("ignore")

# ============================================================================
# 1. 평가 데이터 클래스
# ============================================================================

@dataclass
class TestQuestion:
    id: int
    question: str
    category: str
    expected_keywords: List[str]
    should_use_db: bool
    difficulty: str

@dataclass
class EvalResult:
    question_id: int
    question: str
    answer: str
    execution_time: float
    keyword_score: float
    llm_score: float
    combined_score: float
    answer_length: int
    error: str = ""


# ============================================================================
# 2. 테스트 질문 데이터셋
# ============================================================================

TEST_QUESTIONS = [
     TestQuestion(1, "서울에서 AI 스타트업 창업하는데 지원받을 수 있는 사업 알려줘",
                   "announcement", ["서울", "AI", "지원"], True, "easy"),
    
    TestQuestion(2,"중소기업창업 지원법에서 창업자의 정의가 뭐야?",
                "law", ["창업자", "정의", "법"], True, "easy"),

    TestQuestion(3 , "AI와 블록체인을 결합한 핀테크 스타트업을 서울에서 창업하려는데, 정부 지원금과 엑셀러레이터 프로그램 동시에 받을 수 있나요?",
                 "cases", ["핀테크", "지원금", "프로그램"], True, "easy"),
    
    TestQuestion(4, " 서울 스타트업 창업 지원 사업",
                "announcement", ["서울", "스타트업", "사업"], True, "easy"),
    
    TestQuestion(5, "서울날씨 알려줘",
                "general", [], True, "easy"),
]
# ============================================================================
# 3. RAG 시스템 로더
# ============================================================================

def load_rag_systems():
    """세 가지 RAG 시스템 로드"""
    systems = {}
    
    # rag_base
    try:
        sys.path.insert(0, '.')
        import rag_base
        
        def base_wrapper(question):
            transformed_q = rag_base.qt_chain.invoke({"question": question})
            prompt = rag_base.choose_prompt(question)
            docs = rag_base.retriever.invoke(transformed_q)
            context = rag_base.format_docs(docs)
            answer_chain = prompt | rag_base.llm | rag_base.StrOutputParser()
            return answer_chain.invoke({"context": context, "question": question})
        
        systems['rag_base'] = base_wrapper
        print("✅ rag_base.py 로드 완료")
    except Exception as e:
        print(f"❌ rag_base.py 로드 실패: {e}")
    
    # rag_base_multi
    try:
        import rag_base_multi
        
        def multi_wrapper(question):
            result = rag_base_multi.multi_query_rag_with_qt(question)
            # tuple 반환 처리
            if isinstance(result, tuple):
                return result[0]
            return result
        
        systems['rag_base_multi'] = multi_wrapper
        print("✅ rag_base_multi.py 로드 완료")
    except Exception as e:
        print(f"❌ rag_base_multi.py 로드 실패: {e}")
    
    # rag_base_multi_web
    try:
        import rag_base_multi_web
        
        def web_wrapper(question):
            return rag_base_multi_web.multi_query_rag_with_qt(question)
        
        systems['rag_base_multi_web.py'] = web_wrapper
        print("✅ rag_base_multi_web.py 로드 완료")
    except Exception as e:
        print(f"❌ rag_base_multi_web.py 로드 실패: {e}")
    
    # try:
    #     import rag_base_multi_web_llm
    
    #     def llm_web_wrapper(question):
    #         return rag_base_multi_web_llm.multi_query_rag_with_qt(question)
    
    #     systems['rag_base_multi_web_llm'] = llm_web_wrapper
    #     print("✅ rag_base_multi_web_llm.py 로드 완료")
    # except Exception as e:
    #     print(f"❌ rag_base_multi_web_llm.py 로드 실패: {e}")
    
    try:
        import rag_base_multi_web_cache
        
        def cache_wrapper(question):
            return rag_base_multi_web_cache.multi_query_rag_with_cache(question)
        
        systems['rag_base_multi_web_cache'] = cache_wrapper
        print("✅rag_base_multi_web_cache.py 로드 완료")
    except Exception as e:
        print(f"❌ rag_base_multi_web_cache.py 로드 실패: {e}")   

    return systems


# ============================================================================
# 4. 평가 함수
# ============================================================================

def calculate_keyword_score(answer: str, keywords: List[str]) -> float:
    """키워드 매칭 점수 계산"""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return matched / len(keywords)


def calculate_llm_score(question: str, answer: str) -> float:
    """LLM 기반 답변 품질 평가 (업계 표준 방식)"""
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt = f"""다음 질문에 대한 답변을 0-10점으로 평가하세요.

질문: {question}

답변: {answer}

평가 기준:
1. 관련성 (질문과 얼마나 관련있는가?)
2. 정확성 (정보가 정확한가?)
3. 완전성 (질문에 충분히 답했는가?)
4. 유용성 (실제로 도움이 되는가?)

점수만 숫자로 응답하세요. (예: 8.5)"""

        response = llm.invoke(prompt)
        score = float(response.content.strip())
        return min(max(score / 10.0, 0.0), 1.0)  # 0-1 범위로 정규화
        
    except Exception as e:
        print(f"LLM 평가 오류: {e}")
        return 0.5  # 오류 시 중간값 반환


def evaluate_question(rag_function, question: TestQuestion) -> EvalResult:
    """단일 질문 평가 - 더 엄격한 평가 기준 적용"""
    try:
        # 빈 질문 체크
        if not question.question.strip():
            return EvalResult(
                question_id=question.id,
                question=question.question,
                answer="",
                execution_time=0.0,
                keyword_score=0.0,
                llm_score=0.0,
                combined_score=0.0,
                answer_length=0,
                error="Empty question provided"
            )
        
        start = time.time()
        answer = rag_function(question.question)
        elapsed = time.time() - start
        
        # 타임아웃 체크 (30초 초과)
        if elapsed > 30:
            return EvalResult(
                question_id=question.id,
                question=question.question,
                answer="",
                execution_time=elapsed,
                keyword_score=0.0,
                llm_score=0.0,
                combined_score=0.0,
                answer_length=0,
                error="Execution timeout (>30s)"
            )
        
        # tuple 반환 처리
        if isinstance(answer, tuple):
            answer = answer[0] if answer else ""
        
        # 답변이 None이거나 빈 문자열인 경우
        if answer is None or not str(answer).strip():
            return EvalResult(
                question_id=question.id,
                question=question.question,
                answer="",
                execution_time=elapsed,
                keyword_score=0.0,
                llm_score=0.0,
                combined_score=0.0,
                answer_length=0,
                error="Empty or null response"
            )
        
        answer = str(answer).strip()
        
        # 너무 짧은 답변 체크 (10자 미만)
        if len(answer) < 10:
            return EvalResult(
                question_id=question.id,
                question=question.question,
                answer=answer,
                execution_time=elapsed,
                keyword_score=0.0,
                llm_score=0.0,
                combined_score=0.0,
                answer_length=len(answer),
                error="Response too short (<10 chars)"
            )
        
        # 오류 메시지가 포함된 답변 체크
        error_indicators = [
            "오류가 발생했습니다",
            "error occurred", 
            "failed to",
            "exception",
            "traceback",
            "시스템 오류"
        ]
        
        answer_lower = answer.lower()
        for indicator in error_indicators:
            if indicator in answer_lower:
                return EvalResult(
                    question_id=question.id,
                    question=question.question,
                    answer=answer,
                    execution_time=elapsed,
                    keyword_score=0.0,
                    llm_score=0.0,
                    combined_score=0.0,
                    answer_length=len(answer),
                    error=f"Error message detected: {indicator}"
                )
        
        keyword_score = calculate_keyword_score(answer, question.expected_keywords)
        llm_score = calculate_llm_score(question.question, answer)
        combined_score = (keyword_score * 0.3) + (llm_score * 0.7)  # LLM 점수에 더 높은 가중치
        
        return EvalResult(
            question_id=question.id,
            question=question.question,
            answer=answer,
            execution_time=elapsed,
            keyword_score=keyword_score,
            llm_score=llm_score,
            combined_score=combined_score,
            answer_length=len(answer)
        )
        
    except Exception as e:
        return EvalResult(
            question_id=question.id,
            question=question.question,
            answer="",
            execution_time=0.0,
            keyword_score=0.0,
            llm_score=0.0,
            combined_score=0.0,
            answer_length=0,
            error=f"Exception: {str(e)[:100]}"  # 오류 메시지 길이 제한
        )


def evaluate_system(rag_function, system_name: str, questions: List[TestQuestion]) -> Dict:
    """전체 시스템 평가"""
    print(f"\n{'='*80}")
    print(f"🔍 [{system_name}] 평가 시작 - {len(questions)}개 질문")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Q{q.id}: {q.question[:40]}...")
        result = evaluate_question(rag_function, q)
        results.append(result)
        
        status = "✅" if not result.error else "❌"
        print(f"  {status} {result.execution_time:.2f}초 | 키워드: {result.keyword_score:.0%} | LLM: {result.llm_score:.0%} | 종합: {result.combined_score:.0%} | {result.answer_length}자")
        if result.error:
            print(f"     오류: {result.error}")
        time.sleep(0.3)  # API 제한 고려
    
    # 통계 계산
    total = len(results)
    avg_time = sum(r.execution_time for r in results) / total
    avg_keyword_score = sum(r.keyword_score for r in results) / total
    avg_llm_score = sum(r.llm_score for r in results) / total
    avg_combined_score = sum(r.combined_score for r in results) / total
    error_rate = sum(1 for r in results if r.error) / total
    
    # 카테고리별 점수
    category_scores = {}
    for cat in set(q.category for q in questions):
        cat_results = [results[i] for i, q in enumerate(questions) if q.category == cat]
        cat_keyword_score = sum(r.keyword_score for r in cat_results) / len(cat_results)
        cat_llm_score = sum(r.llm_score for r in cat_results) / len(cat_results)
        cat_combined_score = sum(r.combined_score for r in cat_results) / len(cat_results)
        category_scores[cat] = {
            'keyword': cat_keyword_score,
            'llm': cat_llm_score,
            'combined': cat_combined_score
        }
    
    return {
        'system_name': system_name,
        'results': results,
        'stats': {
            'avg_time': avg_time,
            'avg_keyword_score': avg_keyword_score,
            'avg_llm_score': avg_llm_score,
            'avg_combined_score': avg_combined_score,
            'error_rate': error_rate,
            'category_scores': category_scores
        }
    }


# ============================================================================
# 5. HTML 리포트 생성
# ============================================================================

def generate_html_report(all_results: List[Dict], questions: List[TestQuestion], output_file: str):
    """HTML 리포트 생성"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 시스템 평가 리포트</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .summary-card h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        .stat {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }}
        
        .stat:last-child {{
            border-bottom: none;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .stat-value {{
            font-weight: bold;
            color: #333;
        }}
        
        .winner {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white !important;
        }}
        
        .winner h3 {{
            color: white !important;
        }}
        
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .comparison-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .comparison-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .comparison-table tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        
        .comparison-table tr:hover {{
            background: #f0f0f0;
        }}
        
        .best-score {{
            background: #4CAF50;
            color: white;
            padding: 3px 8px;
            border-radius: 5px;
            font-weight: bold;
        }}
        
        .worst-score {{
            background: #f44336;
            color: white;
            padding: 3px 8px;
            border-radius: 5px;
        }}
        
        .details {{
            margin-top: 40px;
        }}
        
        .question-card {{
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .question-card h4 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .answer-section {{
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 5px;
        }}
        
        .answer-section h5 {{
            color: #764ba2;
            margin-bottom: 8px;
        }}
        
        .answer-text {{
            color: #555;
            white-space: pre-wrap;
            max-height: 150px;
            overflow-y: auto;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        
        .error {{
            color: #f44336;
            font-weight: bold;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: bold;
            margin: 0 5px;
        }}
        
        .badge-easy {{
            background: #4CAF50;
            color: white;
        }}
        
        .badge-medium {{
            background: #FF9800;
            color: white;
        }}
        
        .badge-hard {{
            background: #f44336;
            color: white;
        }}
        
        .category-badge {{
            background: #2196F3;
            color: white;
        }}
        
        .footer {{
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 10px;
        }}
        
        .bar {{
            height: 30px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 5px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 RAG 시스템 평가 리포트</h1>
            <p>생성 시간: {timestamp}</p>
        </div>
        
        <div class="content">
"""
    
    # 시스템별 요약 카드
    html += '<div class="summary">\n'
    
    # 최고 점수 찾기
    best_system = max(all_results, key=lambda x: x['stats']['avg_combined_score'])
    
    for eval_data in all_results:
        name = eval_data['system_name']
        stats = eval_data['stats']
        
        winner_class = ' winner' if name == best_system['system_name'] else ''
        
        html += f"""
            <div class="summary-card{winner_class}">
                <h3>{'🏆 ' if winner_class else ''}{name}</h3>
                <div class="stat">
                    <span class="stat-label">종합 점수</span>
                    <span class="stat-value">{stats['avg_combined_score']:.1%}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">키워드 점수</span>
                    <span class="stat-value">{stats['avg_keyword_score']:.1%}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">LLM 점수</span>
                    <span class="stat-value">{stats['avg_llm_score']:.1%}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">평균 실행 시간</span>
                    <span class="stat-value">{stats['avg_time']:.2f}초</span>
                </div>
                <div class="stat">
                    <span class="stat-label">오류율</span>
                    <span class="stat-value">{stats['error_rate']:.1%}</span>
                </div>
            </div>
"""
    
    html += '</div>\n'
    
    # 비교 테이블
    html += """
        <h2>📊 상세 비교</h2>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>시스템</th>
                    <th>종합 점수</th>
                    <th>키워드 점수</th>
                    <th>LLM 점수</th>
                    <th>실행 시간</th>
                    <th>오류율</th>
                    <th>성공 질문</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for eval_data in all_results:
        name = eval_data['system_name']
        stats = eval_data['stats']
        success = sum(1 for r in eval_data['results'] if not r.error)
        total = len(eval_data['results'])
        
        score_class = 'best-score' if eval_data == best_system else ''
        
        html += f"""
                <tr>
                    <td><strong>{name}</strong></td>
                    <td><span class="{score_class}">{stats['avg_combined_score']:.1%}</span></td>
                    <td>{stats['avg_keyword_score']:.1%}</td>
                    <td>{stats['avg_llm_score']:.1%}</td>
                    <td>{stats['avg_time']:.2f}초</td>
                    <td>{stats['error_rate']:.1%}</td>
                    <td>{success}/{total}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
"""
    
    # 카테고리별 성능
    html += '<h2>📂 카테고리별 성능</h2>\n<div class="chart-container">\n'
    
    all_categories = set()
    for eval_data in all_results:
        all_categories.update(eval_data['stats']['category_scores'].keys())
    
    for category in sorted(all_categories):
        html += f'<h4>{category}</h4>\n'
        for eval_data in all_results:
            if isinstance(eval_data['stats']['category_scores'].get(category), dict):
                score = eval_data['stats']['category_scores'][category]['combined']
            else:
                score = eval_data['stats']['category_scores'].get(category, 0)
            width = int(score * 100)
            html += f'<div class="bar" style="width: {width}%">{eval_data["system_name"]}: {score:.0%}</div>\n'
    
    html += '</div>\n'
    
    # 질문별 상세 결과
    html += '<div class="details">\n<h2>📝 질문별 상세 결과</h2>\n'
    
    for i, q in enumerate(questions):
        difficulty_badge = f'badge-{q.difficulty}'
        html += f"""
            <div class="question-card">
                <h4>Q{q.id}. {q.question}</h4>
                <div>
                    <span class="badge {difficulty_badge}">{q.difficulty}</span>
                    <span class="badge category-badge">{q.category}</span>
                </div>
"""
        
        for eval_data in all_results:
            result = eval_data['results'][i]
            
            if result.error:
                html += f"""
                <div class="answer-section">
                    <h5>{eval_data['system_name']} <span class="error">❌ 오류</span></h5>
                    <p class="error">{result.error}</p>
                </div>
"""
            else:
                html += f"""
                <div class="answer-section">
                    <h5>{eval_data['system_name']} 
                        <span style="color: #4CAF50;">종합: {result.combined_score:.0%}</span>
                        <span style="color: #2196F3;">키워드: {result.keyword_score:.0%}</span>
                        <span style="color: #FF9800;">LLM: {result.llm_score:.0%}</span>
                        <span style="color: #666;">| {result.execution_time:.2f}초 | {result.answer_length}자</span>
                    </h5>
                    <div class="answer-text">{result.answer[:500]}{'...' if len(result.answer) > 500 else ''}</div>
                </div>
"""
        
        html += '</div>\n'
    
    html += '</div>\n'
    
    # Footer
    html += f"""
        </div>
        <div class="footer">
            <p>RAG 시스템 평가 리포트 | 생성 시간: {timestamp}</p>
            <p>총 {len(questions)}개 질문 평가 | {len(all_results)}개 시스템 비교</p>
        </div>
    </div>
</body>
</html>
"""
    
    # 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ HTML 리포트 생성 완료: {output_file}")


# ============================================================================
# 6. 메인 실행
# ============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        🎯 RAG 시스템 통합 평가 (All-in-One)                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 환경 변수 체크
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return
    
    # RAG 시스템 로드
    print("\n📦 RAG 시스템 로드 중...\n")
    rag_systems = load_rag_systems()
    
    if not rag_systems:
        print("\n❌ 로드된 RAG 시스템이 없습니다.")
        return
    
    print(f"\n✅ {len(rag_systems)}개 시스템 로드 완료")
    print(f"📋 {len(TEST_QUESTIONS)}개 질문으로 평가 시작\n")
    
    # 평가 실행
    all_results = []
    
    for name, rag_func in rag_systems.items():
        try:
            result = evaluate_system(rag_func, name, TEST_QUESTIONS)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ {name} 평가 중 오류: {e}\n")
    
    if not all_results:
        print("\n❌ 평가 결과가 없습니다.")
        return
    
    # HTML 리포트 생성
    print("\n" + "="*80)
    print("📄 HTML 리포트 생성 중...")
    print("="*80)
    
    output_file = "rag_evaluation_report.html"
    generate_html_report(all_results, TEST_QUESTIONS, output_file)
    
    # 간단한 요약 출력
    print("\n" + "="*80)
    print("🏆 평가 완료!")
    print("="*80)
    
    best = max(all_results, key=lambda x: x['stats']['avg_combined_score'])
    fastest = min(all_results, key=lambda x: x['stats']['avg_time'])
    
    print(f"\n🥇 최고 정확도: {best['system_name']} ({best['stats']['avg_combined_score']:.1%})")
    print(f"⚡ 최고 속도: {fastest['system_name']} ({fastest['stats']['avg_time']:.2f}초)")
    
    print(f"\n📊 상세 결과는 {output_file} 파일을 확인하세요!")
    print(f"   브라우저에서 열기: file://{os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
