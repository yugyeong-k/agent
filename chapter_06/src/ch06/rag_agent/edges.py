from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 평가(grading)에 사용하는 LLM
llm = ChatOpenAI(model="gpt-4o")

# 문서 관련성 평가 결과를 구조화된 출력으로 받기 위한 스키마
class Grade(BaseModel):
    """ 관련성 확인을 위한 점수 스키마""" 
    binary_score: str = Field(description="문서가 질문과 관련이 있는지 여부, 'yes' 또는 'no")
    
# 조건부 엣지: 검색 문서의 관련성에 따라 다음 노드("generate" / "transform_query")를 결정
def decide_to_generate(state):
    """
    답변을 생성할지, 아니면 질문을 다시 생성할지 결정합니다.
    """
    
    print("--- ASSESS GRADED DOCUMENTS ---")
    # 재시도 3회 이상이면 무한 루프 방지를 위해 평가 없이 바로 답변 생성
    if state.get("retry_num", 0) >=3 :
        return "generate"

    # LLM 출력이 Grade 스키마(yes/no)를 따르도록 설정
    grader = llm.with_structured_output(Grade)
    grader_prompt = ChatPromptTemplate(
        [
            (
                "system",
                """
                당신은 검색된 문서가 사용자 질문과 관련이 있는지 평가하는 평가자입니다.
                문서가 사용자 질문과 관련된 키워드나 의미를 포함하고 있다면 관련성이 있다고 평가하세요.
                엄격한 테스트일 필요는 없습니다. 목표는 잘못된 검색 결과를 필터링하는 것입니다.
                문서가 질문과 관련이 있는지를 나타내는 'yes' 또는 'no'의 이진 점수를 제공하세요.
                """
            ),
            (
                "user",
                "검색된 문서: {context} \n\n 사용자 질문: {question} \n\n 관련성 점수:"
            ),
        ]
    )
    
    chain = grader_prompt | grader

    question = state.get("question", "")
    context = state.get("context", "")

    # 질문이나 컨텍스트가 없으면 평가할 수 없으므로 기본값으로 답변 생성
    if not context or not question:
        print("---ERROR: Missing context or question, defaulting to generate---")
        return "generate"
    
    score = chain.invoke({"question": question, "context": context})
    grade = score.binary_score
    # 관련 없음 -> 질문 재작성 후 재검색 / 관련 있음 -> 답변 생성
    if grade == "no":
        print("-- DECISION: RETRIEVED DOCUMENT ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY ---")
        return "transform_query"
    else:
        print("--- DECISION: GENERATE ---")
        return "generate"
    
# 환각 평가 결과를 구조화된 출력으로 받기 위한 스키마
class GradeHallucinations(BaseModel):
    """생성된 답변의 환각 여부를 판단하기 위한 점수 스키마"""
    
    binary_score: str = Field(description="답변이 사실에 근거하는지 여부, 'yes' 또는 'no'")

# 조건부 엣지: 생성된 답변의 환각 여부 확인 ("support" -> 종료 / "not supported" -> 재생성)
def check_hallucinations(state):
    """
    생성된 답변이 문서에 근거하고 질문에 답하는지 판단합니다.

    """
    print("--- CHECK HALLUCINATIONS ---")
    # 재시도 3회 이상이면 안내용 답변이므로 검증 없이 종료
    if state.get("retry_num", 0) >= 3:
        return "support"

    context = state.get("context", "")
    answer = state.get("answer", "")
    if not context or not answer:
        return "support"

    grader = llm.with_structured_output(GradeHallucinations)
    grader_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 LLM이 생성한 답변이 검색된 문서에 근거하는지 평가하는 평가자입니다.
                답변이 문서의 사실에 근거하면 'yes', 문서에 없는 내용을 지어냈다면 'no'를 반환하세요.
                """,
            ),
            (
                "user",
                "검색된 문서: {context} \n\n 생성된 답변: {answer} \n\n 근거 여부:",
            ),
        ]
    )

    chain = grader_prompt | grader
    score = chain.invoke({"context": context, "answer": answer})

    if score.binary_score == "yes":
        print("--- DECISION: SUPPORTED ---")
        return "support"
    print("--- DECISION: NOT SUPPORTED, REGENERATE ---")
    return "not supported"
