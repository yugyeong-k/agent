from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from operator import add
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END

load_dotenv()

# 기본 그래프 정의 [page 106]
class InputState(TypedDict): # invoke() 입력 값
    question: str
    
class OutputState(TypedDict): # invoke() 출력 값
    answer : str

class OverallState(TypedDict): # 노드 전체 데이터
    messages: Annotated[list[str], add]
    question : str
    answer : str

# 그래프 생성: 내부(Overall) / 입력(Input) / 출력(Output) State를 각각 지정
graph_builder = StateGraph(
    OverallState,
    input_schema= InputState,
    output_schema= OutputState
)

# 챗봇 노드 정의 및 등록 [page 108]
llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: InputState) -> OverallState:
    question = state["question"]        # state question 값 읽기
    response = llm.invoke(question)     # LLM에 질문
    return{
        "answer" : response.content,    # 출력용 답변
        "messages" : [question, response.content]   # 대화 기록(add 리듀서 누적)
    }
    
graph_builder.add_node("chatbot", chatbot) # chatbot 노드 등록

# 챗봇 노드 엣지 연결 [page 108]
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
