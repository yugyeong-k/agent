from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# 상태그래프 정의 [page 110]
class State(TypedDict):
    messages: Annotated[list[str], add]
    question : str
    question_length : int

# 상태그래프 생성 [page 111]
graph_builder = StateGraph(State)

# 질문 길이 저장 에이전트 노드 생성 [page 111]
def guardrail(state : State) -> State:
    question_lenght = len(state["question"])
    return {
        "question_length" : question_lenght
    }
    
graph_builder.add_node("guardrail", guardrail)

# 질문 답변 생성 에이전트 노드 생성 [page 112]
llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State) -> State:
    question = state["question"]
    response = llm.invoke(question)
    return {
        "messages" : [response.content]
    }
    
graph_builder.add_node("chatbot", chatbot)  

# 라우팅 함수 정의 및 조건부 엣지 추가 [page 112]
def routing_function(state: State):
    if state["question_length"] > 3:
        return "chatbot"
    else:
        return END

graph_builder.add_conditional_edges(
    "guardrail", 
    routing_function,
    {"chatbot": "chatbot", END: END}
)

# 엣지 연결하기 [page 113]
graph_builder.add_edge(START, "guardrail")
graph_builder.add_edge("chatbot", END)