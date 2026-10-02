from typing import TypedDict, Annotated
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END

# TypedDict[page 95] 
class UserTD(TypedDict):
    id: int
    name: str
    email: str

# Pydantic[page 97]
class UserPD(BaseModel):
    id: int
    name: str
    email: str

# 직접 만든 add 리듀서 함수[page 99]
def add(left, right):
    return left + right

# 그래프 정의 및 생성[page 101]
class State(TypedDict):
    messages: Annotated[list[str], add]

graph = StateGraph(State)

# 그래프 노드 추가[page 102]
def chatbot(state: State):
    question = state["messages"]
    answer = f"사용자 입력을 그대로 반환하는 챗봇입니다. {question}라는 질문을 받았습니다."
    return {"messages": [answer]}

graph.add_node("chatbot", chatbot)

# 그래프 실행 경로, 엣지 추가[page 103]
# 단일 노드
# graph.add_edge(START, "chatbot")
# graph.add_edge("chatbot", END)

# 다중 노드
# graph.add_node("chatbot_a", chatbot)
# graph.add_node("chatbot_b", chatbot)

# graph.add_edge(START, "chatbot_a")
# graph.add_edge("chatbot_a", "chatbot_b")
# graph.add_edge("chatbot_b", END)

# 그래프 조건부 엣지 추가[page 104]
graph.add_edge(START, "chatbot")    # 시작: chatbot

def summary(state: State):          # 글자수 자르는 노드
    last = state["messages"][-1]
    answer = f"자르기: {last[:100]}"
    return {"messages": [answer]}

graph.add_node("summary", summary)
    
def routing_function(state: State):
    # state["messages"][-1] : chatbot이 만든 답변
    if len(state["messages"][-1]) > 100 :
        return True
    return False

graph.add_conditional_edges(
    "chatbot",
    routing_function,
    {True: "summary", False: END}
)

graph.add_edge("summary", END)