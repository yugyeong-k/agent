from typing import TypedDict, Annotated

import asyncio
import json
from langchain.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

# 실행 위치(cwd)와 상관없이 프로젝트 루트(chapter_06)의 .env를 읽음
load_dotenv()

# ,Tavily 검색 도구 만들기 [page 130]
tool = TavilySearch(max_results=1) # 검색 결과 최대 1개까지 가져옴
tools = [tool] # 사용할 도구들을 리스트로 묶음, 현재 도구는 Tavily만 존재

llm = ChatOpenAI(model="gpt-4o")
llm_with_tavily = llm.bind_tools(tools) # Tavily라는 검색 도구를 사용할 수 있다고 알려줌, 실행은 되지않음

# 상태(State) 정의 [page 130] 
class State(TypedDict):
    messages: Annotated[list, add_messages] # 대화 내용 저장(사용자 질문, AI 도구 사용 요청, 검색 결과, 최종 답변)
    
graph_builder = StateGraph(State) 

# LLM이 답변 또는 도구 호출 정보(tool_calls)를 만드는 노드 [page 131]
def chatbot(state: State):    
    response = llm_with_tavily.invoke(state["messages"]) # 대화 내용을 AI에게 전달
    return {"messages": [response]} # AI가 만든 메세지를 State에 추가

graph_builder.add_node("chatbot", chatbot)

#  LLM이 요청한 도구를 "실제로 실행"하는 클래스 생성 [page 132]
class BasicToolNode:
    """
        마지막 AIMessage에서 요청된 도구를 실행하는 노드
    """
    
    # 객체를 만들 때 한 번 실행됨
    def __init__(self, tools: list) -> None:
        # "이름 → 도구" 저장, 예: {"tavily_search": tool}
        self.tools_by_name = {tool.name: tool for tool in tools}
        
    # input = 현재 상태, 예: {"messages": [HumanMessage(질문), AIMessage]}
    def __call__(self, input: dict):
        # message가 비어 있지 않으면 마지막 메세지 가져오기
        if messages := input.get("messages", []): 
            message = messages[-1] # 마지막 메시지 = 방금 chatbot이 만든 AIMessage
        else:
            raise ValueError("ERROR: 입력에 메세지가 없습니다.")
        
        outputs = [] # 도구 실행 결과(ToolMessage)를 모아 둘 빈 리스트 생성
        
        # AI tool_call 예: {"name": "tavily_search", "args": {"query": "..."}, "id": "call_xxx"}  하나씩 꺼내 반복
        for tool_call in message.tool_calls: 
            
            # 실행 결과(검색 결과 딕셔너리)를 tool_result에 저장
            tool_result = self.tools_by_name[tool_call["name"]].invoke( # tool_call["name"]으로 도구를 찾아서 invoke(실행)
                tool_call["args"] # 찾은 도구에 args를 입력값으로 넣고 실행
            )
        
            outputs.append(
                ToolMessage(
                    content = json.dumps(tool_result, ensure_ascii=False), # json 형식 지정, ensure_ascii=False: 한글이 깨지지 않게 함
                    name = tool_call["name"], # 어떤 도구가 실행됐는지 기록
                    tool_call_id = tool_call["id"], # AI가 요청한 tool_call과 이 결과를 연결하기 위한 ID
                )
            )
        return {"messages": outputs} # 새로 만들어진 ToolMessage를 State에 추가, 기존 메시지 뒤에 이 결과를 추가함

# BasicToolNode에게 우리가 사용할 도구를 알려줌
tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
    
# LLM의 도구 호출 결과에 따라 처리하는 조건부 엣지 만들기 [page 134]
def route_tools(
    state: State,
):
    """
    마지막 메세지에 도구 호출이 있는 경우, ToolNode로 라우팅하고 그렇지 않으면 END로 라우팅
    chatbot이 만든 마지막 AIMessage를 확인해서 도구 사용 요청이 있으면 tools로 이동, 없으면 end로 이동
    """
    
    if isinstance(state, list): # 만약 state 자체가 리스트라면
        ai_message = state[-1] # 리스트의 마지막 메시지를 가져옴
    elif messages := state.get("messages", []): # 일반적인 State 형태라면
        ai_message = messages[-1] # messages의 마지막 메시지를 가져옴
    else:
        raise ValueError(f"ERROR: 입력에 메세지가 없습니다. 상태:{state}")
    
    # ai_message에 tool_calls가 존재하고 tool_calls가 하나라도 있는지 확인
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0: 
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot", route_tools,
    {"tools": "tools", END: END},
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

def main() -> None:
    
    # # invoke: 그래프가 끝까지 다 실행된 뒤, 최종 결과를 한 번에 받음
    # def invoke():
    #     response = graph.invoke(
    #         {
    #             "messages": ["Langgraph가 무엇인가요?"]
    #         }
    #     )

    #     for msg in response["messages"]:
    #         msg.pretty_print()

    # # ainvoke: invoke의 비동기 버전. await로 기다리며, asyncio.run()으로 실행해야 함
    # async def ainvoke():
    #     response = await graph.ainvoke(
    #         {
    #             "messages": ["Langgraph가 무엇인가요?"]
    #         }
    #     )

    #     for msg in response["messages"]:
    #         msg.pretty_print()
        
    # asyncio.run(ainvoke()) 
    
    # # stream: 노드가 하나 끝날 때마다 {노드이름: 그 노드가 만든 결과}를 바로 받음(chatbot → tools → chatbot 순서가 눈에 보임)
    # def stream():
    #     response = graph.stream(
    #         {
    #             "messages": ["Langgraph가 무엇인가요?"]
    #         }
    #     )
    #     for chunk in response:
    #         for node, state in chunk.items():
    #             print("---", node, "---")
    #             print(state)
    #             print("=" * 60)
    
    # stream()      
    
    # # stream_mode="values": 노드가 끝날 때마다 "지금까지 쌓인 전체 상태"를 받음(stream은 새로 추가된 것만, values는 누적된 전체)
    # def stream_values():
    #     response = graph.stream(
    #         {
    #             "messages": ["Langgraph가 무엇인가요?"]
    #         },
    #         stream_mode="values"
    #     )

    #     for chunk in response:
    #         for state_key, state_value in chunk.items():
    #             print("--- 현재 상태 ---")
    #             for msg in state_value:
    #                 print(f"{type(msg).__name__}: {msg.content[:50]}")
    #             if state_key == "messages":
    #                 state_value[-1].pretty_print()
    #             print("=" * 60)
    
    # stream_values()
    
    # stream_mode="messages": AI가 답변을 만드는 중 글자(토큰) 단위로 실시간 받음(ChatGPT처럼 한 글자씩 나오는 방식)
    # def stream_messages():
    #     response = graph.stream(
    #         {
    #             "messages": ["Langgraph가 무엇인가요?"]
    #         },
    #         stream_mode="messages"
    #     )

    #     for token, metadata in response:
    #         print(token.content)
    #         # print(metadata["langgraph_node"])
            
    # stream_messages()
    
    # astream: stream의 비동기 버전. async for로 받고, asyncio.run()으로 실행
    async def astream():
        response = graph.astream(
            {
                "messages": ["Langgraph가 무엇인가요?"]
            }
        )
        async for chunk in response:
            for node, state in chunk.items():
                print("---", node, "---")
                print(state)
                print("=" * 60)
                
    asyncio.run(astream()) 
    
"""
① graph.invoke()
② START
③ chatbot
④ AI가 질문 확인
⑤ "검색할까?"
⑥ route_tools
⑦ tools
⑧ Tavily 실제 검색
⑨ ToolMessage로 검색 결과 저장
⑩ chatbot으로 돌아감
⑪ AI가 검색 결과 확인
⑫ 최종 답변 생성
⑬ route_tools
⑭ tool_calls 없음
⑮ END

사용자 질문
↓
START
↓
[chatbot]
↓
LLM이 판단한다
↓
도구가 필요한가?
↙          ↘
YES           NO
↓             ↓
[route_tools] END
↓
[tools]
↓
Tavily 실제 실행
↓
ToolMessage
↓
[chatbot]
↓
검색 결과를 보고
최종 답변 생성
↓
[route_tools]
↓
tool_calls 없음
↓
END
"""