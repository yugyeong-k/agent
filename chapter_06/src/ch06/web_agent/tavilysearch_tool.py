from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Tavily로 웹 검색 [page 121]
def search_web(query: str) -> dict:
    search = TavilySearch(max_results=2)
    return search.invoke({"query": query})

# 덧셈, 곱셈 도구 만들기 [page 123]
@tool # 함수를 LLM이 호출할 수 있는 도구로 변환(랭체인)
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

tools = [add, multiply]

# llm 생성 + 도구 연결 [page 124]
def get_llm_with_tools():
    llm = ChatOpenAI(model="gpt-4o-mini")
    return llm.bind_tools(tools)

# 웹 검색 도구를 사용하는 LLM 생성 [page 127]
def get_llm_with_tavily():
    search= TavilySearch(max_results=2)
    tavily_tools= [search]
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    return llm.bind_tools(tavily_tools)

# 메세지 목록을 관리하는 그래프 상태 정의 및 생성 [page 130] 
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
############################################################ 130쪽 부터 이어서 하기~~
        
def main() -> None:
    # Tavily search 실행 [page 121]
    # result = search_web("랭그래프가 무엇인가요?")
    # print(result)
    
    # 곱셈, 덧셈 도구 사용 [page 125]
    # llm_with_tools = get_llm_with_tools()
    # response1 = llm_with_tools.invoke("3곱하기 5는 뭔가요? 그리고 2 더하기 4는 뭔가요?") # 관련 질문
    # response2 = llm_with_tools.invoke("안녕하세요.") # 관련 없는 질문
    # print(response1,"\n")
    # print(response2)
    # print(response1.tool_calls)
    # print(response2.tool_calls)  # 도구가 필요 없는 질문이면 빈 리스트
    
    # 웹 검색 도구 사용 [page 129]
    llm_with_tavily = get_llm_with_tavily()
    response3 = llm_with_tavily.invoke("안녕") # 도구 사용 필요 없는 질문
    response4 = llm_with_tavily.invoke("2026년 AI 트렌드는 무엇인가요?") # 최신 정보가 필요한 질문 -> tavily_search 호출 요청
    
    print(response3,"\n")
    print(response4,"\n")
    print(response3.tool_calls,"\n")
    print(response4.tool_calls)
    