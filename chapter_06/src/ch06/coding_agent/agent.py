from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from .tools import python_exec_tool, file_write_tool

# create_agent를 사용하여 에이전트 생성 [page 157]
tools = [python_exec_tool, file_write_tool]

llm = ChatOpenAI(model="gpt-4o")
graph = create_agent(
    model= llm, 
    tools = tools,
    system_prompt="""
    당신은 Python 개발 에이전트입니다.
    코드 작성이 필요한 경우 python_exec_tool을 사용하세요.
    코드 저장이 필요한 경우 file_write_tool을 사용하세요.
    """
    )


def main() -> None:
    response = graph.stream(
        {
            "messages": [
                "numpy로 1부터 10까지의 평균을 계산하는 코드를 작성하고 정상적으로 실행되는지 확인해줘",
                "확인했다면 그 코드는 .py 파일로 src\ch06\coding_agent 폴더 내 저장해줘"
            ]
        }
    )

    for chunk in response:
        for node, value in chunk.items():
            print(f"\n========== {node} ==========")
            
            if "messages" in value:
                for msg in value["messages"]:
                    print("메시지 타입:", type(msg).__name__)
                    
            # AI가 Tool을 호출한 경우
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print("→ AI가 Tool을 호출함")

                        for tool_call in msg.tool_calls:
                            print("Tool 이름:", tool_call["name"])
                            print("Tool 입력값:")
                            print(tool_call["args"])

                    # 일반 메시지 내용
                    if msg.content:
                        print("내용:")
                        print(msg.content)

            print("=" * 40)