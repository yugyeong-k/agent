from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

from .tools import tools

# 모델을 동적으로 선택하는 미들웨어 만들기 [page 176]
basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

# Model을 호출하는 과정에 중간에 개입하는 Middleware
@wrap_model_call 
def dynamic_model_selection(
    request: ModelRequest,  # 현재 Model에게 전달하려는 요청 정보
    handler                 # 원래의 Model 호출을 실행하는 함수
) -> ModelResponse:         # Model 호출 결과를 반환
    """
    대화 복잡도에 따라 모델을 동적으로 선택하는 미들웨어
    """
    
    messages = request.state["messages"]
    message_count = len(messages)

    human_count = sum(
        1 for message in messages
        if message.type == "human"
    )
    
    ai_count = sum(
        1 for message in messages
        if message.type == "ai"
    )

    tool_count = sum(
        1 for message in messages
        if message.type == "tool"
    )
    
    print("\n" + "-" * 40)
    print(f"메시지: {message_count}개")
    print(f"  사용자 : {human_count}")
    print(f"  AI     : {ai_count}")
    print(f"  Tool   : {tool_count}")

    # 메시지가 10개보다 많으면 복잡하다고 판단
    if message_count > 10:
        model = advanced_model
        print(f"메시지 수: {message_count} → gpt-4o")
    else:
        model = basic_model
        print(f"메시지 수: {message_count} → gpt-4o-mini")
        
    # request.override(model=model) 기존 request에서 Model만 새로 선택한 Model로 변경
    # handler(...) → 변경된 request를 가지고 실제 Model을 호출
    return handler(request.override(model=model))    
    
agent = create_agent(
    model= basic_model,
    tools= tools,
    system_prompt="""
    너는 계산 작업을 수행하는 AI Agent이다.

    계산이 필요한 질문을 받으면 반드시 calculator Tool을 사용한다.

    다음 규칙을 반드시 지켜라.

    1. 계산이 필요한 경우 머릿속으로 계산한 결과를 최종 답변에 사용하지 말고,
    반드시 calculator Tool의 실행 결과를 사용한다.

    2. Tool을 호출할 때는 사용자의 요청을 정확하게 해석한다.
    예를 들어 "1부터 10까지 더하기"는
    1 + 2 + 3 + ... + 10을 의미한다.
    1 + 10을 의미하지 않는다.

    3. 여러 숫자를 더해야 하는 경우 필요한 계산을 여러 번 수행하거나,
    정확한 계산 방법을 선택한다.

    4. Tool 결과가 예상과 다르다면 임의로 정답을 만들어내지 않는다.
    계산 과정을 다시 확인하고 필요한 경우 Tool을 다시 호출한다.

    5. 최종 답변에서는 실제로 실행된 Tool 결과를 바탕으로 답변한다.

    6. Tool 결과와 다른 숫자를 임의로 만들어서 답변하지 않는다.
    """,
    middleware= [dynamic_model_selection]
)


def main() -> None:
    response = agent.invoke(
        {
            "messages": [
                "1부터 10까지 더한 후 평균을 계산해줘"
            ]
        }
    )

    print("\n" + "=" * 50)
    print("AGENT 실행 과정")
    print("=" * 50)

    for message in response["messages"]:

        # 사용자 메시지
        if message.type == "human":
            print("\n[사용자]")
            print(message.content)

        # AI 메시지
        elif message.type == "ai":

            # Tool 호출이 있으면
            if message.tool_calls:
                print("\n[AI → TOOL 호출]")

                for tool_call in message.tool_calls:
                    print(f"  Tool : {tool_call['name']}")
                    print(f"  입력 : {tool_call['args']}")

        # Tool 결과
        elif message.type == "tool":
            print("\n[TOOL → 결과]")
            print(f"  {message.content}")

    print("\n" + "=" * 50)
    print("최종 답변")
    print("=" * 50)
    print(response["messages"][-1].content)