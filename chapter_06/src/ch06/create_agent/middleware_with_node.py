from .tools import tools
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, dynamic_prompt, AgentState, ModelRequest
from langgraph.runtime import Runtime

from dotenv import load_dotenv

load_dotenv()

import random

# 모델 호출 전 미들웨어와 프롬프트 선택 미들웨어 구현하기 [page 177]
model = ChatOpenAI(model="gpt-4o-mini")

# 금지어 목록
BLOCKED_WORDS = ["바보", "멍청이", "나쁜말"]

# Model이 호출되기 전에 개입하는 Middleware
@before_model
# state : 현재 Agent의 상태, runtime: 현재 실행 환경 정보
def content_filter_middleware(state: AgentState, runtime: Runtime):
    """
    금지어를 필터링하는 미들웨어
    - 그래프에 'content_filter_middleware' 노드 추가됨
    - 금지어 감지 시 예외 발생으로 중단
    """
    # messages가 존재하는 경우에만 검사
    if state["messages"]:
        # 가장 최근 메시지를 가져온다.
        last_msg = state["messages"][-1]
        content = getattr(
            last_msg,       # 이 객체에서
            'content',      # content를 찾아서
            str(last_msg)   # 없으면 이것을 사용
        )
    
        for word in BLOCKED_WORDS:
            if word in content:
                print(f"[before_model] 금지어 감지: '{word}'")
                raise ValueError(f"부적절한 표현이 감지되었습니다: '{word}'")
            
        print(f"[before_model] 입력 검증 통과")
    
    # before_model에서는 다음 단계로 진행한다.
    return None

# Runtime Context를 사용하는 동적 프롬프트 Middleware [page 181]
class UserContext(TypedDict):
    user_role : str 
    
@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    role = request.runtime.context.get("user_role", "user")
    
    if role == "expert":
        print(f"[dynamic_prompt] 전문가 모드")
        return "전문 용어를 사용하여 상세하게 답변하세요."
    elif role == "beginner":
        print(f"[dynamic_prompt] 초보자 모드")
        return "쉬운말로 간단하게 설명하세요."
    return "친절하게 답변하세요"    

# 랜덤으로 Prompt를 선택하는 동적 프롬프트 Middleware [page 178]
@dynamic_prompt
def random_tone_prompt(request: ModelRequest) -> str:
    """
    랜덤하게 말투를 변경하는 미들웨어
    - 존댓말 또는 반말 프롬프트를 랜덤 선택
    - @wrap_model_call 기반이므로 노드 추가 X
    """
    
    if random.choice([True, False]):
        print(f"[dynamic_prompt] 존댓말 모드")
        return "당신은 친절한 AI 입니다. 항상 존댓말로 정중하게 답변하세요."
    else:
        print(f"[dynamic_prompt] 반말 모드")
        return "너는 친근한 AI야 항상 반말로 편하게 답변해."
    
agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        content_filter_middleware,
        role_based_prompt,
        random_tone_prompt,
    ],
    context_schema=UserContext
)

def main() -> None:
    response = agent.stream(
        {"messages": ["15와 7을 더하고, 컴퓨터가 계산하는 방식을 설명해주세요."]},
        context={
            "user_role": "expert"
        }
        )
    
    print("\n" + "=" * 50)
    print("AGENT 실행 과정")
    print("=" * 50)
    
    for chunk in response:
        for node, value in chunk.items():
            print(f"\n--- {node} ---")
            if value and "messages" in value:
                for message in value["messages"]:
                    if message.type == "tool":
                        print(f"Tool : {message.name}")
                        print(f"결과 : {message.content}")
                    elif message.tool_calls:
                        for call in message.tool_calls:
                            print(f"도구 요청 : {call['name']} {call['args']}")
                    elif message.content:
                        print(f"내용 : {message.content}")
                
    print("\n" + "=" * 50)
    print("금지어 AGENT 실행 과정")
    print("=" * 50)
    
    try:
        response = agent.stream({"messages": ["바보야 10과 5를 더해줘"]})
        for chunk in response:
            for node, value in chunk.items():

                print(f"\n--- {node} ---")
                
                if value and "messages" in value:
                    for message in value["messages"]:          
                        if message.tool_calls:
                            for call in message.tool_calls:
                                print(f"도구 요청 : {call['name']} {call['args']}")
                        elif message.content:
                            print(f"내용 : {message.content}")
    except ValueError as e:
        print(f"차단됨: {e}")

if __name__ == "__main__":
    main()