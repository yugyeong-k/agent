from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool

class ContactInfo(BaseModel):
    name: str = Field(description="이름")
    email: str = Field(description="이메일 주소")
    phone: str = Field(description="전화번호")
    
model = ChatOpenAI(model="gpt-4o-mini")

agent = create_agent(
    model= model,
    tools=[],
    response_format= ToolStrategy(ContactInfo)
)

def main() -> None:
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "다음 텍스트에서 연락처 정보를 추출해줘, 홍길동, ghdrlfehd@example.com, 01012345678"
        }]
    })
    
    # 구조화된 결과
    contact = result["structured_response"]

    print("===== 구조화 전 원본 메시지 =====")
    print(result["messages"][0].content)

    print("\n===== 구조화된 결과 =====")
    print(contact)

    print("\n===== 구조화된 데이터를 하나씩 사용 =====")
    print(f"이름: {contact.name}")
    print(f"이메일: {contact.email}")
    print(f"전화번호: {contact.phone}")