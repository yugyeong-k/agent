from langchain.tools import tool
from pydantic import BaseModel, Field

# tool 도구의 설명을 독스트링에서 추출하기 [page 150]
@tool # 일반 함수를 AI가 쓸 수 있는 "도구"로 바꿔줌
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.""" # 독스트링 = 도구 설명, AI가 이걸 읽고 언제 쓸지 판단함
    return a * b

#도구의 전달인자 스키마 정의하기 [page 152], 스키마 = 입력값 설명서, 각 입력값에 설명(description)을 직접 붙일 수 있음
class CalulatorInput(BaseModel):
    a: int = Field(description="first number")  # 첫 번째 숫자
    b: int = Field(description="second number") # 두 번째 숫자

# args_schema로 위에서 만든 입력값 설명서를 도구에 연결
@tool(args_schema=CalulatorInput)
def multiply_with_schema(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# 구글 스타일 독스트링 [page 154]: parse_docstring=True면 Args: 아래 설명이 입력값 설명이 됨
@tool(parse_docstring=True)
def multiply_with_google(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b
    

def main() -> None:
    print("=========== 독스트링만 쓴 도구 (args에 설명 없음) ===========")
    print(multiply.name)        # 도구 이름: multiply
    print(multiply.description) # 도구 설명: 독스트링 내용
    print(multiply.args)        # 입력값 정보: a, b가 int라는 정보 (타입 힌트에서 자동 생성)

    print("\n=========== args_schema 쓴 도구 (args에 설명 포함) ===========")
    print(multiply_with_schema.name)
    print(multiply_with_schema.description)
    print(multiply_with_schema.args)

    print("\n=========== 구글 스타일의 독스트링 ===========")
    print(multiply_with_google.name)
    print(multiply_with_google.description)
    print(multiply_with_google.args)
    print(multiply_with_google.args_schema.model_json_schema()) # 입력값 스키마 전체를 JSON 형식으로 출력
