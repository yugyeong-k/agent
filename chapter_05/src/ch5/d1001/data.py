from .chapter5_2 import UserTD, UserPD   

# TypeDict  
users_td: list[UserTD] = [
    {"id": 1, "name": "홍길동", "email": "example@gmail.com"},
    {"id": 2, "name": 123, "email": "example@gmail.com"},   # 에러 없음 (밑줄만)
]

# Pydantic
def make_users_pd() -> list[UserPD]:
    return [
        UserPD(id=1, name="홍길동", email="example@gmail.com"),
        UserPD(id=2, name=123, email="example@gmail.com"),   # ValidationError
    ]