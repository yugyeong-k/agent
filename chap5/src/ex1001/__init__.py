# from 같은 경로의 app파일
# improt .app 파일 안에 있는 main()
from .app import main

__all__ = ["main"]

# 프로젝트를 초기화 해준다.
print("프로젝트 초기화 init")