# from .page86 import page86_ai_msg
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from .chapter5_2 import add, chatbot, graph
from .data import users_td, make_users_pd

def main() -> None:
    print(users_td)

    try:
        print(make_users_pd())
    except Exception as e:
        print(e)

    print(add(["안녕"], ["반가워"]))

    # 리듀서는 메세지를 누적하되 같은 ID는 교체
    msg_human = [HumanMessage(content="Hello", id="1")]
    msg_ai = [AIMessage(content="Hi here!", id="2")]
    print(add_messages(msg_human, msg_ai))

    # 노드 함수 직접 호출
    result = chatbot({"messages": ["안녕"]})
    print(result)
    
    # 그래프 실행 (실행 경로 엣지)
    # app = graph.compile()
    # result = app.invoke({"messages": ["안녕"]})
    # print(result)
    
    # 그래프 실행 (조건부 엣지)
    def run_5_2():
        app = graph.compile()
        result = app.invoke({"messages": ["1234567891011"]})
        print(result)
    
    run_5_2()
