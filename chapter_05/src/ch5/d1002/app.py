# from .chapter5_3_1 import graph_builder
from .chapter5_3_2 import graph_builder

def main() -> None:
    print("d1002 시작")
    
    graph = graph_builder.compile()

    result1 = graph.invoke({"question": "한국"})
    result2 = graph.invoke({"question": "한국의 수도는 어디인가요?"})
    print(result1)
    print(result2)

    # # 출력된 mermaid 텍스트를 https://mermaid.live 에 붙여 넣으면 그림으로 볼 수 있음
    print(graph.get_graph().draw_mermaid())

    