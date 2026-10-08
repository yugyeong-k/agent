from langgraph.graph import MessagesState

# RAG 에이전트 그래프의 상태 (messages 필드는 MessagesState에서 상속)
class AgentState(MessagesState):
    question: str   # 현재 질문 (쿼리 변환 시 재작성된 질문으로 갱신됨)
    context: str    # 검색된 문서 내용 (페이지 번호 포함)
    answer: str     # 최종 생성된 답변
    retry_num: int  # 쿼리 재작성 횟수 (3회 이상이면 답변 생성으로 넘어감)
