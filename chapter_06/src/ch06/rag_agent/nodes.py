from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage

from .retriever import retriever, retriever_tool
from .state import AgentState

# 모든 노드에서 공통으로 사용하는 LLM
llm = ChatOpenAI(model="gpt-4o")

# 검색 판단 노드
def chatbot(state: AgentState):
    """
    검색(Retriever) 도구를 바인딩한 LLM 모델에 현재 메시지 상태를 입력하여 응답을 생성합니다.
    질문이 주어지면 검색 도구를 도구 호출하거나 일반 답변하며 종료할지 결정할 수 있습니다.
    """
    print("--- [CHATBOT] ---")
    messages = state["messages"]
    # 검색 도구를 LLM에 바인딩 (LLM이 필요 시 tool_calls를 생성)
    llm_with_tools = llm.bind_tools([retriever_tool])
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        # 마지막 메시지(사용자 질문)를 검색에 쓸 question으로 저장
        "question": messages[-1].content
    }
    
# 검색 노드
def retrieve(state: AgentState):
    """
    현재 질문을 기반으로 관련 문서를 검색합니다.
    """
    print("--- [RETRIEVER] ---")
    question = state["question"]
    relevant_doc = retriever.invoke(question)
    # 검색된 문서를 "Page N: 내용" 형식으로 이어붙임 (metadata의 page는 0부터 시작하므로 +1)
    context = ""
    for doc in relevant_doc:
        context += f"Page {doc.metadata['page']+1}: {doc.page_content}"

    last_message = state["messages"][-1]

    # 직전 메시지가 도구 호출이면, 해당 호출 id에 대응하는 ToolMessage로 결과를 반환
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        tool_call_id = last_message.tool_calls[0]['id']
        tool_message = ToolMessage(
            content= context,
            name = "pdf_search",
            tool_call_id=tool_call_id
        ) 
        return {"messages": [tool_message], "context": context}
    else:
        # 도구 호출이 아닌 경우 일반 메시지로 반환
        return {"messages": [context], "context": context}

# 검색 결과 정리 노드
def context_organizer(state: AgentState):
    """
    검색된 결과를 정리합니다.
    """
    print("--- [CONTEXT ORGANIZER] ---")
    context = state["context"]
    
    context_organizer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 검색증강생성(RAG)을 위한 검색 문서를 정리하는 전문가입니다.
                다음의 검색된 결과 문서를 확인하고, LLM이 해당 문서를 정리된 형태로 참고할 수 있도록
                문서의 불필요한 공백 등을 삭제하거나 정렬을 다시하여 정리된 형태로 반환해주세요.
                내용을 삭제하는 것을 최소로 합니다.
                페이지 번호 정보를 절대 삭제하지 마세요.
                """,
            ),
            (
                "user",
                """
                검색 결과: {context}
                """,
            ),
        ]
    )
    
    # 프롬프트 -> LLM 체인 실행
    context_organizer = context_organizer_prompt | llm
    organized_context = context_organizer.invoke({"context": context})

    # 정리된 내용으로 context를 덮어쓰고 메시지에도 추가
    return {
        "context": organized_context.content,
        "messages": [AIMessage (organized_context.content)]
        }

# 쿼리 재작성 노드
def transform_query(state: AgentState):
    """
    더 나은 질문을 생성하기 위해 쿼리를 변환합니다.
    
    Args:
        state (dict): 현재 그래프 상태
        
    Returns:
        state (dict): 재구성된 질문으로 question 키를 업데이트
    """
    
    print("--- [TRANSFORM QUERY] ---")
    question = state["question"]
    
    system = """
    당신은 질문을 다시 작성하는 전문가입니다.
    입력된 질문을 벡터 저장소 검색에 최적화된 더 나은 버전으로 변환하세요.
    입력을 살펴보고 질문의 핵심적인 의미와 의도를 파악하여 개선된 질문을 만들어주세요.
    """
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "user",
                "다음은 초기 질문입니다.: \n\n {question} \n 한국어로 개선된 질문을 작성해주세요.")
        ]
    )
    
    question_rewriter = re_write_prompt | llm
    
    better_question = question_rewriter.invoke({"question": question})
    return {
        "question" : better_question.content,
        "messages": [better_question],
        # 재시도 횟수 증가 (값이 없으면 1로 시작)
        "retry_num": state["retry_num"] + 1 if state.get("retry_num") else 1
    }

# 답변 생성 노드
def generate(state: AgentState):
    """
    검색된 문서와 질문을 기반으로 답변을 생성합니다.
    """
    
    print("--- [GENERATE] ---")
    question = state["question"]
    context = state["context"]
    
    retry_num = state.get("retry_num", 0)
    
    # 재시도 3회 이상: 검색이 계속 실패한 경우이므로, 답변 대신 양해를 구하고
    # 검색 결과로 답할 수 있는 다른 질문을 안내하는 프롬프트 사용
    if retry_num >= 3:
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 검색된 문서를 통해 해결할 수 있는 질문을 추출하는 어시스턴트입니다.
                    사용자가 해결하고자 한 질문이 있었으나 검색 컨텍스트가 충분하지 않은 상황이므로, 주어진 검색 결과 내에서 답변할 수 있는 질문을 새롭게 작성해 나열하세요.
                    사용자에게 질문에 대한 답변을 하지 못함에 양해를 구하고, 다른 질문의 기회와 선택지를 제공하는 친절한 가이드를 하세요.
                    """,
                ),
                (
                    "user",
                    "질문: {question} \n\n검색 결과: {context} \n\n답변:",
                ),
            ]
        )
    else:
        # 일반적인 경우: 검색 컨텍스트 기반으로 출처(페이지 번호)와 함께 간결히 답변
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 질문-답변 업무를 수행하는 어시스턴트입니다. 검색된 컨텍스트를 사용하여 질문에 답변하세요.
                    답변을 모르는 경우, 모른다고 말하세요.
                    답변은 간결하게 작성하고, 반드시 답변의 출처(페이지 번호)를 함께 명시해주세요.


                    """,
                ),
                (
                    "user",
                    "질문: {question} \n\n검색 결과: {context} \n\n답변:",
                ),
            ]            
        )
        
    rag_chain = rag_prompt | llm
    response = rag_chain.invoke({"question": question, "context": context})
    return {"question": question, "answer": response.content, "messages": [response]}