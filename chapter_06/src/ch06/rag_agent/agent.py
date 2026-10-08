from dotenv import load_dotenv

load_dotenv()

from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END

from ch06.rag_agent.state import AgentState
from ch06.rag_agent.nodes import chatbot, retrieve, context_organizer, generate, transform_query
from ch06.rag_agent.edges import decide_to_generate, check_hallucinations

graph_builder = StateGraph(AgentState, input_schema=MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("retriever", retrieve)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "retriever",
        END: END
    }
)

graph_builder.add_node("context_organizer", context_organizer)
graph_builder.add_node("transform_query", transform_query)
graph_builder.add_node("generate", generate)

graph_builder.add_edge("retriever", "context_organizer")
graph_builder.add_conditional_edges(
    "context_organizer",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
graph_builder.add_edge("transform_query", "retriever")
graph_builder.add_conditional_edges(
    "generate",
    check_hallucinations,
    {
        "not supported": "generate",
        "support": END
    },
)

graph = graph_builder.compile()

