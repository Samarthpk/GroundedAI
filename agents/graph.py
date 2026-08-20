import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from ingestion.vector_store import retrieve


class AgentState(TypedDict):
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    draft_answer: str
    validation_result: bool
    retry_count: int


def retriever_node(state: AgentState):
    chunks = retrieve(state["question"], top_k=5)

    return {
        **state,
        "retrieved_chunks": chunks
    }


def analyst_node(state: AgentState):
    chunks = state["retrieved_chunks"]

    if not chunks:
        return {
            **state,
            "draft_answer": "Not found in the provided documents."
        }

    best = chunks[0]

    # First attempt intentionally uses a bad citation
    # so we can verify the retry loop works.
    if state["retry_count"] == 0:
        answer = (
            "Sensitive information should use local AI systems because "
            "data can remain within organizational infrastructure "
            "[fake-chunk-999]."
        )
    else:
        answer = (
            "Sensitive information should use local AI systems because "
            f"data can remain within organizational infrastructure "
            f"[{best['id']}]."
        )

    return {
        **state,
        "draft_answer": answer
    }


def validator_node(state: AgentState):
    answer = state["draft_answer"]

    valid_ids = {
        chunk["id"]
        for chunk in state["retrieved_chunks"]
    }

    cited_ids = re.findall(r"\[([^\]]+)\]", answer)

    has_citation = len(cited_ids) > 0

    citations_exist = (
        has_citation
        and all(citation in valid_ids for citation in cited_ids)
    )

    return {
        **state,
        "validation_result": citations_exist
    }


def validation_router(state: AgentState):
    if state["validation_result"]:
        return "complete"

    if state["retry_count"] < 1:
        return "retry"

    return "failed"


def increment_retry_node(state: AgentState):
    return {
        **state,
        "retry_count": state["retry_count"] + 1
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("validator", validator_node)
    graph.add_node("increment_retry", increment_retry_node)

    graph.set_entry_point("retriever")

    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "validator")

    graph.add_conditional_edges(
        "validator",
        validation_router,
        {
            "complete": END,
            "retry": "increment_retry",
            "failed": END
        }
    )

    graph.add_edge("increment_retry", "analyst")

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "question": "Why should sensitive information use local AI systems?",
        "retrieved_chunks": [],
        "draft_answer": "",
        "validation_result": False,
        "retry_count": 0
    })

    print("\nQUESTION:")
    print(result["question"])

    print("\nFINAL ANSWER:")
    print(result["draft_answer"])

    print("\nVALIDATION:")
    print(result["validation_result"])

    print("\nRETRY COUNT:")
    print(result["retry_count"])
