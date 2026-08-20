import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from ingestion.vector_store import retrieve

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


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

    context = "\n\n".join(
        f"[{chunk['id']}]\n{chunk['text']}"
        for chunk in chunks
    )

    retry_instruction = ""

    if state["retry_count"] > 0:
        retry_instruction = """
Your previous answer failed citation validation.
Rewrite the answer and make sure every factual statement
contains a valid citation from the provided chunk IDs.
"""

    prompt = f"""
You are a grounded enterprise research assistant.

Answer ONLY from the context below.

Rules:
1. Every factual claim must end with a citation using the exact chunk ID.
2. Example citation: [ai_policy.txt-0]
3. Never invent citation IDs.
4. If the answer is not supported by the context, say:
   "Not found in the provided documents."
5. Do not use outside knowledge.

{retry_instruction}

QUESTION:
{state['question']}

CONTEXT:
{context}
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "draft_answer": response.content.strip()
    }


def validator_node(state: AgentState):
    answer = state["draft_answer"]

    if answer == "Not found in the provided documents.":
        return {
            **state,
            "validation_result": True
        }

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


def failed_node(state: AgentState):
    return {
        **state,
        "draft_answer": "Could not produce a grounded answer with valid citations."
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("validator", validator_node)
    graph.add_node("increment_retry", increment_retry_node)
    graph.add_node("failed", failed_node)

    graph.set_entry_point("retriever")

    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "validator")

    graph.add_conditional_edges(
        "validator",
        validation_router,
        {
            "complete": END,
            "retry": "increment_retry",
            "failed": "failed"
        }
    )

    graph.add_edge("increment_retry", "analyst")
    graph.add_edge("failed", END)

    return graph.compile()
