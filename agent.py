"""LangGraph agent for RAGFlow AI.

Reliability improvements:
- bounded model calls per request
- controlled tool errors and division-by-zero handling
- cached embeddings and FAISS index
- source-aware retrieval results
- observable tool status and duration in ToolMessage metadata
"""
from __future__ import annotations

import operator
import os
import time
from functools import lru_cache
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.getenv("FAISS_INDEX_DIR", os.path.join(BASE_DIR, "faiss_index"))
EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
RETRIEVAL_K = max(1, int(os.getenv("RETRIEVAL_K", "3")))
MAX_MODEL_CALLS = max(1, int(os.getenv("MAX_MODEL_CALLS", "5")))

SYSTEM_PROMPT = """You are RAGFlow AI, a careful knowledge assistant.

Rules:
- For questions about AI/ML, LangGraph, RAG, embeddings, transformers, or the
  project's knowledge base, use search_docs and base knowledge-base claims on
  the returned evidence.
- If the retrieved evidence is insufficient or says NO_EVIDENCE, say that the
  indexed documents do not provide enough information. Do not invent citations.
- The search tool returns source labels with each passage. Refer to those source
  labels when useful, but do not claim a passage supports a statement it does not.
- Use calculator tools for arithmetic.
- If a tool returns TOOL_ERROR, explain that the operation failed; never present
  the error as a successful result.
- Be concise, clear, and transparent about uncertainty.
"""

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def _configuration_error(variable: str) -> str:
    return (
        f"Configuration error: {variable} is missing. "
        "Set it in your local .env file or deployment secrets."
    )


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(_configuration_error("GOOGLE_API_KEY"))
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    if not os.path.isdir(INDEX_DIR):
        raise FileNotFoundError(
            f"FAISS index directory not found: {INDEX_DIR}. Run `python ingest.py`."
        )
    # FAISS local loading uses pickle deserialization. Only load an index you
    # created yourself or otherwise trust.
    return FAISS.load_local(
        INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def divide(a: int, b: int) -> str:
    """Divide a by b. Returns a controlled error for division by zero."""
    if b == 0:
        return "TOOL_ERROR: Division by zero is not allowed."
    return str(a / b)


@tool
def search_docs(query: str) -> str:
    """Search the project's indexed text documents for relevant evidence."""
    try:
        retriever = get_vectorstore().as_retriever(
            search_kwargs={"k": RETRIEVAL_K}
        )
        docs = retriever.invoke(query)
        if not docs:
            return "NO_EVIDENCE: No relevant passages were found in the indexed documents."

        blocks = []
        for rank, doc in enumerate(docs, start=1):
            source = str(doc.metadata.get("source", "unknown source"))
            chunk_id = doc.metadata.get("chunk_id")
            start_index = doc.metadata.get("start_index")
            labels = [f"source={source}"]
            if chunk_id:
                labels.append(f"chunk={chunk_id}")
            if start_index is not None:
                labels.append(f"start_index={start_index}")
            blocks.append(
                f"[Evidence {rank}; {'; '.join(labels)}]\n{doc.page_content.strip()}"
            )
        return "\n\n---\n\n".join(blocks)
    except Exception as exc:
        # Keep credentials and provider response bodies out of the UI.
        return (
            f"TOOL_ERROR: Document search failed ({type(exc).__name__}). "
            "Check the API key, embedding service, and FAISS index."
        )


tools = [add, multiply, divide, search_docs]
tools_by_name = {tool_obj.name: tool_obj for tool_obj in tools}


@lru_cache(maxsize=1)
def get_model_with_tools():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(_configuration_error("GROQ_API_KEY"))
    model = ChatGroq(model=GROQ_MODEL, temperature=0)
    return model.bind_tools(tools)


def llm_call(state: MessagesState) -> dict:
    """Call the LLM while enforcing a per-request call budget."""
    calls_so_far = int(state.get("llm_calls", 0))
    if calls_so_far >= MAX_MODEL_CALLS:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I couldn't finish within the execution limit. "
                        "Please simplify the request and try again."
                    )
                )
            ],
            "llm_calls": calls_so_far,
        }

    try:
        response = get_model_with_tools().invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
    except Exception as exc:
        # Return a controlled final response instead of allowing a UI crash.
        response = AIMessage(
            content=(
                "I couldn't contact the language-model service "
                f"({type(exc).__name__}). Check your Groq configuration and try again."
            )
        )

    next_count = calls_so_far + 1
    # Do not start another tool cycle if the last allowed model call asks for tools.
    if getattr(response, "tool_calls", None) and next_count >= MAX_MODEL_CALLS:
        response = AIMessage(
            content=(
                "I reached the model-call limit before completing this request. "
                "Please simplify the question and try again."
            )
        )

    return {"messages": [response], "llm_calls": next_count}


def tool_node(state: MessagesState) -> dict:
    """Run requested tools and attach observable status/timing metadata."""
    results = []
    last_message = state["messages"][-1]

    for tool_call in getattr(last_message, "tool_calls", []) or []:
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {}) or {}
        started = time.perf_counter()

        try:
            selected_tool = tools_by_name.get(name)
            if selected_tool is None:
                observation = f"TOOL_ERROR: Unknown tool '{name}'."
                tool_status = "failed"
            else:
                observation = selected_tool.invoke(args)
                tool_status = (
                    "failed"
                    if str(observation).startswith("TOOL_ERROR:")
                    else "succeeded"
                )
        except Exception as exc:
            observation = f"TOOL_ERROR: {type(exc).__name__} while executing {name}."
            tool_status = "failed"

        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call.get("id", ""),
                name=name,
                response_metadata={
                    "status": tool_status,
                    "duration_ms": duration_ms,
                },
            )
        )

    return {"messages": results}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Route to tools only when a tool call exists and budget remains."""
    last = state["messages"][-1]
    if (
        getattr(last, "tool_calls", None)
        and state.get("llm_calls", 0) < MAX_MODEL_CALLS
    ):
        return "tool_node"
    return END


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {"tool_node": "tool_node", END: END},
)
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()


def invoke_agent(messages: list[AnyMessage]) -> dict:
    """Invoke the graph with a fresh per-request model-call budget."""
    return agent.invoke(
        {"messages": messages, "llm_calls": 0},
        config={"recursion_limit": max(10, MAX_MODEL_CALLS * 3 + 2)},
    )
