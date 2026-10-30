
"""
Reliability smoke tests for RAGFlow AI.

Run from the project root:
    python evaluation/test_reliability.py

This script reports actual responses for manual review.
It does not use an LLM judge or overwrite existing evaluation results.
"""

import json
import time
import sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent import invoke_agent


TESTS = [
    {
        "id": "R01",
        "category": "grounded_answer",
        "question": "What is an embedding in machine learning?",
        "expected": "answer",
    },
    {
        "id": "R02",
        "category": "unsupported_question",
        "question": (
            "Who created the first version of LangGraph, "
            "and in what month was it released?"
        ),
        "expected": "abstain",
    },
    {
        "id": "R03",
        "category": "ambiguous_question",
        "question": "Does fine-tuning make a model better?",
        "expected": "clarify",
    },
    {
        "id": "R04",
        "category": "calculator_tool",
        "question": "Calculate 125 multiplied by 48.",
        "expected": "answer",
    },
    {
        "id": "R05",
        "category": "missing_context",
        "question": (
            "Which embedding model is best for my application?"
        ),
        "expected": "clarify",
    },
]


def message_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return str(content)


def inspect_state(state):
    messages = state.get("messages", []) or []
    answer = ""
    tool_events = []
    evidence = []

    for message in messages:
        if isinstance(message, AIMessage):
            content = message_text(message.content).strip()

            # Prefer the latest non-empty assistant response.
            if content and not getattr(message, "tool_calls", []):
                answer = content

        elif isinstance(message, ToolMessage):
            content = message_text(message.content)
            metadata = getattr(message, "response_metadata", {}) or {}
            tool_name = getattr(message, "name", None) or "unknown_tool"

            tool_events.append({
                "tool": tool_name,
                "status": metadata.get("status", "unknown"),
                "content_preview": content[:300],
            })

            if tool_name == "search_docs":
                evidence.append(content[:1500])

    return {
        "answer": answer,
        "tool_events": tool_events,
        "evidence": evidence,
        "message_count": len(messages),
        "model_calls": state.get("llm_calls"),
    }


def main():
    results = []

    print("=" * 72)
    print("RAGFlow AI — Reliability Smoke Tests")
    print("=" * 72)

    for test in TESTS:
        print(f"\n[{test['id']}] {test['category']}")

        result = {
            **test,
            "answer": "",
            "tool_events": [],
            "evidence": [],
            "message_count": 0,
            "model_calls": None,
            "execution_error": None,
            "latency_seconds": None,
        }

        start = time.perf_counter()

        try:
            state = invoke_agent([
                HumanMessage(content=test["question"])
            ])

            inspected = inspect_state(state)
            result.update(inspected)

        except Exception as exc:
            result["execution_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        result["latency_seconds"] = round(
            time.perf_counter() - start, 2
        )

        print("Question:", test["question"])
        print("Expected:", test["expected"])
        print("Answer:", result["answer"] or "<EMPTY>")
        print("Tools:", json.dumps(result["tool_events"], ensure_ascii=False))
        print("Evidence chunks:", len(result["evidence"]))
        print("Model calls:", result["model_calls"])
        print("Latency:", result["latency_seconds"], "seconds")

        if result["execution_error"]:
            print("EXECUTION ERROR:", result["execution_error"])

        results.append(result)

    output_path = (
        Path(__file__).resolve().parent
        / "results"
        / "reliability_smoke_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print(f"Saved results to: {output_path}")
    print("Review each answer against its expected behavior.")
    print("These are diagnostic results, not an automated pass rate.")
    print("=" * 72)


if __name__ == "__main__":
    main()
