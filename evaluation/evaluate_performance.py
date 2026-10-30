"""
RAGFlow AI — Latency and Cost Evaluation
Run from project root: python evaluation/evaluate_performance.py

Outputs:
  evaluation/results/performance_results.csv
  evaluation/results/performance_summary.json

Optional environment variables for estimated LLM-only cost:
  MODEL_INPUT_USD_PER_1M
  MODEL_OUTPUT_USD_PER_1M
Set these using current official model pricing. Embedding costs are excluded.
"""
import csv
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent import invoke_agent

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
CSV_PATH = RESULTS_DIR / "performance_results.csv"
SUMMARY_PATH = RESULTS_DIR / "performance_summary.json"

REPEATS = 3
RUN_WARMUP = True

TEST_CASES = [
    {"id": "P01", "category": "grounded_knowledge",
     "question": "What is an embedding in machine learning?"},
    {"id": "P02", "category": "multi_concept_knowledge",
     "question": "Explain how embeddings and a vector database work together in a RAG pipeline."},
    {"id": "P03", "category": "calculator",
     "question": "Calculate 125 multiplied by 48."},
    {"id": "P04", "category": "clarification",
     "question": "Which embedding model is best for my application?"},
    {"id": "P05", "category": "unsupported",
     "question": "Who created the first version of LangGraph, and in what month was it released?"},
]


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 3)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 3)


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_usage(messages):
    """Extract normalized token usage from model responses."""
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    found = False

    for message in messages:
        if not isinstance(message, AIMessage):
            continue

        metadata = getattr(message, "response_metadata", None) or {}
        usage = metadata.get("cost_usage") or {}

        if not usage:
            usage = getattr(message, "usage_metadata", None) or {}

        if not usage:
            provider_usage = metadata.get("token_usage") or {}
            if provider_usage:
                usage = {
                    "input_tokens": provider_usage.get("prompt_tokens", 0),
                    "output_tokens": provider_usage.get("completion_tokens", 0),
                    "total_tokens": provider_usage.get("total_tokens", 0),
                }

        if usage and (
            "input_tokens" in usage or "output_tokens" in usage
        ):
            found = True
            in_count = int(usage.get("input_tokens", 0) or 0)
            out_count = int(usage.get("output_tokens", 0) or 0)
            total_count = int(
                usage.get("total_tokens", in_count + out_count) or 0
            )

            input_tokens += in_count
            output_tokens += out_count
            total_tokens += total_count

    if not found:
        return None, None, None

    return input_tokens, output_tokens, total_tokens


def final_answer(messages):
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = [part.get("text", "") for part in content
                         if isinstance(part, dict) and part.get("text")]
                if parts:
                    return "\n".join(parts).strip()
    return ""


def tool_events(messages):
    events = []
    for message in messages:
        if isinstance(message, ToolMessage):
            metadata = getattr(message, "response_metadata", None) or {}
            events.append({
                "tool": getattr(message, "name", None),
                "status": metadata.get("status"),
                "duration_ms": metadata.get("duration_ms"),
            })
    return events


def model_call_count(state):
    try:
        return int(state.get("llm_calls", 0))
    except (TypeError, ValueError, AttributeError):
        return None


def run_once(case, repeat):
    started = time.perf_counter()
    state, error = {}, ""
    answer = ""
    try:
        state = invoke_agent([HumanMessage(content=case["question"])])
        if not isinstance(state, dict):
            state = {}
        answer = final_answer(state.get("messages", []))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - started
    messages = state.get("messages", []) if isinstance(state, dict) else []
    in_tokens, out_tokens, total_tokens = extract_usage(messages)
    tools = tool_events(messages)
    input_rate = as_number(os.getenv("MODEL_INPUT_USD_PER_1M"))
    output_rate = as_number(os.getenv("MODEL_OUTPUT_USD_PER_1M"))
    cost = None
    if in_tokens is not None and out_tokens is not None and input_rate is not None and output_rate is not None:
        cost = (in_tokens * input_rate + out_tokens * output_rate) / 1_000_000

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_id": case["id"], "category": case["category"], "repeat": repeat,
        "question": case["question"], "latency_seconds": round(latency, 3),
        "success": bool(answer) and not error, "answer_nonempty": bool(answer),
        "error": error, "model_calls": model_call_count(state),
        "tool_count": len(tools), "tools_json": json.dumps(tools, ensure_ascii=False),
        "input_tokens": in_tokens, "output_tokens": out_tokens, "total_tokens": total_tokens,
        "estimated_llm_cost_usd": round(cost, 8) if cost is not None else None,
        "answer_preview": answer[:500],
    }


def mean_or_none(values):
    return round(statistics.mean(values), 3) if values else None


def summarize(rows, warmup_seconds):
    latencies = [r["latency_seconds"] for r in rows]
    calls = [r["model_calls"] for r in rows if r["model_calls"] is not None]
    in_tokens = [r["input_tokens"] for r in rows if r["input_tokens"] is not None]
    out_tokens = [r["output_tokens"] for r in rows if r["output_tokens"] is not None]
    total_tokens = [r["total_tokens"] for r in rows if r["total_tokens"] is not None]
    costs = [r["estimated_llm_cost_usd"] for r in rows if r["estimated_llm_cost_usd"] is not None]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "RAGFlow AI",
        "benchmark_config": {
            "repeats_per_case": REPEATS, "warmup_enabled": RUN_WARMUP,
            "warmup_latency_seconds_unmeasured": round(warmup_seconds, 3) if warmup_seconds is not None else None,
            "test_case_count": len(TEST_CASES), "measured_run_count": len(rows),
        },
        "latency_seconds": {
            "average": mean_or_none(latencies),
            "median_p50": round(statistics.median(latencies), 3) if latencies else None,
            "p95": percentile(latencies, .95),
            "minimum": round(min(latencies), 3) if latencies else None,
            "maximum": round(max(latencies), 3) if latencies else None,
        },
        "reliability": {
            "successful_runs": sum(bool(r["success"]) for r in rows),
            "runs_with_exceptions": sum(bool(r["error"]) for r in rows),
            "nonempty_answers": sum(bool(r["answer_nonempty"]) for r in rows),
        },
        "model_calls": {
            "average_per_run": mean_or_none(calls),
            "total_observed": sum(calls) if calls else None,
            "runs_with_count_available": len(calls),
        },
        "token_usage": {
            "available_runs": len(total_tokens),
            "average_input_tokens_when_available": mean_or_none(in_tokens),
            "average_output_tokens_when_available": mean_or_none(out_tokens),
            "average_total_tokens_when_available": mean_or_none(total_tokens),
            "note": "Provider-reported usage only; missing usage is not estimated.",
        },
        "estimated_llm_cost_usd": {
            "total": round(sum(costs), 8) if costs else None,
            "average_per_run": round(statistics.mean(costs), 8) if costs else None,
            "runs_with_estimate": len(costs),
            "input_rate_usd_per_1m_tokens": as_number(os.getenv("MODEL_INPUT_USD_PER_1M")),
            "output_rate_usd_per_1m_tokens": as_number(os.getenv("MODEL_OUTPUT_USD_PER_1M")),
            "note": "LLM-only estimate; requires token usage and current official rates. Embedding costs are excluded.",
        },
        "by_category": {},
    }
    for category in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == category]
        group_latencies = [r["latency_seconds"] for r in group]
        group_calls = [r["model_calls"] for r in group if r["model_calls"] is not None]
        summary["by_category"][category] = {
            "runs": len(group),
            "average_latency_seconds": mean_or_none(group_latencies),
            "median_latency_seconds": round(statistics.median(group_latencies), 3) if group_latencies else None,
            "successes": sum(bool(r["success"]) for r in group),
            "exceptions": sum(bool(r["error"]) for r in group),
            "average_model_calls": mean_or_none(group_calls),
        }
    return summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    warmup_seconds = None
    if RUN_WARMUP:
        print("Running one unmeasured warm-up request (this still uses API calls)...")
        start = time.perf_counter()
        try:
            invoke_agent([HumanMessage(content="What is RAG?")])
        except Exception as exc:
            print(f"Warm-up failed: {type(exc).__name__}: {exc}")
        warmup_seconds = time.perf_counter() - start
        print(f"Warm-up took {warmup_seconds:.2f}s; excluded from metrics.\n")

    rows = []
    total = len(TEST_CASES) * REPEATS
    for i, case in enumerate(TEST_CASES, start=1):
        for repeat in range(1, REPEATS + 1):
            number = (i - 1) * REPEATS + repeat
            print(f"[{number}/{total}] {case['id']} ({case['category']}) — run {repeat}/{REPEATS}")
            row = run_once(case, repeat)
            rows.append(row)
            status = "OK" if row["success"] else ("ERROR" if row["error"] else "EMPTY ANSWER")
            calls = row["model_calls"] if row["model_calls"] is not None else "n/a"
            print(f"  {status} | {row['latency_seconds']:.2f}s | model calls: {calls} | tools: {row['tool_count']}")
            if row["error"]:
                print(f"  Exception: {row['error']}")

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, warmup_seconds)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("RAGFlow AI — Performance Evaluation Summary")
    print("=" * 72)
    print(f"Measured runs: {summary['benchmark_config']['measured_run_count']}")
    print(f"Average latency: {summary['latency_seconds']['average']} s")
    print(f"Median (P50): {summary['latency_seconds']['median_p50']} s")
    print(f"P95 latency: {summary['latency_seconds']['p95']} s")
    print(f"Successful runs: {summary['reliability']['successful_runs']}")
    print(f"Runs with exceptions: {summary['reliability']['runs_with_exceptions']}")
    print(f"Average model calls/run: {summary['model_calls']['average_per_run']}")
    if summary["estimated_llm_cost_usd"]["total"] is None:
        print("Estimated LLM cost: unavailable (usage and/or pricing rates missing)")
    else:
        print(f"Estimated LLM cost total: ${summary['estimated_llm_cost_usd']['total']:.8f}")
    print(f"\nSaved CSV: {CSV_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print("Note: warm-up uses API calls but is excluded from metrics.")
    print("Cost is an LLM-only estimate, not a complete API bill.")


if __name__ == "__main__":
    main()
