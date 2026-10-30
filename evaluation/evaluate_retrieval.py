"""
Retrieval-only evaluator for RAGFlow AI.

Run from the project root:
python -m evaluation.evaluate_retrieval
"""

import csv
import json
import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ingest import BASE_DIR, INDEX_DIR, EMBEDDING_MODEL, load_and_split


load_dotenv()

DATASET = BASE_DIR / "evaluation" / "dataset.json"
OUT = BASE_DIR / "evaluation" / "results"
K_VALUES = (1, 3, 5)


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Set it in .env or environment."
        )

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    if not (INDEX_DIR / "index.faiss").exists() or not (
        INDEX_DIR / "index.pkl"
    ).exists():
        raise FileNotFoundError(
            f"Expected both index.faiss and index.pkl in {INDEX_DIR}"
        )

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    source_chunks = load_and_split()

    # Build the set of chunk IDs available in the current source documents.
    source_chunk_ids = {
        str(chunk.metadata.get("chunk_id", ""))
        for chunk in source_chunks
        if chunk.metadata.get("chunk_id")
    }

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    OUT.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    evaluated = 0
    skipped = 0

    with (OUT / "retrieval_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "id",
            "type",
            "question",
            "expected_topics",
            "k",
            "retrieved_chunk_ids",
            "gold_chunk_ids",
            "hit_at_k",
            "recall_at_k",
            "precision_at_k",
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in dataset:
            # Exclude questions that should be answered by abstaining
            # or asking for clarification.
            if (
                item["should_abstain"]
                or item["should_clarify"]
                or not item["expected_topics"]
            ):
                skipped += 1
                continue

            if "gold_chunks" not in item:
                raise ValueError(
                    f"{item['id']} is missing 'gold_chunks' in dataset.json."
                )

            gold = set(item["gold_chunks"])

            if not gold:
                raise ValueError(
                    f"{item['id']} is answerable but has no gold chunk IDs."
                )

            # Ensure manual labels match the current source chunk IDs.
            unknown_ids = gold - source_chunk_ids

            if unknown_ids:
                raise ValueError(
                    f"{item['id']} references chunk IDs that do not exist "
                    f"in the current source chunks: {sorted(unknown_ids)}. "
                    "Check ingest.py, chunking settings, and dataset.json."
                )

            hits = store.similarity_search_with_score(
                item["question"],
                k=max(K_VALUES),
            )

            retrieved_ids = [
                str(doc.metadata.get("chunk_id", ""))
                for doc, score in hits
            ]

            evaluated += 1

            for k in K_VALUES:
                top_k_ids = retrieved_ids[:k]
                found = set(top_k_ids) & gold

                row = {
                    "id": item["id"],
                    "type": item["type"],
                    "question": item["question"],
                    "expected_topics": "|".join(item["expected_topics"]),
                    "k": k,
                    "retrieved_chunk_ids": "|".join(top_k_ids),
                    "gold_chunk_ids": "|".join(sorted(gold)),
                    "hit_at_k": int(bool(found)),
                    "recall_at_k": round(len(found) / len(gold), 4),
                    "precision_at_k": round(len(found) / k, 4),
                }

                writer.writerow(row)
                grouped[k].append(row)

    report = {
        "evaluated_questions": evaluated,
        "skipped_questions": skipped,
        "k_values": {},
        "caveats": [
            "Gold chunk IDs are manually assigned in dataset.json "
            "and should be reviewed against the indexed source documents.",
            "The evaluator checks gold IDs against current source chunk IDs, "
            "but this does not guarantee that every label is semantically correct.",
            "Unanswerable and clarification-needed questions are excluded "
            "from retrieval metrics.",
            "These metrics do not measure answer correctness, groundedness, "
            "latency, or tool reliability.",
        ],
    }

    for k, rows in grouped.items():
        count = len(rows)

        report["k_values"][str(k)] = {
            "question_count": count,
            "hit_rate_at_k": round(
                sum(row["hit_at_k"] for row in rows) / count, 4
            ),
            "mean_recall_at_k": round(
                sum(row["recall_at_k"] for row in rows) / count, 4
            ),
            "mean_precision_at_k": round(
                sum(row["precision_at_k"] for row in rows) / count, 4
            ),
        }

    summary_path = OUT / "retrieval_summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Evaluated: {evaluated}; skipped: {skipped}")

    for k, metrics in report["k_values"].items():
        print(
            f"@{k}: "
            f"hit rate={metrics['hit_rate_at_k']:.3f}, "
            f"recall={metrics['mean_recall_at_k']:.3f}, "
            f"precision={metrics['mean_precision_at_k']:.3f}"
        )

    print(f"Detailed CSV: {OUT / 'retrieval_results.csv'}")
    print(f"Summary JSON: {summary_path}")
    print("Review manual gold labels before reporting these metrics.")


if __name__ == "__main__":
    main()
