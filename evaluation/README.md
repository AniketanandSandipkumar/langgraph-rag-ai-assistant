# RAGFlow AI evaluation starter

Run from the project root using your project's Python environment:

```bash
python -m evaluation.evaluate_retrieval
```

Requires the same dependencies and `GOOGLE_API_KEY` environment used by `ingest.py`, plus `faiss_index/index.faiss` and `faiss_index/index.pkl`. The script does not call the LLM and does not modify the index.

The evaluator rebuilds chunks using `ingest.load_and_split()` and estimates relevant chunks with topic-term matching. Inspect `evaluation/results/retrieval_results.csv`, refine `TOPIC_TERMS`, and manually review mismatches before presenting metrics. Ambiguous and unanswerable cases are intentionally reserved for separate clarification/abstention tests. This is not an answer-quality evaluation.
