# ◈ RAGFlow AI

### Intelligent Knowledge Assistant powered by LangGraph, RAG & Streamlit

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF)](https://github.com/facebookresearch/faiss)
[![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-6366F1)]()

> **RAGFlow AI** is an intelligent knowledge assistant combining Retrieval-Augmented Generation (RAG), LangGraph-based agent orchestration, FAISS vector search, and a custom Streamlit conversational interface.
>
> The application processes user requests, retrieves relevant knowledge, executes deterministic calculator operations, and exposes agent activity through an interactive execution-tracing interface.

---

## ✨ Why RAGFlow AI?

Modern AI applications require more than a language model connected to a chat interface.

RAGFlow AI demonstrates an agent-based architecture capable of selecting appropriate tools, retrieving relevant information, and generating evidence-grounded responses.

**Core workflow:**

`User Query → LangGraph Agent → Tool Selection → Knowledge Retrieval / Calculation → Final Response`

The project focuses on:

- Agentic AI and LangGraph orchestration
- Retrieval-Augmented Generation
- Semantic search using embeddings and FAISS
- Tool calling and deterministic computation
- Context-aware response generation
- Agent execution tracing
- Error handling and bounded model execution
- Evaluation of retrieval, response behavior, and performance

---

## 🖥️ Application Preview

> Add screenshots of the actual running application here.

### Main Interface

![RAGFlow AI Interface](visualizations/ragflow-main.png)

### Agent Activity & Tool Trace

![Agent Activity](visualizations/agent-trace.png)

### Knowledge Retrieval

![Knowledge Base Search](visualizations/knowledge-search.png)

---

# 🚀 Key Features

## 🤖 LangGraph Agent

The application uses LangGraph to orchestrate the interaction between the language model and available tools.

The agent can:

- Process user queries.
- Decide when a tool is required.
- Invoke knowledge retrieval or calculator tools.
- Interpret tool outputs.
- Generate a final response.
- Handle tool failures and model-call limits.

The graph uses a state-based execution workflow, with message history and LLM-call tracking.

## 📚 Retrieval-Augmented Generation

RAG allows the assistant to retrieve relevant information from its knowledge base before generating a response.

The retrieval workflow:

```text
User Question
      |
      v
Query Processing
      |
      v
Embedding Generation
      |
      v
FAISS Similarity Search
      |
      v
Relevant Knowledge Chunks
      |
      v
LLM Response Generation
      |
      v
Grounded Answer
```

## 🔎 Semantic Knowledge Retrieval

The knowledge base contains AI and machine-learning concepts, including:

- Transformers
- Retrieval-Augmented Generation
- Embeddings
- Vector Databases
- FAISS
- LangGraph
- Large Language Models
- Fine-tuning
- Prompt Engineering
- AI Agents

Text is divided into chunks using `RecursiveCharacterTextSplitter`, embedded, and indexed for similarity-based retrieval.

## 🧮 Calculator Tool

The agent supports deterministic numerical operations through calculator tools:

- Addition
- Multiplication
- Division

This demonstrates how LLM reasoning can be combined with deterministic tool execution.

## 💬 Conversational Interface

The Streamlit application includes:

- Chat interface
- Conversation history
- Custom avatars
- Suggested prompts
- New conversation functionality
- Custom CSS styling
- Agent activity panels

## 🔍 Agent Activity Monitoring

The interface exposes execution information, including:

- Tool names
- Tool inputs and outputs
- Execution status
- Tool execution duration
- Turn-by-turn activity

This provides visibility into how the agent processes requests.

---

# 🧠 System Architecture

```text
                    +----------------------+
                    |      User Query      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |     Streamlit UI     |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |    LangGraph Agent   |
                    +----------+-----------+
                               |
                   +-----------+-----------+
                   |                       |
                   v                       v
          +----------------+      +----------------+
          | Knowledge Base |      |   Calculator   |
          | Search Tool    |      |     Tools      |
          +-------+--------+      +-------+--------+
                  |                       |
                  +-----------+-----------+
                              |
                              v
                    +----------------------+
                    |    LLM Response      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Conversation & Trace |
                    +----------------------+
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.13 | Application development |
| Streamlit | Conversational interface |
| LangGraph | Agent orchestration |
| LangChain | LLM and tool integration |
| Groq | Language model inference |
| Google Gemini Embeddings | Semantic embedding generation |
| FAISS | Vector similarity search |
| RecursiveCharacterTextSplitter | Document chunking |
| JSON / CSV | Evaluation datasets and results |

---

# 📊 Evaluation & Benchmarking

An important part of RAGFlow AI is its evaluation framework.

Rather than relying only on manual testing, the project includes dedicated evaluation scripts for retrieval quality, answer behavior, reliability, performance, and estimated LLM cost.

The evaluation artifacts are available in the `evaluation/` directory.

## 1. Retrieval Evaluation

**Script:** `evaluation/evaluate_retrieval.py`

The retrieval evaluation measures how effectively the FAISS knowledge base retrieves relevant chunks for a collection of test questions.

The evaluation uses manually validated gold chunk identifiers.

### Results

| Metric | Result |
|---|---:|
| Answerable test cases evaluated | 20 |
| Recall@1 | 0.975 |
| Hit Rate@1 | 1.000 |
| Recall@3 | 1.000 |
| Hit Rate@3 | 1.000 |
| Precision@3 | 0.350 |
| Recall@5 | 1.000 |
| Hit Rate@5 | 1.000 |
| Precision@5 | 0.210 |

**Interpretation:**

- At k=3, all evaluated answerable queries retrieved their relevant gold chunks.
- Recall reached 1.000 at k=3 and remained 1.000 at k=5.
- Precision decreases as more chunks are retrieved, indicating that larger retrieval sets also contain less relevant chunks.

Ten unanswerable or clarification-oriented cases were excluded from this retrieval metric calculation.

## 2. Answer Quality & Behavioral Evaluation

**Script:** `evaluation/evaluate_answers.py`

The answer evaluation tests whether the agent follows the expected response behavior for different categories of questions.

The test dataset contains 30 cases:

| Expected behavior | Test cases |
|---|---:|
| Answer | 20 |
| Abstain | 5 |
| Clarify | 5 |
| **Total** | **30** |

The evaluation uses an LLM-based judge to assess answer correctness, faithfulness, and relevance.

### Results

| Metric | Result |
|---|---:|
| Total test cases | 30 |
| Behavioral matches | 27 |
| Behavioral pass rate | 90% |
| Average correctness on answerable cases | 2.0 / 2 |
| Average faithfulness on answerable cases | 2.0 / 2 |
| Average relevance on answerable cases | 2.0 / 2 |
| Tool errors | 0 |
| Agent exceptions | 0 |
| Judge exceptions | 0 |

The evaluation recorded some mismatches involving unsupported questions and ambiguous questions requiring clarification.

**Important:** The 90% result represents agreement with the expected behavioral categories. The answer-quality scores are based on an LLM judge and should not be interpreted as independent human-verified accuracy.

## 3. Reliability Smoke Testing

**Script:** `evaluation/test_reliability.py`

The reliability tests exercise representative application scenarios:

- Grounded knowledge retrieval
- Unsupported questions
- Ambiguous recommendations
- Calculator operations
- Missing application context

The tests inspect agent responses, tool usage, model-call counts, and execution behavior.

The diagnostic run completed without unhandled Python exceptions. However, the unsupported-question scenario exposed a model-call-limit issue involving repeated retrieval attempts.

These tests are diagnostic smoke tests, not a statistically established reliability rate.

## 4. Performance & Cost Evaluation

**Script:** `evaluation/evaluate_performance.py`

The performance benchmark measures execution latency, response availability, model calls, token usage, and estimated language-model cost.

### Benchmark methodology

- Five representative scenarios.
- Three measured repetitions per scenario.
- Fifteen measured requests.
- One warm-up request excluded from the measurements.

Scenarios include:

1. Grounded knowledge question.
2. Multi-concept RAG question.
3. Calculator operation.
4. Clarification-oriented question.
5. Unsupported knowledge question.

### Performance results

| Metric | Measured result |
|---|---:|
| Measured requests | 15 |
| Non-empty responses | 15 / 15 |
| Unhandled exceptions | 0 |
| Average latency | 10.978 seconds |
| Median latency (P50) | 2.56 seconds |
| P95 latency | 41.67 seconds |
| Average model calls | 2.4 |

The P95 latency is influenced by slower unsupported-question scenarios involving repeated model and retrieval calls.

### Estimated LLM cost

| Metric | Estimated cost |
|---|---:|
| Total cost across 15 requests | $0.00339660 |
| Average cost per request | $0.00022644 |
| Projected cost for 100 similar requests | $0.022644 |
| Projected cost for 1,000 similar requests | $0.22644 |

The estimate uses configured Groq input/output token pricing for the benchmark.

**Cost limitations:**

- These figures represent estimated LLM inference cost only.
- Gemini embedding charges are excluded.
- Hosting and infrastructure expenses are excluded.
- Projections assume a similar request and token-usage distribution.
- Actual costs vary with model usage and provider pricing.

**Performance interpretation:** A non-empty response is not necessarily a correct response. The 15/15 figure measures response availability, not factual accuracy.

---

# 📂 Project Structure

```text
langgraph-rag-ai-assistant/
│
├── app.py
├── agent.py
├── ingest.py
├── requirements.txt
├── .gitignore
│
├── sample_docs/
│   └── ai_concepts.txt
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── evaluation/
│   ├── __init__.py
│   ├── README.md
│   ├── dataset.json
│   ├── reference_answers.json
│   ├── evaluate_retrieval.py
│   ├── evaluate_answers.py
│   ├── evaluate_performance.py
│   ├── test_reliability.py
│   │
│   └── results/
│       ├── retrieval_results.csv
│       ├── retrieval_summary.json
│       ├── answer_results.csv
│       ├── answer_summary.json
│       ├── reliability_smoke_results.json
│       ├── performance_results.csv
│       └── performance_summary.json
│
└── visualizations/
    ├── ragflow-main.png
    ├── agent-trace.png
    └── knowledge-search.png
```

---

# ⚙️ Installation & Setup

## 1. Clone the repository

```bash
git clone https://github.com/AniketanandSandipkumar/langgraph-rag-ai-assistant.git
cd langgraph-rag-ai-assistant
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure API credentials

Configure the required Groq and Google API credentials using environment variables or the secrets configuration supported by your deployment.

Never commit API keys, credentials, or `.env` files to GitHub.

## 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🧪 Running the Evaluation Suite

Run the following commands from the repository root.

### Retrieval evaluation

```bash
python -m evaluation.evaluate_retrieval
```

### Answer evaluation

```bash
python -m evaluation.evaluate_answers
```

### Reliability smoke tests

```bash
python -m evaluation.test_reliability
```

### Performance and cost benchmark

```bash
python -m evaluation.evaluate_performance
```

Evaluation outputs are saved under:

```text
evaluation/results/
```

The CSV files contain individual test or benchmark results, while the JSON summaries provide aggregated metrics.

---

# 🔐 Security & Configuration

- Keep API credentials outside source control.
- Use Streamlit secrets for deployed applications where applicable.
- Exclude `.env`, virtual environments, Python cache files, and other sensitive artifacts from Git.
- Preserve the FAISS index files required by the deployed application.

---

# 💡 Example Queries

**Knowledge Retrieval**

```text
Search the knowledge base and explain the most relevant information.
```

**RAG**

```text
Explain how embeddings and vector databases work together in a RAG pipeline.
```

**Calculation**

```text
Calculate 125 multiplied by 48.
```

**LangGraph**

```text
What is LangGraph and why is it useful for AI agents?
```

---

# 🎯 What This Project Demonstrates

### AI Engineering

- Agentic AI workflows
- LangGraph state-based orchestration
- Retrieval-Augmented Generation
- Embedding-based semantic search
- Vector databases and FAISS
- LLM integration
- Tool calling

### Evaluation & Engineering

- Retrieval evaluation using gold chunk identifiers
- Answer behavior evaluation
- LLM-as-a-judge evaluation
- Reliability smoke testing
- Latency benchmarking
- Token-usage tracking
- Estimated inference cost analysis
- Error handling and bounded model execution

### Application Development

- Python application development
- Streamlit UI development
- Modular architecture
- Conversation state management
- Interactive execution traces
- Deployment configuration

---

# 🚀 Future Improvements

Potential extensions include:

- [ ] Persistent conversation memory
- [ ] Multiple knowledge bases
- [ ] PDF, CSV, and website ingestion
- [ ] Streaming responses
- [ ] Improved source citations
- [ ] Authentication and role-based access
- [ ] Advanced retrieval strategies
- [ ] Human-reviewed evaluation datasets
- [ ] More extensive load and concurrency testing
- [ ] Production monitoring and observability

---

# 👨‍💻 Author

## Aniketanand Sandipkumar

**Aspiring Data Scientist | ML Engineer | AI Enthusiast**

Interested in:

- Artificial Intelligence
- Machine Learning
- Generative AI
- LLM & Agentic Systems
- Data Science
- Retrieval-Augmented Generation

---

# ⭐ Support

If you find this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 🐛 Open an issue
- 💡 Suggest an improvement

---

## ◈ RAGFlow AI

**Agentic Intelligence. Grounded Knowledge. Interactive Experience.**

---    

#APP LINK: https://langgraph-rag-ai-assistant-fgbc2rngjpgaegkjwjd5yc.streamlit.app/
