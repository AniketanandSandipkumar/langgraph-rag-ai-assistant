# ◈ RAGFlow AI

### Intelligent Knowledge Assistant powered by LangGraph, RAG & Streamlit

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1C3C3C?logo=langchain\&logoColor=white)](https://www.langchain.com/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=chainlink\&logoColor=white)](https://www.langchain.com/)
[![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-6366F1)]()
[![AI](https://img.shields.io/badge/AI-Agentic%20AI-8B5CF6)]()
[![GitHub](https://img.shields.io/badge/Repository-GitHub-181717?logo=github\&logoColor=white)](https://github.com/)

> **RAGFlow AI** is an intelligent knowledge assistant that combines **Retrieval-Augmented Generation (RAG)** with a **LangGraph-powered agent architecture** and a premium **Streamlit conversational interface**.
>
> The application intelligently processes user requests, retrieves relevant knowledge when required, performs calculations through tools, and presents the execution flow through an interactive agent-activity interface.

---

## ✨ Why This Project?

Modern AI applications are moving beyond simple chatbot interfaces.

RAGFlow AI demonstrates how an AI assistant can combine:

**User Query → Agent Reasoning → Tool Selection → Knowledge Retrieval / Calculation → Final Response**

Instead of forcing every question through the same pipeline, the application is designed around an **agent-based workflow** capable of deciding when tools are useful.

This project demonstrates practical experience with:

* 🤖 Agentic AI
* 🔗 LangGraph
* 📚 Retrieval-Augmented Generation
* 🔎 Knowledge-base retrieval
* 🛠️ Tool calling
* 🧮 Calculator integration
* 💬 Conversational interfaces
* 🎨 Custom Streamlit UI
* 📊 Agent execution tracing
* 🧠 Stateful conversation handling

---

## 🖥️ Application Preview

> Add screenshots of your actual application here.

### Main Interface

![RAGFlow AI Interface](visualizations/ragflow-main.png)

### Agent Activity & Tool Trace

![Agent Activity](visualizations/agent-trace.png)

### Knowledge Retrieval

![Knowledge Base Search](visualizations/knowledge-search.png)

> **Tip:** For GitHub, screenshots of the actual running application will make this README significantly more impressive than text alone.

---

# 🚀 Key Features

## 🤖 LangGraph Agent

The application uses a LangGraph-based agent to process incoming user requests.

The agent receives the user's message and determines the appropriate execution path instead of relying on a fixed sequence.

```text
User Query
    │
    ▼
┌──────────────────┐
│  LangGraph Agent │
└────────┬─────────┘
         │
         ├───────────────┐
         ▼               ▼
 Knowledge Search    Calculator
         │               │
         └───────┬───────┘
                 ▼
          Final AI Response
```

---

## 📚 Retrieval-Augmented Generation

RAG enables the assistant to retrieve relevant information from a knowledge base before generating an answer.

This helps the system provide responses grounded in available information rather than relying exclusively on the language model's internal knowledge.

### RAG workflow

```text
User Question
      │
      ▼
Query Processing
      │
      ▼
Knowledge Base Search
      │
      ▼
Relevant Information
      │
      ▼
LLM Generation
      │
      ▼
Grounded Response
```

---

## 🔎 Knowledge Base Search

The assistant provides a dedicated knowledge-base search capability.

Users can ask questions requiring information from the application's available knowledge source and allow the agent to retrieve relevant content.

Example:

```text
"Search the knowledge base and explain the most relevant information."
```

---

## 🧮 Calculator Tool

The agent can also route numerical questions to a calculator tool.

Example:

```text
Calculate 125 * 48 and explain the result.
```

This demonstrates the ability to combine **LLM reasoning with deterministic tool execution**.

---

## 💬 Conversational Interface

The application provides a conversational interface built with Streamlit.

It supports:

* User messages
* AI responses
* Conversation history
* Custom avatars
* Chat input
* Suggested prompts
* New conversation functionality

Conversation state is maintained using Streamlit session state.

---

## 🔍 Agent Activity Monitoring

One of the project's standout features is its **agent execution trace**.

The sidebar provides visibility into:

* Tool calls
* Tool names
* Tool inputs
* Tool outputs
* Execution status
* Turn-by-turn activity

This makes the application more transparent and useful for debugging and understanding agent behavior.

Example:

```text
Turn 1
 └── Knowledge Search
      ├── Input
      └── Output

Turn 2
 └── Calculator
      ├── Input
      └── Output
```

---

# 🎨 Premium Streamlit UI

The application goes beyond Streamlit's default appearance with a custom-designed interface.

The UI includes:

* 🌑 Dark theme
* ✨ Gradient visual elements
* 🎯 Custom branding
* 📱 Wide responsive layout
* 🧩 Capability cards
* 📊 Agent activity panels
* 💬 Custom chat styling
* 🔘 Interactive controls
* 🎨 Custom CSS
* ⚡ Processing status indicators

The application is branded as **RAGFlow AI — LangGraph Assistant** and presents capabilities such as Knowledge Base Search, Calculator, LangGraph Agent, and Retrieval Augmented Generation.

---

# 🧠 Architecture

At a high level, the application follows an agentic architecture:

```text
                         ┌──────────────────────┐
                         │      User Query      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Streamlit UI       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   LangGraph Agent    │
                         └──────────┬───────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
              ┌────────────────┐       ┌────────────────┐
              │ Knowledge Base │       │   Calculator   │
              │     Search     │       │      Tool      │
              └───────┬────────┘       └───────┬────────┘
                      │                        │
                      └────────────┬───────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │     AI Response      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Conversation History │
                         │ + Agent Activity     │
                         └──────────────────────┘
```

---

# 🔄 Request Lifecycle

When a user submits a question:

### 1. User Input

The question is captured through Streamlit's chat interface.

### 2. State Initialization

The application creates an agent state containing the user's message and LLM call tracking.

### 3. Agent Execution

The request is passed to the LangGraph agent.

### 4. Tool Selection

Depending on the request, the agent can invoke an appropriate tool.

### 5. Tool Execution

The selected tool performs its task and returns its result.

### 6. Response Generation

The agent produces the final response based on the available information.

### 7. Trace Generation

Tool calls and their outputs are recorded and displayed in the sidebar.

### 8. Conversation Update

The user and assistant messages are stored in the current session.

The Streamlit application initializes and maintains both `chat_history` and `tool_traces` in session state.

---

# 🛠️ Technology Stack

| Technology         | Purpose                                     |
| ------------------ | ------------------------------------------- |
| 🐍 Python 3.13     | Application development                     |
| 🖥️ Streamlit      | Web application & chat UI                   |
| 🔗 LangGraph       | Agent orchestration                         |
| 🦜 LangChain       | LLM/agent ecosystem                         |
| 📚 RAG             | Knowledge-grounded generation               |
| 🔎 Retrieval       | Knowledge-base search                       |
| 🧮 Calculator Tool | Deterministic numerical computation         |
| 🎨 HTML/CSS        | Custom UI styling                           |
| 🧠 LLM             | Natural-language understanding & generation |

---

# 📂 Project Structure

```text
RAGFlow-AI/
│
├── app.py
├── agent.py
├── requirements.txt
├── README.md
│
├── data/
│   └── ...
│
├── knowledge_base/
│   └── ...
│
├── visualizations/
│   ├── ragflow-main.png
│   ├── agent-trace.png
│   └── knowledge-search.png
│
└── docs/
    └── ...
```

> Adjust the folder names above to match your actual repository structure.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
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

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser.

The project's current Streamlit entry point is `app.py`.

---

# 💡 Example Queries

Try asking:

### 📚 Knowledge Retrieval

```text
Search the knowledge base and explain the most relevant information.
```

### 🧠 RAG

```text
Explain how Retrieval Augmented Generation works.
```

### 🧮 Calculation

```text
Calculate 125 * 48 and explain the result.
```

### 🤖 LangGraph

```text
What is LangGraph and why is it useful for AI agents?
```

The application also provides these example prompts directly on its welcome screen.

---

# 📊 What This Project Demonstrates

### Technical Skills

* Python application development
* LangGraph agent workflows
* RAG architecture
* Tool calling
* LLM integration
* State management
* Streamlit application development
* Custom HTML/CSS styling
* Conversational AI
* Agent observability

### AI/ML Concepts

* Retrieval-Augmented Generation
* Agentic workflows
* Tool-augmented LLMs
* Knowledge retrieval
* Context-aware generation
* Deterministic tool execution
* Human-readable agent traces

### Software Engineering

* Modular architecture
* Session-state management
* Error-resistant tool handling
* Interactive UI design
* Separation of agent and interface logic
* Reproducible environment setup

---

# 🔐 Environment Variables

If your `agent.py` or other project files require API keys, create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

Do **not** commit API keys or other secrets to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

> Update this section according to the actual LLM/provider configuration used by your `agent.py`.

---

# 🚀 Future Improvements

Potential extensions for the project include:

* [ ] Persistent conversation history
* [ ] Multiple knowledge bases
* [ ] Document upload functionality
* [ ] PDF/CSV/website ingestion
* [ ] Streaming LLM responses
* [ ] Source citations in RAG responses
* [ ] User authentication
* [ ] Conversation export
* [ ] LangSmith-based observability
* [ ] Cloud deployment
* [ ] Advanced evaluation of retrieval quality
* [ ] RAGAS-based RAG evaluation
* [ ] Multi-agent workflows

---

# 📈 Future Architecture

The project can be extended into a production-oriented AI platform:

```text
                 ┌───────────────────────┐
                 │      Web Client       │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │      AI Gateway       │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   LangGraph Agent     │
                 └───────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
      ┌──────────┐     ┌──────────┐    ┌────────────┐
      │   RAG    │     │  Tools   │    │  Memory    │
      └────┬─────┘     └────┬─────┘    └─────┬──────┘
           │                │                │
           ▼                ▼                ▼
      Vector DB        External APIs    Persistent DB
                             │
                             ▼
                    ┌────────────────┐
                    │  LLM Response  │
                    └────────────────┘
```

---

# 🏆 Project Highlights

| Area            | Implementation          |
| --------------- | ----------------------- |
| Architecture    | Agentic AI              |
| Agent Framework | LangGraph               |
| AI Pattern      | RAG                     |
| Interface       | Streamlit               |
| Retrieval       | Knowledge Base Search   |
| Tool Use        | Calculator              |
| State           | Streamlit Session State |
| Observability   | Tool Execution Traces   |
| UI              | Custom HTML/CSS         |
| Python          | 3.13                    |

---

# 🎯 Why It Matters

This project goes beyond building a basic chatbot.

It demonstrates the transition from:

```text
Traditional Chatbot
       ↓
LLM Application
       ↓
RAG Application
       ↓
Tool-Augmented AI
       ↓
Agentic AI System
```

The focus is on building an AI application that can **reason about the task, use appropriate capabilities, retrieve information when required, execute tools, and present the resulting workflow to the user**.

---

# 👨‍💻 Author

## Aniketanand Sandipkumar

**Aspiring Data Scientist | ML Engineer | AI Enthusiast**

Interested in:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 📚 Generative AI
* 🔗 LLM & Agentic Systems
* 📊 Data Science
* 🔎 Retrieval-Augmented Generation

---

# ⭐ Support

If you find this project useful or interesting:

⭐ **Star the repository**

🍴 **Fork the project**

🐛 **Open an issue**

💡 **Suggest an improvement**

---

## 📌 Final Note

RAGFlow AI was built to explore how modern **LLMs, RAG pipelines, agent orchestration, tool calling, and interactive interfaces** can be combined into a practical AI assistant.

> **From asking a question → to retrieving knowledge → to executing tools → to generating an intelligent response.**

### ◈ RAGFlow AI

**Agentic Intelligence. Grounded Knowledge. Interactive Experience.**
