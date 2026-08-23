"""
LangGraph RAG Assistant — Premium Streamlit UI

Run:
    streamlit run app.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import MessagesState, agent


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="RAGFlow AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------------------ */
    /* Global                                                              */
    /* ------------------------------------------------------------------ */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 50% -20%,
                rgba(99, 102, 241, 0.12),
                transparent 35%
            ),
            #09090b;
        color: #f4f4f5;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 8rem;
    }

    /* Remove Streamlit default top spacing */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ------------------------------------------------------------------ */
    /* Sidebar                                                             */
    /* ------------------------------------------------------------------ */

    section[data-testid="stSidebar"] {
        background: #0d0d10;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.2rem 1rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.8rem;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;

        display: flex;
        align-items: center;
        justify-content: center;

        background:
            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            );

        color: white;
        font-size: 21px;
        font-weight: 700;

        box-shadow:
            0 8px 25px rgba(99,102,241,0.25);
    }

    .brand-title {
        font-size: 16px;
        font-weight: 700;
        color: #fafafa;
        letter-spacing: -0.3px;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #71717a;
        margin-top: 2px;
    }

    .status-card {
        border: 1px solid rgba(255,255,255,0.07);
        background: rgba(255,255,255,0.025);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 20px;
    }

    .status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #a1a1aa;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34,197,94,0.7);
    }

    .section-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #52525b;
        text-transform: uppercase;
        margin: 22px 0 10px 2px;
    }

    .capability {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 4px;
        color: #a1a1aa;
        font-size: 12px;
    }

    .capability-icon {
        width: 27px;
        height: 27px;
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.05);
        font-size: 13px;
    }

    /* ------------------------------------------------------------------ */
    /* Main Header                                                         */
    /* ------------------------------------------------------------------ */

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .topbar-title {
        font-size: 14px;
        font-weight: 600;
        color: #a1a1aa;
    }

    .tech-badges {
        display: flex;
        gap: 7px;
    }

    .tech-badge {
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.025);
        color: #71717a;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 10px;
        font-weight: 500;
    }

    /* ------------------------------------------------------------------ */
    /* Welcome Screen                                                      */
    /* ------------------------------------------------------------------ */

    .welcome {
        text-align: center;
        padding: 5rem 1rem 3rem;
    }

    .welcome-orb {
        width: 72px;
        height: 72px;
        margin: 0 auto 22px;

        border-radius: 22px;

        display: flex;
        align-items: center;
        justify-content: center;

        background:
            linear-gradient(
                135deg,
                rgba(99,102,241,0.95),
                rgba(139,92,246,0.95)
            );

        font-size: 30px;

        box-shadow:
            0 20px 60px rgba(99,102,241,0.22);
    }

    .welcome h1 {
        font-size: 32px;
        letter-spacing: -1px;
        margin-bottom: 10px;
        color: #fafafa;
    }

    .welcome p {
        color: #71717a;
        font-size: 14px;
        max-width: 570px;
        margin: 0 auto;
        line-height: 1.7;
    }

    .prompt-title {
        text-align: left;
        max-width: 760px;
        margin: 2rem auto 10px;
        font-size: 11px;
        color: #52525b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* ------------------------------------------------------------------ */
    /* Chat Messages                                                       */
    /* ------------------------------------------------------------------ */

    div[data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 0.8rem 0;
    }

    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        font-size: 14px;
        line-height: 1.75;
    }

    div[data-testid="stChatMessage"] p {
        margin-bottom: 0.7rem;
    }

    /* Avatar */
    div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon"] {
        border-radius: 10px;
    }

    /* ------------------------------------------------------------------ */
    /* Code Blocks                                                         */
    /* ------------------------------------------------------------------ */

    pre {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
    }

    /* ------------------------------------------------------------------ */
    /* Tool Trace                                                          */
    /* ------------------------------------------------------------------ */

    .tool-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #a1a1aa;
    }

    .tool-icon {
        width: 27px;
        height: 27px;
        border-radius: 7px;
        background: rgba(99,102,241,0.12);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .tool-name {
        color: #d4d4d8;
        font-weight: 600;
    }

    .tool-complete {
        margin-left: auto;
        color: #22c55e;
        font-size: 10px;
    }

    /* ------------------------------------------------------------------ */
    /* Buttons                                                             */
    /* ------------------------------------------------------------------ */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.025);
        color: #a1a1aa;
        font-size: 12px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: rgba(99,102,241,0.5);
        background: rgba(99,102,241,0.08);
        color: #e4e4e7;
    }

    /* ------------------------------------------------------------------ */
    /* Chat Input                                                          */
    /* ------------------------------------------------------------------ */

    div[data-testid="stChatInput"] {
        border-top: 1px solid rgba(255,255,255,0.06);
        background: rgba(9,9,11,0.85);
    }

    div[data-testid="stChatInput"] textarea {
        background: #141417 !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 14px !important;
        color: #fafafa !important;
        font-size: 14px !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 0 0 1px rgba(99,102,241,0.2) !important;
    }

    /* ------------------------------------------------------------------ */
    /* Expander                                                            */
    /* ------------------------------------------------------------------ */

    div[data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        background: rgba(255,255,255,0.015);
    }

    /* ------------------------------------------------------------------ */
    /* Scrollbar                                                           */
    /* ------------------------------------------------------------------ */

    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: #27272a;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE
# ============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "tool_traces" not in st.session_state:
    st.session_state.tool_traces = []


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">◈</div>
            <div>
                <div class="brand-title">RAGFlow AI</div>
                <div class="brand-subtitle">LangGraph Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="status-card">
            <div class="status-row">
                <span class="status-dot"></span>
                <span>Agent online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Capabilities</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="capability">
            <div class="capability-icon">⌕</div>
            <span>Knowledge Base Search</span>
        </div>

        <div class="capability">
            <div class="capability-icon">∑</div>
            <span>Calculator</span>
        </div>

        <div class="capability">
            <div class="capability-icon">◇</div>
            <span>LangGraph Agent</span>
        </div>

        <div class="capability">
            <div class="capability-icon">◉</div>
            <span>Retrieval Augmented Generation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Agent Activity</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.tool_traces:
        st.caption("No activity yet.")
    else:

        for i, trace in enumerate(
            reversed(st.session_state.tool_traces)
        ):

            turn_number = len(st.session_state.tool_traces) - i

            with st.expander(
                f"Turn {turn_number}",
                expanded=(i == 0),
            ):

                if not trace:

                    st.markdown(
                        """
                        <div class="tool-header">
                            <div class="tool-icon">✓</div>
                            <span>No tools required</span>
                            <span class="tool-complete">DIRECT</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    for call in trace:

                        tool_name = call.get("name", "Unknown")
                        args = call.get("args", {})
                        result = call.get("result", "")

                        icon = (
                            "⌕"
                            if "search" in tool_name.lower()
                            else "∑"
                            if "calc" in tool_name.lower()
                            else "◇"
                        )

                        st.markdown(
                            f"""
                            <div class="tool-header">
                                <div class="tool-icon">{icon}</div>
                                <span class="tool-name">{tool_name}</span>
                                <span class="tool-complete">✓ DONE</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        with st.expander("View execution details"):

                            st.markdown(
                                "**Input**"
                            )

                            st.code(
                                json.dumps(
                                    args,
                                    indent=2,
                                    ensure_ascii=False,
                                ),
                                language="json",
                            )

                            st.markdown(
                                "**Output**"
                            )

                            st.code(
                                str(result),
                                language="text",
                            )


    st.markdown(
        '<div class="section-label">Session</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "＋ New conversation",
        use_container_width=True,
    ):

        st.session_state.chat_history = []
        st.session_state.tool_traces = []

        st.rerun()

    st.markdown(
        """
        <div style="
            position: fixed;
            bottom: 20px;
            font-size: 10px;
            color: #3f3f46;
        ">
            LangGraph · RAG · AI Assistant
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown(
    """
    <div class="topbar">
        <div class="topbar-title">
            Intelligent Knowledge Assistant
        </div>

        <div class="tech-badges">
            <span class="tech-badge">LANGGRAPH</span>
            <span class="tech-badge">RAG</span>
            <span class="tech-badge">AI</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# WELCOME SCREEN
# ============================================================================

if not st.session_state.chat_history:

    st.markdown(
        """
        <div class="welcome">

            <div class="welcome-orb">
                ◈
            </div>

            <h1>How can I help you?</h1>

            <p>
                Ask questions about your knowledge base, retrieve relevant
                information, or use the built-in calculator.
                Your request is intelligently routed through a LangGraph agent.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="prompt-title">Try asking</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "⌕  Search my knowledge base",
            use_container_width=True,
        ):
            st.session_state.pending_prompt = (
                "Search the knowledge base and explain the most relevant information."
            )
            st.rerun()

    with col2:

        if st.button(
            "◇  Explain a RAG concept",
            use_container_width=True,
        ):
            st.session_state.pending_prompt = (
                "Explain how Retrieval Augmented Generation works."
            )
            st.rerun()

    col3, col4 = st.columns(2)

    with col3:

        if st.button(
            "∑  Calculate something",
            use_container_width=True,
        ):
            st.session_state.pending_prompt = (
                "Calculate 125 * 48 and explain the result."
            )
            st.rerun()

    with col4:

        if st.button(
            "◉  Explain LangGraph",
            use_container_width=True,
        ):
            st.session_state.pending_prompt = (
                "What is LangGraph and why is it useful for AI agents?"
            )
            st.rerun()


# ============================================================================
# CHAT HISTORY
# ============================================================================

for role, content in st.session_state.chat_history:

    avatar = "👤" if role == "user" else "🤖"

    with st.chat_message(
        role,
        avatar=avatar,
    ):
        st.markdown(content)


# ============================================================================
# INPUT
# ============================================================================

pending_prompt = st.session_state.pop(
    "pending_prompt",
    None,
)

prompt = st.chat_input(
    "Ask about your knowledge base, calculate something, or explore an AI concept..."
)

if pending_prompt:
    prompt = pending_prompt


# ============================================================================
# AGENT EXECUTION
# ============================================================================

if prompt:

    # ---------------------------------------------------------------
    # User message
    # ---------------------------------------------------------------

    st.session_state.chat_history.append(
        ("user", prompt)
    )

    with st.chat_message(
        "user",
        avatar="👤",
    ):
        st.markdown(prompt)

    # ---------------------------------------------------------------
    # Agent
    # ---------------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖",
    ):

        with st.status(
            "Processing your request...",
            expanded=False,
        ) as status:

            initial_state: MessagesState = {
                "messages": [
                    HumanMessage(content=prompt)
                ],
                "llm_calls": 0,
            }

            result = agent.invoke(initial_state)

            status.update(
                label="Response generated",
                state="complete",
                expanded=False,
            )

        # -----------------------------------------------------------
        # Parse messages
        # -----------------------------------------------------------

        messages = result["messages"]

        pending_calls: dict[str, dict] = {}

        for msg in messages:

            if (
                isinstance(msg, AIMessage)
                and msg.tool_calls
            ):

                for tc in msg.tool_calls:

                    pending_calls[tc["id"]] = {
                        "name": tc["name"],
                        "args": tc["args"],
                        "result": "",
                    }

            elif isinstance(msg, ToolMessage):

                if msg.tool_call_id in pending_calls:

                    pending_calls[
                        msg.tool_call_id
                    ]["result"] = msg.content

        trace = list(
            pending_calls.values()
        )

        st.session_state.tool_traces.append(
            trace
        )

        # -----------------------------------------------------------
        # Find final answer
        # -----------------------------------------------------------

        final_answer = ""

        for msg in reversed(messages):

            if (
                isinstance(msg, AIMessage)
                and not msg.tool_calls
            ):

                final_answer = msg.content
                break

        # -----------------------------------------------------------
        # Display response
        # -----------------------------------------------------------

        if final_answer:

            st.markdown(final_answer)

        else:

            st.markdown(
                "_The agent did not return a final response._"
            )

        st.session_state.chat_history.append(
            (
                "assistant",
                final_answer,
            )
        )

    # Refresh sidebar activity
    st.rerun()