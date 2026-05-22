"""Streamlit demo UI for Verbatim.

Run:
    streamlit run src/app.py

Features:
  - Chat interface for asking questions across the interview corpus
  - Pipeline selector (naive_rag vs agent_v1) — useful for the demo,
    lets you show the difference live
  - Citations panel showing which interviews backed each answer
  - Example questions sidebar so a demo viewer doesn't have to think
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


# ---- page setup ----

st.set_page_config(
    page_title="Verbatim",
    page_icon="💬",
    layout="wide",
)

st.title("Verbatim")
st.caption("Talk to your customer research interviews. Agentic search with citations.")


# ---- sidebar: settings + example questions ----

with st.sidebar:
    st.subheader("Settings")
    pipeline_name = st.selectbox(
        "Pipeline",
        ["agent_v1", "naive_rag"],
        index=0,
        help="agent_v1 uses tool-using Claude with multi-step retrieval. naive_rag is the single-shot baseline.",
    )

    st.subheader("Try asking")
    example_questions = [
        "What's the top reason trial users are churning?",
        "Which interviewees mentioned Salesforce being broken?",
        "What's the most-loved feature among Team-plan customers?",
        "Why did Yuki choose SavvyCal over Linkup?",
        "How big is Chloe's team?",
        "Who asked for a Notion integration?",
        "Compare what power users love vs what trial users hated.",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("---")
    st.caption(
        "Built as a portfolio piece. See README.md for the eval-driven "
        "development story."
    )


# ---- pipeline loader (cached so we don't rebuild between turns) ----

@st.cache_resource
def get_pipeline(name: str):
    if name == "naive_rag":
        from .naive_rag import NaiveRagPipeline
        return NaiveRagPipeline()
    if name == "agent_v1":
        from .agent import AgentPipeline
        return AgentPipeline()
    raise ValueError(f"unknown pipeline: {name}")


# ---- chat history ----

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Replay history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"Citations ({len(msg['citations'])})"):
                for cit in msg["citations"]:
                    st.markdown(
                        f"**[{cit['interview_id']}]** {cit.get('speaker', '')} — "
                        f"\"{cit.get('excerpt', '')[:300]}\""
                    )
        if msg.get("metadata"):
            with st.expander("Pipeline metadata"):
                st.json(msg["metadata"])


# ---- input ----

question = st.chat_input("Ask about the interview corpus...")

# Pending question from sidebar button click
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking ({pipeline_name})..."):
            try:
                pipeline = get_pipeline(pipeline_name)
                result = pipeline.answer(question)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.session_state["messages"].append(
                    {"role": "assistant", "content": f"Error: {e}"}
                )
                st.stop()

        st.markdown(result.answer)
        citations_payload = [
            {
                "interview_id": c.interview_id,
                "speaker": c.speaker,
                "excerpt": c.excerpt,
                "score": c.score,
            }
            for c in result.citations
        ]
        if citations_payload:
            with st.expander(f"Citations ({len(citations_payload)})"):
                for cit in citations_payload:
                    st.markdown(
                        f"**[{cit['interview_id']}]** {cit.get('speaker', '')} — "
                        f"\"{cit.get('excerpt', '')[:300]}\""
                    )
        if result.metadata:
            with st.expander("Pipeline metadata"):
                st.json(result.metadata)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result.answer,
            "citations": citations_payload,
            "metadata": result.metadata,
        }
    )
