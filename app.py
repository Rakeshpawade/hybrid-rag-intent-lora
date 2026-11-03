"""Streamlit chat UI for the Hybrid RAG + Fine-Tuned Customer Support Assistant.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from rag.config import settings

st.set_page_config(page_title=settings.app_title, page_icon="💬", layout="wide")

SAMPLES = [
    "Where is my order? It was supposed to arrive yesterday.",
    "How do I get a refund for a damaged item?",
    "I forgot my password and can't log in.",
    "What payment methods do you accept?",
    "I want to talk to a human agent.",
    "How can I cancel my subscription?",
]

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1000px;}
.badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.78rem;
        font-weight:600; margin-right:6px;}
.b-intent {background:#e0ecff; color:#1d4ed8;}
.b-cat {background:#e7f7ee; color:#15803d;}
.b-warn {background:#fff1e0; color:#b45309;}
.src {font-size:0.8rem; color:#6b7280;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ backend
@st.cache_resource(show_spinner="Loading models and vector database (first run only)...")
def load_pipeline():
    from rag import HybridRAG

    return HybridRAG()


def render_meta(meta: dict, show_trace: bool) -> None:
    """Badges + sources + retrieved chunks for one assistant message."""
    if not meta:
        return
    intent, category = meta.get("intent"), meta.get("category")
    badges = ""
    if intent:
        badges += f'<span class="badge b-intent">intent: {intent}</span>'
    if category:
        badges += f'<span class="badge b-cat">category: {category}</span>'
    if not meta.get("json_valid"):
        badges += '<span class="badge b-warn">router fallback: raw query</span>'
    if badges:
        st.markdown(badges, unsafe_allow_html=True)

    chunks = meta.get("chunks", [])
    if chunks:
        srcs = ", ".join(dict.fromkeys(c["source"] for c in chunks))
        st.markdown(f'<div class="src">📎 Sources: {srcs}</div>', unsafe_allow_html=True)
        with st.expander(f"Retrieved context ({len(chunks)} chunks)"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**[{i}] {c['source']}** · similarity `{c['similarity']:.3f}`")
                st.caption(c["text"])
    if show_trace:
        with st.expander("Pipeline trace"):
            st.json(
                {
                    "router_raw_output": meta.get("router_raw"),
                    "search_string": meta.get("search_string"),
                    "timings_s": {k: round(v, 2) for k, v in meta.get("timings", {}).items()},
                }
            )


# ------------------------------------------------------------------ state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

try:
    rag = load_pipeline()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not start the pipeline: {exc}")
    st.info("Run `python scripts/check_setup.py` to diagnose the problem.")
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("⚙️ System")
    for k, v in rag.info().items():
        st.markdown(f"**{k}**  \n{v}")
    if rag.llm.adapter_error:
        st.warning(f"Adapter not loaded: {rag.llm.adapter_error}")
    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 5, settings.top_k)
    show_trace = st.toggle("Show pipeline trace", value=True)
    st.divider()
    st.markdown(
        "**Pipeline**  \n1. Fine-tuned router → intent JSON  \n"
        "2. Intent → SOP-aware ChromaDB search  \n3. Grounded answer from retrieved context"
    )
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

# ------------------------------------------------------------------ main
st.title("💬 " + settings.app_title)
st.caption("Runs 100% locally · fine-tuned intent router + RAG over the SOP knowledge base")

if not st.session_state.messages:
    st.markdown("##### Try one of these")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLES):
        if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_meta(m.get("meta", {}), show_trace)

prompt = st.chat_input("Ask a customer support question...")
if st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Routing intent and retrieving documents..."):
            res = rag.prepare(prompt, top_k=top_k)
        answer = st.write_stream(rag.stream_answer(res, history))
        meta = {
            "intent": res.intent,
            "category": res.category,
            "json_valid": res.json_valid,
            "router_raw": res.router_raw,
            "search_string": res.search_string,
            "timings": res.timings,
            "chunks": [
                {"source": c.source, "similarity": c.similarity, "text": c.text}
                for c in res.chunks
            ],
        }
        render_meta(meta, show_trace)
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
