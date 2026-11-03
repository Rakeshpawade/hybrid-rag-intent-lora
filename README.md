
---
title: Hybrid RAG Customer Support Assistant
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# 🤖 Hybrid RAG Customer Support Assistant

An end-to-end customer support assistant combining **Fine-Tuned Intent Routing (LoRA)** with **Domain-Grounded Retrieval Augmented Generation (ChromaDB)**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 📌 Architecture Overview

```mermaid
flowchart TD
    User([User Query]) --> Router[LoRA Fine-Tuned Qwen-0.5B Intent Router]
    Router -->|Extracted Intent & Category| QueryBuilder[Smart Search Query Formulation]
    QueryBuilder --> VectorDB[(ChromaDB: SOP Knowledge Base)]
    VectorDB -->|Top-k Grounded Chunks| Generator[Base Qwen-0.5B Generator]
    User -.->|Raw User Query| Generator
    Generator --> Response([Grounded, Hallucination-Free Support Answer])
```

1. **Dual-Role Single Model Architecture:** Uses a single `Qwen/Qwen2.5-0.5B-Instruct` backbone.
   * **Adapter Active:** Fast, structured intent classification (extracts user intent and category as JSON).
   * **Adapter Disabled:** Grounded response synthesis using standard operating procedure (SOP) context.
2. **Context-Aware Retrieval:** Semantic vector search with `sentence-transformers/all-MiniLM-L6-v2` over company policy / SOP documents in `ChromaDB`.
3. **Safety & Fallback:** Built-in validation ensures if JSON intent parsing fails, the pipeline gracefully falls back to raw semantic search.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Base LLM** | `Qwen/Qwen2.5-0.5B-Instruct` | Lightweight causal LM for intent classification & generation |
| **Fine-Tuning** | `PEFT / LoRA` | Domain-adapted adapter for customer support intent routing |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Dense sentence representations for retrieval |
| **Vector Store** | `ChromaDB` | Embedded vector database with cosine similarity search |
| **Frontend UI** | `Streamlit` | Interactive chat UI with live intent badges and source attribution |

---

## 🚀 Quickstart (Run Locally)

### 1. Clone repository & set up environment
```bash
git clone https://github.com/<your-username>/hybrid-rag-support.git
cd hybrid-rag-support
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure

```text
├── app.py                      # Streamlit chat interface
├── requirements.txt            # Python dependencies (CPU-optimized PyTorch)
├── .streamlit/
│   └── config.toml             # Custom theme and server settings
├── rag/
│   ├── config.py               # Central configuration & intent-to-doc mappings
│   ├── llm.py                  # Dual-role causal LM loader with LoRA toggle
│   ├── retriever.py            # ChromaDB interface and semantic search
│   └── pipeline.py             # End-to-end RAG orchestrator
├── chroma_db/                  # Persistent ChromaDB vector store
├── models/
│   └── intent_lora_best/       # Fine-tuned LoRA adapter weights (~8.6 MB)
└── README.md
```

---

## 📄 License
This project is licensed under the Apache 2.0 License.
