---
title: Hybrid RAG Customer Support Assistant
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# 💬 Hybrid RAG Customer Support Assistant

An end-to-end customer support assistant combining **Fine-Tuned Intent Routing (LoRA)** with **Domain-Grounded Retrieval Augmented Generation (ChromaDB)**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hybrid-rag-intent-lora-support.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Rakeshpawade/hybrid-rag-intent-lora)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Base_LLM-Qwen2.5--0.5B--Instruct-orange)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![PEFT](https://img.shields.io/badge/Fine--Tuning-LoRA_Adapter-green)](https://github.com/huggingface/peft)

🔗 **Try the Live Web App:** [https://hybrid-rag-intent-lora-support.streamlit.app/](https://hybrid-rag-intent-lora-support.streamlit.app/)

---

## 🖥️ App Overview & UI Features

When a user or recruiter interacts with the app, the pipeline dynamically demonstrates intent classification, semantic retrieval, and policy-grounded generation:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│  💬 Hybrid RAG Customer Support Assistant                                      │
│  Runs 100% locally · fine-tuned intent router + RAG over SOP knowledge base     │
│                                                                                 │
│  👤 User: Where is my order? It was supposed to arrive yesterday.              │
│                                                                                 │
│  🤖 Assistant:                                                                  │
│  ┌─────────────────────────┐  ┌───────────────────────┐                         │
│  │  intent: track_order    │  │   category: order     │                         │
│  └─────────────────────────┘  └───────────────────────┘                         │
│  📎 Sources: order_tracking.md                                                  │
│                                                                                 │
│  To check the status of your order, please log into your account and navigate   │
│  to the 'Order History' section to locate your real-time tracking number.       │
│                                                                                 │
│  ▼ Retrieved context (2 chunks)                                                 │
│    • [1] order_tracking.md · similarity 0.884                                   │
│    • [2] shipping_delays.md · similarity 0.761                                  │
│                                                                                 │
│  ▼ Pipeline trace (JSON router output & latency)                                │
│    • {"router_raw_output": "{\"intent\": \"track_order\"}", "timings_s": 0.42}   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key UI Features:
1. **Interactive Sample Buttons:** One-click customer queries covering order tracking, refunds, password recovery, billing disputes, and human escalations.
2. **Intent Badges:** Color-coded badges indicating the structured JSON output (`intent` and `category`) from the fine-tuned LoRA router.
3. **Source Attribution & Cosine Similarity:** Transparent citations showing exactly which SOP policy file and chunk informed the answer.
4. **Pipeline Trace:** Expandable debugging telemetry showing search query formulations and execution latency.
5. **Configurable Retrieval Sidebar:** Adjust top-k chunk retrieval on the fly and monitor adapter status.

---

## 📌 Architecture Overview

```mermaid
flowchart TD
    User([User Query]) --> Router[LoRA Fine-Tuned Qwen-0.5B Intent Router]
    Router -->|Extracted Intent & Category JSON| QueryBuilder[Smart Search Query Formulation]
    QueryBuilder --> VectorDB[(ChromaDB: SOP Knowledge Base)]
    VectorDB -->|Top-k Grounded Chunks| Generator[Base Qwen-0.5B Generator]
    User -.->|Raw User Query| Generator
    Generator --> Response([Grounded, Hallucination-Free Support Answer])
```

1. **Dual-Role Single Model Architecture:** A single `Qwen/Qwen2.5-0.5B-Instruct` backbone serves both routing and generation:
   * **Adapter Active:** Fast, structured intent classification (extracts user intent and category as JSON).
   * **Adapter Disabled:** Grounded response synthesis using standard operating procedure (SOP) context.
2. **Context-Aware Retrieval:** Semantic vector search with `sentence-transformers/all-MiniLM-L6-v2` over company policy / SOP documents in `ChromaDB`.
3. **Safety & Fallback:** Built-in validation ensures if JSON intent parsing fails, the pipeline gracefully falls back to raw semantic search.

---

## 📊 Model Training & Evaluation Visuals

| Fine-Tuning LoRA Curves | SOP Retrieval Cosine Similarity | Top Customer Support Intents |
|:---:|:---:|:---:|
| ![Training Curves](assets/training_curves.png) | ![SOP Cosine Similarity](assets/sop_cosine_similarity.png) | ![Top Intents](assets/top10_intents.png) |

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Base LLM** | `Qwen/Qwen2.5-0.5B-Instruct` | Lightweight causal LM for intent classification & generation |
| **Fine-Tuning** | `PEFT / LoRA` | Domain-adapted adapter for customer support intent routing |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Dense representations for semantic search |
| **Vector Store** | `ChromaDB` | Embedded vector database with cosine similarity search |
| **Frontend UI** | `Streamlit` | Interactive chat UI with live intent badges and source attribution |

---

## 🚀 Quickstart (Run Locally)

### 1. Clone repository & set up environment
```bash
git clone https://github.com/Rakeshpawade/hybrid-rag-intent-lora.git
cd hybrid-rag-intent-lora
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
├── assets/                     # Architecture diagrams and evaluation plots
├── Hybrid_RAG_App_Creation_and_Deployment_Guide.pdf  # Complete documentation
└── README.md
```

---

## 📄 License
This project is licensed under the Apache 2.0 License.
