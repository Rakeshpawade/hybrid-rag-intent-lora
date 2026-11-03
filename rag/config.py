"""Central configuration. Every value can be overridden from the .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> str:
    p = Path(os.getenv(name, default)).expanduser()
    return str(p if p.is_absolute() else (ROOT / p).resolve())


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Models (must match the fine-tuned LoRA base model) ---
    base_model_id: str = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    adapter_path: str = _path("ADAPTER_PATH", "./models/intent_lora_best")
    use_adapter: bool = _bool("USE_ADAPTER", "true")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Vector DB (must match NB4) ---
    chroma_path: str = _path("CHROMA_PATH", "./chroma_db")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "sop_knowledge_base")
    top_k: int = int(os.getenv("TOP_K", "2"))

    # --- Runtime ---
    device: str = os.getenv("DEVICE", "auto")  # auto | cuda | mps | cpu
    load_in_4bit: bool = _bool("LOAD_IN_4BIT", "true")  # only used on CUDA
    router_max_new_tokens: int = int(os.getenv("ROUTER_MAX_NEW_TOKENS", "40"))
    answer_max_new_tokens: int = int(os.getenv("ANSWER_MAX_NEW_TOKENS", "256"))
    history_turns: int = int(os.getenv("HISTORY_TURNS", "2"))
    hf_token: str | None = os.getenv("HF_TOKEN") or None

    # --- UI ---
    app_title: str = os.getenv("APP_TITLE", "Hybrid RAG Customer Support Assistant")


settings = Settings()

# Intent (Bitext) -> SOP document id (file name without .md).
# Edit this to match how your SOP files should answer each intent.
INTENT_TO_DOC = {
    "cancel_order": "order_tracking",
    "change_order": "order_tracking",
    "change_shipping_address": "shipping_delays",
    "check_cancellation_fee": "subscription_cancellation",
    "check_invoice": "billing_disputes",
    "check_payment_methods": "payment_methods",
    "check_refund_policy": "refund_policy",
    "complaint": "escalation_matrix",
    "contact_customer_service": "working_hours",
    "contact_human_agent": "escalation_matrix",
    "create_account": "account_recovery",
    "delete_account": "data_privacy",
    "delivery_options": "shipping_delays",
    "delivery_period": "shipping_delays",
    "edit_account": "account_recovery",
    "get_invoice": "billing_disputes",
    "get_refund": "refund_policy",
    "newsletter_subscription": "data_privacy",
    "payment_issue": "billing_disputes",
    "place_order": "order_tracking",
    "recover_password": "password_reset",
    "registration_problems": "account_recovery",
    "review": "escalation_matrix",
    "set_up_shipping_address": "shipping_delays",
    "switch_account": "account_recovery",
    "track_order": "order_tracking",
    "track_refund": "refund_policy",
}
