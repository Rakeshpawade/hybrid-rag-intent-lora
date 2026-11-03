"""Pre-flight check: python scripts/check_setup.py"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def report(label: str, good: bool, detail: str = "") -> None:
    global ok
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] {label} {detail}")


for mod in ["streamlit", "torch", "transformers", "peft", "sentence_transformers", "chromadb", "dotenv"]:
    try:
        importlib.import_module(mod)
        report(f"import {mod}", True)
    except Exception as e:  # noqa: BLE001
        report(f"import {mod}", False, str(e))

from rag.config import settings  # noqa: E402

report("chroma_db folder", os.path.isdir(settings.chroma_path), settings.chroma_path)
try:
    import chromadb

    col = chromadb.PersistentClient(path=settings.chroma_path).get_collection(settings.collection_name)
    report("Chroma collection", col.count() > 0, f"'{settings.collection_name}' has {col.count()} chunks")
except Exception as e:  # noqa: BLE001
    report("Chroma collection", False, f"{e}  -> run scripts/rebuild_index.py")

has_weights = os.path.isdir(settings.adapter_path) and any(
    f.startswith("adapter_model") for f in os.listdir(settings.adapter_path)
)
report("LoRA adapter weights", has_weights, settings.adapter_path)
if not has_weights:
    print("     (app still runs with the base model; copy adapter_model.safetensors + adapter_config.json here)")

try:
    import torch

    print(f"[info] CUDA={torch.cuda.is_available()} MPS={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
except Exception:
    pass

print("\nAll good - run: streamlit run app.py" if ok else "\nFix the FAIL items above first.")
