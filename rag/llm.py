"""Loads the base model once and (optionally) attaches the LoRA intent-router adapter.

One model object serves two roles:
  * adapter ON  -> the fine-tuned intent router (emits JSON)
  * adapter OFF -> the grounded answer generator (uses the retrieved SOP context)
This keeps RAM/VRAM usage to a single 1.5B model.
"""
from __future__ import annotations

import os
import threading
from typing import Iterator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from .config import Settings


def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalLLM:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.adapter_loaded = False
        self.adapter_error: str | None = None
        self.quantized = False

        print(f">>> [LLM] Initializing tokenizer for {cfg.base_model_id}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model_id, token=cfg.hf_token, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {"token": cfg.hf_token, "trust_remote_code": True, "low_cpu_mem_usage": True}
        if self.device == "cuda":
            if cfg.load_in_4bit:
                try:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                    kwargs["device_map"] = {"": 0}
                    self.quantized = True
                except Exception:  # bitsandbytes missing
                    kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float16
        elif self.device == "mps":
            kwargs["torch_dtype"] = torch.float16
        else:
            # On CPU, use bfloat16 to halve memory footprint (~0.9GB instead of ~2GB)
            kwargs["torch_dtype"] = torch.bfloat16

        print(f">>> [LLM] Loading weights for {cfg.base_model_id} (low_cpu_mem_usage=True)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, **kwargs)
        if not self.quantized:
            model = model.to(self.device)
        print(">>> [LLM] Base model loaded into memory.", flush=True)

        # Attach the fine-tuned LoRA router if the weights exist.
        if cfg.use_adapter:
            weights_present = os.path.isdir(cfg.adapter_path) and any(
                f.startswith("adapter_model") for f in os.listdir(cfg.adapter_path)
            )
            if weights_present:
                try:
                    from peft import PeftModel

                    print(f">>> [LLM] Attaching LoRA router from {cfg.adapter_path}...", flush=True)
                    model = PeftModel.from_pretrained(model, cfg.adapter_path)
                    self.adapter_loaded = True
                    print(">>> [LLM] LoRA adapter successfully attached!", flush=True)
                except Exception as exc:  # noqa: BLE001
                    self.adapter_error = f"{type(exc).__name__}: {exc}"
                    print(f">>> [LLM Warning] Adapter load failed: {exc}", flush=True)
            else:
                self.adapter_error = (
                    f"No adapter_model.* weights found in {cfg.adapter_path}"
                )
                print(f">>> [LLM Warning] {self.adapter_error}", flush=True)
        model.eval()
        self.model = model

    # ------------------------------------------------------------------ utils
    def _encode(self, messages: list[dict]):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _adapter_ctx(self, enabled: bool):
        """Context manager that turns the adapter on/off for a call."""
        from contextlib import nullcontext

        if not self.adapter_loaded:
            return nullcontext()
        return nullcontext() if enabled else self.model.disable_adapter()

    # ------------------------------------------------------------- generation
    @torch.inference_mode()
    def generate(
        self, messages: list[dict], max_new_tokens: int, use_adapter: bool
    ) -> str:
        inputs = self._encode(messages)
        with self._adapter_ctx(use_adapter):
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(
        self, messages: list[dict], max_new_tokens: int, use_adapter: bool
    ) -> Iterator[str]:
        inputs = self._encode(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )

        def _run():
            with torch.inference_mode(), self._adapter_ctx(use_adapter):
                self.model.generate(**gen_kwargs)

        threading.Thread(target=_run, daemon=True).start()
        for piece in streamer:
            yield piece
