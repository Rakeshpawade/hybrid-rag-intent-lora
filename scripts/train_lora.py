"""REAL LoRA fine-tuning of the intent router (works on CPU, NVIDIA GPU, or Colab).

Usage (from the project folder):
    python scripts/train_lora.py                 # full run
    python scripts/train_lora.py --quick         # 5-minute smoke test (small subset)

Data it looks for (first match wins):
  1) train.jsonl + val.jsonl   (made by Data_Preparation notebook; needs instruction/intent/category)
  2) sampled_data.csv          (instruction,intent,category) -> split here; df_test.csv rows are excluded

Outputs:
  ./intent_lora_best/   best checkpoint by validation loss  (adapter_model.safetensors + adapter_config.json)
  ./intent_lora/        final checkpoint
  training_log.csv, training_curves.png, eval_results.json   (REAL numbers)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are an intent routing assistant. Respond ONLY in valid JSON."


# ----------------------------------------------------------------- data
def _find(names, roots):
    for r in roots:
        for n in names:
            hits = glob.glob(os.path.join(r, "**", n), recursive=True)
            hits = [h for h in hits if ".venv" not in h and "site-packages" not in h]
            if hits:
                return hits[0]
    return None


def load_rows(root):
    roots = [root]
    tr, va = _find(["train.jsonl"], roots), _find(["val.jsonl", "valid.jsonl"], roots)
    if tr and va:
        def rd(p):
            with open(p, encoding="utf-8") as f:
                return [json.loads(l) for l in f if l.strip()]
        print(f"[data] using {tr} and {va}")
        return rd(tr), rd(va)

    csv_path = _find(["sampled_data.csv"], roots)
    if not csv_path:
        raise SystemExit(
            "Could not find train.jsonl/val.jsonl or sampled_data.csv under the project folder.\n"
            "Tell Claude which data files you have in the Dataset folder."
        )
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print(f"[data] using {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=["instruction", "intent", "category"])
    df["intent"] = df["intent"].astype(str).str.lower().str.strip()
    df["category"] = df["category"].astype(str).str.upper().str.strip()
    df["norm"] = df["instruction"].astype(str).str.lower().str.strip()
    df = df.drop_duplicates(subset=["norm"])
    test_csv = _find(["df_test.csv"], roots)
    if test_csv:  # keep the evaluation notebooks' test set unseen
        test_norm = set(pd.read_csv(test_csv)["instruction"].astype(str).str.lower().str.strip())
        before = len(df)
        df = df[~df["norm"].isin(test_norm)]
        print(f"[data] removed {before - len(df)} rows that belong to df_test.csv")
    tr_df, va_df = train_test_split(df, test_size=0.15, stratify=df["intent"], random_state=42)
    return tr_df.to_dict("records"), va_df.to_dict("records")


def encode(tok, row, max_len):
    """prompt tokens are masked (-100); loss only on the JSON answer + <|im_end|>."""
    instr = str(row["instruction"]).strip()
    target = json.dumps({"intent": str(row["intent"]).lower().strip(),
                         "category": str(row["category"]).upper().strip()})
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instr}],
        tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(target + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return ids, labels


def batches(examples, bs, pad_id, shuffle):
    idx = list(range(len(examples)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [examples[j] for j in idx[i:i + bs]]
        m = max(len(c[0]) for c in chunk)
        ids = torch.full((len(chunk), m), pad_id)
        lab = torch.full((len(chunk), m), -100)
        att = torch.zeros((len(chunk), m), dtype=torch.long)
        for k, (a, b) in enumerate(chunk):
            ids[k, :len(a)] = torch.tensor(a)
            lab[k, :len(b)] = torch.tensor(b)
            att[k, :len(a)] = 1
        yield ids, att, lab


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--val-every", type=int, default=50, help="optimizer steps between validations")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--eval-n", type=int, default=100, help="validation rows for final generation check")
    ap.add_argument("--quick", action="store_true", help="tiny run to test that everything works")
    ap.add_argument("--out", default="intent_lora_best")
    ap.add_argument("--final-out", default="intent_lora")
    a = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={dev}  torch={torch.__version__}")

    tr_rows, va_rows = load_rows(a.root)
    if a.quick:
        tr_rows, va_rows, a.epochs, a.val_every, a.eval_n = tr_rows[:160], va_rows[:40], 1, 5, 10
    print(f"[data] train={len(tr_rows)}  val={len(va_rows)}")

    tok = AutoTokenizer.from_pretrained(a.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train = [encode(tok, r, a.max_len) for r in tr_rows]
    val = [encode(tok, r, a.max_len) for r in va_rows]
    trunc = sum(1 for ids, _ in train if len(ids) >= a.max_len)
    print(f"[data] sequences at max_len ({a.max_len}): {trunc}")

    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"[model] loading {a.base_model} ({dtype}) ... first run downloads ~3 GB")
    model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=dtype, trust_remote_code=True)
    model.config.use_cache = False
    model.to(dev)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps_per_epoch = max(1, (len(train) // a.batch) // a.accum)
    total = steps_per_epoch * a.epochs
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    use_amp = dev == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def run_loss(ids, att, lab):
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
            out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev))
        return out.loss

    @torch.no_grad()
    def validate():
        model.eval()
        tot, n = 0.0, 0
        for ids, att, lab in batches(val, a.batch, tok.pad_token_id, False):
            tot += run_loss(ids, att, lab).item() * ids.size(0)
            n += ids.size(0)
        model.train()
        return tot / max(1, n)

    log, best, bad, step, t0 = [], float("inf"), 0, 0, time.time()
    run_tr, run_n = 0.0, 0
    model.train()
    os.makedirs(a.out, exist_ok=True)
    stop = False
    print(f"[train] {total} optimizer steps ({a.epochs} epochs). Validation every {a.val_every} steps.")
    for ep in range(a.epochs):
        micro = 0
        for ids, att, lab in batches(train, a.batch, tok.pad_token_id, True):
            loss = run_loss(ids, att, lab)
            scaler.scale(loss / a.accum).backward()
            run_tr += loss.item(); run_n += 1; micro += 1
            if micro % a.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1
            if step % 10 == 0:
                print(f"  step {step}/{total}  train_loss={run_tr / run_n:.4f}  {time.time() - t0:.0f}s")
            if step % a.val_every == 0 or step == total:
                vl = validate()
                tl = run_tr / max(1, run_n); run_tr, run_n = 0.0, 0
                log.append({"epoch": ep + 1, "step": step, "train_loss": round(tl, 4), "val_loss": round(vl, 4)})
                print(f"[val] step {step}: train={tl:.4f} val={vl:.4f}")
                if vl < best:
                    best, bad = vl, 0
                    model.save_pretrained(a.out)
                    print(f"      new best -> saved to {a.out}/")
                else:
                    bad += 1
                    if bad >= a.patience:
                        print("[train] early stopping")
                        stop = True
                        break
            if step >= total:
                break
        if stop or step >= total:
            break

    if not os.path.exists(os.path.join(a.out, "adapter_model.safetensors")):
        model.save_pretrained(a.out)
    model.save_pretrained(a.final_out)

    with open("training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss", "val_loss"])
        w.writeheader(); w.writerows(log)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot([r["step"] for r in log], [r["train_loss"] for r in log], "b-o", label="train")
        plt.plot([r["step"] for r in log], [r["val_loss"] for r in log], "r--s", label="validation")
        plt.xlabel("optimizer step"); plt.ylabel("loss"); plt.legend(); plt.title("Real fine-tuning loss")
        plt.savefig("training_curves.png", dpi=120, bbox_inches="tight")
    except Exception as e:  # noqa: BLE001
        print("[warn] could not draw curve:", e)

    # ---- real generation check on validation rows
    print("[eval] generating on validation rows ...")
    model.eval(); model.config.use_cache = True
    ok_json = ok_intent = ok_cat = 0
    sample = va_rows[:a.eval_n]
    for r in sample:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": str(r["instruction"]).strip()}],
            tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*?\}", txt, re.DOTALL)
        try:
            d = json.loads(m.group(0)); ok_json += 1
            ok_intent += str(d.get("intent", "")).lower() == str(r["intent"]).lower()
            ok_cat += str(d.get("category", "")).upper() == str(r["category"]).upper()
        except Exception:  # noqa: BLE001
            pass
    n = max(1, len(sample))
    res = {"n_eval": len(sample), "best_val_loss": round(best, 4),
           "json_valid_rate": round(ok_json / n, 3), "intent_accuracy": round(ok_intent / n, 3),
           "category_accuracy": round(ok_cat / n, 3), "base_model": a.base_model,
           "train_rows": len(train), "minutes": round((time.time() - t0) / 60, 1)}
    with open("eval_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("[eval]", json.dumps(res, indent=2))
    print(f"\nDONE. Check that {a.out}/adapter_model.safetensors exists, then set USE_ADAPTER=true in .env")


if __name__ == "__main__":
    main()