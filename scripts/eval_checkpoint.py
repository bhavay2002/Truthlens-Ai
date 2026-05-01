"""
eval_checkpoint.py
------------------
Run the full TruthLens evaluation pipeline against a saved checkpoint.

Usage
-----
    # auto-detect checkpoint in saved_models/
    uv run python scripts/eval_checkpoint.py

    # explicit checkpoint file
    uv run python scripts/eval_checkpoint.py --checkpoint saved_models/checkpoint.pt

    # custom data file (default: data/eval_100.json)
    uv run python scripts/eval_checkpoint.py --data data/eval_100.json

    # save JSON report
    uv run python scripts/eval_checkpoint.py --report reports/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.task_config import TASK_CONFIG
from src.evaluation.evaluate_model import evaluate

logging.basicConfig(
    format="%(levelname)s %(name)s — %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger("eval_checkpoint")


# ─────────────────────────────────────────────────────────────
# CHECKPOINT DISCOVERY
# ─────────────────────────────────────────────────────────────

def find_checkpoint(hint: str | None) -> Path:
    if hint:
        p = Path(hint)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    search_root = ROOT / "saved_models"
    candidates = [
        search_root / "checkpoint.pt",
        search_root / "model.pt",
        search_root / "best_model.pt",
    ]
    for c in candidates:
        if c.exists():
            return c

    step_dirs = sorted(
        (d for d in search_root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0,
        reverse=True,
    ) if search_root.exists() else []

    for d in step_dirs:
        for name in ("checkpoint.pt", "model.pt"):
            p = d / name
            if p.exists():
                return p

    raise FileNotFoundError(
        "No checkpoint found in saved_models/. "
        "Copy your checkpoint.pt there and re-run."
    )


# ─────────────────────────────────────────────────────────────
# CHECKPOINT LOADING
# ─────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    print(f"\nLoading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(ckpt.keys()) if isinstance(ckpt, dict) else ["<raw tensor>"]
    print(f"  Checkpoint keys: {keys}")
    return ckpt


# ─────────────────────────────────────────────────────────────
# MODEL RECONSTRUCTION
# ─────────────────────────────────────────────────────────────

def build_and_load_model(ckpt: dict):
    from src.models.architectures.hybrid_truthlens_model import HybridTruthLensModel
    from src.models.config.model_config import MultiTaskModelConfig, ModelConfigLoader

    model_cfg_path = ROOT / "config" / "model_config.yaml"
    if model_cfg_path.exists():
        cfg = ModelConfigLoader().load(model_cfg_path)
    else:
        cfg = MultiTaskModelConfig()

    model = HybridTruthLensModel(cfg)

    state_key = next(
        (k for k in ("model_state_dict", "state_dict", "model") if k in ckpt),
        None,
    )
    state_dict = ckpt[state_key] if state_key else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  ⚠  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────

def load_tokenizer(ckpt: dict):
    from transformers import AutoTokenizer

    name = (
        ckpt.get("tokenizer_name")
        or ckpt.get("model_name")
        or "roberta-base"
    )
    print(f"  Tokenizer: {name}")
    return AutoTokenizer.from_pretrained(name)


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, texts: list[str], batch_size: int = 16) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_logits: dict[str, list] = {t: [] for t in TASK_CONFIG}

    print(f"\nRunning inference on {len(texts)} texts (device={device}, batch={batch_size})")
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(**enc)

            if isinstance(outputs, dict):
                for task in TASK_CONFIG:
                    if task in outputs:
                        all_logits[task].append(outputs[task].cpu().float().numpy())
            elif hasattr(outputs, "logits"):
                logits_tensor = outputs.logits
                if isinstance(logits_tensor, dict):
                    for task, lg in logits_tensor.items():
                        if task in all_logits:
                            all_logits[task].append(lg.cpu().float().numpy())
                else:
                    all_logits.setdefault("default", []).append(
                        logits_tensor.cpu().float().numpy()
                    )
            else:
                raise RuntimeError(
                    f"Unexpected model output type: {type(outputs)}. "
                    "Expected dict of per-task logits or ModelOutput with .logits"
                )

            if (i // batch_size + 1) % 5 == 0:
                print(f"  processed {i + len(batch_texts)}/{len(texts)}")

    return {t: np.concatenate(v) for t, v in all_logits.items() if v}


# ─────────────────────────────────────────────────────────────
# POSTPROCESS → predictions + probabilities
# ─────────────────────────────────────────────────────────────

def postprocess(logits_map: dict) -> tuple[dict, dict]:
    from scipy.special import softmax, expit

    preds, probs = {}, {}
    for task, logits in logits_map.items():
        cfg = TASK_CONFIG.get(task, {})
        ttype = cfg.get("type", "multiclass")

        if ttype == "multilabel":
            p = expit(logits)
            probs[task] = p
            preds[task] = (p >= 0.5).astype(int)
        else:
            p = softmax(logits, axis=-1)
            probs[task] = p
            preds[task] = np.argmax(p, axis=1)

    return preds, probs


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

def run_evaluation(preds: dict, probs: dict, labels: dict) -> dict:
    results = {}
    tasks = sorted(set(preds) & set(labels))
    print(f"\nEvaluating {len(tasks)} tasks: {tasks}")

    for task in tasks:
        y_true = np.array(labels[task])
        y_pred = np.array(preds[task])
        y_proba = np.array(probs.get(task))

        try:
            r = evaluate(y_true=y_true, y_pred=y_pred, y_proba=y_proba, task=task)
            results[task] = r.get("metrics", r)
        except Exception as exc:
            logger.warning("Evaluation failed for %s: %s", task, exc)
            results[task] = {"error": str(exc)}

    return results


# ─────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────

def print_report(results: dict, n_samples: int) -> None:
    print("\n" + "=" * 60)
    print(f"  TruthLens Evaluation Report  ({n_samples} samples)")
    print("=" * 60)

    for task, m in results.items():
        if "error" in m:
            print(f"\n[{task:<20}]  ERROR: {m['error']}")
            continue
        cfg = TASK_CONFIG.get(task, {})
        ttype = cfg.get("type", "?")
        acc = m.get("accuracy", m.get("subset_accuracy", "n/a"))
        f1 = m.get("f1_macro", m.get("f1", "n/a"))
        prec = m.get("precision", "n/a")
        rec = m.get("recall", "n/a")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        print(
            f"\n[{task:<20}]  type={ttype:<11}"
            f"  acc={fmt(acc)}  f1={fmt(f1)}"
            f"  prec={fmt(prec)}  rec={fmt(rec)}"
        )

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate TruthLens checkpoint")
    ap.add_argument("--checkpoint", default=None, help="Path to checkpoint.pt")
    ap.add_argument("--data", default="data/eval_100.json", help="Labeled eval JSON")
    ap.add_argument("--report", default=None, help="Optional path to save JSON report")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    # ── load data ──────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    data = json.loads(data_path.read_text())
    texts = [d["text"] for d in data]
    labels = {
        task: [d[task] for d in data]
        for task in TASK_CONFIG
        if task in data[0]
    }
    print(f"Loaded {len(texts)} samples from {data_path}")
    print(f"Label tasks: {list(labels.keys())}")

    # ── checkpoint ─────────────────────────────────────────────
    ckpt_path = find_checkpoint(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)

    # ── model + tokenizer ──────────────────────────────────────
    model = build_and_load_model(ckpt)
    tokenizer = load_tokenizer(ckpt)

    # ── inference ──────────────────────────────────────────────
    logits_map = run_inference(model, tokenizer, texts, batch_size=args.batch_size)
    preds, probs = postprocess(logits_map)

    # ── evaluate ───────────────────────────────────────────────
    results = run_evaluation(preds, probs, labels)
    print_report(results, len(texts))

    # ── save report ────────────────────────────────────────────
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"results": results, "n_samples": len(texts)}, indent=2))
        print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
