#!/usr/bin/env python3
"""Evaluate a trained LoRA adapter or full model against its base model."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_utils import load_wikitext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("DATASET_DIR"),
        help="Local wikitext-103-raw-v1 parquet directory; omit to use Hugging Face.",
    )
    parser.add_argument("--train-mode", choices=("lora", "full"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-eval-tokens", type=int, default=131072)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seq_len < 2:
        parser.error("--seq-len must be at least 2")
    if args.max_eval_tokens < args.seq_len:
        parser.error("--max-eval-tokens must be at least --seq-len")
    return args


def build_eval_blocks(tokenizer, split: str, seq_len: int,
                      max_eval_tokens: int,
                      dataset_dir: str | None = None) -> list[list[int]]:
    dataset = load_wikitext(split, dataset_dir)
    eos = tokenizer.eos_token or ""
    text = eos.join(row for row in dataset["text"] if row.strip())
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids = token_ids[:max_eval_tokens]
    usable = (len(token_ids) // seq_len) * seq_len
    if usable == 0:
        raise RuntimeError("Evaluation split did not produce a complete sequence")
    return [token_ids[i:i + seq_len] for i in range(0, usable, seq_len)]


def load_causal_lm(path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.config.use_cache = False
    model.eval()
    return model


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def evaluate(model, blocks: list[list[int]], label: str) -> dict:
    device = input_device(model)
    total_nll = 0.0
    total_predicted_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for index, block in enumerate(blocks, start=1):
            input_ids = torch.tensor([block], dtype=torch.long, device=device)
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            predicted_tokens = input_ids.numel() - input_ids.shape[0]
            total_nll += float(output.loss.detach().float().cpu()) * predicted_tokens
            total_predicted_tokens += predicted_tokens
            if index == 1 or index % 10 == 0 or index == len(blocks):
                print(f"{label}: evaluated {index}/{len(blocks)} blocks", flush=True)

    loss = total_nll / total_predicted_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 20 else float("inf"),
        "evaluated_blocks": len(blocks),
        "evaluated_tokens": total_predicted_tokens,
        "elapsed_seconds": time.perf_counter() - started,
    }


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    blocks = build_eval_blocks(
        tokenizer, args.split, args.seq_len, args.max_eval_tokens, args.dataset_dir
    )

    print("Loading base model", flush=True)
    base_model = load_causal_lm(args.base_model, args.device_map)
    base_metrics = evaluate(base_model, blocks, "base")

    if args.train_mode == "lora":
        adapter_config = Path(args.trained_model) / "adapter_config.json"
        if not adapter_config.is_file():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_config}")
        from peft import PeftModel

        print("Loading LoRA adapter", flush=True)
        trained_model = PeftModel.from_pretrained(base_model, args.trained_model)
        trained_model.eval()
    else:
        base_model = release_model(base_model)
        print("Loading full trained model", flush=True)
        trained_model = load_causal_lm(args.trained_model, args.device_map)

    trained_metrics = evaluate(trained_model, blocks, "trained")
    loss_delta = trained_metrics["loss"] - base_metrics["loss"]
    perplexity_improvement_percent = (
        100.0
        * (base_metrics["perplexity"] - trained_metrics["perplexity"])
        / base_metrics["perplexity"]
    )
    result = {
        "dataset": (
            str(Path(args.dataset_dir).resolve())
            if args.dataset_dir
            else "Salesforce/wikitext/wikitext-103-raw-v1"
        ),
        "split": args.split,
        "seq_len": args.seq_len,
        "max_eval_tokens": args.max_eval_tokens,
        "base_model": args.base_model,
        "trained_model": args.trained_model,
        "train_mode": args.train_mode,
        "base": base_metrics,
        "trained": trained_metrics,
        "loss_delta_trained_minus_base": loss_delta,
        "perplexity_improvement_percent": perplexity_improvement_percent,
        "improved": loss_delta < 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
