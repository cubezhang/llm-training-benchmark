#!/usr/bin/env python3
"""Reproducible Qwen3/WikiText-103 scaling benchmark."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from dataset_utils import load_wikitext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--dataset-dir",
        default=os.environ.get("DATASET_DIR"),
        help="Local wikitext-103-raw-v1 parquet directory; omit to use Hugging Face.",
    )
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--local-batch", type=int, default=32,
                   help="Effective batch per accelerator; must be divisible by micro-batch.")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="Steps excluded from performance aggregation.")
    p.add_argument("--measure-window", type=int, default=400,
                   help="Use this many final post-warmup steps for the headline metrics.")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deepspeed", default="none",
                   help="DeepSpeed JSON path, or 'none' for plain DDP.")
    p.add_argument("--train-mode", choices=("lora", "full"), default="lora")
    p.add_argument(
        "--save-final-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the final trained model under OUTPUT_DIR/final_model (default: enabled).",
    )
    p.add_argument(
        "--final-model-subdir",
        default="final_model",
        help="Subdirectory of output-dir used for the final model.",
    )
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated linear module suffixes shared by dense and MoE Qwen.",
    )
    p.add_argument("--active-parameters-billion", type=float, default=None)
    p.add_argument("--theoretical-tflops-per-device", type=float, default=None)
    args = p.parse_args()
    if not 1 <= args.max_steps <= 2000:
        p.error("--max-steps must be in [1, 2000]")
    if args.local_batch % args.micro_batch:
        p.error("--local-batch must be divisible by --micro-batch")
    if args.warmup_steps >= args.max_steps:
        p.error("--warmup-steps must be smaller than --max-steps")
    return args


class StepMetrics(TrainerCallback):
    def __init__(self, output_dir: str, warmup_steps: int, measure_window: int,
                 global_batch: int, seq_len: int, world_size: int,
                 active_parameters_billion: float | None,
                 theoretical_tflops_per_device: float | None):
        self.output_dir = Path(output_dir)
        self.warmup_steps = warmup_steps
        self.measure_window = measure_window
        self.global_batch = global_batch
        self.seq_len = seq_len
        self.world_size = world_size
        self.active_parameters_billion = active_parameters_billion
        self.theoretical_tflops_per_device = theoretical_tflops_per_device
        self.started = 0.0
        self.records: list[dict] = []
        self.losses: dict[int, float] = {}

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        self.records.append({"step": int(state.global_step), "step_seconds": elapsed})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses[int(state.global_step)] = float(logs["loss"])

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "steps.jsonl").open("w", encoding="utf-8") as f:
            for row in self.records:
                f.write(json.dumps(row) + "\n")

        eligible = [r for r in self.records if r["step"] > self.warmup_steps]
        measured = eligible[-self.measure_window:]
        if not measured:
            raise RuntimeError("No measured steps remain after warmup")
        mean_s = statistics.fmean(r["step_seconds"] for r in measured)
        step_times = sorted(r["step_seconds"] for r in measured)
        p95_s = step_times[min(len(step_times) - 1, math.ceil(0.95 * len(step_times)) - 1)]
        cv_percent = (
            100 * statistics.pstdev(step_times) / mean_s if len(step_times) > 1 else 0.0
        )
        tokens_per_step = self.global_batch * self.seq_len
        throughput = tokens_per_step / mean_s
        per_device = throughput / self.world_size
        mfu = None
        if self.active_parameters_billion and self.theoretical_tflops_per_device:
            achieved_tflops = 6 * self.active_parameters_billion * 1e9 * throughput / 1e12
            mfu = 100 * achieved_tflops / (
                self.world_size * self.theoretical_tflops_per_device
            )

        def loss_at(step: int):
            # Do not report a later checkpoint label before training has
            # actually reached it (for example, a 10-step smoke test must not
            # populate the step-500 or step-1000 columns with final loss).
            if int(state.global_step) < step:
                return None
            candidates = [s for s in self.losses if s <= step]
            return self.losses[max(candidates)] if candidates else None

        metrics = {
            "world_size": self.world_size,
            "global_batch_size": self.global_batch,
            "seq_len": self.seq_len,
            "tokens_per_step": tokens_per_step,
            "measurement_first_step": measured[0]["step"],
            "measurement_last_step": measured[-1]["step"],
            "measured_steps": len(measured),
            "active_parameters_billion": self.active_parameters_billion,
            "mean_step_ms": mean_s * 1000,
            "p95_step_ms": p95_s * 1000,
            "step_time_cv_percent": cv_percent,
            "throughput_tokens_per_sec": throughput,
            "per_device_tokens_per_sec": per_device,
            "estimated_mfu_percent": mfu,
            "peak_memory_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available() else None
            ),
            "peak_memory_reserved_gib": (
                torch.cuda.max_memory_reserved() / 1024**3
                if torch.cuda.is_available() else None
            ),
            "device_memory_total_gib": (
                torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                / 1024**3 if torch.cuda.is_available() else None
            ),
            "loss_step_500": loss_at(500),
            "loss_step_1000": loss_at(1000),
            "final_step": int(state.global_step),
            "final_loss": loss_at(int(state.global_step)),
        }
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def build_dataset(tokenizer, seq_len: int, dataset_dir: str | None = None):
    raw = load_wikitext("train", dataset_dir)
    eos = tokenizer.eos_token or ""

    def tokenize(batch):
        return tokenizer([text + eos for text in batch["text"]],
                         add_special_tokens=False, return_attention_mask=False)

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names,
                        desc="Tokenizing WikiText-103")

    def pack(batch):
        joined = list(itertools.chain.from_iterable(batch["input_ids"]))
        usable = (len(joined) // seq_len) * seq_len
        ids = [joined[i:i + seq_len] for i in range(0, usable, seq_len)]
        return {"input_ids": ids}

    return tokenized.map(pack, batched=True, desc=f"Packing to {seq_len} tokens")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gradient_accumulation = args.local_batch // args.micro_batch
    global_batch = args.local_batch * world_size

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(tokenizer, args.seq_len, args.dataset_dir)

    # Construct this before from_pretrained: Transformers uses the DeepSpeed
    # configuration during model loading (important for ZeRO-3 initialization).
    deepspeed_config = None if args.deepspeed.lower() == "none" else args.deepspeed
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        # Transformers' tf32 flag is NVIDIA-specific and rejects ROCm even on
        # MI300-class hardware. BF16 remains enabled for both backends.
        tf32=(getattr(torch.version, "hip", None) is None),
        gradient_checkpointing=True,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
        data_seed=args.seed,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    if args.train_mode == "lora":
        from peft import LoraConfig, get_peft_model
        target_modules = [
            name.strip() for name in args.lora_target_modules.split(",") if name.strip()
        ]
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
        ))
        # Gradient checkpointing needs at least one grad-requiring input even
        # though the frozen embedding weights themselves are not trainable.
        model.enable_input_require_grads()

    callback = StepMetrics(
        args.output_dir, args.warmup_steps, args.measure_window, global_batch,
        args.seq_len, world_size, args.active_parameters_billion,
        args.theoretical_tflops_per_device,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[callback],
    )
    trainer.train()

    # Saving happens after on_train_end, so model serialization time is not
    # included in the performance window written by StepMetrics.
    if args.save_final_model:
        final_model_dir = Path(args.output_dir) / args.final_model_subdir
        trainer.save_model(str(final_model_dir))
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(final_model_dir)
            (final_model_dir / "training_manifest.json").write_text(
                json.dumps(
                    {
                        "base_model": args.model,
                        "dataset": (
                            str(Path(args.dataset_dir).resolve())
                            if args.dataset_dir
                            else "Salesforce/wikitext/wikitext-103-raw-v1"
                        ),
                        "train_mode": args.train_mode,
                        "max_steps": args.max_steps,
                        "seq_len": args.seq_len,
                        "global_batch_size": global_batch,
                        "deepspeed": args.deepspeed,
                        "saved_at_unix": time.time(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
