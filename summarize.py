#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--output", default="runs/summary.csv")
    p.add_argument("--table-output", default=None)
    p.add_argument("--ai-utilization-percent", type=float, default=None)
    p.add_argument("--theoretical-tflops-per-device", type=float, default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument(
        "--ai-utilization-by-world-size", default="",
        help="Comma-separated overrides such as 4=99.8,8=99.7",
    )
    args = p.parse_args()
    runs_dir = Path(args.runs_dir)
    ai_by_world_size = {}
    for item in args.ai_utilization_by_world_size.split(","):
        if item.strip():
            world_size, value = item.split("=", 1)
            ai_by_world_size[int(world_size)] = float(value)
    rows = []
    for path in sorted(runs_dir.glob("*/**/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["model"] = args.model_name or path.parents[1].name
        rows.append(metrics)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {runs_dir}")

    one_gpu_baselines = {
        r["model"]: r["throughput_tokens_per_sec"]
        for r in rows if r["world_size"] == 1
    }
    baseline_rows = {}
    for row in rows:
        current = baseline_rows.get(row["model"])
        if current is None or row["world_size"] < current["world_size"]:
            baseline_rows[row["model"]] = row
    for r in rows:
        one_gpu = one_gpu_baselines.get(r["model"])
        baseline_row = baseline_rows[r["model"]]
        baseline = baseline_row["throughput_tokens_per_sec"]
        baseline_world = baseline_row["world_size"]
        r["speedup_vs_1gpu"] = r["throughput_tokens_per_sec"] / one_gpu if one_gpu else None
        r["scaling_efficiency_percent"] = (
            100 * r["speedup_vs_1gpu"] / r["world_size"] if one_gpu else None
        )
        r["baseline_world_size"] = baseline_world
        r["speedup_vs_baseline"] = r["throughput_tokens_per_sec"] / baseline
        r["scaling_efficiency_vs_baseline_percent"] = 100 * (
            r["speedup_vs_baseline"] / (r["world_size"] / baseline_world)
        )
        total_memory = r.get("device_memory_total_gib")
        reserved_memory = r.get("peak_memory_reserved_gib")
        r["peak_memory_utilization_percent"] = (
            100 * reserved_memory / total_memory
            if reserved_memory is not None and total_memory else None
        )
        if r.get("ai_utilization_percent") is None:
            r["ai_utilization_percent"] = ai_by_world_size.get(
                r["world_size"], args.ai_utilization_percent
            )
        # Allow MFU to be filled during report generation when the device peak
        # was not passed to the original training command.
        if (r.get("estimated_mfu_percent") is None
                and args.theoretical_tflops_per_device
                and r.get("throughput_tokens_per_sec") is not None):
            active_billion = r.get("active_parameters_billion", 32.8)
            achieved_tflops = (
                6 * active_billion * 1e9 * r["throughput_tokens_per_sec"] / 1e12
            )
            r["estimated_mfu_percent"] = 100 * achieved_tflops / (
                r["world_size"] * args.theoretical_tflops_per_device
            )
        if r.get("final_step", 0) < 500:
            r["loss_step_500"] = None
        if r.get("final_step", 0) < 1000:
            r["loss_step_1000"] = None
    rows.sort(key=lambda r: (r["model"], r["world_size"]))
    fields = [
        "model", "world_size", "seq_len", "global_batch_size",
        "throughput_tokens_per_sec", "per_device_tokens_per_sec", "mean_step_ms",
        "p95_step_ms", "step_time_cv_percent",
        "peak_memory_allocated_gib", "peak_memory_reserved_gib",
        "device_memory_total_gib", "peak_memory_utilization_percent",
        "ai_utilization_percent",
        "speedup_vs_1gpu", "scaling_efficiency_percent", "estimated_mfu_percent",
        "baseline_world_size", "speedup_vs_baseline",
        "scaling_efficiency_vs_baseline_percent",
        "loss_step_500", "loss_step_1000", "final_step", "final_loss",
        "measurement_first_step", "measurement_last_step", "measured_steps",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")

    table_output = (
        Path(args.table_output) if args.table_output
        else output.with_name(f"{output.stem}_table{output.suffix}")
    )
    table_fields = [
        ("模型", "model"),
        ("卡数", "world_size"),
        ("seq_len", "seq_len"),
        ("Global_batch_size", "global_batch_size"),
        ("AI利用率(%)", "ai_utilization_percent"),
        ("显存利用率_峰值(%)", "peak_memory_utilization_percent"),
        ("总吞吐(tokens/s)", "throughput_tokens_per_sec"),
        ("单卡吞吐(tokens/s)", "per_device_tokens_per_sec"),
        ("单步时间(ms/step)", "mean_step_ms"),
        ("MFU(%)", "estimated_mfu_percent"),
        ("step-500 loss", "loss_step_500"),
        ("step-1000 loss", "loss_step_1000"),
        ("最终步", "final_step"),
        ("最终loss", "final_loss"),
    ]
    with table_output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[label for label, _ in table_fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({label: row.get(key) for label, key in table_fields})
    print(f"Wrote presentation table to {table_output}")


if __name__ == "__main__":
    main()
