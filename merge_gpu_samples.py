#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--min-vram-percent", type=float, default=40.0)
    args = parser.parse_args()

    active = {f"card{item.strip()}" for item in args.devices.split(",")}
    gpu_values: list[float] = []
    vram_values: list[float] = []
    with Path(args.samples).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["device"] not in active:
                continue
            vram = float(row["vram_percent"])
            if vram < args.min_vram_percent:
                continue
            gpu_values.append(float(row["gpu_use_percent"]))
            vram_values.append(vram)
    if not gpu_values:
        raise SystemExit("No steady-state GPU samples were collected")

    metrics_path = Path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["ai_utilization_percent"] = sum(gpu_values) / len(gpu_values)
    metrics["gpu_utilization_sample_count"] = len(gpu_values)
    metrics["sampled_peak_vram_percent"] = max(vram_values)
    metrics["gpu_utilization_sampling_interval_seconds"] = 10
    metrics["gpu_utilization_filter_min_vram_percent"] = args.min_vram_percent
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
