# 脚本参数操作手册

项目只保留一个对外训练入口：`run_scaling.sh`。传入一个GPU数量就运行一个场景，
传入多个GPU数量就按顺序运行扩展测试。所有场景使用相同的GPU采样、日志、状态、
模型保存和结果汇总逻辑。

## 0. 进入容器

容器创建或启动在宿主机执行，之后进入容器：

```bash
docker exec -it llm-training-rocm bash
cd /workspace
```

下文命令全部在容器内执行。网络受限时设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 1. 统一训练入口：`run_scaling.sh`

### 1.1 卡数控制

只运行8卡：

```bash
GPU_COUNTS=8 bash run_scaling.sh
```

依次运行1、2、4、8卡：

```bash
GPU_COUNTS="1 2 4 8" bash run_scaling.sh
```

脚本会根据卡数自动设置：

- `torchrun --nproc_per_node=GPU数量`
- `HIP_VISIBLE_DEVICES=0,...,GPU数量-1`
- Global batch = `LOCAL_BATCH × GPU数量`
- 对本次使用的GPU进行定时采样

### 1.2 常用参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `GPU_COUNTS` | `1 2 4 8` | 空格分隔的GPU数量，可以只填一个 |
| `MODELS` | 两个Qwen3模型 | 模型路径，多个模型使用空格分隔 |
| `MODEL_SLUG` | 模型目录名 | 单模型运行时的输出目录名称 |
| `TRAIN_MODE` | `lora` | `lora`或`full` |
| `LOCAL_BATCH` | `32` | 每张卡的有效batch |
| `MICRO_BATCH` | `1` | 每张卡单次前后向batch |
| `SEQ_LEN` | `2048` | 序列长度 |
| `MAX_STEPS` | `2000` | 训练步数，范围1–2000 |
| `WARMUP_STEPS` | `100` | 性能预热步数 |
| `MEASURE_WINDOW` | `400` | 用于主要性能指标的末尾步数 |
| `OUTPUT_ROOT` | `runs/TRAIN_MODE` | 本次测试输出根目录 |
| `SAVE_FINAL_MODEL` | `1` | `1`保存最终模型，`0`不保存 |
| `GPU_SAMPLE_INTERVAL` | `10` | GPU采样间隔，单位秒 |
| `DEEPSPEED_CONFIG` | `none` | 全参数模式未指定时自动使用ZeRO-3 |
| `ACTIVE_PARAMETERS_B` | 自动判断 | 激活参数量，单位B，用于MFU计算 |
| `THEORETICAL_TFLOPS_PER_DEVICE` | 空 | 单卡理论TFLOPS |
| `PORT_BASE` | `29500` | torchrun端口基数 |
| `SKIP_COMPLETED` | `0` | `1`时跳过已有相同步数结果 |
| `GPU_DEVICES` | 自动生成 | 可选的显式设备列表，仅高级场景使用 |

## 2. 常用命令

### 2.1 Qwen3-32B LoRA，8卡10步检查

```bash
mkdir -p /workspace/timing
nohup env \
  GPU_COUNTS=8 \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-LoRA \
  TRAIN_MODE=lora \
  MICRO_BATCH=8 \
  ACTIVE_PARAMETERS_B=32.8 \
  MAX_STEPS=10 \
  WARMUP_STEPS=2 \
  MEASURE_WINDOW=8 \
  SAVE_FINAL_MODEL=0 \
  OUTPUT_ROOT=/workspace/timing/qwen3-32b-lora-smoke \
  bash /workspace/run_scaling.sh \
  > /workspace/timing/qwen3-32b-lora-smoke-launcher.log 2>&1 &
```

### 2.2 Qwen3-32B LoRA，1/2/4/8卡2000步

```bash
nohup env \
  GPU_COUNTS="1 2 4 8" \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-LoRA \
  TRAIN_MODE=lora \
  MICRO_BATCH=8 \
  ACTIVE_PARAMETERS_B=32.8 \
  MAX_STEPS=2000 \
  WARMUP_STEPS=100 \
  MEASURE_WINDOW=1900 \
  SAVE_FINAL_MODEL=1 \
  OUTPUT_ROOT=/workspace/timing/qwen3-32b-lora-scaling \
  bash /workspace/run_scaling.sh \
  > /workspace/timing/qwen3-32b-lora-scaling-launcher.log 2>&1 &
```

### 2.3 Qwen3-32B全参数，4/8卡2000步

```bash
nohup env \
  GPU_COUNTS="4 8" \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-full \
  TRAIN_MODE=full \
  MICRO_BATCH=2 \
  DEEPSPEED_CONFIG=configs/zero3_bf16.json \
  ACTIVE_PARAMETERS_B=32.8 \
  MAX_STEPS=2000 \
  WARMUP_STEPS=100 \
  MEASURE_WINDOW=1900 \
  SAVE_FINAL_MODEL=1 \
  OUTPUT_ROOT=/workspace/timing/qwen3-32b-full-scaling \
  bash /workspace/run_scaling.sh \
  > /workspace/timing/qwen3-32b-full-scaling-launcher.log 2>&1 &
```

## 3. 每个场景自动生成的文件

```text
OUTPUT_ROOT/MODEL_SLUG/NGPU/
├── console.log
├── status.txt
├── metrics.json
├── steps.jsonl
├── gpu_samples.csv
└── final_model/
```

训练结束后，`run_scaling.sh`还会在`OUTPUT_ROOT`生成：

```text
summary.csv
summary_table.csv
```

`gpu_samples.csv`从训练进程启动后持续采样到训练结束。采样结果会自动通过
`merge_gpu_samples.py`写入`metrics.json`，不需要再手动启动采样脚本。

## 4. 查看状态

```bash
tail -f OUTPUT_ROOT/MODEL_SLUG/8gpu/console.log
cat OUTPUT_ROOT/MODEL_SLUG/8gpu/status.txt
watch -n 2 rocm-smi --showuse --showmemuse
```

## 5. 最终模型效果评测

```bash
python evaluate_model.py \
  --base-model /models/Qwen3-32B \
  --trained-model OUTPUT_ROOT/Qwen3-32B-LoRA/8gpu/final_model \
  --train-mode lora \
  --split test \
  --seq-len 2048 \
  --max-eval-tokens 131072 \
  --output OUTPUT_ROOT/Qwen3-32B-LoRA/8gpu/evaluation.json
```

`perplexity_improvement_percent > 0`且`improved=true`表示最终模型在测试集上优于基础模型。

## 6. 内部脚本

- `run_case.sh`：通用单场景执行器，负责torchrun、GPU采样、日志、状态和模型保存。
- `run_scaling.sh`：唯一对外训练入口，循环模型和`GPU_COUNTS`并调用`run_case.sh`。
- `run_8gpu_case.sh`：只为兼容旧命令保留，新任务不再使用。
- `run_8gpu_2000step_plan.sh`：三个固定场景的串行计划，内部也调用`run_scaling.sh`。
- `run_200step_plan.sh`：200步计划，内部也调用`run_scaling.sh`。
