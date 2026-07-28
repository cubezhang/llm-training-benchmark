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
DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 \
GPU_COUNTS=8 \
MODELS=/models/Qwen3-32B \
MODEL_SLUG=Qwen3-32B-LoRA \
TRAIN_MODE=lora \
THEORETICAL_TFLOPS_PER_DEVICE=232.6528 \
bash /workspace/run_scaling.sh
```

依次运行1、2、4、8卡：

```bash
DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 \
GPU_COUNTS="1 2 4 8" \
MODELS=/models/Qwen3-32B \
MODEL_SLUG=Qwen3-32B-LoRA \
TRAIN_MODE=lora \
THEORETICAL_TFLOPS_PER_DEVICE=232.6528 \
bash /workspace/run_scaling.sh
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
| `DATASET_DIR` | 自动检测标准目录 | 本地wikitext-103-raw-v1 Parquet目录 |
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

### 2.0 8卡快捷命令

快捷脚本只保存三个当前场景的参数预设，内部仍然调用`run_scaling.sh`：

```bash
DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 bash /workspace/run_8gpu_case.sh qwen3-32b-lora
DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 bash /workspace/run_8gpu_case.sh qwen3-30b-a3b-lora
DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 bash /workspace/run_8gpu_case.sh qwen3-32b-full
```

第二个位置参数可以指定输出根目录：

```bash
DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 \
  bash /workspace/run_8gpu_case.sh qwen3-32b-lora /workspace/timing/my-run
```

步数等参数仍通过环境变量覆盖，例如：

```bash
DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 \
MAX_STEPS=10 WARMUP_STEPS=2 MEASURE_WINDOW=8 SAVE_FINAL_MODEL=0 \
  bash /workspace/run_8gpu_case.sh qwen3-32b-lora /workspace/timing/smoke
```

### 2.1 Qwen3-32B LoRA，8卡10步检查

```bash
mkdir -p /workspace/timing
nohup env \
  DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 \
  GPU_COUNTS=8 \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-LoRA \
  TRAIN_MODE=lora \
  MICRO_BATCH=8 \
  LOCAL_BATCH=32 \
  SEQ_LEN=2048 \
  ACTIVE_PARAMETERS_B=32.8 \
  THEORETICAL_TFLOPS_PER_DEVICE=232.6528 \
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
  DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 \
  GPU_COUNTS="1 2 4 8" \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-LoRA \
  TRAIN_MODE=lora \
  MICRO_BATCH=8 \
  LOCAL_BATCH=32 \
  SEQ_LEN=2048 \
  ACTIVE_PARAMETERS_B=32.8 \
  THEORETICAL_TFLOPS_PER_DEVICE=232.6528 \
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
  DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 \
  GPU_COUNTS="4 8" \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-full \
  TRAIN_MODE=full \
  MICRO_BATCH=2 \
  LOCAL_BATCH=32 \
  SEQ_LEN=2048 \
  DEEPSPEED_CONFIG=configs/zero3_bf16.json \
  ACTIVE_PARAMETERS_B=32.8 \
  THEORETICAL_TFLOPS_PER_DEVICE=232.6528 \
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
OUTPUT_ROOT=/workspace/timing/qwen3-32b-lora-scaling
MODEL_SLUG=Qwen3-32B-LoRA

tail -f "${OUTPUT_ROOT}/${MODEL_SLUG}/8gpu/console.log"
cat "${OUTPUT_ROOT}/${MODEL_SLUG}/8gpu/status.txt"
watch -n 2 rocm-smi --showuse --showmemuse
```

## 5. 最终模型效果评测

```bash
OUTPUT_ROOT=/workspace/timing/qwen3-32b-lora-scaling

python evaluate_model.py \
  --base-model /models/Qwen3-32B \
  --trained-model "${OUTPUT_ROOT}/Qwen3-32B-LoRA/8gpu/final_model" \
  --dataset-dir /workspace/datasets/wikitext-103-raw-v1 \
  --train-mode lora \
  --split test \
  --seq-len 2048 \
  --max-eval-tokens 131072 \
  --output "${OUTPUT_ROOT}/Qwen3-32B-LoRA/8gpu/evaluation.json"
```

`perplexity_improvement_percent > 0`且`improved=true`表示最终模型在测试集上优于基础模型。

## 6. 内部脚本

- `run_case.sh`：通用单场景执行器，负责torchrun、GPU采样、日志、状态和模型保存。
- `run_scaling.sh`：唯一对外训练入口，循环模型和`GPU_COUNTS`并调用`run_case.sh`。
- `run_8gpu_case.sh`：当前三个场景的短命令预设，内部设置`GPU_COUNTS=8`后调用`run_scaling.sh`。
- `run_8gpu_2000step_plan.sh`：三个固定场景的串行计划，内部也调用`run_scaling.sh`。
- `run_200step_plan.sh`：200步计划，内部也调用`run_scaling.sh`。

## 7. WikiText-103 下载失败与离线缓存

### 7.1 典型现象

训练刚启动就退出，并在`console.log`中出现以下内容之一：

```text
hf_hub_download
httpx
Cannot send a request, as the client has been closed
```

这是数据集尚未缓存，多卡训练进程同时访问Hugging Face时发生的下载失败。应先停止
训练，使用一个Python进程准备共享缓存。

### 7.2 浏览器下载并上传服务器

服务器无法连接 Hugging Face 时，在个人电脑浏览器中下载：

- [train-00000-of-00002.parquet](https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet?download=true)
- [train-00001-of-00002.parquet](https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00001-of-00002.parquet?download=true)
- [validation-00000-of-00001.parquet](https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/validation-00000-of-00001.parquet?download=true)
- [test-00000-of-00001.parquet](https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet?download=true)

在容器内创建接收目录：

```bash
mkdir -p /workspace/datasets/wikitext-103-raw-v1
chmod 777 /workspace/datasets/wikitext-103-raw-v1
```

使用 MobaXterm、SFTP 或其他文件传输工具上传到宿主机：

```text
/volumes/oss5/models/qwen-scaling/datasets/wikitext-103-raw-v1/
```

容器内检查：

```bash
ls -lh /workspace/datasets/wikitext-103-raw-v1/
```

应当看到4个`.parquet`文件。`run_case.sh`会自动检测这个标准目录并让训练脚本
直接读取本地文件，不会访问 Hugging Face。也可以显式设置：

```bash
export DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1
```

如果指定了`DATASET_DIR`但目录不存在，或者当前split所需文件缺失，程序会直接
报告缺少的完整文件路径。

### 7.3 单进程在线下载数据集

以下命令全部在容器内执行：

```bash
cd /workspace
mkdir -p /workspace/hf-cache/datasets

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets

unset HF_DATASETS_OFFLINE
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

python3 - <<'PY'
from datasets import load_dataset

cache_dir = "/workspace/hf-cache/datasets"
for split in ("train", "validation", "test"):
    print("Downloading:", split)
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
        cache_dir=cache_dir,
    )
    print(split, len(dataset))

print("WikiText-103 cache prepared.")
PY
```

只有看到`WikiText-103 cache prepared.`后才能切换离线模式。

### 7.4 离线运行8卡10步检查

```bash
cd /workspace
mkdir -p /workspace/timing

nohup env \
  DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1 \
  HF_HOME=/workspace/hf-cache \
  HF_DATASETS_CACHE=/workspace/hf-cache/datasets \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  GPU_COUNTS=8 \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-LoRA \
  TRAIN_MODE=lora \
  MICRO_BATCH=8 \
  LOCAL_BATCH=32 \
  SEQ_LEN=2048 \
  ACTIVE_PARAMETERS_B=32.8 \
  THEORETICAL_TFLOPS_PER_DEVICE=232.6528 \
  MAX_STEPS=10 \
  WARMUP_STEPS=2 \
  MEASURE_WINDOW=8 \
  SAVE_FINAL_MODEL=0 \
  OUTPUT_ROOT=/workspace/timing/qwen3-32b-lora-smoke \
  bash /workspace/run_scaling.sh \
  > /workspace/timing/qwen3-32b-lora-smoke-launcher.log 2>&1 &
```

### 7.5 查看日志

```bash
tail -f /workspace/timing/qwen3-32b-lora-smoke/Qwen3-32B-LoRA/8gpu/console.log
```

如果任务已经结束，不要只看日志最后十行。提取首次异常：

```bash
grep -nEi -B 10 -A 30 \
  "traceback|error|exception|modulenotfound|runtimeerror|out of memory|failed" \
  /workspace/timing/qwen3-32b-lora-smoke/Qwen3-32B-LoRA/8gpu/console.log | \
  head -200
```

### 7.6 缓存位置与迁移

容器内缓存目录：

```text
/workspace/hf-cache/
```

对应宿主机目录：

```text
/volumes/oss5/models/qwen-scaling/hf-cache/
```

缓存不提交到GitHub。需要迁移到另一台服务器时，在容器内打包：

```bash
cd /workspace
tar -I zstd -cf wikitext-103-cache.tar.zst hf-cache/
```

目标服务器恢复：

```bash
cd /workspace
tar -I zstd -xf wikitext-103-cache.tar.zst
```
