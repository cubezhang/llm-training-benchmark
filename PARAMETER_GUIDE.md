# 脚本参数操作手册

本文档说明项目中训练、8卡场景、扩展测试、最终模型评测和结果汇总脚本的参数。

## 1. 最常用入口：`run_8gpu_case.sh`

调用格式：

```bash
bash /workspace/run_8gpu_case.sh \
  LABEL MODEL OUTPUT_DIR TRAIN_MODE MICRO_BATCH DEEPSPEED ACTIVE_PARAMS_B PORT
```

8个位置参数必须按顺序填写：

| 参数 | 含义 | Qwen3-32B LoRA示例 |
|---|---|---|
| `LABEL` | 本次任务名称，只用于日志和状态 | `qwen3-32b-lora-8gpu-2000steps` |
| `MODEL` | 容器内基础模型路径 | `/models/Qwen3-32B` |
| `OUTPUT_DIR` | 容器内结果目录 | `/workspace/timing/qwen3-32b-lora-8gpu-2000steps` |
| `TRAIN_MODE` | `lora`或`full` | `lora` |
| `MICRO_BATCH` | 每卡每次前后向的样本数 | `8` |
| `DEEPSPEED` | `none`或配置文件路径 | LoRA使用`none` |
| `ACTIVE_PARAMS_B` | 每token激活参数量，单位B，用于估算MFU | Qwen3-32B填`32.8` |
| `PORT` | torchrun主端口，同机并发任务不能重复 | `30118` |

脚本固定使用GPU `0–7`、8个torchrun进程、`seq_len=2048`和单卡有效batch 32。

### 环境变量

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `MAX_STEPS` | `2000` | 训练步数，范围1–2000 |
| `WARMUP_STEPS` | `100` | 不计入性能平均值的前置步数，必须小于MAX_STEPS |
| `MEASURE_WINDOW` | `MAX_STEPS-WARMUP_STEPS` | 从预热后记录中取最后多少步计算平均值 |
| `SAVE_FINAL_MODEL` | `1` | `1`保存最终模型，`0`只测性能 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Hugging Face镜像地址 |
| `HF_HOME` | `/workspace/hf-cache` | Hugging Face缓存目录 |
| `HF_DATASETS_CACHE` | `/workspace/hf-cache/datasets` | 数据集缓存目录 |
| `HF_DATASETS_OFFLINE` | `1` | `1`仅使用本地数据缓存 |
| `HF_HUB_OFFLINE` | `1` | `1`禁止访问Hub |
| `TRANSFORMERS_OFFLINE` | `1` | `1`只加载本地模型文件 |
| `LOG_FILE` | `OUTPUT_DIR/console.log` | 控制台日志路径 |
| `STATUS_FILE` | `OUTPUT_DIR/status.txt` | 任务状态文件路径 |

### 10步检查

```bash
docker exec -d \
  -e MAX_STEPS=10 \
  -e WARMUP_STEPS=2 \
  -e MEASURE_WINDOW=8 \
  -e SAVE_FINAL_MODEL=0 \
  llm-training-rocm \
  bash /workspace/run_8gpu_case.sh \
  qwen3-32b-lora-8gpu-smoke \
  /models/Qwen3-32B \
  /workspace/timing/qwen3-32b-lora-8gpu-smoke \
  lora 8 none 32.8 30118
```

### 2000步正式训练

```bash
docker exec -d \
  -e MAX_STEPS=2000 \
  -e WARMUP_STEPS=100 \
  -e MEASURE_WINDOW=1900 \
  -e SAVE_FINAL_MODEL=1 \
  llm-training-rocm \
  bash /workspace/run_8gpu_case.sh \
  qwen3-32b-lora-8gpu-2000steps \
  /models/Qwen3-32B \
  /workspace/timing/qwen3-32b-lora-8gpu-2000steps \
  lora 8 none 32.8 30118
```

`MICRO_BATCH=8`时，单卡有效batch 32，所以梯度累积为`32/8=4`；8卡Global
batch为`32×8=256`。如果显存不足可降低MICRO_BATCH，但同一组扩展对比必须保持一致。

## 2. Python训练入口：`train_qwen.py`

通常不需要直接调用，`run_8gpu_case.sh`和`run_scaling.sh`会构造这些参数。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | 必填 | 模型目录或Hugging Face模型名 |
| `--output-dir` | 必填 | 指标和最终模型输出目录 |
| `--seq-len` | `2048` | 训练序列长度 |
| `--micro-batch` | `1` | 每卡单次前后向batch |
| `--local-batch` | `32` | 每卡有效batch，必须整除micro-batch |
| `--max-steps` | `2000` | 最大训练步数，限制1–2000 |
| `--warmup-steps` | `100` | 性能预热步数 |
| `--measure-window` | `400` | 主性能指标使用的步数窗口 |
| `--learning-rate` | `2e-5` | 学习率 |
| `--seed` | `42` | 随机种子 |
| `--train-mode` | `lora` | `lora`或`full` |
| `--deepspeed` | `none` | DeepSpeed JSON路径；LoRA通常为none |
| `--save-final-model` | 开启 | 保存最终模型 |
| `--no-save-final-model` | — | 关闭最终模型保存 |
| `--final-model-subdir` | `final_model` | 最终模型子目录名 |
| `--lora-r` | `8` | LoRA rank |
| `--lora-target-modules` | `q_proj,k_proj,v_proj,o_proj` | LoRA目标模块，逗号分隔 |
| `--active-parameters-billion` | 空 | 激活参数量，用于MFU估算 |
| `--theoretical-tflops-per-device` | 空 | 单卡理论TFLOPS，用于MFU估算 |

## 3. 1/2/4/8卡扩展入口：`run_scaling.sh`

通过环境变量控制：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GPU_COUNTS` | `1 2 4 8` | 依次测试的卡数 |
| `MODELS` | 两个Qwen3模型 | 空格分隔的模型路径 |
| `TRAIN_MODE` | `lora` | 训练模式 |
| `LOCAL_BATCH` | `32` | 单卡有效batch |
| `MICRO_BATCH` | `1` | 单卡单次batch |
| `SEQ_LEN` | `2048` | 序列长度 |
| `MAX_STEPS` | `2000` | 训练步数 |
| `WARMUP_STEPS` | `100` | 性能预热步数 |
| `MEASURE_WINDOW` | `400` | 性能统计窗口 |
| `PEAK_TFLOPS` | 空 | 单卡理论TFLOPS |
| `DEEPSPEED_CONFIG` | `none` | DeepSpeed配置；full模式默认ZeRO-3 |

示例：

```bash
docker exec \
  -e MODELS=/models/Qwen3-32B \
  -e GPU_COUNTS="1 2 4 8" \
  -e TRAIN_MODE=lora \
  -e MICRO_BATCH=8 \
  -e MAX_STEPS=2000 \
  llm-training-rocm bash /workspace/run_scaling.sh
```

## 4. 最终模型评测：`evaluate_model.py`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--base-model` | 必填 | 原始基础模型目录 |
| `--trained-model` | 必填 | LoRA Adapter或全参数最终模型目录 |
| `--train-mode` | 必填 | `lora`或`full` |
| `--output` | 必填 | evaluation.json输出路径 |
| `--split` | `test` | `validation`或`test` |
| `--seq-len` | `2048` | 评测序列长度 |
| `--max-eval-tokens` | `131072` | 最多评测token数量 |
| `--device-map` | `auto` | Transformers设备映射方式 |
| `--seed` | `42` | 随机种子 |

LoRA评测示例：

```bash
docker exec llm-training-rocm python /workspace/evaluate_model.py \
  --base-model /models/Qwen3-32B \
  --trained-model /workspace/timing/qwen3-32b-lora-8gpu-2000steps/final_model \
  --train-mode lora \
  --split test \
  --seq-len 2048 \
  --max-eval-tokens 131072 \
  --output /workspace/timing/qwen3-32b-lora-8gpu-2000steps/evaluation.json
```

重点字段：`trained.loss`应低于`base.loss`，`perplexity_improvement_percent`应大于0，
且`improved`应为`true`。

## 5. 汇总结果：`summarize.py`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--runs-dir` | `runs` | 递归查找metrics.json的根目录 |
| `--output` | `runs/summary.csv` | 完整CSV输出 |
| `--table-output` | 空 | 展示用表格CSV输出 |
| `--ai-utilization-percent` | 空 | 所有结果统一AI利用率覆盖值 |
| `--ai-utilization-by-world-size` | 空 | 按卡数覆盖，如`4=99.8,8=99.7` |
| `--theoretical-tflops-per-device` | 空 | 单卡理论TFLOPS，用于补算MFU |
| `--model-name` | 空 | 强制覆盖模型显示名 |

示例：

```bash
docker exec llm-training-rocm python /workspace/summarize.py \
  --runs-dir /workspace/timing \
  --output /workspace/timing/summary.csv \
  --table-output /workspace/timing/summary_table.csv
```

## 6. 输出文件

| 文件 | 内容 |
|---|---|
| `console.log` | 训练控制台日志 |
| `status.txt` | RUNNING、COMPLETE或FAILED状态 |
| `steps.jsonl` | 每个优化器步的耗时 |
| `gpu_samples.csv` | 每10秒GPU和显存利用率采样 |
| `metrics.json` | 吞吐、单步时间、显存、loss和MFU |
| `final_model/` | LoRA Adapter或全参数最终权重 |
| `evaluation.json` | 基础模型与最终模型测试集效果对比 |
| `summary.csv` | 多场景完整汇总表 |
| `summary_table.csv` | 展示用汇总表 |

## 7. 查看运行状态

```bash
tail -f timing/qwen3-32b-lora-8gpu-2000steps/console.log
cat timing/qwen3-32b-lora-8gpu-2000steps/status.txt
watch -n 2 rocm-smi
```
