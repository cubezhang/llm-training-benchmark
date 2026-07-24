# 三个场景8卡2000步测试手册

## 1. 文件结构

宿主机目录：

```text
/volumes/oss5/models/qwen-scaling/
├── train_qwen.py
├── run_case.sh
├── run_scaling.sh
├── run_8gpu_2000step_plan.sh
├── merge_gpu_samples.py
├── summarize.py
├── requirements-full.txt
├── configs/
│   ├── zero2_bf16.json
│   └── zero3_bf16.json
└── timing/
```

模型目录：

```text
/volumes/oss0/models/
├── Qwen3-32B/
└── Qwen3-30B-A3B/
```

容器内挂载关系：

```text
/volumes/oss5/models/qwen-scaling  -> /workspace
/volumes/oss0/models               -> /models
```

测试结果目录：

```text
/volumes/oss5/models/qwen-scaling/timing/qwen-three-scenarios-8gpu-2000steps/
├── scenario1/Qwen3-32B-LoRA/8gpu/
│   └── final_model/
├── scenario2/Qwen3-30B-A3B-LoRA/8gpu/
│   └── final_model/
├── scenario3/Qwen3-32B-full/8gpu/
│   └── final_model/
├── console.log
├── status.txt
├── summary.csv
└── summary_table.csv
```

统一测试参数：

```text
卡数：8
seq_len：2048
单卡有效batch：32
Global batch：256
max_steps：2000
性能预热：前100步
性能统计：跳过前100步，统计第101–2000步，共1900步
数据集：WikiText-103
精度：BF16
```

三个场景的训练部分串行预计约167.18小时，即约7天；训练结束后的模型聚合和
落盘时间不计入性能指标，需要在总计划中额外预留时间。

性能统计不再只取最后400步。现在跳过前100步预热后，使用第101–2000步共1900步计算主平均值、P95、CV和吞吐，能够覆盖整个长时间稳定运行阶段。每一步的原始时间仍保存在 `steps.jsonl`，如需观察训练末段，还可以单独计算最后400步。

## 2. 启动容器

### 2.1 镜像名称

```text
rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
```

### 2.2 第一次创建容器

在宿主机执行：

```bash
docker run -d \
  --name qwen-scaling-rocm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --network=host \
  --shm-size=256g \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /volumes/oss0/models:/models \
  -v /volumes/oss5/models/qwen-scaling:/workspace \
  -w /workspace \
  rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
  sleep infinity
```

进入容器：

```bash
docker exec -it qwen-scaling-rocm bash
```

从这里开始，除非特别标明，后续命令全部在容器内执行：

```bash
cd /workspace
python3 -m pip install -r requirements-full.txt
```

检查环境：

```bash
python3 -c \
  "import torch,transformers,datasets,deepspeed,peft; print(torch.__version__, torch.cuda.device_count())"
```

正常情况下应显示8张GPU。

### 2.3 容器已经存在时

以下两条在宿主机执行：

```bash
docker start qwen-scaling-rocm
docker exec -it qwen-scaling-rocm bash
```

进入容器后检查GPU：

```bash
rocm-smi --showproductname --showuse --showmemuse
```

正式运行前应确认 card0–card7 没有其他任务。

## 3. 启动三个场景串行测试

串行脚本：

```text
/workspace/run_8gpu_2000step_plan.sh
```

添加执行权限：

```bash
cd /workspace
chmod +x run_case.sh run_scaling.sh run_8gpu_2000step_plan.sh
```

后台启动：

```bash
mkdir -p /workspace/timing
nohup bash run_8gpu_2000step_plan.sh > timing/plan-launcher.log 2>&1 &
```

脚本将按以下顺序执行：

```text
1. Qwen3-32B LoRA 8卡2000步
2. Qwen3-30B-A3B LoRA 8卡2000步
3. Qwen3-32B全参数 ZeRO-3 8卡2000步
4. 自动生成summary.csv和summary_table.csv
```

查看当前场景：

```bash
cat /workspace/timing/qwen-three-scenarios-8gpu-2000steps/status.txt
```

查看训练日志：

```bash
tail -f /workspace/timing/qwen-three-scenarios-8gpu-2000steps/console.log
```

查看GPU：

```bash
watch -n 5 rocm-smi --showuse --showmemuse
```

## 4. 分别单独执行

单独执行和串行计划都调用 `run_scaling.sh`，因此使用完全相同的训练、GPU采样、模型保存和指标合并流程。三个命令不要同时执行。

公共设置：

```bash
ROOT=/workspace/timing/qwen-three-scenarios-8gpu-2000steps
mkdir -p "${ROOT}"
```

### 4.1 场景1：Qwen3-32B LoRA 8卡

```bash
nohup env MAX_STEPS=2000 WARMUP_STEPS=100 MEASURE_WINDOW=1900 \
  GPU_COUNTS=8 MODELS=/models/Qwen3-32B MODEL_SLUG=Qwen3-32B-LoRA \
  TRAIN_MODE=lora MICRO_BATCH=8 ACTIVE_PARAMETERS_B=32.8 \
  OUTPUT_ROOT=${ROOT}/scenario1 PORT_BASE=30010 \
  bash /workspace/run_scaling.sh \
  > ${ROOT}/scenario1-launcher.log 2>&1 &
```

### 4.2 场景2：Qwen3-30B-A3B LoRA 8卡

```bash
nohup env MAX_STEPS=2000 WARMUP_STEPS=100 MEASURE_WINDOW=1900 \
  GPU_COUNTS=8 MODELS=/models/Qwen3-30B-A3B MODEL_SLUG=Qwen3-30B-A3B-LoRA \
  TRAIN_MODE=lora MICRO_BATCH=8 ACTIVE_PARAMETERS_B=3.3 \
  OUTPUT_ROOT=${ROOT}/scenario2 PORT_BASE=30020 \
  bash /workspace/run_scaling.sh \
  > ${ROOT}/scenario2-launcher.log 2>&1 &
```

### 4.3 场景3：Qwen3-32B全参数 8卡

```bash
nohup env MAX_STEPS=2000 WARMUP_STEPS=100 MEASURE_WINDOW=1900 \
  GPU_COUNTS=8 MODELS=/models/Qwen3-32B MODEL_SLUG=Qwen3-32B-full \
  TRAIN_MODE=full MICRO_BATCH=2 DEEPSPEED_CONFIG=configs/zero3_bf16.json \
  ACTIVE_PARAMETERS_B=32.8 OUTPUT_ROOT=${ROOT}/scenario3 PORT_BASE=30030 \
  bash /workspace/run_scaling.sh \
  > ${ROOT}/scenario3-launcher.log 2>&1 &
```

每个单独命令完成后都会生成：

```text
metrics.json
steps.jsonl
gpu_samples.csv
console.log
status.txt
final_model/
```

`final_model/` 是最终训练成果。LoRA 场景保存 Adapter 和 tokenizer，加载时仍需
对应的基础模型；全参数 ZeRO-3 场景保存聚合后的 BF16 模型权重和 tokenizer。
默认不保存中间 checkpoint。如某次只做性能测试，在容器内执行任务前设置
`SAVE_FINAL_MODEL=0`，或在`nohup env`后增加`SAVE_FINAL_MODEL=0`。

## 5. 汇总结果

三个场景全部完成后，在容器内执行：

```bash
cd /workspace
ROOT=/workspace/timing/qwen-three-scenarios-8gpu-2000steps

python summarize.py \
  --runs-dir ${ROOT} \
  --output ${ROOT}/summary.csv \
  --table-output ${ROOT}/summary_table.csv \
  --theoretical-tflops-per-device 232.6528
```

结果文件：

```text
${ROOT}/summary.csv
${ROOT}/summary_table.csv
```

检查：

```bash
ls -lh ${ROOT}/summary*.csv
cat ${ROOT}/summary_table.csv
```

最终表包含：

```text
模型
卡数
seq_len
Global batch
AI利用率
峰值显存利用率
总吞吐
单卡吞吐
单步时间
MFU
step-500 loss
step-1000 loss
最终loss
```

串行执行和单独执行都调用同一个 `run_scaling.sh`，而它对每个GPU数量调用通用的 `run_case.sh`，因此都会自动采样GPU利用率并写入 `metrics.json`，最终字段口径完全一致。

模型保存发生在训练计时结束之后，因此不会降低表中的训练吞吐量，也不会计入
平均单步时间。运行前请为全参数 Qwen3-32B 最终权重额外预留充足磁盘空间。
