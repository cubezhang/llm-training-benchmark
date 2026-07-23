# LLM Training Benchmark

面向单节点多卡服务器的可复用大模型训练、扩展效率和稳定性测试工具。
项目支持 LoRA 与全参数训练、GPU 利用率采样、训练性能统计、loss 记录、
最终模型保存及 CSV 汇总。当前首批测试场景使用 Qwen3，后续可继续加入
Kimi、Llama 等模型。

## 文档入口

- [脚本参数操作手册](PARAMETER_GUIDE.md)：逐项解释训练、8卡场景、评测和汇总参数
- [8卡2000步测试手册](THREE_SCENARIO_8GPU_2000STEP_GUIDE.md)：三个既定场景的执行流程

## 当前场景

- Qwen3-32B LoRA：1/2/4/8 卡扩展测试
- Qwen3-30B-A3B LoRA：8 卡测试
- Qwen3-32B 全参数：DeepSpeed ZeRO-3 训练
- WikiText-103，`seq_len=2048`
- `max_steps` 可通过启动命令指定，上限为 2000
- 单卡有效 batch 固定，Global batch 随卡数等比例增大

## 主要文件

```text
train_qwen.py                       训练入口与性能指标记录
run_8gpu_case.sh                    单个8卡场景统一入口
run_8gpu_2000step_plan.sh           三个场景2000步串行测试
run_scaling.sh                      1/2/4/8卡扩展测试
merge_gpu_samples.py                合并GPU采样与训练指标
summarize.py                        生成CSV汇总表
evaluate_model.py                   对比基础模型与最终模型的测试集loss/PPL
PARAMETER_GUIDE.md                  全部脚本参数操作手册
setup_rocm_container.sh             创建ROCm PyTorch容器
configs/zero2_bf16.json             DeepSpeed ZeRO-2配置
configs/zero3_bf16.json             DeepSpeed ZeRO-3配置
THREE_SCENARIO_8GPU_2000STEP_GUIDE.md 详细操作手册
```

## 默认环境

```text
镜像：rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
项目目录：/volumes/oss5/models/qwen-scaling
模型目录：/volumes/oss0/models
容器名称：qwen-scaling-rocm
容器项目目录：/workspace
容器模型目录：/models
```

宿主机目录和容器名称可以在 `setup_rocm_container.sh` 中调整。虽然现有路径
保留了 `qwen-scaling` 名称以兼容已经完成的测试环境，但项目本身不限制模型类型。

## 创建容器

```bash
cd /volumes/oss5/models/qwen-scaling
bash setup_rocm_container.sh
```

首次运行需要缓存 WikiText-103。网络受限时可设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 运行8卡2000步串行计划

```bash
docker exec -d qwen-scaling-rocm \
  bash /workspace/run_8gpu_2000step_plan.sh
```

单独运行某个场景时统一使用 `run_8gpu_case.sh`，这样训练指标、loss、GPU利用率
和显存利用率都会自动生成。详细参数见操作手册。

## 输出

每个场景输出：

```text
metrics.json
steps.jsonl
gpu_samples.csv
console.log
status.txt
final_model/
```

`final_model/` 默认保存最终训练结果：LoRA模式保存Adapter和tokenizer；全参数
ZeRO-3模式保存聚合后的BF16权重。保存发生在性能计时结束之后，不计入训练吞吐。
如只做纯性能测试，可设置：

```bash
SAVE_FINAL_MODEL=0
```

## 汇总

```bash
python summarize.py \
  --runs-dir /workspace/timing/qwen-three-scenarios-8gpu-2000steps \
  --output /workspace/timing/qwen-three-scenarios-8gpu-2000steps/summary.csv \
  --table-output /workspace/timing/qwen-three-scenarios-8gpu-2000steps/summary_table.csv
```

模型权重、数据集缓存、训练结果和服务器凭据不会提交到Git仓库。

## 最终模型效果评测

LoRA训练完成后，在WikiText-103测试集上对比基础模型与最终Adapter：

```bash
python evaluate_model.py \
  --base-model /models/Qwen3-32B \
  --trained-model /workspace/timing/qwen3-32b-lora-8gpu-2000steps/final_model \
  --train-mode lora \
  --split test \
  --seq-len 2048 \
  --max-eval-tokens 131072 \
  --output /workspace/timing/qwen3-32b-lora-8gpu-2000steps/evaluation.json
```

`perplexity_improvement_percent > 0` 且 `improved=true` 表示最终模型在未参与训练的
测试集上优于基础模型。评测应与训练使用相同的tokenizer和序列长度。
