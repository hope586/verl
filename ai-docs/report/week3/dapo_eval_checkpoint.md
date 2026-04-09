# DAPO 模型评测报告：从 Checkpoint 到 GSM8K 评测

> 时间：2026-04-08
> 分支：work-v061
> 平台：verl + evalscope + vllm，isee47 服务器（8× GTX 3090）

---

## 背景

上周（Week 2）完成了 DAPO 算法的 5 个核心任务实现，并在 isee47 服务器上成功启动训练。本次工作目标：**训练完成后，从 FSDP checkpoint 恢复出可评测模型，并用 evalscope 框架完成正式 GSM8K 评测。**

实验配置：
- 模型：Qwen3-4B
- 数据集：GSM8K
- 训练算法：GRPO + DAPO（Overlong Reward Shaping, Dynamic Sampling, Clip-Higher, 无 KL 约束, Token-level Loss）
- 训练脚本：`examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_my_dapo.sh`

---

## 1. 训练结果总览

训练共 87 步（3 个 epoch），每 20 步保存 checkpoint，每 5 步在验证集上评估。

### 训练曲线（验证集 Accuracy）

| Step | Val Acc | 阶段 |
|------|---------|------|
| 0    | 49.2%   | 初始基线（Qwen3-4B 原始能力） |
| 5    | 79.1%   | 快速上升期 |
| 10   | 86.0%   | |
| 20   | 90.1%   | |
| 35   | 93.0%   | |
| 55   | 94.2%   | 进入平台期 |
| **80** | **94.5%** | **全程最高点** |
| 87   | 94.3%   | 训练结束 |

**观察：**
- Step 0→20：急速上升阶段（49.2% → 90.1%），模型快速学会 GSM8K 的推理格式
- Step 20→55：稳步提升（90.1% → 94.2%），逐步提高推理准确度
- Step 55→87：平台期（94.2% ~ 94.5%），准确率趋于饱和，波动 < 1%

### Checkpoint 列表

| Checkpoint | Val Acc | 说明 |
|-----------|---------|------|
| global_step_20 | 90.1% | |
| global_step_40 | 93.5% | |
| global_step_60 | 94.0% | |
| **global_step_80** | **94.5%** | 最佳 checkpoint |
| global_step_87 | 94.3% | 训练结束 |

选择 **global_step_80** 作为最终评测模型（验证集最高准确率）。

---

## 2. Checkpoint 转换：FSDP 分片 → HuggingFace 格式

### 为什么需要转换

verl 使用 FSDP（Fully Sharded Data Parallel）进行分布式训练，模型权重分散存储在 8 个 `.pt` 分片文件中：

```
global_step_80/actor/
├── model_world_size_8_rank_0.pt    ← 8 个 FSDP 权重分片
├── model_world_size_8_rank_1.pt
├── ...
├── model_world_size_8_rank_7.pt
├── optim_world_size_8_rank_*.pt    ← 优化器状态（评测不需要）
├── extra_state_world_size_8_rank_*.pt
├── fsdp_config.json
└── huggingface/                    ← 只有 config + tokenizer，没有模型权重
    ├── config.json
    ├── tokenizer.json
    └── ...
```

评测框架（evalscope、vllm、transformers）需要标准 HuggingFace 格式（完整的 `model.safetensors` 文件）。

### 转换命令

使用 verl 内置的 `model_merger` 工具：

```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/grpo_gsm8k_3090/qwen3_4b_my_dapo_overlong/global_step_80/actor \
    --target_dir /home/data2/zixuan/merged_models/qwen3_4b_my_dapo_step80
```

耗时约 3 分钟，输出标准 HuggingFace 模型目录：

```
qwen3_4b_my_dapo_step80/
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

---

## 3. 评测过程

### 工具选择：evalscope

evalscope（ModelScope 出品）是一个支持多种 benchmark 的模型评测框架，内置 GSM8K 等主流数据集。

安装：`pip install evalscope`（安装在 verl-v061 conda 环境中，无依赖冲突）。

### 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| Native 后端单卡 OOM | Qwen3-4B bfloat16 需要约 8GB，但生成时 KV cache + 激活值超过单卡 24GB | 改用 vllm server 后端 |
| `device_map=auto` 仍然 OOM | 服务器 GPU 0-5 被其他进程占用（仅 GPU 6、7 空闲） | `CUDA_VISIBLE_DEVICES=6,7` 指定空闲卡 |
| evalscope `--use-server` 参数不存在 | evalscope v1.5.2 不支持此参数（文档版本差异） | 手动启动 vllm server + `--api-url` 对接 |

### 最终评测方案（vllm server + evalscope API 模式）

**终端 1：启动 vllm 推理服务**
```bash
CUDA_VISIBLE_DEVICES=6,7 vllm serve /home/data2/zixuan/merged_models/qwen3_4b_my_dapo_step80 \
    --port 8801 \
    --tensor-parallel-size 2 \
    --dtype bfloat16
```

**终端 2：运行 evalscope 评测**
```bash
evalscope eval \
    --model /home/data2/zixuan/merged_models/qwen3_4b_my_dapo_step80 \
    --datasets gsm8k \
    --eval-batch-size 16 \
    --api-url http://localhost:8801/v1 \
    --api-key EMPTY \
    --generation-config '{"max_new_tokens": 2048, "temperature": 0, "do_sample": false}'
```

评测耗时约 22 分钟（1319 题，vllm 推理速度约 75 tokens/s）。

---

## 4. 评测结果

```
+-------------------------+-----------+----------+----------+-------+---------+
| Model                   | Dataset   | Metric   | Subset   |   Num |   Score |
+=========================+===========+==========+==========+=======+=========+
| qwen3_4b_my_dapo_step80 | gsm8k     | mean_acc | main     |  1319 |  0.9507 |
+-------------------------+-----------+----------+----------+-------+---------+
```

### 结果对比

| 模型 | GSM8K Accuracy | 来源 |
|------|---------------|------|
| Qwen3-4B（原始） | 49.2% | 训练 step 0 验证集 |
| DAPO step 80（训练时 val） | 94.5% | 训练日志 |
| **DAPO step 80（evalscope 正式评测）** | **95.07%** | evalscope，1319 题，greedy decoding |

**关键发现：**
- evalscope 正式评测（95.07%）略高于训练时 val acc（94.5%），说明模型泛化良好
- DAPO 训练将 Qwen3-4B 在 GSM8K 上的准确率提升了约 **46 个百分点**（49.2% → 95.1%）
- 评测使用 greedy decoding（temperature=0, do_sample=false），与训练时验证设置一致

---

## 5. 文件与产出

| 文件路径 | 说明 |
|---------|------|
| `isee47:~/verl-v061/checkpoints/grpo_gsm8k_3090/qwen3_4b_my_dapo_overlong/` | 训练 checkpoint（step 20/40/60/80/87） |
| `isee47:/home/data2/zixuan/merged_models/qwen3_4b_my_dapo_step80/` | 合并后的 HuggingFace 模型（可直接用于推理） |
| `isee47:~/verl-v061/outputs/20260408_152247/` | evalscope 评测报告（含 HTML） |

---

## 6. 阶段 1 总结

至此，**阶段 1（自主实现 DAPO）的全部 5 个任务已完成**：

| 任务 | 内容 | 状态 |
|------|------|------|
| Task 1 | Overlong Reward Shaping | ✅ |
| Task 2 | Dynamic Sampling | ✅ |
| Task 3 | Clip-Higher 代码路径理解 | ✅ |
| Task 4 | KL 约束双路径理解 | ✅ |
| Task 5 | 串起脚本，跑通 pipeline | ✅ |
| **评测** | **checkpoint 转换 + evalscope 正式评测** | **✅ 95.07%** |

下一步进入**阶段 2：与 verl 官方 DAPO 实现对比学习**。