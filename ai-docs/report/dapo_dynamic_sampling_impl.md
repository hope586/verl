# DAPO Dynamic Sampling 实现报告

> 创建时间：2026-04-01
> 实现分支：work-v061
> 对应论文：DAPO arxiv:2503.15388，Section 3.3

---

## 一、背景

DAPO 的 Dynamic Sampling 组件来自论文 3.3 节。GRPO 训练中，每个 prompt 生成 n 条 response，形成一个 group，组内 reward 归一化后得到 advantage。若某 group 内的所有 response 要么全对、要么全错：

- **全对组（group_mean_acc == 1）**：prompt 太简单，组内 reward 均相同，归一化后 advantage 全为零，梯度贡献为零但消耗算力
- **全错组（group_mean_acc == 0）**：prompt 太难，同样全零 advantage，且若 std == 0 会导致除零 NaN

Dynamic Sampling 在 `compute_advantage` 之前过滤掉这两类 trivial group，只让有混合正确/错误样本的 group 参与梯度更新，提升训练信号的纯度和数值稳定性。

---

## 二、实现文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `verl/trainer/ppo/ray_trainer.py` | 修改 | 新增 `filter_trivial_groups()` 函数 + 调用点 |
| `verl/workers/reward_manager/my_dapo.py` | 修改 | 补充 Step 1.5：确保 `acc` 写入 `reward_extra_info` |
| `examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_my_dapo.sh` | 修改 | 新增两个配置参数 |

---

## 三、数据流与插入点

### 3.1 训练主循环数据流

```
[ray_trainer.py 行 ~1102]  compute_reward(batch)
                            → reward_manager.__call__() 运行
                            → 返回 reward_tensor（含 overlong penalty）
                               和 reward_extra_infos_dict（含原始 acc）

[ray_trainer.py 行 ~1161]  batch.non_tensor_batch.update(reward_extra_infos_dict)
                            → batch.non_tensor_batch["acc"] 写入，shape (N,)
                            → 此时 acc 是干净的原始正确性标签

              ↓ 插入 filter_trivial_groups ↓

[ray_trainer.py 行 ~1257]  compute_advantage(batch)
                            → GRPO within-group reward 归一化
                            → 过滤后所有保留 group 都有混合信号，std > 0，归一化安全
```

### 3.2 为什么必须用 `acc` 而不是 `reward_tensor`

`reward_tensor` 中每个样本最后一个有效 token 的值是 `base_score + overlong_penalty`。一条"全对但答案太长"的 response，其值可能是 `1.0 + (−0.8) = 0.2`，不再是布尔值。若用 `reward_tensor` 判断 trivial，全对组会因 overlong 惩罚而伪装成混合组，无法被正确过滤。`acc` 保持原始正确性，是正确的过滤依据。

这也是 `acc` 和 `reward_tensor` 分离设计的核心原因之一：**`reward_tensor` 是给优化器看的信号（允许被各种 trick 修改），`acc` 是给研究者看的信号（保持语义纯粹）**。

---

## 四、关键问题：`acc` 字段的可靠性

### 4.1 发现的遗漏

Task 1 实现 `my_dapo.py` 时，继承 `NaiveRewardManager` 并直接透传其 `reward_extra_info`。但 `NaiveRewardManager` 只有在 `compute_score` 返回 dict 时才写 `reward_extra_info`：

```python
# naive.py 中
if isinstance(score, dict):
    for key, value in score.items():
        reward_extra_info[key].append(value)   # acc 进这里
else:
    reward = score
    # ← 不写任何 extra_info
```

GSM8K 的 reward function 返回 float（0.0 或 1.0），不是 dict，因此 `reward_extra_info` 在整个 Task 1 实现中实际上是**空的**。若不修复，`acc` 不会出现在 `batch.non_tensor_batch` 里，`filter_trivial_groups` 中的 `if metric in batch.non_tensor_batch` 检查失败，过滤会静默跳过，训练看似正常但 dynamic sampling 完全不生效。

### 4.2 修复方案（my_dapo.py Step 1.5）

在 overlong penalty 循环之前，若 `acc` 不在 `reward_extra_info` 中，从 `reward_tensor` 逐样本读取 base score（此时 overlong penalty **尚未**叠加，是干净的原始值），写入 `reward_extra_info["acc"]`：

```python
# Step 1.5: 确保 acc 字段存在
if "acc" not in reward_extra_info:
    for i in range(len(data)):
        data_item = data[i]
        response_length = data_item.batch["responses"].shape[-1]
        response_mask = data_item.batch["attention_mask"][-response_length:]
        valid_response_length = int(response_mask.sum())
        if valid_response_length > 0:
            base_score = float(reward_tensor[i, valid_response_length - 1].item())
        else:
            base_score = 0.0
        reward_extra_info["acc"].append(base_score)
```

修复后执行顺序保证了语义正确性：Step 1（base reward） → Step 1.5（读取 acc） → Step 2（叠加 overlong penalty）。

---

## 五、核心函数实现

### `filter_trivial_groups`（ray_trainer.py 约行 181）

```python
def filter_trivial_groups(batch: DataProto, metric: str) -> tuple[DataProto, dict]:
    uids = batch.non_tensor_batch["uid"]           # shape (N,)
    scores = batch.non_tensor_batch[metric].astype(float)  # shape (N,)

    uid_to_indices = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_indices[uid].append(i)

    keep_mask = np.zeros(len(uids), dtype=bool)
    total_groups = len(uid_to_indices)
    kept_groups = 0

    for uid, indices in uid_to_indices.items():
        group_mean = scores[indices].mean()
        if 0.0 < group_mean < 1.0:     # 有混合信号 → 保留
            keep_mask[indices] = True
            kept_groups += 1

    filtered_groups = total_groups - kept_groups
    filter_metrics = {
        "filter_groups/total": total_groups,
        "filter_groups/kept": kept_groups,
        "filter_groups/filtered": filtered_groups,
        "filter_groups/filter_ratio": filtered_groups / total_groups if total_groups > 0 else 0.0,
    }

    if kept_groups == 0:
        print("[filter_trivial_groups] Warning: all groups are trivial. Skipping filter.")
        return batch, filter_metrics

    return batch[keep_mask], filter_metrics
```

### 调用点（ray_trainer.py fit() 内，约行 1249）

```python
filter_cfg = self.config.algorithm.get("filter_groups", None)
if filter_cfg is not None and filter_cfg.enable:
    metric = filter_cfg.metric
    if metric in batch.non_tensor_batch:
        batch, filter_metrics = filter_trivial_groups(batch, metric)
        metrics.update(filter_metrics)
```

---

## 六、DataProto 布尔索引机制

`batch[keep_mask]` 路由到 `DataProto.select_idxs()`，对两种存储分别处理：

| 存储 | 位置 | 索引方式 |
|------|------|---------|
| `batch.batch`（TensorDict） | GPU | `tensor[idxs_torch]`，返回新 TensorDict |
| `batch.non_tensor_batch`（numpy dict） | CPU | `val[idxs_np]`，numpy 布尔索引 |

返回新的 DataProto 对象，不修改原始数据，两种存储保持同步一致。

---

## 七、边界情况与防御性设计

| 情况 | 处理方式 |
|------|---------|
| `metric` 不在 `non_tensor_batch` | `if metric in batch.non_tensor_batch` 保护，静默跳过 |
| 所有 group 都是 trivial | 打印 warning，返回原始 batch，避免空 batch 导致崩溃 |
| `filter_groups` 配置为 None 或未设置 | `.get("filter_groups", None)` 返回 None，整个逻辑跳过 |
| `valid_response_length == 0`（异常样本） | acc 记为 0.0，不影响 group 过滤逻辑 |

---

## 八、配置方式

启动脚本中新增两个参数（`FilterGroupsConfig` 已在框架中预定义，无需修改配置代码）：

```bash
algorithm.filter_groups.enable=True
algorithm.filter_groups.metric=acc
```

完整配置见 `examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_my_dapo.sh`。

---

## 九、与官方实现的差异

本实现是**基础版本**，未包含 `max_num_gen_batches` 功能。官方 DAPO 的完整 Dynamic Sampling 在过滤后若剩余 group 数不足，会重新对被过滤的 prompt 进行 rollout 补充，直到凑够 batch 或达到 `max_num_gen_batches` 上限。本实现直接过滤后继续训练，batch size 每 step 会有波动，但对学习效果影响有限，适合验证组件正确性。

---

## 十、完成标准验证

- [x] 函数能正确按 uid 分组并计算 group_mean
- [x] 过滤后 DataProto tensor/non_tensor 结构同步，布尔索引正确
- [x] 处理极端情况：所有 group 都被过滤时退化为保留全部
- [x] 日志中输出 4 个过滤统计指标（total / kept / filtered / filter_ratio）
- [x] 通过 `filter_groups.enable` 配置开关控制，默认关闭，不影响原有训练逻辑
- [x] 修复 `my_dapo.py` 中 `acc` 字段遗漏问题，确保 filter 实际生效