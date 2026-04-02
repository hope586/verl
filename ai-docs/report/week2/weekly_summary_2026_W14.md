# 本周工作总结：从零实现 DAPO 算法

> 时间：2026 年第 14 周（3月30日 - 4月2日）
> 分支：work-v061
> 平台：verl（ByteDance 开源工业级 RL 训练框架）

---

## 背景与目标

以"框架未内置 DAPO"为前提，在 verl 框架中独立实现 DAPO（Direct Alignment from Preference Optimization）算法的全部核心组件，并在 isee47 服务器（8× GTX 3090）上跑通完整训练 pipeline。

实验设置：Qwen3-4B 模型，GSM8K 数据集，GRPO 训练框架。

---

## 完成的 5 个任务

### Task 1：Overlong Reward Shaping
**目标：** 实现对过长回答的线性惩罚，防止模型无限生成。

**实现：** 新建 `verl/workers/reward_manager/my_dapo.py`，继承 NaiveRewardManager，在 base reward 之上叠加惩罚项。

**惩罚公式：**
```
expected_len = max_resp_len - overlong_buffer_len
若 valid_response_length > expected_len：
    penalty = -min((len - expected_len) / buffer_len, 1.0) × penalty_factor
```

**关键收获：** 掌握 verl 的插件注册机制（`@register` 装饰器 + `**kwargs` 配置透传）。

---

### Task 2：Dynamic Sampling
**目标：** 过滤全对或全错的 trivial group，只对有区分度的样本计算梯度。

**实现：** 在 `ray_trainer.py` 中新增 `filter_trivial_groups()` 函数，按 uid 分组，计算每组 acc 均值，保留 `0 < group_mean_acc < 1` 的 group。

**额外发现：** 追踪数据流时发现 Task 1 中 `acc` 字段从未写入，修复了这个静默 bug。

**副作用：** 过滤减少了 actor 更新的计算量，训练时间从 12 小时缩短至 7.5 小时。

---

### Task 3：Clip-Higher（非对称裁剪）
**目标：** 理解参数从启动脚本流经框架影响 loss 的完整路径。

**追踪路径：**
```
脚本 clip_ratio_high=0.28
  → ActorConfig.clip_ratio_high
    → core_algos.py:924 compute_policy_loss()
      → torch.clamp(ratio, 1-clip_ratio_low, 1+clip_ratio_high)
```

**原理：** 标准 PPO 使用对称裁剪；Clip-Higher 解耦上下界，放宽正向探索的上限，同时保持负向更新的保守性，鼓励模型在 advantage > 0 的方向上更大胆地更新。

---

### Task 4：去掉 KL 约束（双路径）
**目标：** 理解 verl 中 KL 约束的两条独立路径，并正确关闭。

| 路径 | 配置项 | 作用阶段 |
|------|--------|---------|
| 路径 A：KL as Reward Penalty | `algorithm.use_kl_in_reward` | rollout 后，影响 advantage 全链路 |
| 路径 B：KL as Loss Term | `actor.use_kl_loss` | actor 梯度更新时 |

**额外发现：** 原脚本中误设 `use_kl_loss=True`，本次修正。

**替代稳定性手段：** Clip-Higher（限制策略偏移上界）+ Overlong Shaping（防止生成崩溃），无需参考模型前向传播，减少显存开销。

---

### Task 5：整合启动，跑通 Pipeline
**目标：** 将前 4 个任务整合进启动脚本，在服务器上成功启动训练。

**补充配置：** Token-level loss（`loss_agg_mode=token-mean`）——按 token 平均而非按样本平均，使长推理链的梯度信号更稳定。

**修复的 4 个集成 Bug：**

| Bug | 根本原因 | 修复方式 |
|-----|---------|---------|
| token-mean 未设置 | 任务清单遗漏 | 显式加入脚本 |
| `filter_groups.enable` Hydra 报错 | Hydra struct 校验基于实际 YAML，不是 dataclass | 在 `ppo_trainer.yaml` 加入字段 |
| `reward_kwargs.*` Hydra 报错 | 动态参数字典无法预定义 | 改用 `+` 前缀追加新键 |
| `AssertionError: unequal chunk` | 过滤后 batch size 不整除 dp_size | 过滤后截断到 dp_size 整数倍 |

**训练结果：** 成功启动，step 1/2 正常完成，初始 val acc 49.2%（符合预期基线）。

---

## 能力收获

**框架工程能力：**
- verl 插件开发完整工作流：读接口 → 设计 → 实现 → 注册 → 配置
- Hydra 配置系统原理：YAML struct 校验、`+` 追加前缀、config_name 定位
- 大型框架的参数路径追踪能力（脚本 → config → loss 函数）
- DataProto 数据流理解（tensor_batch / non_tensor_batch / 布尔索引）

**AI 辅助研发工作流：**
- 设计先于实现：先用自然语言描述清楚逻辑，再让 Claude 实现
- 人是设计者，Claude 是执行者
- 用 Claude Code 追踪代码路径、生成单元测试、定位配置问题

---

## 关键文件清单

| 文件 | 内容 |
|------|------|
| `verl/workers/reward_manager/my_dapo.py` | Overlong Reward Shaping 实现 |
| `verl/trainer/ppo/ray_trainer.py` | Dynamic Sampling `filter_trivial_groups()` |
| `verl/trainer/config/ppo_trainer.yaml` | 新增 filter_groups 配置节 |
| `examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_my_dapo.sh` | 完整 DAPO 启动脚本 |

---

## 下一阶段计划

**阶段 2：与 verl 官方 DAPO 实现对比学习**
- 对比 `verl/workers/reward_manager/dapo.py` 与 `my_dapo.py` 的差异
- 对比官方启动脚本 `test_dapo_7b_math.sh` 的参数设置
- 输出：实现对比笔记（`ai-docs/report/dapo_comparison_notes.md`）
