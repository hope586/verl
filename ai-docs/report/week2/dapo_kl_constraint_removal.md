# Task 4 实现报告：理解并关闭 KL 约束双路径

## 概述

DAPO 论文 §3.4 明确去掉了 KL 约束。verl 框架中 KL 约束有两条独立路径，本次任务通过代码追踪理解了两条路径的完整调用链，并在脚本中将两者同时关闭。

---

## verl 中 KL 约束的两条路径

### 路径 A：KL as Reward Penalty

**配置项：** `algorithm.use_kl_in_reward`（默认 `False`）

**作用阶段：** rollout 结束后，advantage 计算之前

**调用链：**

```
启动脚本 algorithm.use_kl_in_reward=True
    ↓
main_ppo.py — 初始化 RefPolicy Worker（提供 ref_log_prob）
    ↓
RayPPOTrainer.__init__() [ray_trainer.py:396-397]
    → 创建 KL 控制器 kl_ctrl_in_reward（fixed 或 adaptive）
    ↓
训练循环 [ray_trainer.py:1221-1226]
    apply_kl_penalty(batch, kl_ctrl, kl_penalty)
        ↓
    core_algos.kl_penalty(old_log_probs, ref_log_prob, method)
        → low_var_kl: kld = exp(ref-π) - (ref-π) - 1
        ↓
    token_level_rewards = token_level_scores - β * kld
    kl_ctrl.update(current_kl)   # adaptive 时自动调整 β
    ↓
compute_advantage() — 使用修改后的 token_level_rewards
    GAE : δ_t = reward[t] + γ*V(t+1) - V(t)
    GRPO: outcome = sum(reward)，组内标准化
```

**核心代码位置：**

| 步骤 | 文件 | 行号 |
|------|------|------|
| 配置定义 | `verl/trainer/config/algorithm.py` | 370 |
| KL 控制器初始化 | `verl/trainer/ppo/ray_trainer.py` | 396-397 |
| apply_kl_penalty 调用 | `verl/trainer/ppo/ray_trainer.py` | 1221-1226 |
| apply_kl_penalty 实现 | `verl/trainer/ppo/ray_trainer.py` | 121-160 |
| KL 散度计算（kl_penalty） | `verl/trainer/ppo/core_algos.py` | 1412-1473 |
| Adaptive KL 控制器 | `verl/trainer/ppo/core_algos.py` | 150-210 |

**Adaptive KL 控制器原理：**

```python
# 每批次更新后自动调整 β，使实际 KL 趋近 target_kl
proportional_error = clip((current_kl / target_kl) - 1, -0.2, 0.2)
mult = 1 + proportional_error * batch_size / horizon
β_new = β * mult
# 实际 KL > target → β 增大 → 下一批次惩罚更强
# 实际 KL < target → β 减小 → 下一批次惩罚更弱
```

---

### 路径 B：KL as Loss Term

**配置项：** `actor_rollout_ref.actor.use_kl_loss`（默认 `False`）

**作用阶段：** actor 梯度更新时，直接加入 loss 函数

**调用链：**

```
启动脚本 actor.use_kl_loss=True, kl_loss_coef=0.001, kl_loss_type=low_var_kl
    ↓
main_ppo.py — 检测到 use_kl_loss=True，加载 RefPolicy Worker
    ↓
训练循环 [ray_trainer.py:1194-1201]
    compute_ref_log_prob() → DataProto["ref_log_prob"]
    ↓
update_actor() → dp_actor.py:update_policy()
    log_prob = 当前策略前向传播          [dp_actor.py:442]
    pg_loss  = policy_loss_fn(...)      # PPO clip loss
    policy_loss = pg_loss - entropy_coeff * entropy_loss
    
    if use_kl_loss:                     [dp_actor.py:500-508]
        kld = core_algos.kl_penalty(log_prob, ref_log_prob, kl_loss_type)
        kl_loss = agg_loss(kld, response_mask, loss_agg_mode)
        policy_loss = policy_loss + kl_loss * kl_loss_coef
    
    loss.backward()                     [dp_actor.py:520]
```

**核心代码位置：**

| 步骤 | 文件 | 行号 |
|------|------|------|
| 配置定义 | `verl/workers/config/actor.py` | 109, 111, 112 |
| ref_log_prob 计算 | `verl/trainer/ppo/ray_trainer.py` | 1194-1201 |
| KL loss 组装 | `verl/workers/actor/dp_actor.py` | 500-510 |
| KL 散度计算（同路径 A） | `verl/trainer/ppo/core_algos.py` | 1412-1473 |
| loss 反向传播 | `verl/workers/actor/dp_actor.py` | 520 |

**最终 loss 公式：**

```
Total Loss = pg_loss
           - entropy_coeff × entropy_loss
           + kl_loss_coef × KL(π_ref ‖ π_θ)
```

---

## 两条路径的核心差异

| | 路径 A（In-Reward） | 路径 B（KL Loss） |
|--|--|--|
| **配置项** | `algorithm.use_kl_in_reward` | `actor.use_kl_loss` |
| **作用阶段** | reward → advantage → actor（全链路） | 仅 actor loss 函数 |
| **影响 advantage** | 是 | 否 |
| **β 自适应** | 支持 adaptive 动态调整 | 仅固定系数 |
| **典型算法** | PPO | GRPO |

两条路径共享同一套 KL 散度计算函数（`core_algos.kl_penalty`），支持的类型：

| Type | 公式 | 特点 |
|------|------|------|
| `kl` / `k1` | `log π - log π_ref` | 无偏估计，梯度有偏 |
| `mse` / `k2` | `0.5 × (Δlog p)²` | 梯度无偏 |
| `low_var_kl` / `k3` | `exp(Δlog p) - Δlog p - 1` | 低方差，verl 默认 |
| `abs` | `|Δlog p|` | 鲁棒，无理论保证 |
| `k3+` | k3 + straight-through | 梯度更优 |

---

## DAPO 的做法：同时关闭两条路径

DAPO 论文去掉 KL 约束的理由：KL 约束限制了策略的探索空间，而 Clip-Higher 和 Overlong Reward Shaping 可以在不限制探索的前提下实现等效的稳定性。

**关闭方式（`run_qwen3-4b_gsm8k_3090_my_dapo.sh`）：**

```bash
# 路径 A：默认关闭，显式声明
algorithm.use_kl_in_reward=False

# 路径 B：默认关闭，显式声明（原来误设为 True，本次修正）
actor_rollout_ref.actor.use_kl_loss=False
# kl_loss_coef 和 kl_loss_type 随之移除（use_kl_loss=False 时不起作用）
```

**替代稳定性手段：**

- **Clip-Higher**（Task 3）：`clip_ratio_high=0.28`，限制策略上界偏移，防止训练崩溃
- **Overlong Reward Shaping**（Task 1）：线性惩罚过长回答，防止模型无限生成倾向

两者均不依赖参考模型，避免了 KL 约束的显存和计算开销（无需保持 ref policy 前向传播）。

---

## 脚本改动记录

**文件：** `examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_my_dapo.sh`

**改动前：**
```bash
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
```

**改动后：**
```bash
actor_rollout_ref.actor.use_kl_loss=False \
```

头部注释同步添加了 Task 4 说明，明确标注两条路径的配置项及其含义。

---

## 总结

Task 4 是一个"理解"任务：verl 中有两条独立的 KL 路径，分别在 reward 阶段和 loss 阶段起作用，DAPO 需要同时关闭两者。本次通过代码追踪完整理解了这两条路径后，发现原脚本中 `use_kl_loss=True` 的设置与 DAPO 设计不符，已予以修正。
