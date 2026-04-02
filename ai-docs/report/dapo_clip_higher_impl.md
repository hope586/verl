# DAPO Clip-Higher 实现报告

> 创建时间：2026-04-02
> 实现分支：work-v061
> 对应论文：DAPO arxiv:2503.15388，Section 3.2

---

## 一、背景

标准 PPO 使用对称裁剪（symmetric clipping）：

```
ratio_clipped = clip(ratio, 1 - ε, 1 + ε)    # 默认 ε = 0.2 → [0.8, 1.2]
```

其中 `ratio = π_θ(a|s) / π_θ_old(a|s)` 是新旧策略的重要性比率。

**问题：对称裁剪对"探索"方向过于保守。**

当 advantage > 0（这个 token 应该鼓励），ratio > 1（新策略相对旧策略更倾向于输出此 token），标准 PPO 在 ratio > 1.2 时裁掉梯度，阻止策略做更大幅度的正向更新。DAPO 论文认为这抑制了训练早期的探索，尤其在稀疏奖励场景。

**Clip-Higher 的做法（DAPO Section 3.2）：**

```
ratio_clipped = clip(ratio, 1 - ε_low, 1 + ε_high)
其中 ε_low = 0.2，ε_high = 0.28
```

下界不变（保守地控制负向更新），上界放宽到 1.28（允许更积极的正向更新）。

---

## 二、实现方式

Clip-Higher **无需修改任何 Python 代码**，verl 框架已将 `clip_ratio_high` 作为一等配置项内置。

### 配置项定义

| 字段 | 文件 | 行号 | 默认值 |
|------|------|------|--------|
| `clip_ratio_low` | `verl/workers/config/actor.py` | 102 | 0.2 |
| `clip_ratio_high` | `verl/workers/config/actor.py` | 103 | 0.2 |

YAML 同步声明于 `verl/trainer/config/actor/actor.yaml:35-39`。

默认值 `clip_ratio_low == clip_ratio_high == 0.2` 退化为标准对称 PPO，完全向后兼容。

### 启用方式：脚本中加两行 Hydra override

```bash
actor_rollout_ref.actor.clip_ratio_low=0.2 \
actor_rollout_ref.actor.clip_ratio_high=0.28 \
```

已写入：`examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_my_dapo.sh`

---

## 三、完整调用链

```
Shell Script
  clip_ratio_high=0.28
  → actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
         ↓ Hydra CLI override
  verl/trainer/main_ppo.py（@hydra.main）
         ↓ OmegaConf 合并 actor.yaml 默认值 + CLI override
  verl/workers/config/actor.py
    ActorConfig.clip_ratio_high = 0.28
         ↓ DataParallelPPOActor(config=self.config.actor)
  verl/workers/actor/dp_actor.py
    update_policy() → policy_loss_fn(..., config=self.config)
         ↓ get_policy_loss_fn("vanilla") 分发
  verl/trainer/ppo/core_algos.py
    compute_policy_loss_vanilla()
      clip_ratio_high = config.clip_ratio_high   # = 0.28
      cliprange_high = clip_ratio_high
      pg_losses2 = -advantages * torch.clamp(
          ratio,
          1 - cliprange_low,   # = 0.8
          1 + cliprange_high   # = 1.28  ← Clip-Higher 生效
      )
```

关键代码行：[core_algos.py:949-951](../../verl/trainer/ppo/core_algos.py#L949-L951)

---

## 四、数学效果

### 裁剪区间对比

| 设置 | 裁剪区间 | ratio 上限 |
|------|---------|-----------|
| 标准 PPO（ε=0.2）| [0.8, 1.2] | 1.2 |
| Clip-Higher（DAPO）| [0.8, 1.28] | 1.28 |

### 梯度影响图示

```
advantage > 0 时的 loss（pg_loss = -advantage * clipped_ratio）：

ratio:  0.8  0.9  1.0  1.1  1.2  1.28  1.4  1.6
        ←— 梯度有效 ——→|PPO截止  |Clip-Higher截止
                                  ← 多 8% 的梯度有效区 →
```

- ratio < 1.28：梯度完整传导，策略可以继续增大此 token 的概率
- ratio ≥ 1.28：梯度被截断，防止过度更新

**负 advantage 方向不受影响**（`clip_ratio_low=0.2` 不变），保持对 token 概率下降的保守控制。

### 与 Dual-Clip 的关系

verl 的 vanilla loss 还内置了 Dual-Clip（`clip_ratio_c=3.0`），在 advantage < 0 时加第二重限制。Clip-Higher 与 Dual-Clip 正交，各自在不同 advantage 符号下生效，互不干扰。

---

## 五、实现文件清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_my_dapo.sh` | 修改 | 新增注释说明 + 两行 Hydra override |
| `verl/workers/config/actor.py` | 无需修改 | `clip_ratio_high` 字段已内置 |
| `verl/trainer/ppo/core_algos.py` | 无需修改 | 裁剪逻辑已支持非对称参数 |

---

## 六、与其他 DAPO 组件的关系

| 任务 | 组件 | 作用域 | 是否完成 |
|------|------|--------|---------|
| Task 1 | Overlong Reward Shaping | reward 计算 | ✓ |
| Task 2 | Dynamic Sampling | advantage 计算前的样本过滤 | ✓ |
| **Task 3** | **Clip-Higher** | **policy loss 裁剪** | **✓** |
| Task 4 | KL 约束双路径 | 待学习 | — |
| Task 5 | 串起启动脚本 | 待整合 | — |

三个组件作用于 PPO 训练流水线的不同位置，彼此独立，可组合叠加。

---

## 七、关键结论

1. **Clip-Higher 是纯配置级改动**：verl 框架已内置，无需写任何 Python 代码。
2. **非对称裁剪的核心洞察**：正负 advantage 方向应区别对待——鼓励正向探索（放宽上界），保持负向谨慎（下界不变）。
3. **调用链终点**：`core_algos.py:950` 的 `torch.clamp(ratio, 0.8, 1.28)` 是 Clip-Higher 最终生效的地方。