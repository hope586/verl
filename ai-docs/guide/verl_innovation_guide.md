# verl 强化学习框架创新科研指南

## Context

verl 是字节跳动开源的工业级强化学习训练框架，支持 PPO、GRPO、DAPO 等算法。
用户已完成 GRPO baseline 训练，下一步是基于框架做算法创新（如 DAPO 实现、自定义 loss、reward 等）。
本文件是"如何在 verl 中进行科研创新"的完整指南，从配置调参到重写核心算法全覆盖。

---

## 一、框架架构速览

### HybridFlow 核心设计

```
单进程控制器（Driver Process）
  └─ verl/trainer/ppo/ray_trainer.py → RayPPOTrainer.fit()
       ├─ 算法主循环（顺序执行）
       ├─ 优势计算、KL 惩罚（在 Driver 上运行）
       └─ 调度 Worker 执行计算

多进程计算引擎（Ray Workers）
  ├─ ActorRolloutRefWorker  → 生成序列 + 计算 log_prob + 策略更新
  ├─ CriticWorker           → 计算 value + 更新价值函数
  └─ RewardModelWorker      → 计算奖励分数
```

### 训练一步的调用顺序（ray_trainer.py fit()，行 956-1313）

```
generate_sequences()          [行 1038] → rollout 生成序列
compute_log_prob()            [行 1119] → 重算 log_prob（用于 IS 比值）
compute_ref_log_prob()        [行 1126] → 参考策略 log_prob
compute_values()              [行 1149] → Critic 估值（PPO 用，GRPO 不用）
  ↓
compute_reward()              [行 1102] → RewardManager.__call__()
apply_kl_penalty()            [行 1165] → 可选：在奖励中减去 KL 惩罚
  ↓
compute_advantage()           [行 1193] → GAE / GRPO / REINFORCE++ 等
  ↓
update_critic()               [行 1204] → 更新价值函数（PPO 用）
update_actor()                [行 1212] → 更新策略（policy loss + entropy + KL loss）
```

---

## 二、分层创新地图

| 层次 | 创新类型示例 | 修改内容 | 难度 |
|------|-------------|---------|------|
| **L0 零代码** | 调超参、换 clip 范围、换 KL 类型 | YAML 配置文件 | ★ |
| **L1 轻量扩展** | 自定义奖励函数、overlong 惩罚 | 新建 reward 文件 | ★★ |
| **L2 模块替换** | 自定义优势估计、loss 聚合模式 | 在 core_algos.py 注册新函数 | ★★★ |
| **L3 核心改写** | 新 policy loss（如 GSPO）、新 KL 路径 | core_algos.py + 配置 | ★★★★ |
| **L4 架构扩展** | 新训练范式（如异步奖励、多轮对话） | 新 Worker + ray_trainer 主循环 | ★★★★★ |

---

## 三、各创新维度详解

### 3.1 奖励函数（Reward Function）

**最简方式：配置文件注入自定义函数**

```yaml
# 在训练 YAML 中
custom_reward_function:
  path: /path/to/my_reward.py   # 你的奖励文件
  name: compute_score           # 函数名
```

**函数签名**（`verl/workers/reward_manager/abstract.py:27`）：
```python
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None
) -> float:
```

**更复杂：自定义 RewardManager 类**（继承基类，支持完整 DataProto 处理）

- 基类：`verl/workers/reward_manager/abstract.py:27` → `AbstractRewardManager`
- 必须实现：`__init__()` 和 `__call__(data: DataProto) -> torch.Tensor`
- 注册方式：在类上加 `@register("my_reward")`，配置中设 `reward_model.reward_manager: "my_reward"`
- 参考实现：`verl/workers/reward_manager/naive.py`（最简单）、`dapo.py`（带 overlong 惩罚）

**Overlong 惩罚（DAPO 特性）**：

```yaml
reward_model:
  overlong_buffer:
    enable: True
    len: 4096          # 超过此长度开始惩罚
    penalty_factor: 1.0
```

这是通过 `dapo.py` 的 RewardManager 实现，不需要改代码，只需配置 + 使用对应的 reward_manager。

---

### 3.2 KL 惩罚（KL Penalty）

verl 有**两条 KL 路径**，可独立或同时使用：

#### 路径 A：In-Reward KL（在奖励中减去 KL）

```yaml
algorithm:
  use_kl_in_reward: True        # 启用
  kl_penalty: low_var_kl        # KL 估计类型
  kl_ctrl:
    type: adaptive              # fixed（固定系数）/ adaptive（自动调整）
    kl_coef: 0.001
    target_kl: 0.1              # adaptive 模式的目标 KL
```

- 计算位置：`ray_trainer.py:121` → `apply_kl_penalty()`
- 效果：`reward = score - β * KL(π || π_ref)`
- KL 类型选项（`core_algos.py:1450`）：`kl`（标准）、`abs`、`mse`、`low_var_kl`（低方差，推荐）

#### 路径 B：KL Loss（在 loss 函数中加 KL 项）

```yaml
actor_rollout_ref:
  actor:
    use_kl_loss: True           # GRPO 用这个
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
```

- 计算位置：`dp_actor.py:500` → 在 `update_policy()` 内直接加入 loss
- 效果：`total_loss = policy_loss + kl_coef * KL(π || π_ref)`

| 对比 | 路径 A（In-Reward） | 路径 B（KL Loss） |
|------|-------------------|-----------------|
| 适用算法 | PPO | GRPO |
| 控制方式 | Adaptive / Fixed | 固定系数 |
| 作用时机 | rollout 后计算奖励时 | actor 更新时 |

---

### 3.3 Clip 裁剪范围

**标准 PPO**（对称裁剪）：
```yaml
actor_rollout_ref.actor:
  clip_ratio: 0.2   # ratio ∈ [0.8, 1.2]
```

**DAPO 风格：非对称裁剪（Clip-Higher）**：
```yaml
actor_rollout_ref.actor:
  clip_ratio_low: 0.2    # 下界：ratio ≥ 0.8
  clip_ratio_high: 0.28  # 上界：ratio ≤ 1.28（放宽上界，允许策略更大幅度探索）
```

**Dual-clip PPO**（防止 advantages < 0 时 ratio 过大）：
```yaml
actor_rollout_ref.actor:
  clip_ratio_c: 3.0   # pg_losses3 = -advantages * 3.0 的下界
```

- 代码位置：`core_algos.py:922-975`（`compute_policy_loss_vanilla()`）
- 原理：`ratio = exp(log_π - log_π_old)`，裁剪为 `[1-low, 1+high]`

---

### 3.4 Loss 函数

#### 修改聚合模式（无需改代码）

```yaml
actor_rollout_ref.actor:
  loss_agg_mode: token-mean   # 选项见下
```

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `token-mean` | 对所有 token 平均 | 默认 PPO |
| `seq-mean-token-sum` | 先按句子 sum，再平均 | DAPO |
| `seq-mean-token-mean` | 先按句子 mean，再平均 | |
| `seq-mean-token-sum-norm` | 除以固定 response_length | DrGRPO |

#### 替换 Policy Loss 函数（L3 级别）

verl 已内置多种 loss，通过配置切换：

```yaml
actor_rollout_ref.actor:
  policy_loss:
    loss_mode: gspo    # vanilla / gspo / gpg / clip_cov / kl_cov / geo_mean
```

| loss_mode | 文件位置 | 特点 |
|-----------|---------|------|
| `vanilla` | `core_algos.py:888` | 标准 PPO clip |
| `gspo` | `core_algos.py:979` | 序列级重要性采样 |
| `gpg` | `core_algos.py:1053` | 简化梯度 PPO |
| `clip_cov` | `core_algos.py:1086` | 覆盖值裁剪 |
| `geo_mean` | `core_algos.py:1268` | 几何平均 PPO |

#### 自定义 Policy Loss（L3 级别，需改代码）

在 `verl/trainer/ppo/core_algos.py` 中添加：

```python
@register_policy_loss("my_loss")
def compute_policy_loss_my(
    old_log_prob, log_prob, advantages, response_mask,
    loss_agg_mode="token-mean", config=None, rollout_is_weights=None
) -> tuple[torch.Tensor, dict]:
    # 你的 loss 实现
    return loss_scalar, {"actor/my_metric": value}
```

配置：`policy_loss.loss_mode: "my_loss"`

---

### 3.5 优势估计（Advantage Estimator）

通过配置切换（无需改代码）：
```yaml
algorithm:
  adv_estimator: grpo   # gae / grpo / reinforce_plus_plus / rloo / opo / gpg
```

| 估计器 | 适用算法 | 特点 |
|--------|---------|------|
| `gae` | PPO | Generalized Advantage Estimation，需要 Critic |
| `grpo` | GRPO/DAPO | 组内相对奖励，无需 Critic |
| `reinforce_plus_plus` | REINFORCE++ | 无 Critic，带 baseline |
| `rloo` | RLOO | Leave-one-out 估计 |

自定义新估计器（L3 级别）：在 `core_algos.py` 中用 `@register_adv_est("my_est")` 注册，函数返回 `(advantages, returns)`。

---

### 3.6 动态采样（Dynamic Sampling，DAPO 特性）

```yaml
algorithm:
  filter_groups:
    enable: True
    metric: acc        # 过滤指标（acc / reward）
    max_num_gen_batches: 10  # 最多重新采样几轮
```

这会在 `ray_trainer.py` 中触发 `filter_trivial_groups()` 逻辑，过滤掉组内所有样本都正确或都错误的 group，只保留有学习信号的样本。

---

## 四、如何使自定义内容生效

### 配置驱动（L0-L1，最简单）

直接修改训练 YAML 文件中的相关字段，重新运行训练脚本即可。

### 注册机制（L2-L3，需改代码）

verl 的 reward、loss、advantage estimator 都用注册表管理：

```python
# 注册新 policy loss
@register_policy_loss("my_loss")    # 在 core_algos.py 中

# 注册新 advantage estimator  
@register_adv_est("my_est")         # 在 core_algos.py 中

# 注册新 reward manager
@register("my_reward")              # 在你的 reward 文件中
```

注册后在 YAML 中通过名字引用，框架自动找到对应实现。

### Worker 级别扩展（L4，架构级）

参考 `docs/advance/dpo_extension.rst` 的 3 步法：
1. 在 Worker 中用 `@register(dispatch_mode=...)` 定义分布式计算 API
2. 在 `ray_trainer.py` 的 `fit()` 主循环中调用新 Worker 方法
3. 通过 `DataProto` 传递数据

---

## 五、关键文件索引

| 功能 | 关键文件 | 说明 |
|------|---------|------|
| 训练入口 | `verl/trainer/main_ppo.py:35` | Hydra 入口，Ray 初始化 |
| **训练主循环** | `verl/trainer/ppo/ray_trainer.py:956` | `fit()` 方法，修改算法逻辑入口 |
| **核心算法** | `verl/trainer/ppo/core_algos.py` | loss、KL、clip、优势估计全在这里 |
| Actor 更新 | `verl/workers/actor/dp_actor.py:372` | `update_policy()`，policy loss 调用点 |
| Critic 更新 | `verl/workers/critic/dp_critic.py:192` | `update_critic()`，value loss 调用点 |
| **Reward 基类** | `verl/workers/reward_manager/abstract.py:27` | 自定义 RewardManager 必须继承 |
| Reward 注册 | `verl/workers/reward_manager/registry.py` | `@register()` 装饰器 |
| 算法配置 | `verl/trainer/config/algorithm.py:334` | adv_estimator、KL 等配置定义 |
| Actor 配置 | `verl/workers/config/actor.py:55` | clip_ratio、loss_mode 等配置定义 |
| DAPO Reward | `verl/workers/reward_manager/dapo.py` | overlong 惩罚参考实现 |

---

## 六、快速上手路径（科研工作流）

### 第一步：改奖励（最常见的入手点）

1. 新建 `my_reward.py`，实现 `compute_score(data_source, solution_str, ground_truth, extra_info) -> float`
2. 在 YAML 中配置 `custom_reward_function.path` 和 `name`
3. 或：继承 `AbstractRewardManager`，用 `@register` 注册，配置 `reward_model.reward_manager`

### 第二步：改训练目标（调整 loss）

1. 先通过配置尝试已有选项：`clip_ratio_high`、`loss_agg_mode`、`loss_mode`
2. 需要新 loss 时：在 `core_algos.py` 中加 `@register_policy_loss("name")` 函数

### 第三步：改采样策略（如 DAPO 动态采样）

1. 配置 `algorithm.filter_groups.enable: True`
2. 可在 `ray_trainer.py` 中修改 `filter_trivial_groups()` 逻辑

### 第四步：改整体训练范式

1. 理解 `ray_trainer.py` 的 `fit()` 主循环
2. 在关键步骤之间插入自定义逻辑（如额外的 reward 信号、多模型交互等）

---

## 七、验证方式

- **配置级修改**：直接重跑训练脚本，观察 wandb/console 中 `actor/pg_clipfrac`、`actor/ppo_kl`、reward 等指标
- **代码级修改**：
  1. 单元测试：`pytest tests/` 或针对 `core_algos.py` 写小脚本验证 loss 值
  2. 少量数据快速验证：缩小 `total_epochs=1`、`train_batch_size` 跑通 pipeline
  3. 对比实验：与 GRPO baseline 脚本结果对比，确认改动方向符合预期
- **关键指标**：
  - `actor/pg_clipfrac`：clip 触发率（太高说明 learning rate 过大）
  - `actor/ppo_kl`：KL 散度（监控策略漂移）
  - `critic/vf_clipfrac`：价值函数 clip 率
  - reward mean/std：奖励信号质量
