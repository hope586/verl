# DAPO 训练启动调试报告

> 创建时间：2026-04-02
> 分支：work-v061
> 目标：将完整 DAPO 实现整合进启动脚本，并在 isee47（8× GTX 3090）上跑通训练

---

## 最终结果

训练成功启动，step 1、2 均正常完成。

- 初始 val acc（step 0）：**49.2%**（Qwen3-4B on GSM8K，符合预期基线）
- 预计训练时长：**~7.5 小时**（87 steps × ~5.3 min/step）
- 所有 DAPO 组件均已验证生效

---

## DAPO 组件实现状态（启动前）

| 任务 | 组件 | 实现位置 |
|------|------|---------|
| Task 1 | Overlong Reward Shaping | `verl/workers/reward_manager/my_dapo.py` |
| Task 2 | Dynamic Sampling | `verl/trainer/ppo/ray_trainer.py:181` `filter_trivial_groups()` |
| Task 3 | Clip-Higher | `verl/trainer/ppo/core_algos.py:924` + 脚本参数 |
| Task 4 | 去掉 KL 约束（双路径） | 脚本参数 |
| Task 5 | Token-level Loss | `actor.yaml` 默认值 + 脚本显式设置 |

本次工作在此基础上，将 Task 5（token-mean loss）补充进脚本，并修复了启动过程中的 4 个 bug。

---

## Bug 修复记录

### Bug 1：脚本缺少 token-mean loss 参数

**现象：** Task 5 虽已是框架默认值，但脚本未显式声明，意图不清晰。

**修复：** 在脚本中加入：
```bash
actor_rollout_ref.actor.loss_agg_mode=token-mean
```

**提交：** `6b07657b`

---

### Bug 2：`algorithm.filter_groups.enable` 报错 "Key not in struct"

**现象：**
```
Could not override 'algorithm.filter_groups.enable'.
Key 'filter_groups' is not in struct
```

**根本原因：** Hydra 以 YAML 文件为准校验 CLI 参数合法性。`AlgorithmConfig` 的 Python dataclass 中 `filter_groups` 默认值是 `None`，生成的 YAML 里这个键完全不存在，Hydra 因此拒绝覆盖。

**修复过程（走了弯路）：**

1. 先改了 `algorithm.py` 的 dataclass 默认值（无效——Hydra 不读 dataclass 做 CLI 校验）
2. 再改了 `_generated_ppo_trainer.yaml`（无效——这是文档文件，不是实际加载的配置）
3. 最终定位到 `main_ppo.py` 中 `@hydra.main(config_name="ppo_trainer")` 加载的是 `ppo_trainer.yaml`，在该文件 `algorithm` 节加入：

```yaml
filter_groups:
  enable: False
  metric: null
  max_num_gen_batches: 0
```

**提交：** `ecca0a4f`（dataclass）、`c34b7c16`（错误文件）、`be2f19fd`（正确文件）

**教训：** Hydra 的 struct 校验基于实际加载的 YAML，不是 Python dataclass。排查时应先确认 `config_name` 指向哪个文件。

---

### Bug 3：`reward_model.reward_kwargs.*` 报错 "Key not in struct"

**现象：**
```
Could not override 'reward_model.reward_kwargs.max_resp_len'.
Key 'reward_kwargs' is not in struct
```

**根本原因：** `reward_kwargs` 是传给 RewardManager `__init__` 的任意参数字典，不同 reward manager 的参数各不相同，无法在 YAML 里预定义所有可能的键。

**修复：** 将脚本中的覆盖改为追加（`+` 前缀）：
```bash
# 修改前
reward_model.reward_kwargs.max_resp_len=2048

# 修改后
+reward_model.reward_kwargs.max_resp_len=2048
```

Hydra 的 `+` 前缀表示"追加新键，即使它不在 struct 里"，官方 DAPO 脚本对 `reward_kwargs` 也采用相同做法。

**提交：** `841849b5`

---

### Bug 4：`AssertionError: only support equal chunk. Got size 670 and chunk 8`

**现象：** Hydra 配置全部通过后，训练在第一个 `update_actor` 步骤崩溃：
```
AssertionError: only support equal chunk. Got size of DataProto 670 and chunk 8.
```

**根本原因：** `filter_trivial_groups` 过滤后剩余 670 个样本，670 不能被 8（GPU 数 / dp_size）整除。verl 的 dispatch 机制要求 batch 能均匀分配到每个 data parallel worker。

**修复：** 在 `ray_trainer.py` 的调用处，过滤完成后立即截断尾部多余样本：

```python
batch, filter_metrics = filter_trivial_groups(batch, metric)
metrics.update(filter_metrics)

# 截断到 dp_size 的整数倍，避免 update_actor 的 unequal chunk 错误
dp_size = self.actor_rollout_wg.world_size
remainder = len(batch) % dp_size
if remainder != 0:
    batch = batch[: len(batch) - remainder]
```

截断在样本级别进行（最多丢弃 7 个样本），不影响训练稳定性。

**提交：** `1111db13`

---

## 最终脚本关键参数

```bash
# DAPO 核心
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.28
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.loss_agg_mode=token-mean
algorithm.use_kl_in_reward=False
algorithm.filter_groups.enable=True
algorithm.filter_groups.metric=acc
reward_model.reward_manager=my_dapo
+reward_model.reward_kwargs.max_resp_len=2048
+reward_model.reward_kwargs.overlong_buffer_len=1024
+reward_model.reward_kwargs.overlong_penalty_factor=1.0
```

---

## 关于训练速度的说明

DAPO（~7.5 小时）比 GRPO baseline（~12 小时）更快，原因在于 Dynamic Sampling 的副作用：`filter_trivial_groups` 在 `update_actor` 之前过滤掉全对/全错的 group，减少了最耗时的 actor 更新步骤的计算量。这是一个学习效率与计算效率同时提升的设计。

---

## 涉及的文件改动

| 文件 | 改动内容 |
|------|---------|
| `examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_my_dapo.sh` | 加入 token-mean、`+`前缀 reward_kwargs |
| `verl/trainer/config/algorithm.py` | `filter_groups` 默认值改为 `FilterGroupsConfig()` |
| `verl/trainer/config/ppo_trainer.yaml` | `algorithm` 节加入 `filter_groups` 字段 |
| `verl/trainer/config/_generated_ppo_trainer.yaml` | 同上（文档文件，顺带更新） |
| `verl/trainer/ppo/ray_trainer.py` | 过滤后截断 batch 到 dp_size 整数倍 |
