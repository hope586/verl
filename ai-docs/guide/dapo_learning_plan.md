# 学习规划：掌握 verl 框架下的科研与开发方法

> 创建时间：2026-03-30
> 最后更新：2026-03-30
> 目标：以实现 DAPO 算法为主线，掌握在 verl 工业级 RL 框架下进行算法开发、实验设计和科研迭代的完整方法论。

---

## 总体路线

```
阶段 0：框架理解（已完成）
    → 阶段 1：实现 DAPO（当前任务）
        → 阶段 2：与标准实现对比学习
            → 阶段 3：算法创新
```

---

## 阶段 0：框架理解 ✅ 已完成（2026-03-22 前）

### 完成的阅读
- [x] `examples/grpo_trainer/run_qwen3-8b.sh` — 理解训练脚本结构和参数体系
- [x] `verl/trainer/main_ppo.py` — 理解程序入口和配置加载
- [x] `verl/trainer/ppo/ray_trainer.py`（全文）— 理解训练主循环
  - 前半段：精读（rollout、reward、advantage 计算流程）
  - 后半段：流程级阅读（actor update、logging、checkpoint）

### 收获
- 理解 verl 的"分层阅读"方法论（地图级 / 流程级 / 实现级）
- 理解 Ray 分布式训练的基本概念
- 理解 GRPO 算法的工程实现框架

---

## 阶段 1：自主实现 DAPO 🔄 进行中

> 原则：先自己实现，再和 verl 标准实现（`examples/gmpo_trainer/test_dapo_7b_math.sh`）对比学习，理解设计取舍。

### DAPO 的 5 个核心组件

| 组件 | 论文 Section | 关键参数/位置 | 状态 |
|------|-------------|--------------|------|
| Clip-Higher（非对称裁剪） | Sec 3.2 | `clip_ratio_low=0.2`, `clip_ratio_high=0.28` | ⬜ 待实现 |
| Dynamic Sampling（动态过滤） | Sec 3.3 | `filter_trivial_groups()` in ray_trainer | ⬜ 待实现 |
| Token-level 损失 | Sec 3.1 | `loss_agg_mode="token-mean"` | ⬜ 待理解 |
| 去掉 KL 约束 | Sec 3.4 | `use_kl_in_reward=False`, `use_kl_loss=False` | ⬜ 待理解 |
| Overlong Reward Shaping | Sec 3.5 | 新建 `my_dapo.py` RewardManager | ⬜ 待实现 |

### 任务 1：实现 Overlong Reward Shaping

**目标：** 学会 verl 的 RewardManager 插件系统，写自己的第一个框架扩展。

**需要阅读：**
- `verl/workers/reward_manager/abstract.py` — 接口规范
- `verl/workers/reward_manager/naive.py` — 最简参考实现
- `verl/workers/reward_manager/registry.py` — 注册机制

**需要编写：**
- `verl/workers/reward_manager/my_dapo.py`
- 实现 `@register("my_dapo")` 类，包含 overlong 线性惩罚逻辑

**惩罚公式：**
```
expected_len = max_len - buffer_len
若 valid_response_length > expected_len:
    penalty = -min((len - expected_len) / buffer_len, 1.0) * penalty_factor
```

**完成标准：**
- [ ] 类注册成功（`reward_manager.name=my_dapo` 不报错）
- [ ] 边界值测试通过（len=expected_len, len=max_len, len=中间值）
- [ ] 理解 `reward_tensor[i, valid_response_length - 1] = reward` 这行的含义

---

### 任务 2：实现 Dynamic Sampling

**目标：** 理解训练循环数据流，学会在 ray_trainer 中插入自定义的数据过滤逻辑。

**需要阅读：**
- `ray_trainer.py:137-231` — `compute_advantage()` 函数全文
- 搜索 `fit` 方法，找到 `compute_advantage` 的调用位置
- 理解 `data.non_tensor_batch["uid"]` 的含义

**需要编写：**
- 在 `ray_trainer.py` 中新增 `filter_trivial_groups(batch, metric)` 函数
- 在训练循环合适位置调用它

**过滤逻辑：**
```python
# 对每个 uid group 计算 acc 的均值
# 保留 0 < group_mean_acc < 1 的 group（有正有负样本）
# 丢弃全对 group（group_mean_acc == 1）和全错 group（group_mean_acc == 0）
```

**完成标准：**
- [ ] 理解 DataProto 的布尔索引方式
- [ ] 函数能正确识别并过滤 trivial groups
- [ ] 处理极端情况（所有 group 都被过滤的空 batch）
- [ ] 在日志中打印被过滤掉的 group 数量，便于监控

---

### 任务 3：理解 Clip-Higher 的代码路径

**目标：** 不写新代码，而是读懂参数如何流经框架影响 loss，培养"追踪参数路径"的能力。

**追踪路径：**
```
启动脚本 clip_ratio_high=0.28
  → ActorConfig.clip_ratio_high  (workers/config/actor.py:152)
  → compute_policy_loss_vanilla() (core_algos.py:1313-1342)
  → cliprange_high 影响 pg_losses2 的上界
```

**需要回答的问题（写在代码注释里）：**
1. `clip_ratio_low < clip_ratio_high` 为什么能鼓励正 advantage 方向的探索？
2. 非对称裁剪和对称裁剪的梯度区别在哪里？

**完成标准：**
- [ ] 能用自己的语言解释 clip-higher 的作用
- [ ] 找到代码中具体的那一行并加注释

---

### 任务 4：理解 KL 约束的双路径

**目标：** 搞清楚 verl 中 KL 的两种用法，理解去掉它的含义和风险。

**两条路径：**
```
路径 A：KL as reward penalty（在 compute_reward 里扣分）
  → 搜索 ray_trainer.py 中的 "use_kl_in_reward"

路径 B：KL as loss term（在 actor update 里加惩罚）
  → 搜索 dp_actor.py 中的 "kl_loss_coef"
```

**需要回答（写成注释或文档）：**
- 去掉 KL 约束后，训练可能产生什么问题？
- DAPO 用什么来替代 KL 的稳定作用？（提示：看 overlong shaping + clip-higher 的组合效果）

**完成标准：**
- [ ] 找到两条 KL 路径在代码里的具体位置
- [ ] 写出 100 字以内的理解笔记

---

### 任务 5：串起来——写启动脚本并跑通 pipeline

**目标：** 把前 4 个任务的实现整合，完整跑通一次 DAPO 训练。

**新建文件：**
- `examples/my_dapo/run_dapo_gsm8k_3090.sh`

**关键配置（适配 8x GTX 3090）：**
```bash
# 自己实现的组件
reward.reward_manager.name=my_dapo
algorithm.filter_groups.enable=True
algorithm.filter_groups.metric=acc

# DAPO 核心参数
adv_estimator=grpo
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

# 3090 显存适配
max_response_length=$((1024 * 4))
n_resp_per_prompt=8
NNODES=1
NGPUS_PER_NODE=8
offload=True
```

**完成标准：**
- [ ] 脚本在 isee47 服务器上能无报错启动
- [ ] 训练日志中能看到 filter_groups 的过滤统计
- [ ] 训练 loss 曲线正常下降

---

## 阶段 2：与标准实现对比学习（阶段 1 完成后）

### 对比维度
1. **Reward Manager**：我的实现 vs `reward_manager/dapo.py`
   - 结构差异？边界处理是否一致？
   - verl 的实现有哪些我没考虑到的细节？

2. **Dynamic Sampling**：我的 `filter_trivial_groups` vs verl 的官方实现（如果有的话）
   - 过滤时机是否相同？
   - `max_num_gen_batches` 参数的作用是什么？

3. **整体配置**：我的启动脚本 vs `test_dapo_7b_math.sh`
   - 哪些参数我没有设置？它们的默认值是什么？
   - `loss_mode=geo_mean` 和 `loss_mode=vanilla` 的区别？

### 对比后的输出
- 写一篇"实现对比笔记"（放在 `.claude/log/dapo_comparison_notes.md`）

---

## 阶段 3：算法创新（阶段 2 完成后）

> 在理解 DAPO 工程实现的基础上，识别可以改进的点，提出并实现自己的算法变体。

### 可能的创新方向（供参考，需要结合文献调研）

**方向 A：改进 Dynamic Sampling 策略**
- 现在是硬过滤（全对/全错 → 丢弃）
- 改为软权重？（部分困难 group 降权而非丢弃）

**方向 B：改进 Overlong Shaping**
- 现在是对最后一个 token 给 penalty
- 能否设计 token-level 的渐进式惩罚？

**方向 C：改进 Clip-Higher 策略**
- 固定的非对称 ε 是否可以动态调整？
- 能否基于当前 policy 的熵动态调整 clip range？

### 创新工作流
1. 文献调研 → 找到相关工作和未解决问题
2. 提出假设 → 用 verl 实现最小可验证版本
3. 小规模实验 → GSM8K + 小模型验证效果
4. 对比分析 → 写实验报告

---

## 框架开发能力检查清单

完成 DAPO 实现后，你应该能回答以下问题：

**框架理解：**
- [ ] verl 中 DataProto 是什么？batch 和 non_tensor_batch 的区别？
- [ ] reward_manager 的 `__call__` 方法在训练循环的哪一步被调用？
- [ ] `@register_adv_est` 装饰器如何让新算法被框架自动识别？
- [ ] `loss_agg_mode` 在哪里被解析？影响的是哪个 tensor 的哪个操作？

**工程实践：**
- [ ] 如何在不改动核心框架的情况下添加新算法？
- [ ] 如何在训练日志中添加自定义 metric？
- [ ] 如何用 `verl/trainer/config/algorithm.py` 中的 dataclass 添加新配置项？

**科研方法：**
- [ ] 如何设计一个"最小可验证实验"来证明某个组件有效？
- [ ] 如何用 wandb 对比两次实验的训练曲线？
- [ ] 如何 ablation 各个组件的贡献？

---

## 进度记录

| 日期 | 完成事项 | 备注 |
|------|---------|------|
| 2026-03-22 前 | 完成阶段 0 全部阅读 | ray_trainer.py 后半段流程级阅读 |
| 2026-03-30 | 完成 DAPO 实现方案设计 | 明确 5 个任务和实现顺序 |
| （待更新） | | |

---

## 参考资源

- **DAPO 论文**：arxiv:2503.15388
- **GRPO 论文**：arxiv:2402.03300
- **verl 标准 DAPO 脚本**：`examples/gmpo_trainer/test_dapo_7b_math.sh`
- **verl 标准 DAPO 异步脚本**：`verl/experimental/fully_async_policy/shell/dapo_30b_a3b_base_math_fsdp.sh`
- **isee47 训练日志**：`~/train.log`
