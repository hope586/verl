# MHPO 实现报告

**日期：** 2026-04-09  
**论文：** MHPO: Modulated Hazard-aware Policy Optimization for Stable Reinforcement Learning  
**arXiv：** https://arxiv.org/abs/2603.16929  
**涉及文件：**
- `verl/workers/config/actor.py`
- `verl/trainer/ppo/core_algos.py`
- `examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_mhpo.sh`

---

## 1. 论文背景：GRPO 的不稳定性问题

GRPO 用 **重要性比率**（importance ratio）$r = \pi_\theta / \pi_{\theta_\text{old}}$ 来衡量新旧策略之间的偏差。在长序列（CoT）场景下，token 级别的 $r$ 会跨越数个数量级，引发梯度爆炸，导致训练不稳定。

现有方案的局限：

| 方法 | 做法 | 缺陷 |
|------|------|------|
| PPO / GRPO | 对称硬截断 `clip(r, 1-ε, 1+ε)` | 边界处梯度不连续，截断外梯度为零 |
| DAPO | 非对称硬截断 `[1-ε_low, 1+ε_high]` | 同上，只是边界不对称 |
| SAPO | sigmoid 软门控 | 恢复了连续性，但无法对正负方向分别控制 |
| GSPO | 序列级比率控制 | 降低方差，但丢失 token 级别的精细信号 |

MHPO 同时解决了两个问题：**梯度保真度**（不引入硬截断）+ **方向感知惩罚**（对正负偏移分别处理）。

---

## 2. MHPO 的两个核心组件

### 2.1 Log-Fidelity Modulator（LFM，对数保真调制器）

$$\psi(r) = c \cdot \tanh\!\left(\frac{\log r}{c}\right)$$

**直觉：** 把无界的 $\log r \in (-\infty, +\infty)$ 通过 tanh 压缩到 $(-c, c)$，同时保持在 $r=1$ 附近的线性行为。

三个关键性质：

- **局部保真（P1）：** $r \approx 1$ 时，$\psi(r) \approx \log r$，梯度退化为标准策略梯度，无偏差。
- **平滑衰减（P2）：** $r \to \infty$ 或 $r \to 0$ 时，梯度通过 $\text{sech}^2$ 项优雅地趋近于零，而非硬截断为零。outlier token 依然参与梯度，只是影响被压制。
- **高阶可微（P3）：** $C^\infty$ 光滑，Adam 的一阶/二阶矩估计不会被"数学冲击"破坏。

梯度上界（Theorem 1）：$\sup_r |\mathcal{M}(r)| \leq e^c$，默认 $c=1.5$ 时上界约为 4.5。

### 2.2 Decoupled Hazard Penalty（DHP，解耦危险惩罚）

$$\zeta(r) = \left(\frac{s(\text{sg}[\psi])}{\ \lambda_+\ }\right)^{k_+} + \left(\frac{s(-\text{sg}[\psi])}{\lambda_-}\right)^{k_-}$$

其中 $s(x) = \log(1+e^x)$（softplus），$\text{sg}[\cdot]$ 是 stop-gradient。

**直觉：** 两项分别负责正向偏移（$r>1$，概率增大）和负向偏移（$r<1$，概率减小），用 Weibull 累积危险函数的形状来控制惩罚的"激活时机"和"加速速度"。

- **$\lambda$（尺度）：** 惩罚从什么偏移量开始显著。$\lambda_- < \lambda_+$ 意味着负向偏移更早被压制。
- **$k$（形状）：** $k>1$ 时，惩罚超线性加速，大偏移受到更激烈的惩罚。$k_- > k_+$ 意味着负向惩罚加速更猛。
- **stop-gradient 的作用：** $\zeta$ 只作为衰减系数 $\exp(-\zeta) \in (0,1]$ 缩放梯度幅值，不参与梯度方向的计算，防止反传路径扭曲策略梯度方向。

**为什么要非对称？** 正向偏移（探索新行为）和负向偏移（抑制已有行为）的风险天然不对称：过度正向偏移导致 mode collapse，过度负向偏移导致不可逆的 policy erosion。

### 2.3 完整目标函数

$$\mathcal{L}_\text{MHPO}(\theta) = -\mathbb{E}\left[\frac{1}{K}\sum_i \frac{1}{T_i}\sum_t \exp\!\bigl(\psi(r^i_t) - \zeta(r^i_t)\bigr)\cdot \hat{A}^i_t\right]$$

与 GRPO 的对比：

```
GRPO:  -min(r·A,  clip(r, 1-ε, 1+ε)·A)    # 硬截断 + min
MHPO:  -exp(ψ(r) - sg[ζ(r)]) · A           # 软调制，全局可微
```

---

## 3. 在 verl 中的实现

### 3.1 框架接入方式

verl 用**注册表模式**管理 policy loss：

```
PolicyLossConfig (声明超参 + 默认值)
       ↓ YAML / CLI 赋值
core_algos.py (@register_policy_loss 注册)
       ↓ get_policy_loss_fn(loss_mode)
losses.py (ppo_loss 统一调用)
       ↓
actor.py (self.loss_fn)
```

新增 loss 只需：① 在 Config 声明超参；② 用装饰器注册函数；③ YAML 写 `loss_mode=mhpo`。无需改核心 trainer。

### 3.2 修改 `verl/workers/config/actor.py`

在 `PolicyLossConfig` 末尾追加 5 个超参，带论文默认值：

```python
mhpo_c: float = 1.5           # LFM bound
mhpo_k_pos: float = 1.5       # 正向 Weibull 形状
mhpo_lambda_pos: float = 1.0  # 正向 Weibull 尺度
mhpo_k_neg: float = 2.0       # 负向 Weibull 形状（更激进）
mhpo_lambda_neg: float = 0.8  # 负向 Weibull 尺度（更早触发）
```

### 3.3 注册 `compute_policy_loss_mhpo`（`verl/trainer/ppo/core_algos.py`）

核心计算流程（约 20 行有效代码）：

```python
# 1. log-ratio（在 log-space 计算，避免 exp 上溢）
log_r = clamp(log_prob - old_log_prob, -20, 20)

# 2. LFM
psi = c * tanh(log_r / c)

# 3. DHP（stop-gradient 阻断反传）
psi_sg = psi.detach()
zeta = (softplus(psi_sg)/lp)**kp + (softplus(-psi_sg)/lm)**km

# 4. 目标
loss = agg_loss(-exp(psi - zeta) * advantages, mask, mode)
```

监控指标：`ppo_kl`、`mhpo_psi_mean`、`mhpo_zeta_mean`、`mhpo_survival_weight`（$\exp(-\zeta)$ 的均值，反映当前惩罚强度）。

### 3.4 训练脚本

`examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_mhpo.sh`，在 GRPO baseline 基础上的关键改动：

```bash
# 新增：指定 loss 模式和 MHPO 超参
actor_rollout_ref.actor.policy_loss.loss_mode=mhpo
actor_rollout_ref.actor.policy_loss.mhpo_c=1.5
actor_rollout_ref.actor.policy_loss.mhpo_k_pos=1.5
actor_rollout_ref.actor.policy_loss.mhpo_lambda_pos=1.0
actor_rollout_ref.actor.policy_loss.mhpo_k_neg=2.0
actor_rollout_ref.actor.policy_loss.mhpo_lambda_neg=0.8

# 移除：clip_ratio_low / clip_ratio_high（MHPO 不用硬截断）
# 移除：use_kl_loss=True / kl_loss_coef（LFM 已提供稳定性保证）
actor_rollout_ref.actor.use_kl_loss=False
```

---

## 4. 超参说明与调参建议

| 超参 | 默认值 | 作用 | 调小 | 调大 |
|------|--------|------|------|------|
| `c` | 1.5 | LFM 饱和范围，梯度上界 $e^c$ | 更保守，梯度更早饱和 | 更宽松，接近不限制 |
| `k_pos` | 1.5 | 正向惩罚加速度 | 平缓惩罚 | 更激进地抑制过度探索 |
| `lambda_pos` | 1.0 | 正向惩罚触发阈值 | 更早惩罚 | 更晚惩罚，允许更多探索 |
| `k_neg` | 2.0 | 负向惩罚加速度 | 放松对 policy erosion 的限制 | 更严格保护已有行为 |
| `lambda_neg` | 0.8 | 负向惩罚触发阈值 | 更早压制负向偏移 | 允许更多负向偏移 |

论文消融结论：
- `c=1.5` 是最优；过小（0.5）导致梯度信息丢失，过大退化为无约束。
- `k≥2.5` 的超线性惩罚显著好于 `k=1`（线性），验证了尾部衰减的必要性。
- `lambda=0.8`（早触发）好于 `lambda=2.0`（晚触发），早触发才能在梯度崩溃前介入。

---

## 5. 与其他方法的对比定位

```
方法         梯度连续  正负解耦  token级  有理论界
──────────────────────────────────────────────
GRPO/PPO      ✗        ✗        ✓        ✗
DAPO          ✗        部分      ✓        ✗
SAPO          ✓        ✗        ✓        ✗
GSPO          ✗        ✗        ✗        ✗
MHPO          ✓        ✓        ✓        ✓
```

MHPO 是目前同时满足全部四个条件的唯一方法（截至论文发布时）。

---

## 6. 后续可探索方向

1. **与 DAPO 组合：** MHPO 替换 loss，DAPO 的 Overlong Reward Shaping + Dynamic Sampling 继续保留，两者正交。
2. **消融实验：** 单独跑"只有 LFM，无 DHP"（把 $k$ 设为 1、$\lambda$ 设很大）验证两个组件各自的贡献。
3. **超参扫描：** 论文建议先固定 `c=1.5`，再调 `k_neg` 和 `lambda_neg`，因为负向惩罚对稳定性影响更大。
