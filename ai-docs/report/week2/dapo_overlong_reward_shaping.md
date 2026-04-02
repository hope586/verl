# DAPO Overlong Reward Shaping 实现报告

## 背景

本次对话完成了 DAPO 五个任务中的**任务 1：Overlong Reward Shaping**。

目标是在不依赖 verl 内置 `dapo.py` 的前提下，从框架扩展接口出发，自主实现带超长惩罚的 Reward Manager，并接入现有训练流程。

---

## 实现内容

### 1. 核心文件：`verl/workers/reward_manager/my_dapo.py`

继承 `NaiveRewardManager`，在其基础上叠加 DAPO 论文中的超长惩罚项 R_length(y)。

**超长惩罚公式（DAPO 论文）：**

$$R_{\text{length}}(y) = \begin{cases} 0, & |y| \leq L_{\max} - L_{\text{cache}} \\ \dfrac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < |y| \leq L_{\max} \\ -1, & |y| > L_{\max} \end{cases}$$

**最终奖励：**

$$R_{\text{total}} = R_{\text{base}} + \text{factor} \times R_{\text{length}}(y)$$

**init 参数（在 naive 基础上新增）：**

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `max_resp_len` | L_max，训练时配置的最大响应长度 | 0（禁用）|
| `overlong_buffer_len` | L_cache，线性惩罚缓冲区长度 | 0（禁用）|
| `overlong_penalty_factor` | 惩罚缩放系数，1.0 还原论文公式 | 1.0 |
| `**kwargs` | 透传给父类，保持兼容性 | — |

**关键实现细节：**
- `L_max` 通过显式参数 `max_resp_len` 传入，而非从张量维度推断，避免 padding 引起的歧义
- 调用 `super().__call__(data, return_dict=True)` 获取基础奖励，对返回值做防御性类型检查（dict / 裸 tensor 均处理）
- 惩罚加在 `reward_tensor[i, valid_response_length - 1]`，与 naive 的奖励落点一致
- `L_cache=0` 或 `max_resp_len=0` 时自动跳过惩罚循环，不影响基础功能

### 2. 注册：`verl/workers/reward_manager/__init__.py`

新增导入与 `__all__` 条目，模块加载时自动完成 `@register("my_dapo")` 注册，yaml 中写 `reward_manager: my_dapo` 即可。

### 3. 训练脚本：`examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_my_dapo.sh`

在 Qwen3-4B GSM8K GRPO baseline 脚本基础上，新增 4 行 reward 配置：

```bash
reward_model.reward_manager=my_dapo
reward_model.reward_kwargs.max_resp_len=2048
reward_model.reward_kwargs.overlong_buffer_len=1024
reward_model.reward_kwargs.overlong_penalty_factor=1.0
```

对应惩罚区间：响应长度 ≤ 1024 时无惩罚，1025~2048 线性从 0 降到 -1，超过 2048 直接 -1。

---

## 参数传递链路

```
shell: reward_model.reward_kwargs.max_resp_len=2048
  → Hydra config: config.reward_model.reward_kwargs = {max_resp_len: 2048, ...}
  → verl/trainer/ppo/reward.py: load_reward_manager(..., **reward_kwargs)
  → MyDAPORewardManager(max_resp_len=2048, overlong_buffer_len=1024, ...)
```

关键函数：`verl/trainer/ppo/reward.py:load_reward_manager()`，用 `config.reward_model.get("reward_kwargs", {})` 取出额外参数并展开传入。

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `verl/workers/reward_manager/my_dapo.py` | 新建，核心实现 |
| `verl/workers/reward_manager/__init__.py` | 修改，新增 import 和 `__all__` |
| `examples/grpo_trainer/run_qwen3-4b_gsm8k_3090_my_dapo.sh` | 新建，训练脚本 |