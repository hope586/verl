# verl 升级至 v0.6.1 指导文件

## 背景

调查时间：2026-03-30
当前运行版本：v0.6.0
目标版本：v0.6.1
服务器环境：Ubuntu 20.04 / GLIBC 2.31 / CUDA Driver 560.35.03（≤CUDA 12.6）/ 8× GTX 3090

## 兼容性结论

v0.6.1 对当前服务器硬件属于**条件兼容**：

- flash-attn、liger-kernel、ray 均无障碍兼容
- **核心问题**：v0.6.1 的 vllm 依赖范围是 `>=0.8.5,<=0.11.0`，pip 默认会安装最新版（0.11.0），而 vllm 0.9.0 起引入 breaking change，强制依赖 torch 2.7+（服务器当前为 torch 2.6.0+cu124），无法满足
- **解决方法**：手动固定 `vllm==0.8.5`，绕过 pip 的自动选版行为

## 升级步骤

```bash
ssh isee47
conda activate verl-v060
cd /home/data2/zixuan/verl

# 1. 拉取最新标签
git fetch --tags

# 2. 切换到 v0.6.1
git checkout v0.6.1

# 3. 关键步骤：手动固定 vllm==0.8.5，阻止 pip 选择 0.9.0+
pip install "vllm==0.8.5" --no-deps

# 4. 安装 verl 其余依赖（--no-deps 防止 vllm 被重新升级）
pip install -e ".[vllm]" --no-deps

# 5. 验证 vllm 版本未被升级
python -c "import vllm; print(vllm.__version__)"
# 预期输出：0.8.5 或 0.8.5.post1
```

## 注意事项

- `pip install -e ".[vllm]" --no-deps` 会跳过所有依赖安装，如果 v0.6.1 引入了新的纯 Python 依赖，需要手动补装。可先用 `pip install -e ".[vllm]"` 查看哪些包会被升级，再决定是否加 `--no-deps`
- 升级后建议用 GSM8K GRPO baseline 脚本验证训练是否正常启动

## 更高版本的限制

| verl 版本 | vllm 范围 | 最低 torch 要求 | 当前环境能否满足 |
|-----------|----------|----------------|--------------|
| v0.6.1    | ≥0.8.5, ≤0.11.0 | torch 2.6（固定 vllm==0.8.5）| 条件满足 |
| v0.7.0    | ≥0.8.5, ≤0.12.0 | torch 2.7+（vllm ≥0.9.0 要求）| 不满足 |
| v0.7.1    | ≥0.8.5, ≤0.12.0 | torch 2.7+                   | 不满足 |

若将来需要升级到 v0.7.x，需先将 torch 升级至 2.7+（CUDA 12.8 wheel），届时需重新评估 flash-attn 等依赖的兼容性。