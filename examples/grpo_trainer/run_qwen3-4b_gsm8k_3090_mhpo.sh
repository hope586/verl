set -x

# MHPO: Modulated Hazard-aware Policy Optimization (https://arxiv.org/abs/2603.16929)
# Model: Qwen3-4B on GSM8K
# Hardware: 8x GTX 3090 (24GB VRAM each)
# verl version: v0.6.1
#
# MHPO replaces GRPO's hard clip+min with a fully differentiable objective:
#
#   L = -E[ exp(ψ(r) - sg[ζ(r)]) · Â ]
#
#   ψ(r) = c · tanh(log(r)/c)          Log-Fidelity Modulator (LFM)
#   ζ(r) = (softplus(sg(ψ))/λ+)^k+    Decoupled Hazard Penalty (DHP)
#            + (softplus(-sg(ψ))/λ-)^k-
#
# Key properties vs GRPO:
#   - No hard clip boundary: gradient is smooth and non-zero everywhere
#   - LFM bounds gradient multiplier by e^c ≈ 4.5 (c=1.5), preventing gradient spikes
#   - DHP applies asymmetric Weibull penalties:
#       positive shift (r>1): lighter regulation (k+=1.5, λ+=1.0), encourages exploration
#       negative shift (r<1): stricter suppression (k-=2.0, λ-=0.8), prevents policy erosion
#   - KL loss disabled: stability is provided by the bounded gradient multiplier instead
#
# MHPO hyperparameters (paper Section 4.1 defaults):
#   mhpo_c=1.5            LFM bound; controls max gradient multiplier e^c ≈ 4.5
#   mhpo_k_pos=1.5        Weibull shape for positive shift penalty
#   mhpo_lambda_pos=1.0   Weibull scale for positive shift (onset threshold)
#   mhpo_k_neg=2.0        Weibull shape for negative shift penalty (superlinear)
#   mhpo_lambda_neg=0.8   Weibull scale for negative shift (earlier onset than positive)

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

model_path=$HOME/models/Qwen/Qwen3-4B

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$gsm8k_train_path']" \
    data.val_files="['$gsm8k_test_path']" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.policy_loss.loss_mode=mhpo \
    actor_rollout_ref.actor.policy_loss.mhpo_c=1.5 \
    actor_rollout_ref.actor.policy_loss.mhpo_k_pos=1.5 \
    actor_rollout_ref.actor.policy_loss.mhpo_lambda_pos=1.0 \
    actor_rollout_ref.actor.policy_loss.mhpo_k_neg=2.0 \
    actor_rollout_ref.actor.policy_loss.mhpo_lambda_neg=0.8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='grpo_gsm8k_3090' \
    trainer.experiment_name='qwen3_4b_mhpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 $@
