set -x

# DAPO (overlong reward shaping + dynamic sampling): Qwen3-4B on GSM8K
# Hardware: 8x GTX 3090 (24GB VRAM each)
# verl version: v0.6.1
# Reward manager: my_dapo (NaiveRewardManager + R_length overlong penalty)
#
# DAPO components enabled:
#   [Task 1] Overlong Reward Shaping: linear penalty for responses in (L_max-L_cache, L_max]
#   [Task 2] Dynamic Sampling: filter trivial groups (all-correct / all-wrong) before advantage
#   [Task 3] Clip-Higher: asymmetric PPO clip [1-clip_ratio_low, 1+clip_ratio_high]
#   [Task 4] Remove KL Constraints: both KL paths disabled (DAPO paper §3.4)
#   [Task 5] Token-level Loss: loss_agg_mode=token-mean (DAPO paper §3.1)
#             Path A (in-reward):  algorithm.use_kl_in_reward=False  (already default)
#             Path B (loss term):  actor.use_kl_loss=False           (set explicitly)
#             Stability is maintained by Clip-Higher + Overlong Shaping instead of KL
#
# Overlong penalty params:
#   max_resp_len=2048         L_max: matches data.max_response_length
#   overlong_buffer_len=1024  L_cache: linear penalty zone [1024, 2048]
#   overlong_penalty_factor=1.0  factor=1 reproduces exact DAPO paper formula
#
# Dynamic sampling params:
#   filter_groups.enable=True   enable group filtering
#   filter_groups.metric=acc    use raw correctness signal (not penalized reward)
#
# Clip-Higher params:
#   clip_ratio_low=0.2    lower bound: ratio >= 1-0.2=0.8 (same as standard PPO)
#   clip_ratio_high=0.28  upper bound: ratio <= 1+0.28=1.28 (relaxed from 1.2)
#                         encourages exploration on positive-advantage tokens

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
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
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
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    reward_model.reward_manager=my_dapo \
    reward_model.reward_kwargs.max_resp_len=2048 \
    reward_model.reward_kwargs.overlong_buffer_len=1024 \
    reward_model.reward_kwargs.overlong_penalty_factor=1.0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='grpo_gsm8k_3090' \
    trainer.experiment_name='qwen3_4b_my_dapo_overlong' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 $@