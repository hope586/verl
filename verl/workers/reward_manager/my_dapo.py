# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.naive import NaiveRewardManager


@register("my_dapo")
class MyDAPORewardManager(NaiveRewardManager):
    """DAPO reward manager: NaiveRewardManager + overlong penalty R_length(y).

    The overlong penalty formula (DAPO paper):
        R_length(y) = 0,                                     if |y| <= L_max - L_cache
                      ((L_max - L_cache) - |y|) / L_cache,  if L_max - L_cache < |y| <= L_max
                      -1,                                    if |y| > L_max

    The penalty is scaled by overlong_penalty_factor (factor=1 reproduces the paper formula).
    Final reward = base_reward + overlong_penalty_factor * R_length(y).

    Args:
        max_resp_len: L_max in the formula. The maximum response length used during training
            (i.e. the rollout.response_length config value). Must be set explicitly.
        overlong_buffer_len: L_cache in the formula. Responses longer than (L_max - L_cache)
            start receiving a linear penalty. If 0, no penalty is applied.
        overlong_penalty_factor: Scales R_length. Set to 1.0 for the exact paper formula.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len: int = 0,
        overlong_buffer_len: int = 0,
        overlong_penalty_factor: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key, **kwargs)
        self.max_resp_len = max_resp_len              # L_max
        self.overlong_buffer_len = overlong_buffer_len  # L_cache
        self.overlong_penalty_factor = overlong_penalty_factor

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # Step 1: get base rewards from NaiveRewardManager
        base_result = super().__call__(data, return_dict=True)

        # Defensive: super() should always return a dict here (return_dict=True),
        # but handle a bare tensor just in case.
        if isinstance(base_result, dict):
            reward_tensor = base_result["reward_tensor"]
            reward_extra_info = base_result.get("reward_extra_info", defaultdict(list))
        else:
            reward_tensor = base_result
            reward_extra_info = defaultdict(list)

        # Step 2: apply overlong penalty per sample
        L_max = self.max_resp_len
        L_cache = self.overlong_buffer_len
        factor = self.overlong_penalty_factor

        # Skip if penalty is disabled (buffer=0 would cause division by zero, factor=0 is a no-op)
        if L_max > 0 and L_cache > 0 and factor != 0.0:
            for i in range(len(data)):
                data_item = data[i]

                response_length = data_item.batch["responses"].shape[-1]
                response_mask = data_item.batch["attention_mask"][-response_length:]
                valid_response_length = int(response_mask.sum())

                if valid_response_length == 0:
                    continue

                y_len = valid_response_length  # |y|

                # Compute R_length(y) per the three-segment formula
                if y_len <= L_max - L_cache:
                    r_length = 0.0
                elif y_len <= L_max:
                    r_length = ((L_max - L_cache) - y_len) / L_cache
                else:
                    r_length = -1.0

                # Add scaled penalty at the same position as the base reward
                reward_tensor[i, valid_response_length - 1] += factor * r_length

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor