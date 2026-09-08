import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch.nn.functional as F
from collections.abc import Callable

def tokenize_prompt_and_output(
        prompt_strs: list[str], 
        output_strs: list[str], 
        tokenizer: PreTrainedTokenizer, 
) -> dict[str, torch.Tensor]:
    sequence_tokens = []
    sequence_masks = []
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        # encode 反回的是一维列表，因此要先转换为 tensor
        prompt_ids = tokenizer.encode(
            prompt_str,
            add_special_tokens=False,
        )
        output_ids = tokenizer.encode(
            output_str,
            add_special_tokens=False,
        )
        tokens = torch.tensor(
            prompt_ids + output_ids, 
            dtype=torch.long
        )
        combined_mask = torch.tensor(
            [False] * len(prompt_ids) + [True] * len(output_ids)
        )

        sequence_tokens.append(tokens)
        sequence_masks.append(combined_mask)

    max_length = max(sequence.numel() for sequence in sequence_tokens)
    batch_size = len(sequence_tokens)

    padded_tokens = torch.full(
        (batch_size, max_length),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
    )

    padded_response_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
    )

    for index, (tokens, mask) in enumerate(
        zip(sequence_tokens, sequence_masks)
    ):
        sequence_length = tokens.numel()
        padded_tokens[index, :sequence_length] = tokens
        padded_response_mask[index, :sequence_length] = mask

    return {
        "input_ids": padded_tokens[:, :-1],
        "labels": padded_tokens[:, 1:],
        "response_mask": padded_response_mask[:, 1:],
    }


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    # return_token_entropy 为 True 的位置显示 token 的概率
    logits = model(input_ids).logits
    all_log_probs = F.log_softmax(logits, dim=-1)

    # labels: (batch_size, sequence_length)
    # labels.unsqueeze(-1): (batch_size, sequence_length, 1)
    # 从词表维度中取出 labels 指定的 token 的 log probability
    token_log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    # 与 crsoss entropy 的区别就是这个得到的是 log p(label) 而前者得到的是 -log p(label)

    result = {
        "log_probs": token_log_probs,
    }

    if return_token_entropy:
        probabilities = all_log_probs.exp()
        token_entropy = -(
            probabilities * all_log_probs
        ).sum(dim=-1)
        result["token_entropy"] = token_entropy

    return result
    # entropy 不是正确 token 的概率。
    # entropy 越大，表示模型在多个候选 token 之间越不确定；
    # entropy 越小，表示模型的预测越集中。


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    rollout_batch_size = len(rollout_responses)
    raw_rewards = torch.zeros(rollout_batch_size, )
    index = 0
    total_rewards = 0
    total_format_rewards = 0
    for (response, ground_truth) in zip(rollout_responses, repeated_ground_truths):
        reward = reward_fn(response, ground_truth)
        raw_rewards[index] = reward["reward"]
        total_rewards += reward["reward"]
        total_format_rewards += reward["format_reward"]
        index += 1

    return (raw_rewards, 
    {
        "mean total rewards": total_rewards / rollout_batch_size, 
        "mean total format rewards": total_format_rewards / rollout_batch_size
    })







