import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch.nn.functional as F
from collections.abc import Callable
from typing import Literal
from einops import rearrange
from torch.optim import Optimizer


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


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,  
    group_size: int, 
    baseline: Literal["mean", "none"] = "mean", 
    advantage_eps: float = 1e-6, 
    advantage_normalizer: Literal["std", "none", "mean"] = "std", 
):
    # rollout_batch_size = n_prompts_per_rollout_batch * group_size
    if baseline != "mean":
        raise NotImplementedError
    if advantage_normalizer != "std":
        raise NotImplementedError

    grouped_rewards = rearrange(
        raw_rewards,
        "(num_groups group_size) -> num_groups group_size",
        group_size=group_size,
    )

    group_mean = grouped_rewards.mean(-1, keepdim=True)
    group_std = grouped_rewards.std(-1, correction=1, keepdim=True)
    max_rewards = torch.max(raw_rewards, dim=-1)
    min_rewards = torch.min(raw_rewards, dim=-1)

    normalized_rewards = (grouped_rewards - group_mean) / (group_std + advantage_eps)

    result = rearrange(normalized_rewards, "n g -> (n g)")

    return (result, {"metadata":{"group_mean": group_mean, "group_std": group_std, "group_max": max_rewards, "group_min": min_rewards}})

    
def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, 
    policy_log_probs: torch.Tensor, 
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none", 
    old_log_probs: torch.Tensor | None = None, 
    cliprange: float | None = None, 
    response_mask: torch.Tensor | None = None, 
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # raw_rewards_or_advantages: Shape (batch_size,) or (batch_size, 1)
    # policy_log_probs: (batch_size, sequence_length)
    if importance_reweighting_method != "none":
        raise NotImplementedError

    advantages = raw_rewards_or_advantages.unsqueeze(-1) if raw_rewards_or_advantages.ndim == 1 else raw_rewards_or_advantages
    per_token_policy_loss = -(advantages * policy_log_probs)

    return (per_token_policy_loss, {})


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor, 
    mask: torch.Tensor, 
    loss_normalization : Literal["sequence", "constant"] = "sequence", 
    normalization_constant: int | None = None, 
) -> torch.Tensor:
    # per_token_policy_gradient_loss: batch_size, sequence_length
    # mask: batch_size, sequence_length
    sequence_lengths = mask.sum(dim=-1)
    masked_per_token_loss = per_token_policy_gradient_loss.masked_fill(~mask, 0)
    if loss_normalization == "sequence":
        total_loss = torch.mean(torch.sum(masked_per_token_loss, dim=-1) / sequence_lengths, dim=-1)
    if loss_normalization == "constant":
        total_loss = torch.sum(torch.sum(per_token_policy_gradient_loss, dim=-1), dim=-1)
        total_loss = total_loss / normalization_constant

    return total_loss


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int, 
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization 
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
)-> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    if importance_reweighting_method != "none":
        raise NotImplementedError

    if loss_normalization != "sequence":
        raise NotImplementedError

    tokenizer_response = tokenize_prompt_and_output(
        prompt_strs=repeated_prompts, 
        output_strs=rollout_responses, 
        tokenizer=tokenizer
    )

    rollout_rewards = compute_rollout_rewards(
        reward_fn=reward_fn, 
        rollout_responses=rollout_responses, 
        repeated_ground_truths=repeated_ground_truths
    )
    raw_rewards = rollout_rewards[0]
    mean_rewards = rollout_rewards[1]["mean total rewards"]
    mean_formal_rewards = rollout_rewards[1]["mean total format rewards"]

    device = next(model.parameters()).device
    input_ids = tokenizer_response["input_ids"].to(device)
    labels = tokenizer_response["labels"].to(device)
    mask = tokenizer_response["response_mask"].to(
        device=device,
        dtype=torch.bool,
    )

    # Parameters to return or log
    total_loss = []
    entropy_sum = 0
    valid_token_count = 0

    microbatch_size = len(input_ids) // gradient_accumulation_steps

    group_normalized_rewards = compute_group_normalized_rewards(
        raw_rewards=raw_rewards, 
        group_size=group_size, 
        baseline=baseline, 
        advantage_eps=advantage_eps, 
        advantage_normalizer=advantage_normalizer
    )
    group_normalized_rewards = group_normalized_rewards[0].to(
        device=device,
        dtype=torch.float32,
    )

    for i in range(0, len(input_ids), microbatch_size):
        inputs_microbatch = input_ids[i:i+microbatch_size]
        labels_microbatch = labels[i:i+microbatch_size]
        rewards_microbatch = group_normalized_rewards[i:i+microbatch_size]
        masks = mask[i:i+microbatch_size]

        response_log_probs = get_response_log_probs(
            model=model, 
            input_ids = inputs_microbatch, 
            labels = labels_microbatch, 
            return_token_entropy=True
        )
        log_probs = response_log_probs["log_probs"]

        per_token_entropy = response_log_probs["token_entropy"]
        entropy_sum += (
            per_token_entropy.masked_fill(~masks, 0.0).sum().detach()
        )
        valid_token_count += masks.sum().detach()

        policy_gradient_loss = compute_policy_gradient_loss(
            raw_rewards_or_advantages=rewards_microbatch, 
            policy_log_probs=log_probs, 
            importance_reweighting_method=importance_reweighting_method
        )
        per_token_policy_loss = policy_gradient_loss[0]

        aggregated_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_policy_loss, 
            mask=masks, 
            loss_normalization=loss_normalization, 
            normalization_constant=normalization_constant
        ) * (len(inputs_microbatch) / len(input_ids))
        aggregated_loss.backward()
        total_loss.append(aggregated_loss.detach())
        # detach 创建一个与原 Tensor 共享数值、但脱离 autograd 计算图的 Tensor

    # 对累积梯度进行裁剪而不是在每个 microbatch 上进行裁剪
    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_grad_norm
        ).detach()

    optimizer.step()
    optimizer.zero_grad()
    batch_loss = torch.stack(total_loss).sum()
    batch_token_entropy = entropy_sum / valid_token_count.clamp_min(1)

    return (batch_loss, {
        "loss": batch_loss.item(), 
        "gradient_norm": grad_norm.item(), 
        "token_entropy": batch_token_entropy.item(),
        "train_rewards": (mean_rewards, mean_formal_rewards)
    })






