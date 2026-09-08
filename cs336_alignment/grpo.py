import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

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
    # return_token_entropy 为 True 的位置使用
    





