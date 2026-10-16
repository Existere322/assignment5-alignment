import torch
import json
from vllm_utils import VLLMServer, VLLMCompletion
from transformers import AutoModelForCausalLM, AutoTokenizer
from grpo import tokenize_prompt_and_output
from pathlib import Path
from checkpoint import get_model_and_tokenizer
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from grpo import grpo_train_step
import random
import numpy as np
import gc
import pandas as pd
import matplotlib.pyplot as plt

# DIR settings
current_folder = Path(__file__).resolve().parent
root_folder = current_folder.parent
gsm8k_test = Path.joinpath(root_folder, "data/gsm8k/test.jsonl")
gsm8k_train = Path.joinpath(root_folder, "data/gsm8k/train.jsonl")
OUTPUT_DIR = current_folder / "exp_results" / "standard_grpo_on_policy"
CHECKPOINTS = current_folder / "model_checkpoints"
zero_path = Path.joinpath(current_folder, "prompts/r1_zero.prompt")
question_only_path = Path.joinpath(current_folder, "prompts/question_only.prompt")
three_shot_path = Path.joinpath(current_folder, "prompts/r1_zero_three_shot_gsm8k.prompt")


# Hyperparameter initialization
n_train_examples = 6400 
n_val_examples = 1024 
num_rollout_steps = 200 
each_step_prompts = n_train_examples // num_rollout_steps
LEARNING_RATE_TUNING = {
    "low": 5e-6, 
    "medium": 7e-6, 
    "default": 1e-5, 
    "high": 2e-5, 
}
exp_name = ["lr_low", "lr_high", "default", "medium"]
variants = ["noclip", "grpo", "gspo"]
learning_rate = LEARNING_RATE_TUNING["default"]
prompt_types = ["r1_zero"]

# on policy params
# gradient_accumulation_steps = 64 
# rollout_batch_size = train_batch_size = 256 

# off policy params
Off_policy = True
gradient_accumulation_steps = 1
rollout_batch_size = 256  # 一次推理生成的结果数量
train_batch_size = 8  # 训练一个 batch 使用的样例数量
standard_cliprange = 0.2
gspo_cliprange = 3e-4
group_size = 8 
sampling_temperature = 1.0 
sampling_max_tokens = 512 
max_grad_norm = 1.0 
baseline = "mean"
advantage_eps = 1e-6
importance_reweighting_method = "none"
loss_normalization = "sequence"
normalization_constant = sampling_max_tokens * train_batch_size

# Model and tokenizer settings
MODEL_ID = "allenai/OLMo-2-0425-1B"
TRAIN_DEVICE = torch.device("cuda:0")
INTERENCE_DEVICE = torch.device("cuda:1")

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_gsm8k_test():
    questions = []
    answers = []
    count = 0

    with open(gsm8k_test, "r", encoding="utf-8") as test:
        for one_line in test:
            dict_line = json.loads(one_line)
            questions.append(dict_line["question"])
            _, ground_truth = dict_line["answer"].rsplit("####", 1)
            answers.append(ground_truth.strip())
            count += 1
            if count == 1024: break

    return (questions, answers)


def construct_prompt(data, prompt_type):
    if prompt_type == "r1_zero":
        zero_prompt = zero_path.read_text(encoding="utf-8")
        r1_style_prompts = []

        for question in data:
            r1_style_prompts.append(zero_prompt.format(question=question))

        return r1_style_prompts

    if prompt_type == "question_only":
        question_only_prompt = question_only_path.read_text(encoding="utf-8")
        question_only_prompts = []

        for question in data:
            question_only_prompts.append(question_only_prompt.format(question=question))

        return question_only_prompts

    if prompt_type == "r1_three_shot":
        three_shot_prompt = three_shot_path.read_text(encoding="utf-8")
        three_shot_prompts = []

        for question in data:
            three_shot_prompts.append(three_shot_prompt.format(question=question))

        return three_shot_prompts


def read_gsm8k_train():
    questions = []
    answers = []
    count = 0
    
    with open(gsm8k_train, "r", encoding="utf-8") as train:
        for one_line in train:
            dict_line = json.loads(one_line)
            questions.append(dict_line["question"])
            _, ground_truth = dict_line["answer"].rsplit("####", 1)
            answers.append(ground_truth.strip())
            count += 1
            if count == 6400: break

    return (questions, answers)


def evaluate_process(
        seed: int,
        server: VLLMServer,
        tokenizer, 
        prompt_type
    ):
    eval_sampling_params = {
        "temperature": sampling_temperature, 
        "top_p": 1.0, 
        "max_tokens": sampling_max_tokens, 
        "n": 1, 
        "seed": seed, 
        "stop": "</answer>", 
        "include_stop_str_in_output": True, 
    }

    prompts, ground_truths = read_gsm8k_test()
    formated_prompts = construct_prompt(prompts, prompt_type)
    completions = server.generate_completions(
        prompts=formated_prompts, 
        sampling_params=eval_sampling_params, 
        batch_size=32
    )

    grade_results = []

    for ground_truth, completion in zip(ground_truths, completions):
        if prompt_type != "question_only":
            grade_result = r1_zero_reward_fn(
                response=completion.text, 
                ground_truth=ground_truth
            )
        else:
            grade_result = question_only_reward_fn(
                response=completion.text, 
                ground_truth=ground_truth
            )

        grade_results.append(grade_result)

    response_lengths = [
        len(completion.token_ids)
        if completion.token_ids
        else len(
            tokenizer.encode(
                completion.text,
                add_special_tokens=False,
            )
        )
        for completion in completions
    ]

    accuracy = sum(result["answer_reward"] for result in grade_results) / n_val_examples
    format_rate = sum(result["format_reward"] for result in grade_results) / n_val_examples
    
    return accuracy, format_rate, np.mean(response_lengths)


def completions_to_old_log_probs(
    completions: list[VLLMCompletion],
    response_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    completions[i].old_log_probs:
        第 i 个 response 的紧凑 log-probs，长度为 response token 数。

    response_mask:
        完整 shifted sequence mask，形状 [B, S]。

    Returns:
        full-aligned old_log_probs，形状 [B, S]。
    """
    response_mask = response_mask.to(
        device=device,
        dtype=torch.bool,
    )

    batch_size, _ = response_mask.shape

    if len(completions) != batch_size:
        raise ValueError(
            f"Number of completions ({len(completions)}) "
            f"does not match batch size ({batch_size})."
        )

    old_log_probs = torch.zeros(
        response_mask.shape,
        dtype=torch.float32,
        device=device,
    )

    for i, completion in enumerate(completions):
        sequence_log_probs = torch.as_tensor(
            completion.old_log_probs,
            dtype=torch.float32,
            device=device,
        )

        expected_length = int(response_mask[i].sum().item())
        actual_length = sequence_log_probs.numel()

        if actual_length != expected_length:
            raise ValueError(
                f"Completion {i} token/log-prob length mismatch: "
                f"vLLM returned {actual_length}, "
                f"but response_mask contains {expected_length} "
                "valid response positions."
            )

        old_log_probs[i, response_mask[i]] = sequence_log_probs

    return old_log_probs


def grpo_traning_process(seed: int, prompt_type: str, variant: str, offpolicy_variant: str):

    set_random_seed(seed)
    
    run_dir = OUTPUT_DIR / f"seed_{seed}_{variant}_{prompt_type}_{exp_name[2]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = open(run_dir / "metrics.jsonl", "w", encoding="utf-8")
    response_file = open(run_dir / "response.jsonl", "w", encoding="utf-8")

    policy, tokenizer = get_model_and_tokenizer(
            model_id_or_dir=MODEL_ID, device="cuda:0"
        )
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    policy.train()
    
    # device = next(policy.parameters()).device
    server = VLLMServer(model_id=MODEL_ID, seed=seed, gpu=1, gpu_memory_utilization=0.75)
    optimizer = torch.optim.AdamW(  
        policy.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95), 
        weight_decay=0.0 
    )

    # GRPO variants:
    baseline = "mean"
    if variant == "GRPO_constant":
        advantage_normalizer = "std"
        loss_normalization = "constant"
    if variant == "Dr_GRPO":
        advantage_normalizer = "none"
        loss_normalization = "constant"
    if variant == "RFT":
        baseline = "none"
        advantage_normalizer = "none"
        loss_normalization = "constant"
    if variant == "MaxRL":
        advantage_normalizer = "mean"
        loss_normalization = "constant"

    server.start()
    server.init_weight_sync(policy_device="cuda:0")
    server.sync_policy_weights(policy=policy)
    prompts, answers = read_gsm8k_train()
    formated_prompts = construct_prompt(prompts, prompt_type)

    step_logs = []

    step = 0

    for i in range(0, n_train_examples, each_step_prompts):
        batch_single_prompts = formated_prompts[i:i+each_step_prompts]
        batch_single_answers = answers[i:i+each_step_prompts]

        batch_prompts = [
            prompt
            for prompt in batch_single_prompts
            for _ in range(group_size)
        ]

        batch_ground_truths = [
            answer
            for answer in batch_single_answers
            for _ in range(group_size)
        ]

        sampling_params = {
            "temperature": sampling_temperature, 
            "top_p": 1.0, 
            "max_tokens": sampling_max_tokens, 
            "n": group_size, 
            "seed": seed + step * group_size, 
            "stop": "</answer>", 
            "include_stop_str_in_output": True, 
        }

        vllm_completion = server.generate_completions(
            prompts=batch_single_prompts, 
            sampling_params=sampling_params, 
            batch_size=each_step_prompts
        )

        rollout_responses = [
            completion.text
            for completion in vllm_completion
        ]

        num_rollout_responses = len(vllm_completion)

        for train_start in range(
            0,
            num_rollout_responses,
            train_batch_size,
        ):
            train_end = train_start + train_batch_size

            train_completions = vllm_completion[
                train_start:train_end
            ]

            train_prompts = batch_prompts[
                train_start:train_end
            ]

            train_responses = rollout_responses[
                train_start:train_end
            ]

            train_ground_truths = batch_ground_truths[
                train_start:train_end
            ]

            # 先用与 grpo_train_step 相同的函数得到 response mask。
            train_tokenized = tokenize_prompt_and_output(
                prompt_strs=train_prompts,
                output_strs=train_responses,
                tokenizer=tokenizer,
            )

            train_response_mask = train_tokenized[
                "response_mask"
            ]

            # 将 list[list[float]] 转换成 full-aligned tensor [8, S]。
            train_old_log_probs = completions_to_old_log_probs(
                completions=train_completions,
                response_mask=train_response_mask,
                device=TRAIN_DEVICE,
            )

            optimizer.zero_grad()

            if prompt_type == "question_only":
                reward_fn = question_only_reward_fn
            else:
                reward_fn = r1_zero_reward_fn

            if offpolicy_variant == "gspo":
                clip_range = gspo_cliprange
            else:
                clip_range = standard_cliprange

            step_result = grpo_train_step(
                model=policy, 
                tokenizer=tokenizer, 
                optimizer=optimizer, 
                gradient_accumulation_steps=gradient_accumulation_steps, 
                max_grad_norm=max_grad_norm, 
                reward_fn=reward_fn, 
                repeated_prompts=train_prompts, 
                rollout_responses=train_responses, 
                repeated_ground_truths=train_ground_truths, 
                group_size=group_size, 
                baseline=baseline, 
                advantage_eps=advantage_eps, 
                advantage_normalizer = advantage_normalizer, 
                importance_reweighting_method=offpolicy_variant, 
                old_log_probs=train_old_log_probs, 
                clip_range = clip_range, 
                loss_normalization=loss_normalization, 
                normalization_constant = normalization_constant
            )

            log_entry = {
                "seed": seed, 
                "step": step, 
                "loss": step_result[1]["loss"], 
                "gradient_norm": step_result[1]["gradient_norm"], 
                "train_reward_total": step_result[1]["train_rewards"][0], 
                "train_reward_format": step_result[1]["train_rewards"][1], 
                "token_entropy": step_result[1]["token_entropy"], 
                "valid_seq_num": step_result[1]["num_valid"]
            }

            step += 1

            if step % 10 == 0:
                accuracy, format_rate, mean_response_length = evaluate_process(seed, server, tokenizer, prompt_type)
                policy.save_pretrained(CHECKPOINTS / f"model_seed_{seed}", safe_serialization=True)
                log_entry.update(
                    {
                        "val_reward_total": accuracy, 
                        "val_reward_format": format_rate, 
                        "val_response_length": mean_response_length
                    }
                )

            output_file.write(
                json.dumps(log_entry, ensure_ascii=False) + "\n"
            )
            output_file.flush()
            step_logs.append(log_entry)

            if step % 40 == 0:
                response_file.writelines(
                    json.dumps(response, ensure_ascii=False) + "\n"
                    for response in rollout_responses
                )
                response_file.flush()

        server.sync_policy_weights(policy=policy)
                

    server.stop()
    output_file.close()
    response_file.close()

    return step_logs



METRICS = {
    "loss": "Policy-gradient loss",
    "gradient_norm": "Gradient norm",
    "token_entropy": "Token entropy",
    "train_reward_total": "Train reward (total)",
    "train_reward_format": "Train reward (format)",
    "val_reward_total": "Validation reward (total)",
    "val_reward_format": "Validation reward (format)",
    "val_response_length": "Validation response length",
}


def plot_training_metrics(
    all_step_logs: list[dict],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_step_logs)

    # 保存四次运行的原始指标
    df.to_csv(
        output_dir / "all_seed_metrics.csv",
        index=False,
    )

    figure, axes = plt.subplots(
        4,
        2,
        figsize=(14, 18),
    )
    axes = axes.flatten()

    all_summaries = []

    for axis, (metric, title) in zip(
        axes,
        METRICS.items(),
    ):
        metric_df = df.dropna(subset=[metric])

        # 画每个 seed 的实际曲线
        for seed, seed_df in metric_df.groupby("seed"):
            seed_df = seed_df.sort_values("step")

            axis.plot(
                seed_df["step"],
                seed_df[metric],
                linewidth=1,
                alpha=0.3,
                linestyle="--",
                label=f"seed {seed}",
            )

        summary = (
            metric_df
            .groupby("step")[metric]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

        summary["metric"] = metric

        standard_error = (
            summary["std"].fillna(0)
            / np.sqrt(summary["count"])
        )

        summary["lower"] = (
            summary["mean"] - 1.96 * standard_error
        )
        summary["upper"] = (
            summary["mean"] + 1.96 * standard_error
        )

        all_summaries.append(summary)

        steps = summary["step"].to_numpy()
        means = summary["mean"].to_numpy(dtype=float)
        lower = summary["lower"].to_numpy(dtype=float)
        upper = summary["upper"].to_numpy(dtype=float)

        axis.plot(
            steps,
            means,
            color="black",
            linewidth=2,
            label="Mean",
        )

        axis.fill_between(
            steps,
            lower,
            upper,
            color="black",
            alpha=0.15,
            label="95% CI",
        )

        axis.set_title(title)
        axis.set_xlabel("Rollout step")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(
        output_dir / "training_metrics.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    summary_df = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    summary_df.to_csv(
        output_dir / "metric_summary.csv",
        index=False,
    )


def main():
    SEEDS = [224, 229, 333, 336]
    TRAINED = False
    all_step_logs = []
    variant_types = ["GRPO_constant", "Dr_GRPO", "RFT", "MaxRL"]  # on policy variants
    off_policy_variants = ["noclip", "grpo", "gspo"]

    variant = variant_types[1]
    for offpolicy_variant in off_policy_variants:
        prompt_type = "r1_zero"
        for seed in SEEDS:
            if not TRAINED:
                logs = grpo_traning_process(seed, prompt_type, variant, offpolicy_variant)
                all_step_logs.extend(logs)

                # 释放上一轮训练模型占用的 GPU 0 显存
                gc.collect()
                torch.cuda.empty_cache()
            else:
                run_dir = OUTPUT_DIR / f"seed_{seed}_{variant}_{exp_name[2]}_{offpolicy_variant}"
                with open(run_dir / "metrics.jsonl", "r", encoding="utf-8") as result:
                    for line in result:
                        if not line.strip():
                            continue

                        log_entry = json.loads(line)
                        log_entry.setdefault("seed", seed)
                        all_step_logs.append(log_entry)

        plot_training_metrics(
            all_step_logs=all_step_logs,
            output_dir=OUTPUT_DIR / f"_{variant}_{exp_name[2]}_{offpolicy_variant}",
        )

        all_step_logs = []


if __name__ == "__main__": 
    main()