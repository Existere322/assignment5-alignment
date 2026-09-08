import os
from pathlib import Path
from vllm_utils import VLLMServer
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
import json


"""
download models prefix settings:
export NO_PROXY=hf-mirror.com,.hf-mirror.com
export no_proxy=hf-mirror.com,.hf-mirror.com
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

hf download allenai/OLMo-2-0425-1B

=====zero_shot=====
Number of examples: 1319
Accuracy: 0.0008
Format rate: 0.5413
Saved generations to /home/zs/assignment5-alignment/cs336_alignment/zero_shot_evaluation_result.jsonl

=====three_shot=====
Number of examples: 1319
Accuracy: 0.2138
Format rate: 0.9636
Saved generations to /home/zs/assignment5-alignment/cs336_alignment/three_shot_evaluation_result.jsonl

=====question_only=====
Number of examples: 1319
Accuracy: 0.0076
Format rate: 0.3040
Saved generations to /home/zs/assignment5-alignment/cs336_alignment/question_only_evaluation_result.jsonl
"""


current_folder = Path(__file__).resolve().parent
root_folder = current_folder.parent
gsm8k_test = Path.joinpath(root_folder, "data/gsm8k/test.jsonl")
question_only_path = Path.joinpath(current_folder, "prompts/question_only.prompt")
three_shot_path = Path.joinpath(current_folder, "prompts/r1_zero_three_shot_gsm8k.prompt")
zero_path = Path.joinpath(current_folder, "prompts/r1_zero.prompt")
MODEL_ID = "allenai/OLMo-2-0425-1B"
prompt_type = "question_only"
OUTPUT_DIR = current_folder
batch_size = 32


def read_gsm8k():
    questions = []
    answers = []
    
    with open(gsm8k_test, "r", encoding="utf-8") as test:
        for one_line in test:
            dict_line = json.loads(one_line)
            questions.append(dict_line["question"])
            _, ground_truth = dict_line["answer"].rsplit("####", 1)
            answers.append(ground_truth.strip())

    return (questions, answers)


def construct_prompt(test_data):

    question_only_prompt = question_only_path.read_text(encoding="utf-8")
    three_shot_prompt = three_shot_path.read_text(encoding="utf-8")
    zero_prompt = zero_path.read_text(encoding="utf-8")

    question_only_prompts = []
    three_shot_prompts = []
    zero_prompts = []

    for one_question in test_data:
        question_only_prompts.append(question_only_prompt.format(question=one_question))
        three_shot_prompts.append(three_shot_prompt.format(question=one_question))
        zero_prompts.append(zero_prompt.format(question=one_question))

    result = {"question_only": question_only_prompts, "three_shot": three_shot_prompts, "zero_shot": zero_prompts}

    return result


def evaluate_result(
    server: VLLMServer, 
    prompts: dict[str, list], 
    prompt_type: str, 
    batch_size: int,
    answers: list[str],  
):
    sampling_params = {
        "temperature": 1.0, 
        "top_p": 1.0, 
        "max_tokens": 512, 
        "n": 1, 
        "seed": 336, 
    }

    if prompt_type in {"three_shot", "zero_shot"}:
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True

    completions = server.generate_completions(
        prompts=prompts[prompt_type],
        sampling_params=sampling_params,
        batch_size = batch_size
        )

    grade_results = []
    results = []

    for prompt, ground_truth, completion in zip(prompts[prompt_type], answers, completions):

        if prompt_type == "question_only":
            grade_result = question_only_reward_fn(
                response=completion.text, 
                ground_truth=ground_truth, 
            )
        else:
            grade_result = r1_zero_reward_fn(
                response=completion.text, 
                ground_truth=ground_truth, 
            )

        grade_results.append(grade_result)

        results.append(
            {
                "prompt": prompt, 
                "ground_truth": ground_truth, 
                "response": completion.text, 
                "finish_reason": completion.finish_reason,
                "num_generated_tokens": len(completion.token_ids),
                **grade_result, 
            }
        )

    accuracy = sum(result["answer_reward"] for result in grade_results) / len(grade_results)
    format_rate = sum(result["format_reward"] for result in grade_results) / len(grade_results)

    print(f"\n====={prompt_type}=====")
    print(f"Number of examples: {len(grade_results)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Format rate: {format_rate:.4f}")

    output_path = OUTPUT_DIR / f"{prompt_type}_evaluation_result.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved generations to {output_path}")


def analyze(prompt_types):
    for prompt_type in prompt_types:
        both_true_num = 0
        format_only_num = 0
        answer_only_num = 0
        both_false_num = 0
        output_path = OUTPUT_DIR / f"{prompt_type}_evaluation_result.jsonl"
        with open(output_path, encoding="utf-8") as eval_result:
            for result in eval_result:
                result = json.loads(result)
                if result["format_reward"] == 1.0 and result["answer_reward"] == 1.0:
                    both_true_num += 1
                elif result["format_reward"] == 1.0 and result["answer_reward"] == 0.0:
                    format_only_num += 1
                elif result["answer_reward"] == 1.0 and result["format_reward"] == 0.0:
                    answer_only_num +=  1
                else:
                    both_false_num += 1

        print(f"========{prompt_type}==========")
        print(f"both true num: {both_true_num}")
        print(f"format only num: {format_only_num}")
        print(f"answer only num: {answer_only_num}")
        print(f"both false num: {both_false_num}")


    
def main():
    # server = VLLMServer(model_id=MODEL_ID, gpu=0)
    # server.start()
    # examples = read_gsm8k()
    # prompts = construct_prompt(examples[0])
    prompt_types = ["question_only", "three_shot", "zero_shot"]

    # try:
    #     for prompt_type in prompt_types:
    #         evaluate_result(
    #             server, 
    #             prompts, 
    #             prompt_type, 
    #             batch_size, 
    #             examples[1]
    #         )

    # finally:
    #     server.stop()

    analyze(prompt_types)


if __name__ == "__main__": 
    main()
