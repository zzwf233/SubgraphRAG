import os
import csv
import json
import wandb
import random
import argparse
from tqdm import tqdm
from pathlib import Path

from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import get_prompts_for_data
from llm_utils import llm_init, llm_inf_all

from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original


def get_defined_prompts(prompt_mode, model_name, llm_mode):
    if 'gpt' in model_name or 'gpt' in prompt_mode:
        if 'gptLabel' in prompt_mode:
            from prompts import sys_prompt_gpt, cot_prompt_gpt
            return sys_prompt_gpt, cot_prompt_gpt
        else:
            from prompts import icl_sys_prompt, icl_cot_prompt
            return icl_sys_prompt, icl_cot_prompt
    elif 'noevi' in prompt_mode:
        from prompts import noevi_sys_prompt, noevi_cot_prompt
        return noevi_sys_prompt, noevi_cot_prompt
    elif 'icl' in llm_mode:
        from prompts import icl_sys_prompt, icl_cot_prompt
        return icl_sys_prompt, icl_cot_prompt
    else:
        from prompts import sys_prompt, cot_prompt
        return sys_prompt, cot_prompt


def save_checkpoint(file_handle, data):
    file_handle.write(json.dumps(data) + "\n")


def load_checkpoint(file_path):
    if os.path.exists(file_path):
        print("*" * 50)
        print(f"Resuming from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            ckpt = [json.loads(line) for line in f]
        try:
            print(f"Last processed item: {ckpt[-1]['id']}")
        except IndexError:
            pass
        print("*" * 50)
        return ckpt
    return []


def eval_all(pred_file_path, run, subset, split=None, eval_hops=-1):

    print("=" * 50)
    print("=" * 50)
    print(f"Evaluating on subset: {subset}")

    print("Results:")
    hit1, f1, prec, recall, em, tw, mi_f1, mi_prec, mi_recall, total_cnt, no_ans_cnt, no_ans_ratio, hal_score, stats = eval_results_corrected(str(pred_file_path), cal_f1=True, subset=subset, split=split, eval_hops=eval_hops)
    if subset:
        postfix = "_sub"
    else:
        postfix = ""
    run.log({f"results{postfix}/hit@1": hit1,
             f"results{postfix}/macro_f1": f1,
             f"results{postfix}/macro_precision": prec,
             f"results{postfix}/macro_recall": recall,
             f"results{postfix}/exact_match": em,
             f"results{postfix}/totally_wrong": tw,
             f"results{postfix}/micro_f1": mi_f1,
             f"results{postfix}/micro_precision": mi_prec,
             f"results{postfix}/micro_recall": mi_recall,
             f"results{postfix}/total_cnt": total_cnt,
             f"results{postfix}/no_ans_cnt": no_ans_cnt,
             f"results{postfix}/no_ans_ratio": no_ans_ratio,
             f"results{postfix}/hal_score": hal_score})  # score_h in the paper
    if stats is not None:
        for k, v in stats.items():
            run.log({f"stats{postfix}/{k}": v})

    hit, _, _, _ = eval_results_original(str(pred_file_path), cal_f1=True, subset=subset, eval_hops=eval_hops)
    run.log({f"results{postfix}/hit": hit})
    print("=" * 50)
    print("=" * 50)
    return {
        "subset": subset,
        "hit_at_1": hit1,
        "macro_f1": f1,
        "macro_precision": prec,
        "macro_recall": recall,
        "exact_match": em,
        "totally_wrong": tw,
        "micro_f1": mi_f1,
        "micro_precision": mi_prec,
        "micro_recall": mi_recall,
        "hit": hit,
        "hal_score": hal_score,
        "total_cnt": total_cnt,
        "no_ans_cnt": no_ans_cnt,
        "no_ans_ratio": no_ans_ratio,
    }


def append_summary(summary_file, row):
    if summary_file is None:
        return

    summary_path = Path(summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "split", "model", "prompt_mode", "llm_mode", "frequency_penalty",
        "thres", "attack", "attack_budget", "rand_budget", "ours_budget", "seed", "subset",
        "hit", "macro_f1", "macro_precision", "macro_recall", "hit_at_1",
        "exact_match", "micro_f1", "hal_score", "total_cnt", "no_ans_cnt",
        "no_ans_ratio", "prediction_file",
    ]
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="RAG for KGQA")
    parser.add_argument("-d", "--dataset_name", type=str, default="cwq", help="Dataset name")
    parser.add_argument("--prompt_mode", type=str, default="scored_100", help="Prompt mode")
    parser.add_argument("-p", "--score_dict_path", type=str)
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name")
    # parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model context length")
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2, help="Max sequence length to capture")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0.16, help="Frequency penalty")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold")
    parser.add_argument("--safety_attack", choices=["clean", "rand", "ours"], default="clean",
                        help="RAG Safety Table 3 scenario")
    parser.add_argument("--safety_rand_budget", type=int, default=20,
                        help="Number of Rand corrupted triples inserted per question")
    parser.add_argument("--safety_ours_budget", type=int, default=20,
                        help="Maximum number of injected triples kept for Ours per question")
    parser.add_argument("--summary_file", type=str, default=None,
                        help="CSV file for clean/rand summary metrics")
    parser.add_argument("--reuse_predictions", action="store_true",
                        help="Reuse an existing predictions file and only run evaluation/summary")
    parser.add_argument("--limit_samples", type=int, default=0,
                        help="Only run the first N samples for smoke testing; 0 means all samples")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    tensor_parallel_size = args.tensor_parallel_size
    max_model_len = args.max_model_len
    max_seq_len_to_capture = args.max_seq_len_to_capture
    max_tokens = args.max_tokens
    seed = args.seed
    temperature = args.temperature
    frequency_penalty = args.frequency_penalty
    thres = args.thres
    safety_attack = args.safety_attack
    safety_rand_budget = args.safety_rand_budget
    safety_ours_budget = args.safety_ours_budget

    pred_file_path = f"./results/KGQA/{dataset_name}/RoG/{split}/results_gen_rule_path_RoG-{dataset_name}_RoG_{split}_predictions_3_False_jsonl/predictions.jsonl"
    run_name = f"{model_name}-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-{safety_attack}"
    os.environ.setdefault("WANDB_MODE", "offline")
    run = wandb.init(project=f"RAG-{dataset_name}", name=run_name, config=args)

    if args.score_dict_path is None:
        if dataset_name == "webqsp":
            assert split == "test"
            score_dict_path = "./scored_triples/webqsp_240912_unidir_test.pth"
        elif dataset_name == "cwq":
            assert split == "test"
            score_dict_path = "./scored_triples/cwq_240907_unidir_test.pth"
    else:
        score_dict_path = args.score_dict_path

    raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/{args.model_name.split('/')[-1]}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    if safety_attack == "clean":
        attack_suffix = ""
    elif safety_attack == "rand":
        attack_suffix = f"-safety_{safety_attack}{safety_rand_budget}_seed{seed}"
    else:
        attack_suffix = f"-safety_{safety_attack}{safety_ours_budget}_seed{seed}"
    raw_pred_file_path = raw_pred_folder_path / f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}{attack_suffix}-predictions-resume.jsonl"
    final_pred_file_path = raw_pred_file_path.with_name(raw_pred_file_path.stem.replace("-resume", "") + raw_pred_file_path.suffix)

    if args.limit_samples > 0:
        raw_pred_file_path = raw_pred_file_path.with_name(raw_pred_file_path.stem + f"-limit{args.limit_samples}" + raw_pred_file_path.suffix)
        final_pred_file_path = final_pred_file_path.with_name(final_pred_file_path.stem + f"-limit{args.limit_samples}" + final_pred_file_path.suffix)

    if args.reuse_predictions and final_pred_file_path.exists():
        if args.limit_samples <= 0:
            eval_all(final_pred_file_path, run, subset=True)
            full_metrics = eval_all(final_pred_file_path, run, subset=False)
        else:
            full_metrics = {}
        append_summary(args.summary_file, {
            **full_metrics,
            "dataset": dataset_name,
            "split": split,
            "model": args.model_name.split("/")[-1],
            "prompt_mode": prompt_mode,
            "llm_mode": llm_mode,
            "frequency_penalty": frequency_penalty,
            "thres": thres,
            "attack": safety_attack,
            "attack_budget": safety_rand_budget if safety_attack == "rand" else (safety_ours_budget if safety_attack == "ours" else 0),
            "rand_budget": safety_rand_budget if safety_attack == "rand" else 0,
            "ours_budget": safety_ours_budget if safety_attack == "ours" else 0,
            "seed": seed,
            "prediction_file": str(final_pred_file_path),
        })
        return

    llm = llm_init(model_name, tensor_parallel_size, max_seq_len_to_capture, max_tokens, seed, temperature, frequency_penalty, max_model_len)
    data = get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode, limit_samples=args.limit_samples)
    sys_prompt, cot_prompt = get_defined_prompts(prompt_mode, model_name, llm_mode)
    print("Generating prompts...")
    data = get_prompts_for_data(data, prompt_mode, sys_prompt, cot_prompt, thres)

    print("Starting inference...")
    start_idx = len(load_checkpoint(raw_pred_file_path))
    with open(raw_pred_file_path, "a", encoding="utf-8") as pred_file:
        for idx, each_qa in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
            res = llm_inf_all(llm, each_qa, llm_mode, model_name)

            del each_qa["graph"], each_qa["good_paths_rog"], each_qa["good_triplets_rog"], each_qa["scored_triplets"]

            each_qa["prediction"] = res[0]
            if each_qa.get("chain_trace"):
                each_qa["chain_trace"] = each_qa["chain_trace"]
            if each_qa.get("chain_final_user_query"):
                each_qa["chain_final_user_query"] = each_qa["chain_final_user_query"]
            save_checkpoint(pred_file, each_qa)

    # If the processing completes, rename the files to remove the "resume" flag
    os.rename(raw_pred_file_path, final_pred_file_path)
    if args.limit_samples <= 0:
        sub_metrics = eval_all(final_pred_file_path, run, subset=True)
        full_metrics = eval_all(final_pred_file_path, run, subset=False)
        append_summary(args.summary_file, {
            **full_metrics,
            "dataset": dataset_name,
            "split": split,
            "model": args.model_name.split("/")[-1],
            "prompt_mode": prompt_mode,
            "llm_mode": llm_mode,
            "frequency_penalty": frequency_penalty,
            "thres": thres,
            "attack": safety_attack,
            "attack_budget": safety_rand_budget if safety_attack == "rand" else (safety_ours_budget if safety_attack == "ours" else 0),
            "rand_budget": safety_rand_budget if safety_attack == "rand" else 0,
            "ours_budget": safety_ours_budget if safety_attack == "ours" else 0,
            "seed": seed,
            "prediction_file": str(final_pred_file_path),
        })


if __name__ == "__main__":
    main()
