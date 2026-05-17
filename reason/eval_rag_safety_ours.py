import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

from metrics.evaluate_results import eval_results as eval_results_original
from metrics.evaluate_results_corrected import eval_results as eval_results_corrected, get_pred


def normalize_answer(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(text or "")).split())


def load_prediction_map(path: str) -> Dict[str, dict]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def load_attack_targets(path: str) -> Dict[str, Set[str]]:
    targets: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("parent_id", row.get("id", "")))
            row_targets = row.get("attack_targets") or row.get("cascade_poison_targets") or row.get("poison_targets", [])
            if isinstance(row_targets, str):
                row_targets = [row_targets]
            if not row_targets and isinstance(row.get("poison_target"), str):
                row_targets = [row.get("poison_target")]
            for target in row_targets:
                if not isinstance(target, str) or not target.strip():
                    continue
                targets.setdefault(sid, set()).add(normalize_answer(target))
    return targets


def extract_predictions(row: dict) -> List[str]:
    prediction = row.get("prediction", "")
    preds = get_pred(prediction, None)
    if not preds:
        text = str(prediction or "").strip()
        return [text] if text else []
    extracted = []
    for p in preds:
        s = str(p)
        if "ans:" in s.lower():
            s = s.split("ans:")[-1].strip()
        extracted.append(s.strip())
    return [x for x in extracted if x]


def compute_attack_metrics(pred_rows: Dict[str, dict], attack_targets: Dict[str, Set[str]]):
    ap, ah1, amrr = [], [], []
    for sid, target_set in attack_targets.items():
        row = pred_rows.get(sid)
        if row is None:
            continue
        preds = [normalize_answer(x) for x in extract_predictions(row) if x.strip()]
        if not preds:
            ap.append(0.0)
            ah1.append(0.0)
            amrr.append(0.0)
            continue

        match_count = sum(1 for p in preds if p in target_set)
        ap.append(match_count / max(1, len(preds)))
        ah1.append(1.0 if preds[0] in target_set else 0.0)

        rr = 0.0
        for rank, pred in enumerate(preds, start=1):
            if pred in target_set:
                rr = 1.0 / rank
                break
        amrr.append(rr)

    n = max(1, len(ap))
    return {
        "A-Precision": sum(ap) * 100 / n,
        "A-H@1": sum(ah1) * 100 / n,
        "A-MRR": sum(amrr) * 100 / n,
        "covered_samples": len(ap),
    }


def append_summary(summary_file: str, row: dict):
    if not summary_file:
        return
    path = Path(summary_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "model", "prompt_mode", "llm_mode", "frequency_penalty", "seed", "attack", "attack_budget",
        "Hit", "F1", "Precision", "Recall", "Hits@1", "EM", "A-Precision", "A-H@1", "A-MRR",
        "predict_file", "poison_file",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Evaluate SubgraphRAG ours with standard and attack metrics.")
    parser.add_argument("--predict_file", required=True)
    parser.add_argument("--poison_file", required=True)
    parser.add_argument("--dataset", required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt_mode", default="")
    parser.add_argument("--llm_mode", default="")
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack_budget", type=int, default=20)
    parser.add_argument("--summary_file", default="")
    parser.add_argument("--output_prefix", default="")
    args = parser.parse_args()

    hit1, f1, prec, recall, em, _tw, _mi_f1, _mi_prec, _mi_recall, _total_cnt, _no_ans_cnt, _no_ans_ratio, _hal_score, _stats = eval_results_corrected(
        args.predict_file, cal_f1=True, subset=False
    )
    hit, _, _, _ = eval_results_original(args.predict_file, cal_f1=True, subset=False)
    pred_rows = load_prediction_map(args.predict_file)
    attack_targets = load_attack_targets(args.poison_file)
    attack = compute_attack_metrics(pred_rows, attack_targets)

    std_str = (
        "Hit\tF1\tPrecision\tRecall\tHits@1\tEM\n"
        f"{hit:.4f}\t{f1:.4f}\t{prec:.4f}\t{recall:.4f}\t{hit1:.4f}\t{em:.4f}"
    )
    atk_str = (
        "A-Precision\tA-H@1\tA-MRR\n"
        f"{attack['A-Precision']:.4f}\t{attack['A-H@1']:.4f}\t{attack['A-MRR']:.4f}"
    )
    print(std_str)
    print()
    print(atk_str)

    prefix = args.output_prefix or str(Path(args.predict_file).with_suffix(""))
    with open(f"{prefix}_paper_metrics.txt", "w", encoding="utf-8") as f:
        f.write(std_str + "\n\n" + atk_str + "\n")
    with open(f"{prefix}_paper_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "standard": {
                "Hit": hit,
                "F1": f1,
                "Precision": prec,
                "Recall": recall,
                "Hits@1": hit1,
                "EM": em,
            },
            "attack": attack,
        }, f, ensure_ascii=False, indent=2)

    append_summary(args.summary_file, {
        "dataset": args.dataset,
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "llm_mode": args.llm_mode,
        "frequency_penalty": args.frequency_penalty,
        "seed": args.seed,
        "attack": "ours",
        "attack_budget": args.attack_budget,
        "Hit": hit,
        "F1": f1,
        "Precision": prec,
        "Recall": recall,
        "Hits@1": hit1,
        "EM": em,
        "A-Precision": attack["A-Precision"],
        "A-H@1": attack["A-H@1"],
        "A-MRR": attack["A-MRR"],
        "predict_file": args.predict_file,
        "poison_file": args.poison_file,
    })

    print(f"[Saved] {prefix}_paper_metrics.txt")
    print(f"[Saved] {prefix}_paper_metrics.json")
    if args.summary_file:
        print(f"[Saved] {args.summary_file}")


if __name__ == "__main__":
    main()
