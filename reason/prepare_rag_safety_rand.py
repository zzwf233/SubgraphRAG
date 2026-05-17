import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


def unique_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def random_entity(entity_pool, forbidden, rng):
    candidates = [entity for entity in entity_pool if entity not in forbidden]
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def corrupt_triple(source, q_entities, entity_pool, rng):
    h, r, t = source[:3]
    q_entity_set = set(q_entities)
    if h in q_entity_set and t not in q_entity_set:
        replacement = random_entity(entity_pool, {h, t}, rng)
        return (h, r, replacement) if replacement is not None else None
    if t in q_entity_set and h not in q_entity_set:
        replacement = random_entity(entity_pool, {h, t}, rng)
        return (replacement, r, t) if replacement is not None else None
    if h in q_entity_set and t in q_entity_set:
        replacement = random_entity(entity_pool, {h, t}, rng)
        return (h, r, replacement) if replacement is not None else None
    return None


def build_rand_triples(sample, scored_triples, budget, rng):
    graph = [tuple(triplet[:3]) for triplet in sample["graph"]]
    q_entities = sample.get("q_entity", [])
    q_entity_set = set(q_entities)
    entity_pool = unique_preserve_order([triplet[0] for triplet in graph] + [triplet[2] for triplet in graph])

    source_triples = [triplet for triplet in graph if triplet[0] in q_entity_set or triplet[2] in q_entity_set]
    if not source_triples:
        source_triples = graph

    score_by_triple = {tuple(triplet[:3]): float(triplet[3]) for triplet in scored_triples}
    fallback_score = min([float(triplet[3]) for triplet in scored_triples], default=0.0)
    existing = set(graph)
    generated = []
    generated_set = set()
    attempts = 0
    max_attempts = max(1000, budget * 200)

    while len(generated) < budget and attempts < max_attempts and source_triples:
        attempts += 1
        source = source_triples[int(rng.integers(len(source_triples)))]
        corrupted = corrupt_triple(source, q_entities, entity_pool, rng)
        if corrupted is None or corrupted in existing or corrupted in generated_set:
            continue

        source_score = score_by_triple.get(tuple(source[:3]), fallback_score)
        generated.append((corrupted[0], corrupted[1], corrupted[2], source_score))
        generated_set.add(corrupted)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Create RAG Safety Rand poisoned scored triples.")
    parser.add_argument("-d", "--dataset_name", choices=["webqsp", "cwq"], required=True)
    parser.add_argument("--input", required=True, help="Clean scored_triples .pth")
    parser.add_argument("--output", required=True, help="Output poisoned scored_triples .pth")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    clean_scores = torch.load(args.input, weights_only=False)
    subgraphs = load_dataset("rmanluo/RoG-" + args.dataset_name, split="test")
    subgraph_by_id = {sample["id"]: sample for sample in subgraphs}

    poisoned_scores = {}
    for sample_id, sample_scores in tqdm(clean_scores.items()):
        if sample_id not in subgraph_by_id:
            continue

        poisoned_sample = dict(sample_scores)
        clean_scored_triples = list(sample_scores["scored_triples"])
        rand_triples = build_rand_triples(
            subgraph_by_id[sample_id], clean_scored_triples, args.budget, rng)
        poisoned_scored_triples = clean_scored_triples + rand_triples
        poisoned_scored_triples = sorted(
            poisoned_scored_triples, key=lambda triplet: float(triplet[3]), reverse=True)
        poisoned_sample["scored_triples"] = poisoned_scored_triples
        poisoned_sample["safety_rand_triples"] = rand_triples
        poisoned_scores[sample_id] = poisoned_sample

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(poisoned_scores, args.output)
    print(f"Saved {len(poisoned_scores)} samples to {args.output}")


if __name__ == "__main__":
    main()
