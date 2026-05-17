import os
import re
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from .prepare_prompts import unique_preserve_order


def get_subgraphs(dataset_name, split):
    input_file = os.path.join("rmanluo", f"RoG-{dataset_name}")
    return load_dataset(input_file, split=split)


def extract_reasoning_paths(text):
    pattern = r"Reasoning Paths:(.*?)\n\nQuestion:"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        reasoning_paths = match.group(1).strip()
        return reasoning_paths
    else:
        return None


def add_good_triplets_from_rog(data):
    print("Adding good triplets from ROG...")
    total_good_triplets = 0
    total_good_triplets_in_graph = 0
    total_good_triplets_not_in_graph = 0
    for idx, each_qa in enumerate(tqdm(data)):
        all_paths = extract_reasoning_paths(each_qa["input"]).split("\n")
        data[idx]["good_paths_rog"] = all_paths
        all_good_triplets = []
        for each_path in all_paths:
            each_path = each_path.split(" -> ")
            good_triplets = []
            i = 0
            while i < len(each_path):
                if i + 2 < len(each_path):
                    triplet = (each_path[i], each_path[i + 1], each_path[i + 2])
                    temp_triplet = (each_path[i + 2], each_path[i + 1], each_path[i])
                    total_good_triplets += 1
                    # if triplet in each_qa["graph"] or temp_triplet in each_qa["graph"]:
                    #     total_good_triplets_in_graph += 1
                    # else:
                    #     total_good_triplets_not_in_graph += 1
                    good_triplets.append(triplet)
                i += 2
            all_good_triplets.extend(good_triplets)
        data[idx]["good_triplets_rog"] = unique_preserve_order(all_good_triplets)
    return data


def add_gt_if_not_present(triple_score_dict):
    st = [','.join(list(each)[:3]) for each in triple_score_dict['scored_triples']]
    tt = [','.join(list(each)[:3]) for each in triple_score_dict['target_relevant_triples']]
    for each in tt:
        if each in st:
            continue
        else:
            # put at the beginning
            triple_score_dict["scored_triples"].insert(0, tuple(each.split(',')))
    return triple_score_dict["scored_triples"]


def add_scored_triplets(data, score_dict_path, prompt_mode):
    print("Adding scored triplets...")
    new_data = []
    cnt = 0
    triple_score_dict = torch.load(score_dict_path, weights_only=False)

    running_baselines = False
    if 'triples' in triple_score_dict[next(iter(triple_score_dict))]:
        running_baselines = True
        for k, v in tqdm(triple_score_dict.items()):
            triple_score_dict[k]['scored_triples'] = v['triples']

    for each_qa in tqdm(data):
        if each_qa["id"] in triple_score_dict:
            sample_score_dict = triple_score_dict[each_qa["id"]]
            if 'gt' in prompt_mode:
                scored_triples = add_gt_if_not_present(sample_score_dict)
            else:
                scored_triples = sample_score_dict["scored_triples"]
            each_qa['scored_triplets'] = scored_triples
            for key, value in sample_score_dict.items():
                if key == "scored_triples":
                    continue
                each_qa[key] = value
            new_data.append(each_qa)
        else:
            print(f"Triplets not found for {each_qa['id']}")
            if running_baselines:
                each_qa['scored_triplets'] = [('', '', '')]
                new_data.append(each_qa)
            elif 'gt' not in prompt_mode:
                raise ValueError
            else:
                cnt += 1
    print(f"Triplets not found for {cnt} questions")
    return new_data


def get_processed_data_path(dataset_name, split):
    return os.path.join("..", "retrieve", "data_files", dataset_name, "processed", f"{split}.pkl")


def load_question_entities(dataset_name, split):
    processed_path = get_processed_data_path(dataset_name, split)
    if not os.path.exists(processed_path):
        return {}

    with open(processed_path, "rb") as f:
        processed_data = pickle.load(f)
    return {sample["id"]: sample.get("q_entity", []) for sample in processed_data}


def infer_question_entities(each_qa):
    if each_qa.get("q_entity"):
        return each_qa["q_entity"]
    for path in each_qa.get("good_paths_rog", []):
        parts = path.split(" -> ")
        if parts and parts[0]:
            return [parts[0]]
    return []


def add_question_entities(data, dataset_name, split):
    q_entities_by_id = load_question_entities(dataset_name, split)
    for each_qa in data:
        each_qa["q_entity"] = q_entities_by_id.get(each_qa["id"], infer_question_entities(each_qa))
    return data


def random_entity(entity_pool, forbidden, rng):
    candidates = [entity for entity in entity_pool if entity not in forbidden]
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def make_rand_corrupted_triple(triple, q_entities, entity_pool, rng):
    h, r, t = triple[:3]
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


def add_rag_safety_rand_triplets(data, budget=20, seed=0):
    """RAG Safety Rand: insert structure-preserving corrupted triples.

    For each question, use triples involving the question entity and randomly
    replace the other entity. The paper constrains Rand to 20 inserted triples
    per question.
    """
    print(f"Adding RAG Safety Rand triplets ({budget} per question)...")
    rng = np.random.default_rng(seed)
    for each_qa in tqdm(data):
        graph = [tuple(triplet[:3]) for triplet in each_qa["graph"]]
        entity_pool = unique_preserve_order([triplet[0] for triplet in graph] + [triplet[2] for triplet in graph])
        q_entities = each_qa.get("q_entity") or infer_question_entities(each_qa)
        q_entity_set = set(q_entities)

        source_triples = [triplet for triplet in graph if triplet[0] in q_entity_set or triplet[2] in q_entity_set]
        if not source_triples:
            source_triples = graph

        existing = set(graph)
        rand_triplets = []
        attempts = 0
        max_attempts = max(1000, budget * 100)
        while len(rand_triplets) < budget and attempts < max_attempts and source_triples:
            attempts += 1
            source = source_triples[int(rng.integers(len(source_triples)))]
            corrupted = make_rand_corrupted_triple(source, q_entities, entity_pool, rng)

            if corrupted is None and q_entities and entity_pool:
                q_entity = q_entities[int(rng.integers(len(q_entities)))]
                replacement = random_entity(entity_pool, {q_entity}, rng)
                if replacement is not None:
                    corrupted = (q_entity, source[1], replacement)

            if corrupted is None or corrupted in existing or corrupted in rand_triplets:
                continue
            rand_triplets.append(corrupted)

        each_qa["safety_attack"] = "rand"
        each_qa["safety_rand_triplets"] = rand_triplets
    return data


def sample_random_triplets(data, num_triplets, seed=0):
    print(f"Sampling {num_triplets} random triplets...")
    np.random.seed(seed)
    for idx, each_qa in enumerate(tqdm(data)):
        all_triplets = np.array(each_qa["graph"])
        sampled_triplets = np.random.permutation(all_triplets)[:num_triplets]
        data[idx][f"sampled_triplets_{num_triplets}"] = sampled_triplets.tolist()
    return data


def get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode, seed=0, triplets_to_sample=[50, 100, 200, 300], limit_samples=0):
    with open(pred_file_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]
    if limit_samples and limit_samples > 0:
        raw_data = raw_data[:limit_samples]

    print("Loading subgraphs...")
    subgraphs = get_subgraphs(dataset_name, split)
    if limit_samples and limit_samples > 0:
        subgraphs = subgraphs.select(range(min(limit_samples, len(subgraphs))))

    print("Adding subgraphs to data...")
    data = []
    for i, each_qa in enumerate(tqdm(raw_data)):
        assert each_qa["id"] == subgraphs[i]["id"]
        each_qa["graph"] = [tuple(each) for each in subgraphs[i]["graph"]]
        each_qa['q_entity'] = subgraphs[i].get('q_entity', [])
        each_qa['a_entity'] = subgraphs[i]['a_entity']
        data.append(each_qa)
    # data = raw_data

    data = add_good_triplets_from_rog(data)
    data = add_question_entities(data, dataset_name, split)
    data = add_scored_triplets(data, score_dict_path, prompt_mode)
    # for num_triplets in triplets_to_sample:
    #     data = sample_random_triplets(data, num_triplets, seed)

    return data
