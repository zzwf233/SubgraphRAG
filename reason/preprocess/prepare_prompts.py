import numpy as np
from collections import defaultdict


def triplet_to_str(triplet):
    return f"({triplet[0]},{triplet[1]},{triplet[2]})"


def canonical_triplet(triplet):
    return tuple(str(v) for v in triplet[:3])


def unique_preserve_order(input_list):
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list


def normalize_text(text):
    return " ".join(str(text or "").strip().lower().split())


def triplet_mentions_entity(triple, entities):
    entity_norm = {normalize_text(x) for x in entities if str(x).strip()}
    if not entity_norm:
        return False
    return normalize_text(triple[0]) in entity_norm or normalize_text(triple[2]) in entity_norm


def relation_in_rule_paths(relation, rule_paths):
    rel_norm = normalize_text(relation)
    for path in rule_paths:
        if any(normalize_text(x) == rel_norm for x in path if x):
            return True
    return False


def is_local_gold_support(triple, each_qa):
    rows = each_qa.get("subquestion_decomposition", [])
    poisoned_rows = [row for row in rows if row.get("is_poisoned")]
    if not poisoned_rows:
        return False
    rule_paths = [row.get("rule_path", []) for row in poisoned_rows if row.get("rule_path")]
    starts = [row.get("start_node") for row in poisoned_rows if row.get("start_node")]
    answers = each_qa.get("a_entity", []) or each_qa.get("answer", []) or each_qa.get("answers", [])
    q_entities = each_qa.get("q_entity", [])
    if not relation_in_rule_paths(triple[1], rule_paths):
        return False
    if not triplet_mentions_entity(triple, starts):
        return False
    return triplet_mentions_entity(triple, answers) or triplet_mentions_entity(triple, q_entities)


def build_dependency_plan_block(each_qa):
    rows = each_qa.get("subquestion_decomposition", [])
    if not rows:
        return ""
    rows = sorted(rows, key=lambda row: int(row.get("sub_id", 0)))
    if len(rows) <= 1:
        return ""

    lines = [
        "Dependency-Aware Subquestions:",
        "Solve the following subquestions in order before answering the original question.",
    ]
    for row in rows:
        sub_id = int(row.get("sub_id", 0))
        dep_prev = row.get("dep_prev_sub_id")
        dep_desc = "independent"
        if row.get("needs_prev_answer"):
            dep_desc = f"depends on step {int(dep_prev) + 1}"
        dep_type = row.get("dep_type", "none")
        lines.append(
            f"[Step {sub_id + 1}] {row.get('question', '').strip()} "
            f"(dependency: {dep_desc}; type: {dep_type})"
        )
    lines.append("Use the answer from each earlier step when a later step depends on it.")
    return "\n".join(lines)


def prioritize_poison_scored_triplets(triplets, each_qa):
    poison_targets = each_qa.get("poison_targets", [])
    if each_qa.get("poison_target"):
        poison_targets = [each_qa["poison_target"]] + list(poison_targets)
    poison_targets = [normalize_text(x) for x in poison_targets if str(x).strip()]
    poison_front = [
        tuple(str(v) for v in triplet[:3])
        for triplet in each_qa.get("poison_front_triples", [])
        if len(triplet) >= 3
    ]
    injected = {
        tuple(str(v) for v in triplet[:3])
        for triplet in each_qa.get("safety_ours_injected_triples", [])
        if len(triplet) >= 3
    }
    if not poison_targets and not injected and not poison_front:
        return triplets

    front_rank = {triple: idx for idx, triple in enumerate(poison_front)}

    def triplet_key(triplet):
        triple = tuple(str(v) for v in triplet[:3])
        text = " ".join(normalize_text(x) for x in triple)
        front_hit = 1 if triple in front_rank else 0
        front_order = -front_rank.get(triple, 10**9)
        injected_hit = 1 if triple in injected else 0
        poison_hit = 1 if any(target and target in text for target in poison_targets) else 0
        local_gold_penalty = 1 if is_local_gold_support(triple, each_qa) else 0
        score = float(triplet[3]) if len(triplet) > 3 else 0.0
        return (front_hit, front_order, injected_hit, poison_hit, -local_gold_penalty, score)

    return sorted(triplets, key=triplet_key, reverse=True)


def build_structured_triplet_buckets(triplets, each_qa):
    scored_triplets = [triplet for triplet in triplets if len(triplet) >= 3]
    if not scored_triplets:
        return [], {}, {}

    q_entities = each_qa.get("q_entity", [])
    answers = each_qa.get("a_entity", []) or each_qa.get("answer", []) or each_qa.get("answers", [])
    rows = each_qa.get("subquestion_decomposition", [])
    poisoned_rows = [row for row in rows if row.get("is_poisoned")]
    starts = [row.get("start_node") for row in poisoned_rows if row.get("start_node")]
    rule_paths = [row.get("rule_path", []) for row in poisoned_rows if row.get("rule_path")]
    support_triples = {
        canonical_triplet(triplet)
        for triplet in each_qa.get("safety_ours_support_triples", [])
        if len(triplet) >= 3
    }
    rog_triples = {
        canonical_triplet(triplet)
        for triplet in each_qa.get("good_triplets_rog", [])
        if len(triplet) >= 3
    }
    poison_front = {
        canonical_triplet(triplet)
        for triplet in each_qa.get("poison_front_triples", [])
        if len(triplet) >= 3
    }
    injected = {
        canonical_triplet(triplet)
        for triplet in each_qa.get("safety_ours_injected_triples", [])
        if len(triplet) >= 3
    }

    scored_sorted = list(scored_triplets)
    if each_qa.get("safety_attack") == "ours":
        scored_sorted = prioritize_poison_scored_triplets(scored_sorted, each_qa)

    triplet_infos = []
    adjacency = defaultdict(list)
    seed_entities = set()
    for entity in list(q_entities) + list(starts):
        norm = normalize_text(entity)
        if norm:
            seed_entities.add(norm)

    for idx, triplet in enumerate(scored_sorted):
        triple = canonical_triplet(triplet)
        score = float(triplet[3]) if len(triplet) > 3 else 0.0
        head_norm = normalize_text(triple[0])
        tail_norm = normalize_text(triple[2])
        mentions_seed = head_norm in seed_entities or tail_norm in seed_entities
        mentions_q = triplet_mentions_entity(triple, q_entities)
        mentions_a = triplet_mentions_entity(triple, answers)
        mentions_start = triplet_mentions_entity(triple, starts)
        local_gold = is_local_gold_support(triple, each_qa)
        if each_qa.get("safety_attack") == "ours" and local_gold:
            continue
        rule_hit = relation_in_rule_paths(triple[1], rule_paths)
        info = {
            "idx": idx,
            "triplet": triplet,
            "triple": triple,
            "score": score,
            "mentions_q": mentions_q,
            "mentions_a": mentions_a,
            "mentions_start": mentions_start,
            "mentions_seed": mentions_seed,
            "local_gold": local_gold,
            "rule_hit": rule_hit,
            "rog_hit": triple in rog_triples,
            "support_hit": triple in support_triples,
            "poison_front_hit": triple in poison_front,
            "injected_hit": triple in injected,
        }
        triplet_infos.append(info)
        adjacency[head_norm].append(info)
        adjacency[tail_norm].append(info)

    def rank_key(info):
        return (
            1 if info["poison_front_hit"] else 0,
            1 if info["injected_hit"] else 0,
            1 if info["local_gold"] else 0,
            1 if info["rule_hit"] and info["mentions_start"] else 0,
            1 if info["support_hit"] else 0,
            1 if info["rog_hit"] else 0,
            1 if info["mentions_q"] else 0,
            1 if info["mentions_a"] else 0,
            info["score"],
            -info["idx"],
        )

    def add_unique(selected, seen, info):
        triple = info["triple"]
        if triple in seen:
            return False
        selected.append(info)
        seen.add(triple)
        return True

    selected = []
    seen = set()

    skeleton_pool = [
        info for info in triplet_infos
        if info["poison_front_hit"]
        or info["injected_hit"]
        or info["support_hit"]
        or info["local_gold"]
        or info["rog_hit"]
        or (info["rule_hit"] and (info["mentions_start"] or info["mentions_q"] or info["mentions_a"]))
    ]
    for info in sorted(skeleton_pool, key=rank_key, reverse=True):
        add_unique(selected, seen, info)

    frontier_entities = set(seed_entities)
    for info in selected:
        frontier_entities.add(normalize_text(info["triple"][0]))
        frontier_entities.add(normalize_text(info["triple"][2]))

    def neighbor_sort_key(info):
        head_norm = normalize_text(info["triple"][0])
        tail_norm = normalize_text(info["triple"][2])
        introduces_new = int(head_norm not in frontier_entities or tail_norm not in frontier_entities)
        return (
            introduces_new,
            1 if info["rule_hit"] else 0,
            1 if info["mentions_seed"] else 0,
            1 if info["mentions_q"] else 0,
            1 if info["mentions_a"] else 0,
            info["score"],
            -info["idx"],
        )

    expanded = True
    while expanded:
        expanded = False
        neighbor_candidates = []
        for entity_norm in list(frontier_entities):
            if not entity_norm:
                continue
            neighbor_candidates.extend(adjacency.get(entity_norm, []))
        for info in sorted(neighbor_candidates, key=neighbor_sort_key, reverse=True):
            if add_unique(selected, seen, info):
                frontier_entities.add(normalize_text(info["triple"][0]))
                frontier_entities.add(normalize_text(info["triple"][2]))
                expanded = True

    for info in sorted(triplet_infos, key=rank_key, reverse=True):
        add_unique(selected, seen, info)

    priority_map = {}
    bucket_map = {}
    for rank, info in enumerate(selected):
        triple_text = triplet_to_str(info["triple"])
        if info["poison_front_hit"] or info["injected_hit"]:
            bucket = "attack"
        elif info["local_gold"] or (info["rule_hit"] and (info["mentions_start"] or info["mentions_q"])):
            bucket = "skeleton"
        elif info["support_hit"] or info["rog_hit"]:
            bucket = "support"
        elif info["mentions_seed"] or info["mentions_q"] or info["mentions_a"]:
            bucket = "neighbor"
        else:
            bucket = "filler"
        priority_map[triple_text] = len(selected) - rank
        bucket_map[triple_text] = bucket

    ordered_triplets = [info["triplet"] for info in selected]
    return ordered_triplets, priority_map, bucket_map


def remove_same_head_tail(triplets, mode):
    if 'rmht' not in mode:
        return triplets

    new_triplets = []
    seen = set()
    for triplet in triplets:
        item_1 = ','.join([str(triplet[0]), str(triplet[2])])
        item_2 = ','.join([str(triplet[2]), str(triplet[0])])
        if item_1 not in seen and item_2 not in seen:
            seen.add(item_1)
            seen.add(item_2)
            new_triplets.append(triplet)
    return new_triplets


def merge_tuples(tuple_list, mode=0):
    if mode == 0:
        merged_dict = defaultdict(lambda: [[], None, None])
        for t in tuple_list:
            key = (t[1], t[2])  # Group by the second and third elements
            merged_dict[key][0].append(t[0])  # Append the first element to the list
            merged_dict[key][1] = t[1]  # Set the second element
            merged_dict[key][2] = t[2]  # Set the third element

        # Convert the dictionary back to a list of merged tuples
        return [('[' + ','.join(v[0]) + ']', v[1], v[2]) for v in merged_dict.values()]
    else:
        assert mode == 2
        merged_dict = defaultdict(lambda: [None, None, []])
        for t in tuple_list:
            key = (t[0], t[1])
            merged_dict[key][2].append(t[2])
            merged_dict[key][0] = t[0]
            merged_dict[key][1] = t[1]
        return [(v[0], v[1], '[' + ','.join(v[2]) + ']') for v in merged_dict.values()]


def get_prompts(each_qa, mode, sys_prompt, cot_prompt, thres, seed=0):
    plan_block = ""
    if each_qa.get("safety_attack") == "ours":
        plan_block = build_dependency_plan_block(each_qa)

    question_sections = []
    if plan_block:
        question_sections.append(plan_block)
    question_sections.append("Question:\n" + each_qa['question'])
    question_prompt = "\n\n".join(question_sections)
    if question_prompt[-1] != '?':
        question_prompt += '?'

    if 'rog' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        good_triplets_rog = each_qa['good_triplets_rog']
        input_triplets = remove_same_head_tail(good_triplets_rog, mode)
        # sampled_triplets = np.array(each_qa[f'sampled_triplets_{num_sampled_triplets}'])
        # input_triplets = np.concatenate([good_triplets_rog, sampled_triplets]) if len(good_triplets_rog) > 0 else sampled_triplets
        input_triplets = [triplet_to_str(triplet) for triplet in input_triplets]
        other_triplets = remove_same_head_tail(each_qa['scored_triplets'], mode)
        other_triplets = [triplet_to_str(triplet) for triplet in other_triplets]
        input_triplets = unique_preserve_order(input_triplets + other_triplets)
        input_triplets = input_triplets[:num_sampled_triplets]
        # input_triplets = np.random.permutation(input_triplets)
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    elif 'scored' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        input_triplets = each_qa['scored_triplets']
        input_triplets, priority_map, bucket_map = build_structured_triplet_buckets(input_triplets, each_qa)
        if thres:
            input_triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in input_triplets if triplet[3] >= thres]
        else:
            input_triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in input_triplets]

        input_triplets = unique_preserve_order(input_triplets)
        input_triplets = input_triplets[:num_sampled_triplets]
        input_triplets = [triplet_to_str(triplet) for triplet in input_triplets]
        if 'rev' in mode:
            input_triplets.reverse()
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
        each_qa["triplet_priority_map"] = {
            triplet: priority_map.get(triplet, 0) for triplet in input_triplets
        }
        each_qa["triplet_bucket_map"] = {
            triplet: bucket_map.get(triplet, "filler") for triplet in input_triplets
        }

    elif 'rand' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        np.random.seed(seed)
        input_triplets = np.random.permutation(np.array(each_qa['graph']))
        if 'randNoA' in mode:
            for each_a in each_qa['a_entity']:
                input_triplets = [triplet for triplet in input_triplets if each_a not in triplet[0] and each_a not in triplet[2]]

        input_triplets = unique_preserve_order([triplet_to_str(triplet) for triplet in input_triplets])
        input_triplets = input_triplets[:num_sampled_triplets]
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    elif 'noevi' in mode:
        triplet_prompt = ''
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if 'firstq' in mode:
        all_query = "\n\n".join([sys_prompt, question_prompt, triplet_prompt])
        user_query = "\n\n".join([question_prompt, triplet_prompt])
    else:
        all_query = "\n\n".join([sys_prompt, triplet_prompt, question_prompt])
        user_query = "\n\n".join([triplet_prompt, question_prompt])
        if triplet_prompt == '':
            user_query = question_prompt

    each_qa['sys_query'] = sys_prompt
    each_qa['user_query'] = user_query
    each_qa['all_query'] = all_query
    each_qa['cot_query'] = cot_prompt
    if "triplet_priority_map" not in each_qa:
        each_qa["triplet_priority_map"] = {}
    if "triplet_bucket_map" not in each_qa:
        each_qa["triplet_bucket_map"] = {}
    return each_qa


def get_prompts_for_data(data, mode, sys_prompt, cot_prompt, thres):
    new_data = []
    for each_qa in data:
        new_data.append(get_prompts(each_qa, mode, sys_prompt, cot_prompt, thres))
    return new_data
