import os
import time
import json
import inspect
import openai
from vllm import LLM, SamplingParams
from openai import OpenAI
from functools import partial
from transformers import AutoTokenizer
from prompts import icl_user_prompt, icl_ass_prompt


LLAMA2_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% set messages = messages[1:] %}"
    "{% else %}"
    "{% set system_message = false %}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if loop.index0 == 0 and system_message != false %}"
    "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
    "{% else %}"
    "{% set content = message['content'] %}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + content.strip() + ' ' + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)


def get_chat_template(model_name):
    tokenizer_config_path = os.path.join(model_name, "tokenizer_config.json")
    has_chat_template = False
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            has_chat_template = bool(json.load(f).get("chat_template"))

    model_name_lower = model_name.lower()
    if not has_chat_template and "llama-2" in model_name_lower and "chat" in model_name_lower:
        return LLAMA2_CHAT_TEMPLATE
    return None


_TOKENIZER_CACHE = {}


def is_api_model(model_name):
    model_name_lower = str(model_name).lower()
    return (
        "gpt" in model_name_lower
        or "deepseek" in model_name_lower
        or "openai-compatible" in model_name_lower
    )


def get_tokenizer(model_name):
    tokenizer = _TOKENIZER_CACHE.get(model_name)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        chat_template = get_chat_template(model_name)
        if chat_template and not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = chat_template
        _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def count_chat_tokens(messages, model_name):
    tokenizer = get_tokenizer(model_name)
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


def split_user_query(user_query):
    marker = "\n\nQuestion:\n"
    if marker not in user_query or not user_query.startswith("Triplets:\n"):
        return None, user_query
    triplet_block, question_block = user_query.split(marker, 1)
    triplet_lines = [line for line in triplet_block.splitlines()[1:] if line.strip()]
    return triplet_lines, "Question:\n" + question_block


def build_user_query(triplet_lines, question_block):
    if not triplet_lines:
        return question_block
    return "Triplets:\n" + "\n".join(triplet_lines) + "\n\n" + question_block


def extract_triplet_text(line):
    return str(line).strip()


def drop_low_priority_triplets(triplet_lines, drop_count, triplet_priority_map=None, triplet_bucket_map=None):
    scored_indices = []
    for idx, line in enumerate(triplet_lines):
        text = extract_triplet_text(line)
        priority = (triplet_priority_map or {}).get(text, 0)
        bucket = (triplet_bucket_map or {}).get(text, "filler")
        bucket_penalty = {
            "filler": 0,
            "neighbor": 1,
            "support": 2,
            "skeleton": 3,
            "attack": 4,
        }.get(bucket, 0)
        scored_indices.append((priority, bucket_penalty, idx))
    drop_indices = {
        idx for _priority, _bucket_penalty, idx in sorted(scored_indices)[:drop_count]
    }
    return [line for idx, line in enumerate(triplet_lines) if idx not in drop_indices]


def extract_answer_lines(text):
    if not text:
        return []
    answers = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("ans:"):
            answers.append(stripped.split(":", 1)[1].strip())
    return [ans for ans in answers if ans]


def build_triplet_prompt(triplet_lines):
    if not triplet_lines:
        return ""
    return "Triplets:\n" + "\n".join(triplet_lines)


def normalize_chain_text(text):
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(text or "")).split())


def triplet_mentions_chain_answers(line, answers):
    line_norm = normalize_chain_text(line)
    if not line_norm:
        return False
    for answer in answers:
        ans_norm = normalize_chain_text(answer)
        if ans_norm and ans_norm in line_norm:
            return True
    return False


def select_chain_step_triplets(triplet_lines, row, previous_answers):
    if not triplet_lines:
        return triplet_lines
    if not row.get("needs_prev_answer") or not previous_answers:
        return triplet_lines

    flat_prev = [ans for answers in previous_answers for ans in answers if str(ans).strip()]
    if not flat_prev:
        return triplet_lines

    focused = [line for line in triplet_lines if triplet_mentions_chain_answers(line, flat_prev)]
    if len(focused) < 6:
        return triplet_lines

    tail_budget = min(12, max(4, len(triplet_lines) // 8))
    tail = [line for line in triplet_lines if line not in set(focused)][:tail_budget]
    return focused + tail


def build_chain_step_user_query(triplet_lines, step_idx, question, previous_answers):
    sections = []
    triplet_prompt = build_triplet_prompt(triplet_lines)
    if triplet_prompt:
        sections.append(triplet_prompt)
    sections.append(f"Subquestion Step {step_idx + 1}:\n{question}")
    if previous_answers:
        prev_lines = [
            f"Step {idx + 1}: {', '.join(ans) if ans else 'not available'}"
            for idx, ans in enumerate(previous_answers)
        ]
        sections.append("Previous step answers:\n" + "\n".join(prev_lines))
    sections.append('Return answers as separate lines prefixed with "ans:".')
    return "\n\n".join(sections)


def build_chain_final_user_query(triplet_lines, original_question, previous_answers):
    sections = []
    triplet_prompt = build_triplet_prompt(triplet_lines)
    if triplet_prompt:
        sections.append(triplet_prompt)
    if previous_answers:
        prev_lines = [
            f"Step {idx + 1}: {', '.join(ans) if ans else 'not available'}"
            for idx, ans in enumerate(previous_answers)
        ]
        sections.append("Intermediate subquestion answers:\n" + "\n".join(prev_lines))
    sections.append("Original Question:\n" + original_question)
    sections.append('Use the intermediate answers when relevant and return final answers as separate lines prefixed with "ans:".')
    return "\n\n".join(sections)


def find_triplet_user_index(conversation):
    for idx in range(len(conversation) - 1, -1, -1):
        message = conversation[idx]
        if message.get("role") == "user" and str(message.get("content", "")).startswith("Triplets:\n"):
            return idx
    return None


def truncate_conversation_to_fit(conversation, model_name, max_model_len, reserve_tokens=32, triplet_priority_map=None, triplet_bucket_map=None):
    if is_api_model(model_name):
        return conversation, 0

    budget = max(1, max_model_len - reserve_tokens)
    current_tokens = count_chat_tokens(conversation, model_name)
    if current_tokens <= budget:
        return conversation, 0

    new_conversation = [dict(message) for message in conversation]
    user_idx = find_triplet_user_index(new_conversation)
    if user_idx is None:
        return new_conversation, 0
    triplet_lines, question_block = split_user_query(new_conversation[user_idx]["content"])
    if triplet_lines is None:
        return new_conversation, 0

    removed = 0
    while triplet_lines and current_tokens > budget:
        overflow = current_tokens - budget
        drop_count = max(1, min(len(triplet_lines), overflow // 24 + 1))
        triplet_lines = drop_low_priority_triplets(
            triplet_lines,
            drop_count,
            triplet_priority_map=triplet_priority_map,
            triplet_bucket_map=triplet_bucket_map,
        )
        removed += drop_count
        new_conversation[user_idx]["content"] = build_user_query(triplet_lines, question_block)
        current_tokens = count_chat_tokens(new_conversation, model_name)

    if current_tokens > budget:
        while triplet_lines and current_tokens > budget:
            triplet_lines = drop_low_priority_triplets(
                triplet_lines,
                1,
                triplet_priority_map=triplet_priority_map,
                triplet_bucket_map=triplet_bucket_map,
            )
            removed += 1
            new_conversation[user_idx]["content"] = build_user_query(triplet_lines, question_block)
            current_tokens = count_chat_tokens(new_conversation, model_name)

    return new_conversation, removed


def strip_triplets_from_conversation(conversation):
    new_conversation = [dict(message) for message in conversation]
    user_idx = find_triplet_user_index(new_conversation)
    if user_idx is None:
        return new_conversation, 0
    triplet_lines, question_block = split_user_query(new_conversation[user_idx]["content"])
    if triplet_lines is None or not triplet_lines:
        return new_conversation, 0
    removed = len(triplet_lines)
    new_conversation[user_idx]["content"] = question_block
    return new_conversation, removed


def trim_assistant_context(conversation, keep_chars=1200):
    new_conversation = [dict(message) for message in conversation]
    for idx in range(len(new_conversation) - 1, -1, -1):
        if new_conversation[idx].get("role") != "assistant":
            continue
        content = str(new_conversation[idx].get("content", ""))
        if len(content) <= keep_chars:
            continue
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        ans_lines = [line for line in lines if line.lower().startswith("ans:")]
        if ans_lines:
            new_conversation[idx]["content"] = "\n".join(ans_lines[:10])
        else:
            new_conversation[idx]["content"] = content[-keep_chars:]
        return new_conversation, True
    return new_conversation, False


def infer_with_auto_truncation(llm, conversation, model_name, triplet_priority_map=None, triplet_bucket_map=None):
    max_model_len = getattr(llm, "max_model_len", 8192)
    reserve_tokens = min(256, getattr(llm, "max_tokens", 0) or 0)
    total_removed = 0
    working_conversation = conversation

    while True:
        working_conversation, removed_triplets = truncate_conversation_to_fit(
            working_conversation,
            model_name,
            max_model_len,
            reserve_tokens=reserve_tokens,
            triplet_priority_map=triplet_priority_map,
            triplet_bucket_map=triplet_bucket_map,
        )
        total_removed += removed_triplets
        if removed_triplets:
            print(f"[prompt-truncate] Removed {removed_triplets} triplets to fit context window.")
        try:
            return get_outputs(llm(messages=working_conversation), model_name)
        except Exception as exc:
            message = str(exc)
            is_context_error = (
                "maximum context length" in message.lower()
                or "input tokens" in message.lower()
                or exc.__class__.__name__ == "VLLMValidationError"
            )
            if not is_context_error:
                raise

            user_idx = find_triplet_user_index(working_conversation)
            if user_idx is not None:
                triplet_lines, question_block = split_user_query(working_conversation[user_idx]["content"])
                if triplet_lines:
                    fallback_drop = min(8, len(triplet_lines))
                    triplet_lines = drop_low_priority_triplets(
                        triplet_lines,
                        fallback_drop,
                        triplet_priority_map=triplet_priority_map,
                        triplet_bucket_map=triplet_bucket_map,
                    )
                    total_removed += fallback_drop
                    working_conversation[user_idx]["content"] = build_user_query(triplet_lines, question_block)
                    print(f"[prompt-truncate-retry] Extra removed {fallback_drop} triplets after validation overflow.")
                    continue

            stripped_conversation, stripped = strip_triplets_from_conversation(working_conversation)
            if stripped:
                total_removed += stripped
                working_conversation = stripped_conversation
                print(f"[prompt-truncate-retry] Dropped all remaining {stripped} triplets for overflow recovery.")
                continue

            trimmed_conversation, trimmed = trim_assistant_context(working_conversation)
            if trimmed:
                working_conversation = trimmed_conversation
                print("[prompt-truncate-retry] Trimmed prior assistant context for overflow recovery.")
                continue

            raise


def llm_init(model_name, tensor_parallel_size=1, max_seq_len_to_capture=8192, max_tokens=4000, seed=0, temperature=0, frequency_penalty=0, max_model_len=8192):
    if not is_api_model(model_name):
        vllm_gpu_memory_utilization = float(
            os.environ.get(
                "SUBGRAPHRAG_VLLM_GPU_MEMORY_UTILIZATION",
                os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90"),
            )
        )
        vllm_max_num_seqs = int(
            os.environ.get(
                "SUBGRAPHRAG_VLLM_MAX_NUM_SEQS",
                os.environ.get("VLLM_MAX_NUM_SEQS", "32"),
            )
        )
        vllm_enforce_eager = os.environ.get(
            "SUBGRAPHRAG_VLLM_ENFORCE_EAGER",
            os.environ.get("VLLM_ENFORCE_EAGER", "0"),
        ).strip().lower() in {"1", "true", "yes"}
        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": vllm_gpu_memory_utilization,
            "max_num_seqs": vllm_max_num_seqs,
            "enforce_eager": vllm_enforce_eager,
        }
        llm_sig = inspect.signature(LLM.__init__)
        if "max_seq_len_to_capture" in llm_sig.parameters:
            llm_kwargs["max_seq_len_to_capture"] = max_seq_len_to_capture
        client = LLM(**llm_kwargs)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                         frequency_penalty=frequency_penalty)
        llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False,
                      chat_template=get_chat_template(model_name))
        llm.max_model_len = max_model_len
        llm.max_tokens = max_tokens
    else:
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("SILICONFLOW_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
        )
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY, SILICONFLOW_API_KEY, or DEEPSEEK_API_KEY for API models.")
        api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE") or "https://api.siliconflow.cn/v1"
        client = OpenAI(api_key=api_key, base_url=api_base)
        api_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "gpt" in str(model_name).lower():
            api_kwargs["seed"] = seed
        llm = partial(client.chat.completions.create, **api_kwargs)
        llm.max_model_len = max_model_len
        llm.max_tokens = max_tokens
    return llm


def get_outputs(outputs, model_name):
    if not is_api_model(model_name):
        return outputs[0].outputs[0].text
    else:
        return outputs.choices[0].message.content


def llm_inf(llm, prompts, mode, model_name):
    res = []
    triplet_priority_map = prompts.get("triplet_priority_map", {})
    triplet_bucket_map = prompts.get("triplet_bucket_map", {})
    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        outputs = infer_with_auto_truncation(
            llm,
            conversation,
            model_name,
            triplet_priority_map=triplet_priority_map,
            triplet_bucket_map=triplet_bucket_map,
        )
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs = infer_with_auto_truncation(
            llm,
            conversation,
            model_name,
            triplet_priority_map=triplet_priority_map,
            triplet_bucket_map=triplet_bucket_map,
        )
        res.append(outputs)
    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = infer_with_auto_truncation(
                llm,
                conversation,
                model_name,
                triplet_priority_map=triplet_priority_map,
                triplet_bucket_map=triplet_bucket_map,
            )
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_subquestion_chain(llm, prompts, mode, model_name):
    subquestions = sorted(prompts.get("subquestion_decomposition", []), key=lambda row: int(row.get("sub_id", 0)))
    if len(subquestions) <= 1:
        return llm_inf(llm, prompts, mode, model_name)

    base_conversation = []
    if 'sys' in mode:
        base_conversation.append({"role": "system", "content": prompts['sys_query']})
    if 'icl' in mode:
        base_conversation.append({"role": "user", "content": icl_user_prompt})
        base_conversation.append({"role": "assistant", "content": icl_ass_prompt})

    triplet_lines, _question_block = split_user_query(prompts['user_query'])
    triplet_lines = triplet_lines or []
    triplet_priority_map = prompts.get("triplet_priority_map", {})
    triplet_bucket_map = prompts.get("triplet_bucket_map", {})
    step_answers = []
    chain_trace = []

    for row in subquestions:
        step_idx = int(row.get("sub_id", 0))
        step_triplet_lines = select_chain_step_triplets(triplet_lines, row, step_answers)
        user_query = build_chain_step_user_query(
            step_triplet_lines,
            step_idx=step_idx,
            question=row.get("question", ""),
            previous_answers=step_answers,
        )
        conversation = list(base_conversation) + [{"role": "user", "content": user_query}]
        outputs = infer_with_auto_truncation(
            llm,
            conversation,
            model_name,
            triplet_priority_map=triplet_priority_map,
            triplet_bucket_map=triplet_bucket_map,
        )
        answers = extract_answer_lines(outputs)
        step_answers.append(answers)
        chain_trace.append({
            "sub_id": step_idx,
            "question": row.get("question", ""),
            "prediction": outputs,
            "answers": answers,
        })

    final_query = build_chain_final_user_query(
        triplet_lines,
        original_question=prompts.get("question", ""),
        previous_answers=step_answers,
    )
    final_conversation = list(base_conversation) + [{"role": "user", "content": final_query}]
    final_outputs = infer_with_auto_truncation(
        llm,
        final_conversation,
        model_name,
        triplet_priority_map=triplet_priority_map,
        triplet_bucket_map=triplet_bucket_map,
    )
    prompts["chain_trace"] = chain_trace
    prompts["chain_final_user_query"] = final_query
    return [final_outputs, ""]


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name)
        except openai.RateLimitError as e:
            wait_time = (2 ** retries) * 5  # Exponential backoff
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5):
    if each_qa.get("safety_attack") == "ours" and len(each_qa.get("subquestion_decomposition", [])) > 1:
        return llm_inf_subquestion_chain(llm, each_qa, llm_mode, model_name)
    if is_api_model(model_name):
        return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)
    else:
        return llm_inf(llm, each_qa, llm_mode, model_name)
