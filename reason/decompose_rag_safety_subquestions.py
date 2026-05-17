import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import load_dataset


MULTIHOP_CUES = [
    r"\bwho\b.*\b(whose|that|which)\b",
    r"\bwhat\b.*\b(whose|that|which)\b",
    r"\bwhere\b.*\b(whose|that|which)\b",
    r"\bwhich\b.*\b(whose|that|which)\b",
    r"\bwhose\b",
    r"\bwho\b.*\b(with|that|which)\b",
    r"\bwhat\b.*\b(with|that|which)\b",
    r"\bwhich\b.*\b(with|that|which)\b",
    r"\bcontains?\b.*\b(that|which|who)\b",
    r"\bhas\b.*\b(that|which|who)\b",
    r"\bwith\b.*\b(that|which|who)\b",
    r"\bafter\b",
    r"\bbefore\b",
    r"\bthen\b",
    r"\bfirst\b",
    r"\bof\b.*\bof\b",
    r"\bfrom\b.*\bto\b",
]

SPLIT_PATTERNS = [
    r",\s*and\s+",
    r"\s+and\s+(?=(who|what|where|when|which|how)\b)",
    r"\sand\s+then\s+",
    r"\sthen\s+",
    r"\safter\s+",
    r"\sbefore\s+",
    r"\swhich\s+",
    r"\sthat\s+",
    r"\swhose\s+",
]

COREF_PATTERNS = [
    r"\bit\b",
    r"\bits\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\bhe\b",
    r"\bshe\b",
    r"\bthat one\b",
    r"\bthis one\b",
    r"\bthat country\b",
    r"\bthat city\b",
    r"\bthat person\b",
]


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def to_full_question(text: str) -> str:
    q = (text or "").strip(" ,.;，。；")
    if not q:
        return q
    return q if q.endswith("?") else f"{q}?"


def is_multihop_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    lower = q.lower()
    if any(x in lower for x in ["what's your name", "who are you", "tell me about yourself", " your ", " my "]):
        return False
    return any(re.search(cue, q, flags=re.IGNORECASE) for cue in MULTIHOP_CUES)


def is_low_information_subquestion(text: str) -> bool:
    q = (text or "").strip(" ?？,.;，。；")
    if not q:
        return True
    tokens = [t for t in re.split(r"\s+", q) if t]
    if len(tokens) <= 1:
        return True
    lower = q.lower()
    if lower.startswith("what is "):
        tail = lower[len("what is "):].strip()
        if len([t for t in tail.split() if t]) <= 1:
            return True
    return False


def has_dependency_signal(text: str) -> bool:
    q = (text or "").lower()
    if re.search(r"\[[A-Z]\]", text or ""):
        return True
    return any(x in f" {q} " for x in [" it ", " its ", " they ", " them ", " he ", " she "])


def low_quality_followup(text: str) -> bool:
    q = normalize_text(text)
    if not q:
        return True
    bad_prefixes = [
        "[b] was",
        "[b] are",
        "[b] is",
        "what happened before [b]",
        "what happened after [b]",
    ]
    if any(q.startswith(prefix) for prefix in bad_prefixes):
        return True
    if q.count("[b]") + q.count("[c]") > 1:
        return True
    tokens = q.replace("[b]", "entity").replace("[c]", "entity").split()
    if len(tokens) < 4:
        return True
    return False


def upgrade_low_information_subquestion(text: str, prev_placeholder: Optional[str] = None) -> str:
    q = (text or "").strip(" ?？,.;，。；")
    if not q:
        return text
    lower = q.lower()
    if prev_placeholder:
        if any(k in lower for k in ["college", "university", "school"]):
            return f"Which college did {prev_placeholder} attend?"
        if any(k in lower for k in ["city", "place", "location", "country"]):
            return f"Where is {prev_placeholder} located?"
        if any(k in lower for k in ["born", "birth"]):
            return f"Where was {prev_placeholder} born?"
        return f"Which entity related to {prev_placeholder} is associated with {q}?"
    return f"Which entity is related to {q}?"


def ensure_dependency(candidate: str, raw_fragment: str, sub_id: int) -> str:
    if sub_id <= 0:
        return candidate
    if has_dependency_signal(candidate):
        return candidate
    prev_placeholder = f"[{chr(ord('B') + sub_id - 1)}]"
    return upgrade_low_information_subquestion(raw_fragment, prev_placeholder=prev_placeholder)


def canonicalize_dependency_placeholder(candidate: str, sub_id: int) -> str:
    if sub_id <= 0:
        return candidate
    expected = f"[{chr(ord('B') + sub_id - 1)}]"
    q = str(candidate or "")
    if re.search(r"\[[A-Z]\]", q):
        return re.sub(r"\[[A-Z]\]", expected, q)
    for pat in COREF_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return re.sub(pat, expected, q, flags=re.IGNORECASE)
    q = q[:-1].strip() if q.endswith("?") else q
    return f"{q} of {expected}?"


def infer_dep_type(subq: str, sub_id: int) -> str:
    if sub_id == 0:
        return "none"
    q = f" {(subq or '').lower()} "
    if any(x in q for x in ["[b]", "[c]", " it ", " its ", " they ", " them ", " he ", " she ", " that one ", " this one "]):
        return "coref"
    if any(x in q for x in [" which ", " what ", " where ", " when ", " who ", " whose "]) and any(
        x in q for x in [" in ", " on ", " at ", " from ", " among ", " within ", " of "]
    ):
        return "filter"
    return "bridge"


def strip_wh_prefix(text: str) -> str:
    value = str(text or "").strip(" ?？,.;，。；")
    return re.sub(
        r"^(what|which|who|where|when|how)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()


def extract_focus_phrase(text: str) -> str:
    value = strip_wh_prefix(text)
    if not value:
        return ""

    verb_markers = [
        " contains ",
        " contain ",
        " has ",
        " have ",
        " with ",
        " featuring ",
        " feature ",
        " starring ",
        " star ",
        " serves ",
        " serve ",
        " includes ",
        " include ",
    ]
    lower = f" {value.lower()} "
    for marker in verb_markers:
        if marker in lower:
            tail = value[lower.index(marker) + len(marker) - 1:].strip()
            return tail.strip(" ,.;")

    tokens = [tok for tok in value.split() if tok]
    if len(tokens) <= 4:
        return value
    return " ".join(tokens[-4:])


def build_followup_from_fragment(fragment: str, focus_phrase: str, prev_placeholder: str) -> str:
    fragment = str(fragment or "").strip(" ?？,.;，。；")
    if not fragment:
        return ""

    if has_dependency_signal(fragment):
        candidate = canonicalize_dependency_placeholder(fragment, 1)
        return to_full_question(candidate)

    lower = fragment.lower()
    focus = focus_phrase.strip()
    focus = re.sub(r"^(an?|the)\s+", "", focus, flags=re.IGNORECASE)
    if re.fullmatch(r"(he|she|they|it)\s+was\s+president", lower):
        return to_full_question(f"What did {prev_placeholder} do before becoming president")
    if re.fullmatch(r"(he|she|they|it)\s+died", lower):
        return to_full_question(f"Where did {prev_placeholder} die")
    if re.fullmatch(r"(he|she|they|it)\s+was\s+born", lower):
        return to_full_question(f"Where was {prev_placeholder} born")
    if re.match(r"^(serves?|served|contains?|contained|located|born|founded|written|directed|produced|starring)\b", lower):
        if focus:
            return to_full_question(f"Which {focus} of {prev_placeholder} {fragment}")
        return to_full_question(f"Which entity related to {prev_placeholder} {fragment}")
    if re.match(r"^(is|was|were|are|did|do|does|has|have|had)\b", lower):
        return to_full_question(f"{prev_placeholder} {fragment}")
    if focus:
        return to_full_question(f"Which {focus} of {prev_placeholder} is associated with {fragment}")
    return to_full_question(f"Which entity related to {prev_placeholder} is associated with {fragment}")


def decompose_relative_clause(question: str) -> List[Dict[str, Any]]:
    text = (question or "").strip()
    if not text:
        return []

    match = re.search(r"\b(that|which|who|whose)\b", text, flags=re.IGNORECASE)
    if not match:
        return []

    left = text[:match.start()].strip(" ,.;，。；")
    right = text[match.end():].strip(" ,.;，。；")
    if not left or not right:
        return []
    if is_low_information_subquestion(left) or len(left.split()) < 4:
        return []

    prev_placeholder = "[B]"
    focus_phrase = extract_focus_phrase(left)
    follow_up = build_followup_from_fragment(right, focus_phrase, prev_placeholder)
    if not follow_up:
        return []

    first = {
        "question": to_full_question(left),
        "sub_id": 0,
        "dep_prev_sub_id": None,
        "dep_type": "none",
        "needs_prev_answer": False,
    }
    second = {
        "question": follow_up,
        "sub_id": 1,
        "dep_prev_sub_id": 0,
        "dep_type": infer_dep_type(follow_up, 1),
        "needs_prev_answer": True,
    }
    if low_quality_followup(second["question"]):
        return []
    return [first, second]


def decompose_temporal_clause(question: str) -> List[Dict[str, Any]]:
    text = (question or "").strip(" ?？")
    if not text:
        return []

    match = re.search(r"\b(before|after)\b", text, flags=re.IGNORECASE)
    if not match:
        return []

    left = text[:match.start()].strip(" ,.;，。；")
    right = text[match.end():].strip(" ,.;，。；")
    if not left or not right:
        return []
    if is_low_information_subquestion(left) or len(left.split()) < 4:
        return []

    marker = match.group(1).lower()
    prev_placeholder = "[B]"
    right_lower = right.lower().strip()
    if re.fullmatch(r"(he|she|they|it)\s+was\s+president", right_lower):
        return []
    first = {
        "question": to_full_question(left),
        "sub_id": 0,
        "dep_prev_sub_id": None,
        "dep_type": "none",
        "needs_prev_answer": False,
    }
    second_q = f"What happened {marker} {prev_placeholder} {right}"
    second_q = canonicalize_dependency_placeholder(second_q, 1)
    second = {
        "question": to_full_question(second_q),
        "sub_id": 1,
        "dep_prev_sub_id": 0,
        "dep_type": infer_dep_type(second_q, 1),
        "needs_prev_answer": True,
    }
    if low_quality_followup(second["question"]):
        return []
    return [first, second]


def split_question(question: str, max_subquestions: int = 3) -> List[Dict[str, Any]]:
    text = (question or "").strip()
    if not text:
        return []
    if not is_multihop_question(text):
        return [{
            "question": to_full_question(text),
            "sub_id": 0,
            "dep_prev_sub_id": None,
            "dep_type": "none",
            "needs_prev_answer": False,
        }]

    relative_rows = decompose_relative_clause(text)
    if relative_rows:
        return relative_rows[:max_subquestions]

    temporal_rows = decompose_temporal_clause(text)
    if temporal_rows:
        return temporal_rows[:max_subquestions]

    merged = text
    for pattern in SPLIT_PATTERNS:
        merged = re.sub(pattern, " [SPLIT] ", merged, flags=re.IGNORECASE)

    parts = [p.strip(" ,.;，。；") for p in merged.split("[SPLIT]")]
    parts = [p for p in parts if len(p) > 3] or [text]

    out = []
    seen = set()
    previous_focus = ""
    for idx, part in enumerate(parts[:max_subquestions]):
        key = normalize_text(part)
        if key in seen:
            continue
        seen.add(key)
        if idx == 0:
            candidate = to_full_question(part)
            if is_low_information_subquestion(candidate):
                candidate = upgrade_low_information_subquestion(part, prev_placeholder=None)
        else:
            prev_placeholder = f"[{chr(ord('B') + idx - 1)}]"
            candidate = build_followup_from_fragment(part, previous_focus, prev_placeholder)
            if not candidate:
                candidate = upgrade_low_information_subquestion(part, prev_placeholder=prev_placeholder)
            candidate = ensure_dependency(candidate, part, idx)
            candidate = canonicalize_dependency_placeholder(candidate, idx)

        previous_focus = extract_focus_phrase(part) or previous_focus
        out.append({
            "question": to_full_question(candidate),
            "sub_id": idx,
            "dep_prev_sub_id": idx - 1 if idx > 0 else None,
            "dep_type": infer_dep_type(candidate, idx),
            "needs_prev_answer": idx > 0,
        })

    if any(low_quality_followup(row["question"]) for row in out[1:]):
        return [{
            "question": to_full_question(text),
            "sub_id": 0,
            "dep_prev_sub_id": None,
            "dep_type": "none",
            "needs_prev_answer": False,
        }]

    if not out:
        out = [{
            "question": to_full_question(text),
            "sub_id": 0,
            "dep_prev_sub_id": None,
            "dep_type": "none",
            "needs_prev_answer": False,
        }]
    return out[:max_subquestions]


def build_decomposed_rows(item: Dict[str, Any], max_subquestions: int = 3) -> List[Dict[str, Any]]:
    parent_id = str(item.get("id", ""))
    rows = []
    for subq in split_question(item.get("question", ""), max_subquestions=max_subquestions):
        row = dict(item)
        row["parent_id"] = parent_id
        row["sub_id"] = subq["sub_id"]
        row["dep_prev_sub_id"] = subq["dep_prev_sub_id"]
        row["dep_type"] = subq["dep_type"]
        row["needs_prev_answer"] = subq["needs_prev_answer"]
        row["question"] = subq["question"]
        row["id"] = parent_id if subq["sub_id"] == 0 and not subq["needs_prev_answer"] else f"{parent_id}_{subq['sub_id']}"
        rows.append(row)
    return rows


def iter_input_rows(input_jsonl: Optional[str], dataset_name: Optional[str]):
    if input_jsonl:
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    if dataset_name:
        for item in load_dataset("rmanluo/RoG-" + dataset_name, split="test"):
            yield dict(item)
        return
    raise ValueError("Provide --input_jsonl or --dataset_name.")


def main():
    parser = argparse.ArgumentParser(description="Decompose KGQA questions into dependency-aware subquestions.")
    parser.add_argument("-d", "--dataset_name", choices=["webqsp", "cwq"], default=None)
    parser.add_argument("--input_jsonl", type=str, default=None)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--max_subquestions", type=int, default=3)
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    out_rows = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for item in iter_input_rows(args.input_jsonl, args.dataset_name):
            total += 1
            rows = build_decomposed_rows(item, max_subquestions=args.max_subquestions)
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_rows += 1

    print(f"Input samples: {total}")
    print(f"Output sub-question samples: {out_rows}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
