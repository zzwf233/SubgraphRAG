#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET="${DATASET:-cwq}"
MODEL_PATH="${MODEL_PATH:-/public/xmz/SubgraphRAG/Llama-3.1-8B-Instruct}"
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-2,3}}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_COUNT="$(python - <<'PY'
import os
val = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if not val:
    print(0)
else:
    print(len([x for x in val.split(",") if x.strip()]))
PY
)"
else
  GPU_COUNT="$(python - <<'PY'
val = "2,3"
print(len([x for x in val.split(",") if x.strip()]))
PY
)"
fi
DEFAULT_TP_SIZE=1
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TP_SIZE}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SUMMARY_FILE="${SUMMARY_FILE:-./results/rag_safety_table3_subgraphrag_poisoned_retrieval_summary.csv}"
PAPER_METRICS_SUMMARY="${PAPER_METRICS_SUMMARY:-./results/rag_safety_table3_subgraphrag_ours_paper_metrics.csv}"
PROMPT_MODE="${PROMPT_MODE:-scored_100}"
LLM_MODE="${LLM_MODE:-sys_icl_dc_repro}"
SEED="${SEED:-0}"
OURS_BUDGET="${OURS_BUDGET:-20}"
ATTACK_TARGET_COUNT="${ATTACK_TARGET_COUNT:-5}"
TRIPLES_PER_TARGET="${TRIPLES_PER_TARGET:-4}"
GOLD_DEMOTION="${GOLD_DEMOTION:-0.35}"
RULE_PATH_DEMOTION="${RULE_PATH_DEMOTION:-0.75}"
PRIMARY_TARGET_LIMIT="${PRIMARY_TARGET_LIMIT:-2}"
AUXILIARY_POISON_BOOST="${AUXILIARY_POISON_BOOST:-0.10}"
PRIMARY_REPEAT="${PRIMARY_REPEAT:-4}"
AUXILIARY_REPEAT="${AUXILIARY_REPEAT:-2}"
DEPENDENCY_REPEAT_BONUS="${DEPENDENCY_REPEAT_BONUS:-2}"
MAX_SUBQUESTIONS="${MAX_SUBQUESTIONS:-3}"
FORCE_RERUN="${FORCE_RERUN:-0}"
DECOMPOSE_ONLY="${DECOMPOSE_ONLY:-0}"
START_FROM="${START_FROM:-1}"
PLANNER_MODEL="${PLANNER_MODEL:-${RAG_SAFETY_PLANNER_MODEL:-deepseek-ai/DeepSeek-V3.2}}"
PLANNER_API_BASE="${PLANNER_API_BASE:-${RAG_SAFETY_PLANNER_API_BASE:-https://api.siliconflow.cn/v1}}"
PLANNER_API_KEY="${PLANNER_API_KEY:-${RAG_SAFETY_PLANNER_API_KEY:-}}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

case "$DATASET" in
  cwq)
    FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.16}"
    SCORED_TRIPLES="./scored_triples/cwq_240907_unidir_test.pth"
    ROG_PREDICTIONS="./results/KGQA/cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/predictions.jsonl"
    ;;
  webqsp)
    FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.14}"
    SCORED_TRIPLES="./scored_triples/webqsp_240912_unidir_test.pth"
    ROG_PREDICTIONS="./results/KGQA/webqsp/RoG/test/results_gen_rule_path_RoG-webqsp_RoG_test_predictions_3_False_jsonl/predictions.jsonl"
    ;;
  *)
    echo "Unsupported DATASET: $DATASET" >&2
    exit 1
    ;;
esac

MODEL_BASENAME="$(basename "$MODEL_PATH")"
MODEL_PATH_LOWER="$(printf '%s' "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
IS_API_MODEL=0
if [[ "$MODEL_PATH_LOWER" == *gpt* || "$MODEL_PATH_LOWER" == *deepseek* || "$MODEL_PATH_LOWER" == *openai-compatible* ]]; then
  IS_API_MODEL=1
fi
DECOMPOSED_FILE="./results/rag_safety_ours/${DATASET}_test_decomposed.jsonl"
POISON_META_FILE="./results/rag_safety_ours/${DATASET}_test_poison_meta.jsonl"
OURS_SCORED_TRIPLES="./scored_triples/${DATASET}_rag_safety_ours${OURS_BUDGET}_seed${SEED}.pth"
OURS_PRED="./results/KGQA/${DATASET}/SubgraphRAG/${MODEL_BASENAME}/${PROMPT_MODE}-${LLM_MODE}-${FREQUENCY_PENALTY}-thres_0.0-test-safety_ours${OURS_BUDGET}_seed${SEED}-predictions.jsonl"
OURS_RESUME="./results/KGQA/${DATASET}/SubgraphRAG/${MODEL_BASENAME}/${PROMPT_MODE}-${LLM_MODE}-${FREQUENCY_PENALTY}-thres_0.0-test-safety_ours${OURS_BUDGET}_seed${SEED}-predictions-resume.jsonl"

if [[ "$IS_API_MODEL" == "1" ]]; then
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${API_BASE:-https://api.siliconflow.cn/v1}}"
  if [[ -z "${OPENAI_API_KEY:-${SILICONFLOW_API_KEY:-${DEEPSEEK_API_KEY:-}}}" ]]; then
    read -rsp "Enter API key for ${OPENAI_BASE_URL}: " OPENAI_API_KEY
    echo
    export OPENAI_API_KEY
  fi
elif [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model directory: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$SCORED_TRIPLES" ]]; then
  echo "Missing scored triples: $SCORED_TRIPLES" >&2
  exit 1
fi

if [[ ! -f "$ROG_PREDICTIONS" ]]; then
  echo "Missing RoG predictions: $ROG_PREDICTIONS" >&2
  exit 1
fi

if [[ ! "$START_FROM" =~ ^[1-4]$ ]]; then
  echo "START_FROM must be one of 1, 2, 3, 4" >&2
  exit 1
fi

if [[ "$FORCE_RERUN" == "1" ]]; then
  echo "FORCE_RERUN=1: removing previous ours artifacts..."
  if [[ "$START_FROM" -le 1 ]]; then
    rm -f "$DECOMPOSED_FILE" "$POISON_META_FILE" "$OURS_SCORED_TRIPLES" "$OURS_PRED" "$OURS_RESUME"
  elif [[ "$START_FROM" -le 2 ]]; then
    rm -f "$POISON_META_FILE" "$OURS_SCORED_TRIPLES" "$OURS_PRED" "$OURS_RESUME"
  else
    rm -f "$OURS_PRED" "$OURS_RESUME"
  fi
fi

mkdir -p "$(dirname "$DECOMPOSED_FILE")"

if [[ "$START_FROM" -le 1 ]]; then
  echo "Step 1/4: Decomposing ${DATASET} questions..."
  python decompose_rag_safety_subquestions.py \
    -d "$DATASET" \
    --output_file "$DECOMPOSED_FILE" \
    --max_subquestions "$MAX_SUBQUESTIONS"
elif [[ ! -f "$DECOMPOSED_FILE" ]]; then
  echo "Missing decomposition file for START_FROM=$START_FROM: $DECOMPOSED_FILE" >&2
  exit 1
fi

if [[ "$DECOMPOSE_ONLY" == "1" ]]; then
  echo "Decomposition finished: $DECOMPOSED_FILE"
  exit 0
fi

if [[ "$START_FROM" -le 2 && ! -f "$OURS_SCORED_TRIPLES" ]]; then
  echo "Step 2/4: Building ours poisoned scored triples..."
  PREP_ARGS=(
    -d "$DATASET"
    --input "$SCORED_TRIPLES"
    --output "$OURS_SCORED_TRIPLES"
    --decompose_output "$DECOMPOSED_FILE"
    --poison_meta_output "$POISON_META_FILE"
    --budget "$OURS_BUDGET"
    --attack_target_count "$ATTACK_TARGET_COUNT"
    --triples_per_target "$TRIPLES_PER_TARGET"
    --gold_demotion "$GOLD_DEMOTION"
    --rule_path_demotion "$RULE_PATH_DEMOTION"
    --primary_target_limit "$PRIMARY_TARGET_LIMIT"
    --auxiliary_poison_boost "$AUXILIARY_POISON_BOOST"
    --primary_repeat "$PRIMARY_REPEAT"
    --auxiliary_repeat "$AUXILIARY_REPEAT"
    --dependency_repeat_bonus "$DEPENDENCY_REPEAT_BONUS"
    --max_subquestions "$MAX_SUBQUESTIONS"
    --seed "$SEED"
  )
  if [[ -n "$PLANNER_MODEL" ]]; then
    PREP_ARGS+=(--planner_model "$PLANNER_MODEL")
  fi
  if [[ -n "$PLANNER_API_BASE" ]]; then
    PREP_ARGS+=(--planner_api_base "$PLANNER_API_BASE")
  fi
  if [[ -n "$PLANNER_API_KEY" ]]; then
    PREP_ARGS+=(--planner_api_key "$PLANNER_API_KEY")
  fi
  python poison_rag_safety_ours.py "${PREP_ARGS[@]}"
elif [[ "$START_FROM" -ge 3 && ! -f "$OURS_SCORED_TRIPLES" ]]; then
  echo "Missing poisoned scored triples for START_FROM=$START_FROM: $OURS_SCORED_TRIPLES" >&2
  exit 1
fi

if [[ "$START_FROM" -le 3 ]]; then
echo "Step 3/4: Running SubgraphRAG ours inference..."
COMMON_ARGS=(
  -d "$DATASET"
  --prompt_mode "$PROMPT_MODE"
  --model_name "$MODEL_PATH"
  --llm_mode "$LLM_MODE"
  --frequency_penalty "$FREQUENCY_PENALTY"
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
  --max_model_len "$MAX_MODEL_LEN"
  --seed "$SEED"
  --summary_file "$SUMMARY_FILE"
)

export WANDB_MODE="${WANDB_MODE:-offline}"
# Keep accepting legacy VLLM_* shell inputs, but forward repo-scoped names so
# vLLM itself does not warn about unknown environment variables.
export SUBGRAPHRAG_VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION"
export SUBGRAPHRAG_VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS"
export SUBGRAPHRAG_VLLM_ENFORCE_EAGER="$VLLM_ENFORCE_EAGER"
unset VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_SEQS VLLM_ENFORCE_EAGER
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python main.py \
  "${COMMON_ARGS[@]}" \
  -p "$OURS_SCORED_TRIPLES" \
  --safety_attack ours \
  --safety_ours_budget "$OURS_BUDGET"
elif [[ ! -f "$OURS_PRED" ]]; then
  echo "Missing prediction file for START_FROM=$START_FROM: $OURS_PRED" >&2
  exit 1
fi

if [[ "$START_FROM" -le 4 ]]; then
  if [[ ! -f "$POISON_META_FILE" ]]; then
    echo "Missing poison meta file for evaluation: $POISON_META_FILE" >&2
    exit 1
  fi
  if [[ ! -f "$OURS_PRED" ]]; then
    echo "Missing prediction file for evaluation: $OURS_PRED" >&2
    exit 1
  fi
  echo "Step 4/4: Evaluating paper metrics..."
  python eval_rag_safety_ours.py \
    --predict_file "$OURS_PRED" \
    --poison_file "$POISON_META_FILE" \
    --dataset "$DATASET" \
    --model "$MODEL_BASENAME" \
    --prompt_mode "$PROMPT_MODE" \
    --llm_mode "$LLM_MODE" \
    --frequency_penalty "$FREQUENCY_PENALTY" \
    --seed "$SEED" \
    --attack_budget "$OURS_BUDGET" \
    --summary_file "$PAPER_METRICS_SUMMARY"
fi

echo "Done."
echo "Decomposed file: $DECOMPOSED_FILE"
echo "Poison meta file: $POISON_META_FILE"
echo "Poisoned scores: $OURS_SCORED_TRIPLES"
echo "Prediction file: $OURS_PRED"
echo "Paper summary file: $PAPER_METRICS_SUMMARY"
