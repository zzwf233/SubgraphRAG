#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL_PATH="${MODEL_PATH:-/public/xmz/SubgraphRAG/Llama-3.1-8B-Instruct}"
CUDA_DEVICES="${CUDA_DEVICES:-2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SUMMARY_FILE="${SUMMARY_FILE:-./results/rag_safety_table3_subgraphrag_poisoned_retrieval_summary.csv}"
PROMPT_MODE="${PROMPT_MODE:-scored_100}"
LLM_MODE="${LLM_MODE:-sys_icl_dc_repro}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.14}"
RAND_BUDGET="${RAND_BUDGET:-20}"
SEED="${SEED:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SCORED_TRIPLES="./scored_triples/webqsp_240912_unidir_test.pth"
RAND_SCORED_TRIPLES="./scored_triples/webqsp_240912_unidir_test-rag_safety_rand${RAND_BUDGET}_seed${SEED}.pth"
ROG_PREDICTIONS="./results/KGQA/webqsp/RoG/test/results_gen_rule_path_RoG-webqsp_RoG_test_predictions_3_False_jsonl/predictions.jsonl"
MODEL_BASENAME="$(basename "$MODEL_PATH")"
CLEAN_PRED="./results/KGQA/webqsp/SubgraphRAG/${MODEL_BASENAME}/${PROMPT_MODE}-${LLM_MODE}-${FREQUENCY_PENALTY}-thres_0.0-test-predictions.jsonl"
CLEAN_RESUME="./results/KGQA/webqsp/SubgraphRAG/${MODEL_BASENAME}/${PROMPT_MODE}-${LLM_MODE}-${FREQUENCY_PENALTY}-thres_0.0-test-predictions-resume.jsonl"
RAND_PRED="./results/KGQA/webqsp/SubgraphRAG/${MODEL_BASENAME}/${PROMPT_MODE}-${LLM_MODE}-${FREQUENCY_PENALTY}-thres_0.0-test-safety_rand${RAND_BUDGET}_seed${SEED}-predictions.jsonl"
RAND_RESUME="./results/KGQA/webqsp/SubgraphRAG/${MODEL_BASENAME}/${PROMPT_MODE}-${LLM_MODE}-${FREQUENCY_PENALTY}-thres_0.0-test-safety_rand${RAND_BUDGET}_seed${SEED}-predictions-resume.jsonl"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model directory: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$ROG_PREDICTIONS" ]]; then
  echo "Missing WebQSP RoG predictions: $ROG_PREDICTIONS" >&2
  exit 1
fi

if [[ ! -f "$SCORED_TRIPLES" ]]; then
  echo "Missing WebQSP scored triples: $SCORED_TRIPLES" >&2
  exit 1
fi

COMMON_ARGS=(
  -d webqsp
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

if [[ "$FORCE_RERUN" == "1" ]]; then
  echo "FORCE_RERUN=1: removing previous WebQSP clean/rand predictions and poisoned scored triples..."
  rm -f "$CLEAN_PRED" "$CLEAN_RESUME" "$RAND_PRED" "$RAND_RESUME" "$RAND_SCORED_TRIPLES"
fi

echo "Running WebQSP clean..."
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python main.py \
  "${COMMON_ARGS[@]}" \
  -p "$SCORED_TRIPLES" \
  --safety_attack clean

if [[ ! -f "$RAND_SCORED_TRIPLES" ]]; then
  echo "Preparing RAG Safety Rand poisoned scored triples..."
  python prepare_rag_safety_rand.py \
    -d webqsp \
    --input "$SCORED_TRIPLES" \
    --output "$RAND_SCORED_TRIPLES" \
    --budget "$RAND_BUDGET" \
    --seed "$SEED"
fi

echo "Running WebQSP rand with budget ${RAND_BUDGET}..."
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python main.py \
  "${COMMON_ARGS[@]}" \
  -p "$RAND_SCORED_TRIPLES" \
  --safety_attack rand \
  --safety_rand_budget "$RAND_BUDGET"

echo "Done. Summary written to: $SUMMARY_FILE"
