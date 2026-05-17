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
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.16}"
RAND_BUDGET="${RAND_BUDGET:-20}"
SEED="${SEED:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SCORED_TRIPLES="./scored_triples/cwq_240907_unidir_test.pth"
RAND_SCORED_TRIPLES="./scored_triples/cwq_240907_unidir_test-rag_safety_rand${RAND_BUDGET}_seed${SEED}.pth"
ROG_PREDICTIONS="./results/KGQA/cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/predictions.jsonl"
MODEL_BASENAME="$(basename "$MODEL_PATH")"
CLEAN_PRED="./results/KGQA/cwq/SubgraphRAG/${MODEL_BASENAME}/${PROMPT_MODE}-${LLM_MODE}-${FREQUENCY_PENALTY}-thres_0.0-test-predictions.jsonl"
CLEAN_RESUME="./results/KGQA/cwq/SubgraphRAG/${MODEL_BASENAME}/${PROMPT_MODE}-${LLM_MODE}-${FREQUENCY_PENALTY}-thres_0.0-test-predictions-resume.jsonl"
RAND_PRED="./results/KGQA/cwq/SubgraphRAG/${MODEL_BASENAME}/${PROMPT_MODE}-${LLM_MODE}-${FREQUENCY_PENALTY}-thres_0.0-test-safety_rand${RAND_BUDGET}_seed${SEED}-predictions.jsonl"
RAND_RESUME="./results/KGQA/cwq/SubgraphRAG/${MODEL_BASENAME}/${PROMPT_MODE}-${LLM_MODE}-${FREQUENCY_PENALTY}-thres_0.0-test-safety_rand${RAND_BUDGET}_seed${SEED}-predictions-resume.jsonl"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model directory: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$ROG_PREDICTIONS" ]]; then
  echo "Missing CWQ RoG predictions: $ROG_PREDICTIONS" >&2
  exit 1
fi

if [[ ! -f "$SCORED_TRIPLES" ]]; then
  echo "Missing CWQ scored triples: $SCORED_TRIPLES" >&2
  echo "Download the official preprocessed results first:" >&2
  echo "  huggingface-cli download siqim311/SubgraphRAG --revision main --local-dir ./" >&2
  exit 1
fi

if [[ "$FORCE_RERUN" == "1" ]]; then
  echo "FORCE_RERUN=1: removing previous CWQ clean/rand predictions and poisoned scored triples..."
  rm -f "$CLEAN_PRED" "$CLEAN_RESUME" "$RAND_PRED" "$RAND_RESUME" "$RAND_SCORED_TRIPLES"
fi

if [[ "$FORCE_RERUN" != "1" && -f "$CLEAN_PRED" ]]; then
  REUSE_CLEAN=(--reuse_predictions)
else
  REUSE_CLEAN=()
fi

COMMON_ARGS=(
  -d cwq
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

echo "Running CWQ clean..."
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python main.py \
  "${COMMON_ARGS[@]}" \
  -p "$SCORED_TRIPLES" \
  --safety_attack clean \
  "${REUSE_CLEAN[@]}"

if [[ ! -f "$RAND_SCORED_TRIPLES" ]]; then
  echo "Preparing RAG Safety Rand poisoned scored triples..."
  python prepare_rag_safety_rand.py \
    -d cwq \
    --input "$SCORED_TRIPLES" \
    --output "$RAND_SCORED_TRIPLES" \
    --budget "$RAND_BUDGET" \
    --seed "$SEED"
fi

echo "Running CWQ rand with budget ${RAND_BUDGET}..."
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python main.py \
  "${COMMON_ARGS[@]}" \
  -p "$RAND_SCORED_TRIPLES" \
  --safety_attack rand \
  --safety_rand_budget "$RAND_BUDGET"

echo "Done. Summary written to: $SUMMARY_FILE"
