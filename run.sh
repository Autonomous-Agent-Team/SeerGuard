#!/bin/bash

# Set variables
MODEL="qwen3-vl-8b-instruct"
MODE="guard"
PROMT="basic"
GUARD="SeerGuard"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_NAME="${MODEL}_${MODE}_${PROMT}_${GUARD}_${TIMESTAMP}.log"

# Execute and log
python experiment/evaluate_all_task.py \
  --agent_model $MODEL \
  --mode $MODE \
  --prompt_mode $PROMT \
  --guard_model $GUARD
  2>&1 | tee "$LOG_NAME"