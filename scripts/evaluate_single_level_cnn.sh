#!/bin/bash

SIMS=4
EPISODES=1000
MODEL_FILES=$(ls logs/test/ | grep MaskableCnnPPO_)
EVAL_FILE_NAME=trained_maskable_ppo
FROM=101

for MOD in $MODEL_FILES; do
  # scripts/run_sims.sh $SIMS
  LN=$(echo $MOD | sed 's/^[^_]*_//' | sed 's/_.*$//')
  TS=$(echo $MOD | sed 's/^.*\///')
  if (($LN >= FROM)); then
    python3 evaluate.py --seed 20142012 --episodes $EPISODES --n_envs $SIMS --train_session $TS $EVAL_FILE_NAME $LN
  fi
done

cat logs/eval/${EVAL_FILE_NAME}_* > logs/eval/${EVAL_FILE_NAME}-merged.csv
rm -f logs/eval/${EVAL_FILE_NAME}_*