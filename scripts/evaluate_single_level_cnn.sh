#!/bin/bash

SIMS=6
EPISODES=100
MODEL_FILES=$(ls logs/test/ | grep CnnPPO_1)

for MOD in $MODEL_FILES; do
  scripts/run_sims.sh $SIMS
  LN=$(echo $MOD | sed 's/^[^_]*_//' | sed 's/_.*$//')
  TS=$(echo $MOD | sed 's/^.*\///')
  python3 evaluate.py --seed 20142012 --episodes $EPISODES --n_envs $SIMS --train_session $TS trained_cnn_ppo $LN
done