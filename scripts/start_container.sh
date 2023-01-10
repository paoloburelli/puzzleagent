#!/bin/bash

PYTHONPATH=.
LC_CTYPE=en_US.UTF-8

FLAGS="--job_id $$ --n_envs $ENVS"

if [ -n "$SEED" ]; then
  FLAGS="$FLAGS --seed $SEED"
fi

source venv/bin/activate
tensorboard --logdir logs --bind_all &
./scripts/run_sims.sh $ENVS
sleep 10

for ((level_id = $FIRST; level_id <= $LAST; level_id++)); do
  for ((i = 1; i <= $REPS; i++)); do
    python3 train.py $FLAGS $level_id
  done
done