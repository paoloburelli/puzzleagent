#!/bin/bash

PYTHONPATH=.
LC_CTYPE=en_US.UTF-8

FLAGS="--job_id $$ --n_envs $ENVS"

if [ -n "$SEED" ]; then
  FLAGS="$FLAGS --seed $SEED"
fi

killall tensorboard
tensorboard --logdir logs --port=80 --bind_all &
./scripts/run_sims.sh $ENVS
sleep 10
python3 train.py $FLAGS $LEVEL