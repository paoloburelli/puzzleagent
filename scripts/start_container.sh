#!/bin/bash

PYTHONPATH=.
LC_CTYPE=en_US.UTF-8

tensorboard --logdir logs --port=80 --bind_all &
./scripts/run_sims.sh $ENVS
python3 train.py --seed $SEED --job_id $$ --n_envs $ENVS $LEVEL