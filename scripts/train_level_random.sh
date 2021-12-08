#!/bin/bash

LEVELID=$1
REPS=$2
N_ENV=$2

for i in $(seq $REPS); do
  echo "Resetting envs"
  scripts/run_sims.sh $N_ENV
  sleep 5
  SEED=$(($RANDOM % 1000))
  echo "Training level $LEVELID with seed $SEED"
  python3 train.py --seed $SEED --job_id $$ --n_envs $N_ENV $LEVELID
done
