#!/bin/bash

BEGIN=$1
END=$2
N_ENV=$3

for i in $(seq $BEGIN $END); do
  echo "Resetting envs"
  scripts/run_sims.sh $N_ENV
  sleep 5
  echo "Training level $i"
  python3 train.py --seed 20142012 --job_id $$ --n_envs $N_ENV $i
done
