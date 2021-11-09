#!/bin/bash

sbatch --export=START_LEVEL=101,END_LEVEL=110,RANDOM_SEED=20142012,EPISODES=1000,POLICY=masked_uniform_random -w cn3 evaluate.job
sbatch --export=START_LEVEL=111,END_LEVEL=120,RANDOM_SEED=20142012,EPISODES=1000,POLICY=masked_uniform_random -w cn3 evaluate.job
sbatch --export=START_LEVEL=121,END_LEVEL=130,RANDOM_SEED=20142012,EPISODES=1000,POLICY=masked_uniform_random -w cn4 evaluate.job
sbatch --export=START_LEVEL=131,END_LEVEL=140,RANDOM_SEED=20142012,EPISODES=1000,POLICY=masked_uniform_random -w cn4 evaluate.job
sbatch --export=START_LEVEL=141,END_LEVEL=150,RANDOM_SEED=20142012,EPISODES=1000,POLICY=masked_uniform_random -w cn4 evaluate.job
sbatch --export=START_LEVEL=151,END_LEVEL=160,RANDOM_SEED=20142012,EPISODES=1000,POLICY=masked_uniform_random -w cn4 evaluate.job
sbatch --export=START_LEVEL=161,END_LEVEL=170,RANDOM_SEED=20142012,EPISODES=1000,POLICY=masked_uniform_random -w cn5 evaluate.job
sbatch --export=START_LEVEL=171,END_LEVEL=180,RANDOM_SEED=20142012,EPISODES=1000,POLICY=masked_uniform_random -w cn5 evaluate.job
sbatch --export=START_LEVEL=181,END_LEVEL=190,RANDOM_SEED=20142012,EPISODES=1000,POLICY=masked_uniform_random -w cn5 evaluate.job
sbatch --export=START_LEVEL=191,END_LEVEL=200,RANDOM_SEED=20142012,EPISODES=1000,POLICY=masked_uniform_random -w cn5 evaluate.job
