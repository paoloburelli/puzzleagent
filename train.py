import argparse
import gym
import numpy as np
from numpy.random import seed
from numpy.random import randint
from datetime import datetime
import torch as th

import lg_gym
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib import ppo_mask
from stable_baselines3.common import env_checker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent in a level')
    parser.add_argument('--job_id', dest='job_id', default="local", type=str)
    parser.add_argument('--n_envs', type=int, default=32)
    parser.add_argument('--multi_seed', action='store_true')
    parser.add_argument('--load_model', dest='model_file_name', default=None, type=str)
    parser.add_argument('level_id', type=int)
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    envs_number = args.n_envs
    level_id = args.level_id
    model_file_name = args.model_file_name


    def make_env(n, seed, train=True):
        return lambda: Monitor(
            gym.make('lgenv_flat-v0', train=train, level_id=level_id, seed=seed, port=8080 + n))


    # env_checker.check_env(gym.make('lgenv_flat-v0', train=True, level_id=level_id, seed=1719, port=8080))

    if args.multi_seed:
        seeds_array = randint(100, 10000, envs_number)
        indexed_seeds = [(i, int(seeds_array[i])) for i in range(envs_number)]
        seeds_signature = np.array2string(seeds_array, separator=",", max_line_width=10000)
    else:
        seeds_array = randint(100, 10000, 1)
        indexed_seeds = [(i, int(seeds_array[0])) for i in range(envs_number)]
        seeds_signature = np.array2string(seeds_array, separator=",", max_line_width=10000)

    env = SubprocVecEnv([make_env(i, s, True) for i, s in indexed_seeds])
    eval_env = SubprocVecEnv([make_env(i, s, False) for i, s in indexed_seeds])

    if model_file_name is not None:
        model = ppo_mask.MaskablePPO.load(model_file_name, env)
    else:
        model = ppo_mask.MaskablePPO(
            ppo_mask.policies.MaskableActorCriticCnnPolicy,
            env,
            tensorboard_log="logs/train/",
            n_steps=4096 // envs_number
        )

    log_name = f'{model.__class__.__name__}_{str(level_id).zfill(3)}_{seeds_signature}_{args.job_id}_{timestamp}'

    callback_on_best = StopTrainingOnRewardThreshold(0.9, verbose=1)

    eval_callback = MaskableEvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1,
                                         best_model_save_path=f'logs/test/{log_name}/',
                                         log_path='logs/test/', eval_freq=4096 // envs_number,
                                         deterministic=False, render=False,
                                         n_eval_episodes=40)

    check_callback = CheckpointCallback(4096 // envs_number,
                                        f"logs/train/{log_name}_1/")

    model.learn(100000, callback=[eval_callback, check_callback], tb_log_name=log_name)
    env.close()
