import argparse
from numpy.random import seed
from numpy.random import randint
from datetime import datetime

import gym
import lg_gym
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, CheckpointCallback, BaseCallback
from utils.callbacks import StopTrainingOnEpisodeLengthThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import safe_mean
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from models.cnnrl import MaskableCnnPPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent in a level')
    parser.add_argument('--job_id', dest='job_id', default="local", type=str)
    parser.add_argument('--n_envs', type=int, default=32)
    parser.add_argument('--load_model', dest='model_file_name', default=None, type=str)
    parser.add_argument('level_id', type=int)
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    envs_number = args.n_envs
    environment = 'lgenv_full-v0'
    level_id = args.level_id


    def make_env(n, seed, train=True):
        return lambda: Monitor(
            gym.make(environment, train=train, level_id=level_id, seed=seed, port=8080 + n))


    seed(1719)
    seeds_array = randint(100, 10000, envs_number)
    indexed_seeds = [(i, int(seeds_array[i])) for i in range(envs_number)]

    env = SubprocVecEnv([make_env(i, s) for i, s in indexed_seeds])
    eval_env = SubprocVecEnv([make_env(i, s, False) for i, s in indexed_seeds])

    if args.model_file_name is not None:
        model = MaskableCnnPPO.load(args.model_file_name, env)
    else:
        model = MaskableCnnPPO(env=env, verbose=1, tensorboard_log="logs/train/", n_steps=4096 // envs_number)

    level_name = level_id if type(level_id) is not list else f"({args.start_level}-{args.end_level})"

    # callback_on_best = StopTrainingOnEpisodeLengthThreshold(episode_length_threshold=25, verbose=1)
    callback_on_best = StopTrainingOnRewardThreshold(0.7, verbose=1)
    eval_callback = MaskableEvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1,
                                         best_model_save_path=f'logs/test/{model.__class__.__name__}_{level_name}_{seeds_array}_{args.job_id}_{timestamp}_1/',
                                         log_path='logs/test/', eval_freq=4096 // envs_number,
                                         deterministic=False, render=False,
                                         n_eval_episodes=max(100 // envs_number, 10))

    check_callback = CheckpointCallback(4096 // envs_number,
                                        f"logs/train/{model.__class__.__name__}_{level_name}_{seeds_array}_{args.job_id}_{timestamp}_1/")

    model.learn(5000000, callback=[eval_callback, check_callback],
                tb_log_name=f'{model.__class__.__name__}_{level_name}_{seeds_array}_{args.job_id}_{timestamp}')
    env.close()
