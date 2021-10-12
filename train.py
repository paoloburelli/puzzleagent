import argparse
from datetime import datetime

import gym
import lg_gym
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.ppo import PPO
from models.cnnrl import CnnPPO, MaskableCnnPPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent in a level')
    parser.add_argument('--job_id', dest='job_id', default="local", type=str)
    parser.add_argument('--seed', dest='seed', default=None, type=int)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--dockersim', action='store_true')
    parser.add_argument('--subprocsim', action='store_true')
    parser.add_argument('start_level', type=int)
    parser.add_argument('end_level', type=int, default=None, nargs='?')
    # parser.add_argument('level_id', default=1, type=int, nargs='?',
    #                     help="level on which the moves are collected, default is 1")
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    env_n = args.n_envs
    environment = 'lgenv_small-v0'

    if args.end_level is None:
        level_id = args.start_level
    else:
        level_id = (args.start_level, args.end_level)


    def make_env(n):
        return lambda: Monitor(
            gym.make(environment, dockersim=args.dockersim, subprocsim=args.subprocsim, level_id=level_id,
                     seed=args.seed, port=8080 + n, extra_moves=5))


    env = SubprocVecEnv([make_env(i) for i in range(env_n)])

    eval_env = SubprocVecEnv([make_env(i) for i in range(env_n // 2)])

    # model = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log="logs/train/")

    # model = MaskablePPO(MaskableActorCriticPolicy, env=env, verbose=1, tensorboard_log="logs/train/")

    # model = CnnPPO(env=env, verbose=1, tensorboard_log="logs/train/", n_steps=2048)

    model = MaskableCnnPPO(env=env, verbose=1, tensorboard_log="logs/train/", n_steps=2048)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1,
                                 best_model_save_path=f'logs/test/{model.__class__.__name__}_{level_id}_{args.job_id}_{timestamp}_1/',
                                 log_path='logs/test/', eval_freq=4096,
                                 deterministic=False, render=False, n_eval_episodes=10)

    model.learn(100000000, callback=eval_callback,
                tb_log_name=f'{model.__class__.__name__}_{level_id}_{args.job_id}_{timestamp}')
    model.save(f"models/saved/{model.__class__.__name__}_{level_id}_{args.job_id}_{timestamp}.zip")
    env.close()
