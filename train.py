import argparse
import random
from datetime import datetime

import gym
import lg_gym
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.ppo import PPO
from models.cnnrl import CnnPPO, MaskableCnnPPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent in a level')
    parser.add_argument('--job_id', dest='job_id', default="local", type=str)
    parser.add_argument('--seed', dest='seed', default=None, type=int)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--dockersim', action='store_true')
    parser.add_argument('--subprocsim', action='store_true')
    parser.add_argument('--random_order', action='store_true')
    parser.add_argument('--load_model', dest='model_file_name', default=None, type=str)
    parser.add_argument('start_level', type=int)
    parser.add_argument('end_level', type=int, default=None, nargs='?')
    parser.add_argument('episodes_per_level', type=int, default=100, nargs='?')
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    env_n = args.n_envs
    environment = 'lgenv_full-v0'

    if args.end_level is None:
        level_id = args.start_level  # Just one level
    else:
        sequence = list(range(args.start_level, args.end_level))
        if args.random_order:
            random.shuffle(sequence)
        level_id = [{'level_id': l, 'episodes': args.episodes_per_level} for l in sequence]


    def make_env(n):
        return lambda: Monitor(
            gym.make(environment, dockersim=args.dockersim, subprocsim=args.subprocsim, level_id=level_id,
                     seed=args.seed, port=8080 + n, extra_moves=5))


    def make_eval_env(n):
        if type(level_id) is list:
            env_level_id = [{'level_id': l['level_id'], 'episodes': 1} for l in level_id]
        else:
            env_level_id = level_id
        return lambda: Monitor(
            gym.make(environment, train=False, dockersim=args.dockersim, subprocsim=args.subprocsim,
                     level_id=env_level_id,
                     seed=args.seed, port=8080 + n))


    env = SubprocVecEnv([make_env(i) for i in range(env_n)])

    eval_env = SubprocVecEnv([make_eval_env(i) for i in range(env_n)])

    if args.model_file_name is not None:
        model = MaskableCnnPPO.load(args.model_file_name, env)

    else:
        # model = PPO(policy="MlpPolicy", policy_kwargs={'net_arch': [128, 64]}, env=env, verbose=1,
        # tensorboard_log = "logs/train/")

        # model = MaskablePPO(MaskableActorCriticPolicy, env=env, verbose=1, tensorboard_log="logs/train/")

        # model = CnnPPO(env=env, verbose=1, tensorboard_log="logs/train/")

        model = MaskableCnnPPO(env=env, verbose=1, tensorboard_log="logs/train/", n_steps=2048)

    level_name = level_id if type(level_id) is not list else f"({args.start_level}-{args.end_level})"

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1,
                                 best_model_save_path=f'logs/test/{model.__class__.__name__}_{level_name}_{args.job_id}_{timestamp}_1/',
                                 log_path='logs/test/', eval_freq=4096 * env_n,
                                 deterministic=False, render=False,
                                 n_eval_episodes=(len(level_id) if type(level_id) is list else 10) * env_n)

    check_callback = CheckpointCallback(4096 * env_n,
                                        f"logs/train/{model.__class__.__name__}_{level_name}_{args.job_id}_{timestamp}_1/")

    model.learn(2000000, callback=[check_callback, eval_callback],
                tb_log_name=f'{model.__class__.__name__}_{level_name}_{args.job_id}_{timestamp}')
    # model.save(f"models/saved/{model.__class__.__name__}_{level_name}_{args.job_id}_{timestamp}.zip")
    env.close()
