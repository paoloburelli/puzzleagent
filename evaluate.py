import gym
import lg_gym
from policies import Policies
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test random agent on a level range.')
    parser.add_argument('start_level', type=int)
    parser.add_argument('end_level', type=int, default=None, nargs='?')
    parser.add_argument('--job_id', dest='job_id', default="debug", type=str)
    parser.add_argument('--seed', dest='seed', default=None, type=int)
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    reps = 1000

    start_level = args.start_level
    end_level = args.end_level + 1 if args.end_level is not None else start_level + 1

    for level_id in range(args.start_level, end_level):
        play_log_filename = f"logs/eval/gaussian_random_{timestamp}_{args.job_id}_{start_level}-{end_level - 1}.csv"

        env = make_vec_env('lgenv_small-v0',
                           env_kwargs={'level_id': level_id, 'seed': args.seed,
                                       'log_file': play_log_filename},
                           n_envs=5)

        # env = DummyVecEnv([lambda: gym.make('lgenv_small-v0', level_id=level_id, seed=args.seed,
        #                                     log_file=play_log_filename, port=8080, docker_control=True),
        #                    lambda: gym.make('lgenv_small-v0', level_id=level_id, seed=args.seed,
        #                                     log_file=play_log_filename, port=8081, docker_control=True),
        #                    lambda: gym.make('lgenv_small-v0', level_id=level_id, seed=args.seed,
        #                                     log_file=play_log_filename, port=8082, docker_control=True),
        #                    lambda: gym.make('lgenv_small-v0', level_id=level_id, seed=args.seed,
        #                                     log_file=play_log_filename, port=8083, docker_control=True)
        #                    ])

        # policy = Policies.trained_ppo("PPO_61_54740_20210928145611_1", env)
        # policy = Policies.uniform_random(env)
        policy = Policies.gaussian_random(env)

        print(f"Testing {policy.name()} on level {level_id} with seed {args.seed}")
        mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=reps, deterministic=False)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        env.close()
