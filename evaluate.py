import gym
import lg_gym
from policies import Policies
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test random agent on a level range.')
    parser.add_argument('--job_id', dest='job_id', default="debug", type=str)
    parser.add_argument('--seed', dest='seed', default=None, type=int)
    parser.add_argument('--episodes', type=int, default=1000, help="number of episodes to collect")
    parser.add_argument('start_level', type=int)
    parser.add_argument('end_level', type=int, default=None, nargs='?')
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    episodes = args.episodes
    environment = 'lgenv_small-v0'
    n_envs = 4

    start_level = args.start_level
    end_level = args.end_level + 1 if args.end_level is not None else start_level + 1

    for level_id in range(args.start_level, end_level):
        play_log_filename = f"logs/eval/gaussian_random_{timestamp}_{args.job_id}_{start_level}-{end_level - 1}.csv"


        # def make_env(n):
        #     return lambda: Monitor(gym.make(environment, level_id=level_id, seed=args.seed, port=8080,
        #                             log_file=play_log_filename))

        def make_env(n):
            return lambda: Monitor(gym.make(environment, level_id=level_id, seed=args.seed, port=8080 + n,
                                            docker_control=True, log_file=play_log_filename))


        env = SubprocVecEnv([make_env(n) for n in range(n_envs)])

        # policy = Policies.trained_ppo("PPO_61_54788_20210929162609_1", env)
        # policy = Policies.uniform_random(env)
        policy = Policies.gaussian_random(env)

        print(f"Testing {policy.name()} on level {level_id} with seed {args.seed}")
        mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=episodes, deterministic=False)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        env.close()
