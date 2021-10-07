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
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--dockersim', action='store_true')
    parser.add_argument('--subprocsim', action='store_true')
    parser.add_argument('policy', type=str)
    parser.add_argument('start_level', type=int)
    parser.add_argument('end_level', type=int, default=None, nargs='?')
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    episodes = args.episodes
    environment = 'lgenv_small-v0'
    n_envs = args.n_envs
    extra_moves = 100

    start_level = args.start_level
    end_level = args.end_level + 1 if args.end_level is not None else start_level + 1

    play_log_filename = f"logs/eval/{args.policy}_{timestamp}_{args.job_id}_{start_level}-{end_level - 1}.csv"

    for level_id in range(args.start_level, end_level):
        def make_env(n):
            return lambda: Monitor(
                gym.make(environment, dockersim=args.dockersim, subprocsim=args.subprocsim, level_id=level_id,
                         seed=args.seed, log_file=play_log_filename,
                         extra_moves=extra_moves, port=8080 + n))


        env = SubprocVecEnv([make_env(n) for n in range(n_envs)])

        policy = getattr(Policies, args.policy)(env)

        print(f"Testing {policy.policy_name} on level {level_id} with seed {args.seed}")
        mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=episodes, deterministic=False)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        env.close()
