import gym
import lg_gym
from stable_baselines3.common.env_util import make_vec_env, Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import logging
import os
from imitation.data.rollout import generate_trajectories, min_episodes
from imitation.data.types import TrajectoryWithRew, save
from policies import Policies
import numpy as np
import argparse
from datetime import datetime

environment = 'lgenv_small-v0'
n_env = 4


def filter_trajectory(trajectory):
    filtered = list(zip(*list(filter(lambda x: x[2]['click_successful'] or 'terminal_observation' in x[2],
                                     zip(trajectory.rews, trajectory.acts, trajectory.infos, trajectory.obs[1:])))))
    rews = np.array(filtered[0])
    acts = np.array(filtered[1])
    infos = np.array(filtered[2])
    obs = np.append([trajectory.obs[0]], np.array(filtered[3]), axis=0)

    return TrajectoryWithRew(obs, acts, infos, rews)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Collect valid moves rollouts from a level.')
    parser.add_argument('--job_id', dest='job_id', default="debug", type=str, help="slurm job id, default is debug")
    parser.add_argument('--seed', dest='seed', default=None, type=int)
    parser.add_argument('--best_fraction', dest='best_fraction', default=0.5, type=float)
    parser.add_argument('level_id', type=int, nargs='?', help="level on which the moves are collected")
    parser.add_argument('episodes', type=int, nargs='?', default=1000, help="number of episodes to collect")
    args = parser.parse_args()

    logging.basicConfig(format=f'{args.job_id}:%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.INFO)

    best_fraction = args.best_fraction
    episodes = args.episodes

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    if not os.path.exists('logs/rollouts'):
        os.makedirs('logs/rollouts')

    playtraces_filename = f"logs/rollouts/level_{args.level_id}_{args.seed}_{environment}_{timestamp}_{args.job_id}.pkl"
    log_filename = f"logs/rollouts/level_{args.level_id}_{args.seed}_{environment}_{timestamp}_{args.job_id}.csv"


    def make_env(n):
        return lambda: gym.make(environment, level_id=args.level_id, seed=args.seed, port=8080 + n, docker_control=True,
                                log_file=log_filename)


    env = SubprocVecEnv([make_env(n) for n in range(n_env)])

    episodes_to_run = int(episodes / best_fraction)
    trajectories = generate_trajectories(Policies.gaussian_random(env), env,
                                         min_episodes(episodes_to_run))

    rank = np.flip(np.argsort([sum(trajectory.rews) for trajectory in trajectories]))

    best_trajectories = [filter_trajectory(trajectories[i]) for i in rank[:episodes]]

    save(playtraces_filename, best_trajectories)

    env.env_method("close")
