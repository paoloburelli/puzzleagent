import gym
import lg_gym
from stable_baselines3.ppo import PPO
from datetime import datetime
import argparse
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent in a level')
    parser.add_argument('--job_id', dest='job_id', default="local", type=str)
    parser.add_argument('--seed', dest='seed', default=None, type=int)
    parser.add_argument('level_id', default=1, type=int, nargs='?',
                        help="level on which the moves are collected, default is 1")
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    env_n = 5
    environment = 'lgenv_small-v0'

    env = SubprocVecEnv([lambda: Monitor(
        gym.make(environment, level_id=args.level_id, seed=args.seed,
                 extra_moves=5)) for i in range(env_n)])

    eval_env = Monitor(gym.make(environment, level_id=args.level_id, seed=args.seed))

    model = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log="logs/train/", n_steps=2000)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1,
                                 best_model_save_path=f'logs/test/{model.__class__.__name__}_{args.level_id}_{args.job_id}_{timestamp}_1/',
                                 log_path='logs/test/', eval_freq=2000,
                                 deterministic=False, render=False, n_eval_episodes=10)

    model.learn(100000000, callback=eval_callback,
                tb_log_name=f'{model.__class__.__name__}_{args.level_id}_{args.job_id}_{timestamp}')
    model.save(f"models/saved/{model.__class__.__name__}_{args.level_id}_{args.job_id}_{timestamp}.zip")
    env.close()

    # from models.cnnrl import CustomCnnPPO, CustomCnnA2C
    # from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
    # env = make_vec_env('lgenvsmall-v0', env_kwargs={"host": "http://localhost:8080", "level_id": args.level_id},
    #                    n_envs=2)
    # env = make_vec_env('lgenvsmall-v0', env_kwargs={"host": "http://localhost:8080", "level_id": args.level_id})
    # eval_env = VecTransposeImage(
    #     make_vec_env('lgenv-v0', env_kwargs={"host": "http://localhost:8080", "level_id": args.level_id}, n_envs=2))
    # eval_env = make_vec_env('lgenvsmall-v0', env_kwargs={"host": "http://localhost:8080", "level_id": args.level_id},
    #                         n_envs=2)
    # eval_env = make_vec_env('lgenvsmall-v0', env_kwargs={"host": "http://localhost:8080", "level_id": args.level_id})
    # model = CustomCnnPPO(env=env, verbose=1, tensorboard_log="logs/train/", n_steps=envs[0].clicks_remaining,
    #                 batch_size=envs[0].clicks_remaining // 10)
    # model = CustomCnnA2C(env=env, verbose=1, tensorboard_log="logs/train/", n_steps=envs[0].clicks_remaining,
    #                 batch_size=envs[0].clicks_remaining // 10)
    # model = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log="logs/train/", n_steps=envs[0].clicks_remaining,
    #             batch_size=envs[0].clicks_remaining // 10)
