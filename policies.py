import math

import numpy as np
from stable_baselines3.ppo import PPO
from models.cnnrl import CnnPPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
import random
from scipy.stats import truncnorm
from numpy.random import choice


class Policies:
    class __RandomGaussianPolicy:

        def __init__(self, env):
            self.sample_size = 2000000

            width = env.get_attr("board_width")[0]
            height = env.get_attr("board_height")[0]

            X = truncnorm(a=-2, b=2, scale=width // 4).rvs(size=self.sample_size)
            self.distribution_x = X.round().astype(int).tolist()

            Y = truncnorm(a=-2, b=2, scale=height // 4).rvs(size=self.sample_size)
            self.distribution_y = Y.round().astype(int).tolist()

            self.env = env
            self.policy_name = "gaussian_random"

        def __action(self, env_index):
            random_index = random.randint(0, self.sample_size - 1)
            x = self.distribution_x[random_index]
            y = self.distribution_y[random_index]
            return self.env.env_method('click_to_action', x, y, indices=env_index)[0]

        def predict(self, *args, **kwargs):
            actions = [self.__action(n) for n in range(self.env.num_envs)]
            return actions, None

    class __Greedy:
        def __init__(self, env, simulate=False, budget=np.Inf):
            self.env = env
            self.simulate = simulate
            if simulate:
                self.policy_name = "greedy_sim"
            else:
                self.policy_name = "greedy"

            self.budget = budget

        def predict(self, obs, state, deterministic):
            valid_actions = self.env.get_attr("valid_action_list")
            actions = [self.__single_prediction(o, am, i, deterministic) for am, o, i in
                       zip(valid_actions, obs, range(len(valid_actions)))]
            return actions, None

        def __score(self, obs, action, index):
            if self.simulate:
                return self.env.env_method("simulate_click", action, indices=index)[0]
            else:
                x, y = self.env.env_method('action_to_board_index', action, indices=index)[0]
                return 0.1 + obs[x, y, 1] * 100 + math.ceil(obs[x, y, 2])

        def __single_prediction(self, obs, valid_actions_list, index, deterministric):
            val = valid_actions_list
            random.shuffle(val)
            val = val[:min(self.budget, len(val))]

            evaluated_actions = [{'action': a, 'score': pow(self.__score(obs, a, index), 2)} for a in val]

            if len(evaluated_actions) == 0:
                return self.env.get_attr('action_space')[index].sample()

            if deterministric:
                evaluated_actions.sort(key=lambda m: m['score'])
                return evaluated_actions[0]['action']
            else:
                total_score = sum([c['score'] for c in evaluated_actions])
                probability = [c['score'] / total_score for c in evaluated_actions]
                return choice(evaluated_actions, p=probability)['action']

    class __RandomUniformPolicy:
        def __init__(self, env):
            self.env = env
            self.policy_name = "random_uniform"

        def predict(self, *args, **kwargs):
            actions = [ap.sample() for ap in self.env.get_attr("action_space")]
            return actions, None

    class __MaskedRandomUniformPolicy:
        def __init__(self, env):
            self.env = env
            self.policy_name = "masked_random_uniform"

        def predict(self, obs, state, deterministic):
            valid_action_lists = self.env.get_attr("valid_action_list")
            actions = [random.choice(val) for val in valid_action_lists]
            return actions, None

    @staticmethod
    def trained_ppo(train_session, env):
        model_filename = f"logs/test/{train_session}/best_model.zip"
        base = PPO("MlpPolicy", env)
        model = PPO.load(model_filename, env,
                         custom_objects={'lr_schedule': base.lr_schedule, 'clip_range': base.clip_range})
        model.policy_name = f"trained_ppo[{train_session}]"
        return model

    @staticmethod
    def trained_cnn_ppo(train_session, env):
        model_filename = f"logs/test/{train_session}/best_model.zip"
        base = CnnPPO(env)
        model = CnnPPO.load(model_filename, env,
                            custom_objects={'lr_schedule': base.lr_schedule, 'clip_range': base.clip_range})
        model.policy_name = f"trained_cnn_ppo[{train_session}]"
        return model

    @staticmethod
    def trained_maskable_ppo(model_filename, env):
        base = MaskablePPO(MaskableActorCriticPolicy, env)
        model = MaskablePPO.load(model_filename, env,
                                 custom_objects={'lr_schedule': base.lr_schedule, 'clip_range': base.clip_range})
        model.policy_name = f"trained_maskable_ppo"
        return model

    @staticmethod
    def uniform_random(env):
        return Policies.__RandomUniformPolicy(env)

    @staticmethod
    def masked_uniform_random(env):
        return Policies.__MaskedRandomUniformPolicy(env)

    @staticmethod
    def gaussian_random(env):
        return Policies.__RandomGaussianPolicy(env)

    @staticmethod
    def greedy(env):
        return Policies.__Greedy(env)

    @staticmethod
    def greedy_sim(env):
        return Policies.__Greedy(env, simulate=True, budget=10)
