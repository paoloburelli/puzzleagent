import numpy as np
from stable_baselines3.ppo import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
import random
from scipy.stats import truncnorm
from numpy.random import choice
import logging


class Policies:
    class __RandomGaussianPolicy:

        def __init__(self, env):
            self.sample_size = 2000000

            width = env.get_attr("width")[0]
            height = env.get_attr("height")[0]

            X = truncnorm(a=-2, b=2, scale=width // 4).rvs(size=self.sample_size) + width // 2
            self.distribution_x = X.round().astype(int).tolist()

            Y = truncnorm(a=-2, b=2, scale=height // 4).rvs(size=self.sample_size) + height // 2
            self.distribution_y = Y.round().astype(int).tolist()

            self.env = env
            self.policy_name = "gaussian_random"

        def __action(self):
            random_index = random.randint(0, self.sample_size - 1)
            x = self.distribution_x[random_index]
            y = self.distribution_y[random_index]
            return [x, y]

        def predict(self, *args, **kwargs):
            actions = [self.__action() for n in range(self.env.num_envs)]
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
            action_masks = self.env.get_attr("action_mask")
            actions = [self.__single_prediction(o, am, i, deterministic) for am, o, i in
                       zip(action_masks, obs, range(len(action_masks)))]
            return actions, None

        def __score(self, obs, x, y, index):
            if self.simulate:
                return self.env.env_method("simulate_click", (x, y), indices=index)[0]
            else:
                return 1 + obs[x, y, 0] * (1 if obs[x, y, 1] > 0 else 0) + obs[x, y, 2]

        def __single_prediction(self, obs, action_mask, index, deterministric):
            budget_used = 0

            width = action_mask.shape[0]
            height = action_mask.shape[1]

            potentially_valid_moves = []
            x_seq = list(range(width))
            random.shuffle(x_seq)

            y_seq = list(range(height))
            random.shuffle(y_seq)

            for x in x_seq:
                for y in y_seq:
                    if action_mask[x, y] and (budget_used < self.budget or len(potentially_valid_moves) == 0):
                        if budget_used >= self.budget:  # go over budget if no good solution is found yet
                            logging.warning(
                                f"{self.__class__.__name__}: estimation budget exceeded ({budget_used}/{self.budget})")
                        score = self.__score(obs, x, y, index)
                        budget_used += 1
                        if score > 0:
                            potentially_valid_moves.append({'move': (x, y), 'score': score * score})

            if len(potentially_valid_moves) > 0:
                if deterministric:
                    return potentially_valid_moves[0]['move']
                else:
                    total_score = sum([c['score'] for c in potentially_valid_moves])
                    probability = [c['score'] / total_score for c in potentially_valid_moves]
                    return choice(potentially_valid_moves, p=probability)['move']
            else:
                return random.randint(0, width - 1), random.randint(0, height - 1)

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
            self.policy_name = "random_uniform"

        def __single_prediction(self, obs, action_mask):
            width = action_mask.shape[0]
            height = action_mask.shape[1]

            potentially_valid_moves = []
            for x in range(width):
                for y in range(height):
                    if action_mask[x, y]:
                        potentially_valid_moves.append((x, y))

            if len(potentially_valid_moves) > 0:
                return random.sample(potentially_valid_moves, 1)[0]
            else:
                return random.randint(0, width - 1), random.randint(0, height - 1)

        def predict(self, obs, state, deterministic):
            action_masks = self.env.get_attr("action_mask")
            actions = [self.__single_prediction(o, am) for am, o in zip(action_masks, obs)]
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
    def trained_maskable_ppo(train_session, env):
        model_filename = f"logs/test/{train_session}/best_model.zip"
        base = MaskablePPO(MaskableActorCriticPolicy, env)
        model = MaskablePPO.load(model_filename, env,
                                 custom_objects={'lr_schedule': base.lr_schedule, 'clip_range': base.clip_range})
        model.policy_name = f"trained_maskable_ppo[{train_session}]"
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
