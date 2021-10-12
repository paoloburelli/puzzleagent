from stable_baselines3.ppo import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
import random
from scipy.stats import truncnorm


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
        def __init__(self, env, simulate=False):
            self.env = env
            self.simulate = simulate
            if simulate:
                self.policy_name = "greedy_sim"
            else:
                self.policy_name = "greedy"

        def predict(self, obs, state, deterministic):
            actions = [self.__single_prediction(o, am, deterministic) for am, o in
                       zip(self.env.get_attr("action_mask"), obs)]
            return actions, None

        def __score(self, obs, x, y):
            if self.simulate:
                return self.env.env_method("simulate_click", (x, y))[0]
            else:
                return (obs[x, y, 0] + 3 * obs[x, y, 2]) * (1 + obs[x, y, 1])

        def __single_prediction(self, obs, action_mask, deterministric):
            potentially_valid_moves = []
            for x in range(action_mask.shape[0]):
                for y in range(action_mask.shape[1]):
                    if action_mask[x, y] > 0:
                        score = self.__score(obs, x, y)
                        potentially_valid_moves.append({'move': (x, y), 'score': score * score})

            potentially_valid_moves.sort(key=lambda a: a['score'], reverse=True)
            if deterministric:
                return potentially_valid_moves[0]['move']
            else:
                total_score = sum([a['score'] for a in potentially_valid_moves])
                r = random.random() * total_score
                rolling_sum = 0
                index = -1
                while rolling_sum <= r and index < len(potentially_valid_moves) - 1:
                    index += 1
                    rolling_sum += potentially_valid_moves[index]['score']

                cl_index = max(0, min(len(potentially_valid_moves) - 1, index))
                return potentially_valid_moves[cl_index]['move']

    class __RandomUniformPolicy:
        def __init__(self, env):
            self.env = env
            self.policy_name = "random_uniform"

        def predict(self, *args, **kwargs):
            actions = [ap.sample() for ap in self.env.get_attr("action_space")]
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
    def gaussian_random(env):
        return Policies.__RandomGaussianPolicy(env)

    @staticmethod
    def greedy(env):
        return Policies.__Greedy(env)

    @staticmethod
    def greedy_sim(env):
        return Policies.__Greedy(env, simulate=True)
