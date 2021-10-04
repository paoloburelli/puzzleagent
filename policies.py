from stable_baselines3.ppo import PPO
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

        def __action(self):
            random_index = random.randint(0, self.sample_size - 1)
            x = self.distribution_x[random_index]
            y = self.distribution_y[random_index]
            return [x, y]

        def predict(self, *args, **kwargs):
            actions = [self.__action() for n in range(self.env.num_envs)]
            return actions, None

        @staticmethod
        def name():
            return "gaussian_random"

    class __RandomUniformPolicy:
        def __init__(self, env):
            self.env = env

        def predict(self, *args, **kwargs):
            actions = [ap.sample() for ap in self.env.get_attr("action_space")]
            return actions, None

        @staticmethod
        def name():
            return "uniform_random"

    @staticmethod
    def trained_ppo(train_session, env):
        model_filename = f"logs/test/{train_session}/best_model.zip"
        model = PPO.load(model_filename, env)
        model.name = lambda: f"trained_ppo[{train_session}]"
        return model

    @staticmethod
    def uniform_random(env):
        return Policies.__RandomUniformPolicy(env)

    @staticmethod
    def gaussian_random(env):
        return Policies.__RandomGaussianPolicy(env)
