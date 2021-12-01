import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from sb3_contrib.ppo_mask import MaskablePPO


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


class MaskableCnnPPO(MaskablePPO):
    def __init__(self, *args, **kwargs):
        # If loading from saved model
        if "policy" in kwargs or "policy_kwargs" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(MaskableActorCriticCnnPolicy, policy_kwargs=policy_kwargs, *args, **kwargs)


class CnnPPO(PPO):
    def __init__(self, *args, **kwargs):
        # If loading from saved model
        if "policy" in kwargs or "policy_kwargs" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__('CnnPolicy', policy_kwargs=policy_kwargs, *args, **kwargs)


class CnnA2C(A2C):
    def __init__(self, *args, **kwargs):
        # If loading from saved model
        if "policy" in kwargs or "policy_kwargs" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__('CnnPolicy', policy_kwargs=policy_kwargs, *args, **kwargs)
