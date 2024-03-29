import gym
from stable_baselines3 import PPO
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)


class CustomCnnPPO(PPO):
    def __init__(self, *args, **kwargs):
        # If loading from saved model
        if "policy" in kwargs or "policy_kwargs" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__('CnnPolicy', policy_kwargs=policy_kwargs, *args, **kwargs)
