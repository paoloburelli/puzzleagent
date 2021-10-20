import numpy as np
from lg_gym.envs.lgenv import LGEnv


class LGEnvFull(LGEnv):
    def __init__(self, *args, **kwargs):
        super(LGEnvFull, self).__init__(*args, **kwargs)

    def channels(self):
        return self.input_channels

    def processed_observation_space(self):
        return self.board / np.array(self.board_info['normList'])
