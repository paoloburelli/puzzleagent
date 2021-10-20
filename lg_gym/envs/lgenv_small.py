import numpy as np
from lg_gym.envs.lgenv import LGEnv


class LGEnvSmall(LGEnv):
    def __init__(self, *args, **kwargs):
        super(LGEnvSmall, self).__init__(*args, **kwargs)

    def channels(self):
        return 4

    def processed_observation_space(self):
        obs = np.zeros((self.board_width, self.board_height, self.channels()), dtype=np.float32)

        colours = np.zeros((self.board_width, self.board_height), dtype=np.uint8)

        for x in range(self.board_width):
            for y in range(self.board_height):
                for c in LGEnv.COLOUR_CHANNELS:
                    if self.board[x, y, c] > 0:
                        colours[x, y] = c

        for x in range(self.board_width):
            for y in range(self.board_height):
                if self.action_mask[x, y] > 0:
                    for c in LGEnv.COLOUR_CHANNELS:
                        if self.board[x, y, c] > 0:
                            obs[x, y, 0] = (self.board[x + 1, y, c] if x + 1 < self.board_width else 0) + \
                                           (self.board[x - 1, y, c] if x > 0 else 0) + \
                                           (self.board[x, y + 1, c] if y + 1 < self.board_height else 0) + \
                                           (self.board[x, y - 1, c] if y > 0 else 0)
                            break
                        else:
                            obs[x, y, 0] = 0

        for x in range(self.board_width):
            for y in range(self.board_height):
                if self.board[x, y, LGEnv.COLLECT_GOAL_CHANNEL] > 0:
                    if self.board[x, y, LGEnv.HITTABLE_BY_NEIGHBOUR] > 0:

                        if x + 1 < self.board_width and colours[x + 1, y] == colours[x, y]:
                            obs[x + 1, y, 1] += self.board[x, y, LGEnv.COLLECT_GOAL_CHANNEL] / self.collect_goals_max
                        if y + 1 < self.board_height and colours[x, y + 1] == colours[x, y]:
                            obs[x, y + 1, 1] += self.board[x, y, LGEnv.COLLECT_GOAL_CHANNEL] / self.collect_goals_max
                        if x > 0 and colours[x - 1, y] == colours[x, y]:
                            obs[x - 1, y, 1] += self.board[x, y, LGEnv.COLLECT_GOAL_CHANNEL] / self.collect_goals_max
                        if y > 0 and colours[x, y - 1] == colours[x, y]:
                            obs[x, y - 1, 1] += self.board[x, y, LGEnv.COLLECT_GOAL_CHANNEL] / self.collect_goals_max
                    else:
                        obs[x, y, 1] += self.board[x, y, LGEnv.COLLECT_GOAL_CHANNEL] / self.collect_goals_max

            obs[:, :, 0] = np.clip(self.board[:, :, LGEnv.CLICKABLE_CHANNELS], 1, 0) * (
                    1 - np.clip(self.board[:, :, LGEnv.NOT_CLICKABLE_CHANNEL], 1, 0))  # clickable tiles
            obs[:, :, 1] /= 4  # adjacent of the same colour
            obs[:, :, 2] /= 4  # goals remaining
            obs[:, :, 3] = (self.board[:, :, 11] + self.board[:, :, 12] + self.board[:, :, 13] + self.board[:, :,
                                                                                                 14])  # power pieces
        return obs
