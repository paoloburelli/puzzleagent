import gym
from gym import spaces
import numpy as np
import json
import logging
from lg_gym.simulator import Simulator

CLICKS_MULTIPLIER = 100


class LGEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def get_seed(self):
        return np.random.randint(1, 2 ** 31 - 1) if self._seed is None else self._seed

    def __init__(self, host, level_id, seed=None):
        super().__init__()

        self._seed = seed

        self.simulator = Simulator(host)
        self.level_id = level_id

        self.game = self.simulator.session_create(self.level_id, self.get_seed())
        self.board_state = json.loads(self.game["multichannelArrayState"])

        self.clicks_remaining = self.game['levelMoveLimit'] * CLICKS_MULTIPLIER
        self.valid_moves_remaining = self.game['levelMoveLimit']
        self.collect_goal_remaining = self.board_state['collectGoalRemaining']

        self.width = self.board_state['boardSize'][0]
        self.height = self.board_state['boardSize'][1]
        self.channels = self.board_state['boardSize'][2]

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, self.channels),
                                            dtype=np.uint8)
        self.action_space = spaces.Discrete(self.width * self.height)

        self.valid_moves_reward = 0.2 / self.valid_moves_remaining
        self.goal_collection_reward = 0.2 / self.collect_goal_remaining
        self.completion_reward = 0.4

    @staticmethod
    def _observation_from_state(board_state):
        obs = np.array(board_state['board'],
                       dtype=np.float64).reshape(board_state['boardSize'], order='F') * 255
        obs //= np.array(board_state['normList'])
        return obs  # np.transpose(obs, (2, 0, 1))

    def step(self, action):
        x = int((action % self.width) - (self.width // 2))
        y = int((action // self.width) - (self.height // 2))

        reward = 0

        self.clicks_remaining -= 1
        try:
            result = self.simulator.session_click(self.game['sessionId'], x, y, False)
            try:
                self.board_state = json.loads(result["multichannelArrayState"])
            except Exception as e:
                logging.error(f"click:parse: {e}")

            if result['clickSuccessful']:

                self.valid_moves_remaining -= 1
                reward += self.valid_moves_reward

                if self.collect_goal_remaining > self.board_state['collectGoalRemaining']:
                    reward += self.goal_collection_reward

                self.collect_goal_remaining = self.board_state['collectGoalRemaining']

        except Exception as e:
            logging.error(f"click: {e}")

        if self.board_state['collectGoalRemaining'] < 1:
            reward += self.completion_reward + 2 * self.valid_moves_reward * self.valid_moves_remaining

        done = self.board_state['collectGoalRemaining'] < 1 or \
               self.clicks_remaining < 1 or \
               self.valid_moves_remaining < 1

        obs = self._observation_from_state(self.board_state)

        return obs, reward, done, {}

    def reset(self):
        try:
            self.simulator.session_destroy(self.game['sessionId'])
        except Exception as e:
            logging.error(str(e))

        self.game = self.simulator.session_create(self.level_id, self.get_seed())
        self.board_state = json.loads(self.game["multichannelArrayState"])
        self.clicks_remaining = self.game['levelMoveLimit'] * CLICKS_MULTIPLIER
        self.valid_moves_remaining = self.game['levelMoveLimit']

        return LGEnv._observation_from_state(self.board_state)

    def close(self):
        try:
            self.simulator.session_destroy(self.game['sessionId'])
        except Exception as e:
            logging.error(str(e))

    def render(self, mode='human', close=False):
        pass
