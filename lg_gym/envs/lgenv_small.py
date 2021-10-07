import random

import gym
from gym import spaces
import numpy as np
import json
import logging
from lg_gym.simulator import Simulator
from datetime import datetime

CLICKS_MULTIPLIER = 100
PIECES_CHANNEL = 0
COLOUR_CHANNELS = [1, 2, 3, 4, 5, 6]
CLICKABLE_CHANNELS = 7
NOT_CLICKABLE_CHANNEL = 8
COLLECT_GOAL_CHANNEL = 9
BASIC_PIECE_CHANNEL = 10
HITTABLE_BY_NEIGHBOUR = 21


class LGEnvSmall(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    reward_range = (-1, 1)

    def get_seed(self):
        return np.random.randint(100000) if self._seed is None else self._seed

    def __init__(self, level_id, host="localhost", port=8080, seed=None, log_file=None, extra_moves=0,
                 dockersim=False, subprocsim=False):
        super().__init__()

        self.simulator_docker = Simulator.start_container(port) if dockersim else None
        self.simulator_subprocess = Simulator.start_process(port) if subprocsim else None

        self.log_file = log_file
        self._seed = seed
        self.extra_moves = extra_moves

        self.simulator = Simulator(host, port)
        self._level_id = level_id

        self.game = self.simulator.session_create(self._get_level_id(), self.get_seed())
        self.board_info = json.loads(self.game["multichannelArrayState"])

        self.clicks_limit = self.game['levelMoveLimit'] * CLICKS_MULTIPLIER
        self.valid_moves_limit = self.game['levelMoveLimit']
        self.collect_goals = self.board_info['collectGoalRemaining']

        self.width = self.board_info['boardSize'][0]
        self.height = self.board_info['boardSize'][1]
        self.input_channels = self.board_info['boardSize'][2]
        self.channels = 3

        self.board = np.array(self.board_info["board"],
                              dtype=np.uint8).reshape((self.width, self.height, self.input_channels), order='F')
        self.action_mask = (1 - np.clip(self.board[:, :, NOT_CLICKABLE_CHANNEL], 0, 1)) * np.clip(self.board[:, :,
                                                                                                  PIECES_CHANNEL], 0, 1)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.width, self.height, self.channels),
                                            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.width, self.height])

        self.valid_moves_reward = 0.2 / self.valid_moves_limit
        self.goal_collection_reward = 0.2 / self.collect_goals
        self.completion_reward = 0.4
        self.invalid_action_penalty = -1 / self.clicks_limit

        self.clicks = 0
        self.valid_moves = 0
        self.goals_collected = 0
        self.cumulative_reward = 0

    def _get_level_id(self):
        if type(self._level_id) is tuple:
            return random.randint(self._level_id[0], self._level_id[1])
        else:
            return self._level_id

    def _observation_from_board(self, board):

        obs = np.zeros((self.width, self.height, self.channels), dtype=np.float32)

        for x in range(self.width):
            for y in range(self.height):
                if board[x, y, NOT_CLICKABLE_CHANNEL] == 0 and board[x, y, PIECES_CHANNEL] == 1:
                    for c in COLOUR_CHANNELS:
                        if board[x, y, c] > 0:
                            obs[x, y, 0] = (board[x + 1, y, c] if x + 1 < self.width else 0) + (
                                board[x - 1, y, c] if x - 1 > 0 else 0) + (
                                               board[x, y + 1, c] if y + 1 < self.height else 0) + (
                                               board[x, y - 1, c] if y - 1 > 0 else 0) + (board[x, y, c])
                            break
                        else:
                            obs[x, y, 0] = 0

        for x in range(self.width):
            for y in range(self.height):
                if board[x, y, COLLECT_GOAL_CHANNEL] > 0:
                    if board[x, y, HITTABLE_BY_NEIGHBOUR] > 0:
                        if x < self.width - 1:
                            obs[x + 1, y, 1] += board[x, y, COLLECT_GOAL_CHANNEL]
                        if y < self.height - 1:
                            obs[x, y + 1, 1] += board[x, y, COLLECT_GOAL_CHANNEL]
                        if x > 0:
                            obs[x - 1, y, 1] += board[x, y, COLLECT_GOAL_CHANNEL]
                        if y > 0:
                            obs[x, y - 1, 1] += board[x, y, COLLECT_GOAL_CHANNEL]
                    else:
                        obs[x, y, 1] += board[x, y, COLLECT_GOAL_CHANNEL]

        obs[:, :, 0] /= 5  # adjacent of the same colour
        obs[:, :, 1] /= 800  # goals remaining
        obs[:, :, 2] = (board[:, :, 11] + board[:, :, 12] + board[:, :, 13] + board[:, :, 14])  # power pieces
        return obs

    def simulate_click(self, action):
        x = int(action[0] - (self.width // 2))
        y = int(action[1] - (self.height // 2))

        reward = 0

        if self.action_mask[action[0], action[1]] > 0:
            try:
                result = self.simulator.session_click(self.game['sessionId'], x, y, dry_run=True)
                board_info = json.loads(result["multichannelArrayState"])
                if result['clickSuccessful']:
                    reward += self.valid_moves_reward

                    if self.goals_collected < self.collect_goals - board_info['collectGoalRemaining']:
                        reward += self.goal_collection_reward
                else:
                    reward += self.invalid_action_penalty

                if board_info['collectGoalRemaining'] < 1:
                    reward += self.completion_reward + 2 * self.valid_moves_reward * (
                            self.valid_moves_limit - self.valid_moves)
            except Exception as e:
                logging.error(f"simulate_click: {e}")
        else:
            reward += self.invalid_action_penalty

        return reward

    def step(self, action):
        x = int(action[0] - (self.width // 2))
        y = int(action[1] - (self.height // 2))
        reward = 0

        self.clicks += 1

        click_successfull = False

        if self.action_mask[action[0], action[1]] > 0:
            try:
                result = self.simulator.session_click(self.game['sessionId'], x, y, False)
                try:
                    self.board_info = json.loads(result["multichannelArrayState"])
                    self.board = np.array(self.board_info["board"],
                                          dtype=np.uint8).reshape((self.width, self.height, self.input_channels),
                                                                  order='F')
                    self.action_mask = 1 - self.board[:, :, NOT_CLICKABLE_CHANNEL]
                except Exception as e:
                    logging.error(f"click:parse: {e}")

                if result['clickSuccessful']:
                    click_successfull = True
                    self.valid_moves += 1
                    reward += self.valid_moves_reward

                    if self.goals_collected < self.collect_goals - self.board_info['collectGoalRemaining']:
                        reward += self.goal_collection_reward

                    self.goals_collected = self.collect_goals - self.board_info['collectGoalRemaining']
                else:
                    reward += self.invalid_action_penalty

            except Exception as e:
                logging.error(f"click: {e}")

            if self.board_info['collectGoalRemaining'] < 1:
                reward += self.completion_reward + 2 * self.valid_moves_reward * (
                        self.valid_moves_limit - self.valid_moves)
        else:
            reward += self.invalid_action_penalty

        done = self.goals_collected >= self.collect_goals or \
               self.clicks >= self.clicks_limit or \
               self.valid_moves >= self.valid_moves_limit + self.extra_moves

        obs = self._observation_from_board(self.board)

        self.cumulative_reward += reward
        if done and self.log_file:
            with open(self.log_file, 'a+') as f:
                f.write(
                    f"""{datetime.now().strftime('%Y%m%d%H%M%S')},{self.get_seed()},{self.spec.id},{self._get_level_id()},{self.valid_moves_limit},{self.clicks_limit},{self.collect_goals},{self.valid_moves},{self.clicks},{self.goals_collected},{self.cumulative_reward}\n""")
                f.close()

        return obs, reward, done, {'x': x, 'y': y, 'click_successful': click_successfull,
                                   'completed': self.board_info['collectGoalRemaining'] < 1,
                                   'failed': self.clicks_limit < 1 or self.valid_moves_limit < 1}

    def reset(self):
        try:
            self.simulator.session_destroy(self.game['sessionId'])
        except Exception as e:
            logging.error(f"reset: {e}")

        self.game = self.simulator.session_create(self._get_level_id(), self.get_seed())
        self.board_info = json.loads(self.game["multichannelArrayState"])

        self.clicks = 0
        self.valid_moves = 0
        self.goals_collected = 0
        self.cumulative_reward = 0

        return self._observation_from_board(self.board)

    def close(self):
        try:
            self.simulator.session_destroy(self.game['sessionId'])
        except Exception as e:
            logging.error(f"close: {e}")

        if self.simulator_docker is not None:
            Simulator.stop_container(self.simulator_docker)

        if self.simulator_subprocess is not None:
            Simulator.stop_process(self.simulator_subprocess)

    def render(self, mode='human', close=False):
        pass
