import random

import gym
from gym import spaces
import numpy as np
import json
import logging
from lg_gym.simulator import Simulator
from datetime import datetime
from abc import *


class LGEnv(gym.Env, ABC):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    reward_range = (-1, 1)

    CLICKS_MULTIPLIER = 100
    PIECES_CHANNEL = 0
    COLOUR_CHANNELS = [1, 2, 3, 4, 5, 6]
    CLICKABLE_CHANNELS = 7
    NOT_CLICKABLE_CHANNEL = 8
    COLLECT_GOAL_CHANNEL = 9
    BASIC_PIECE_CHANNEL = 10
    HITTABLE_BY_NEIGHBOUR = 21

    @property
    def level_seed(self):
        return np.random.randint(100000) if self._level_seed is None else self._level_seed

    @property
    def action_mask(self):
        return self.board[:, :, LGEnv.CLICKABLE_CHANNELS] > 0

    @abstractmethod
    def channels(self):
        return NotImplemented

    @abstractmethod
    def processed_observation_space(self):
        return NotImplemented

    def __init__(self, level_id, host="localhost", port=8080, seed=None, log_file=None, extra_moves=0,
                 dockersim=False, subprocsim=False, train=True):
        super().__init__()

        self.simulator_docker = Simulator.start_container(port) if dockersim else None
        self.simulator_subprocess = Simulator.start_process(port) if subprocsim else None

        self.log_file = log_file
        self._level_seed = seed
        self.extra_moves = extra_moves

        self.simulator = Simulator(host, port)
        self._level_id_config = level_id
        self.train = train
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

        self._reset(init=True)

    def _reset(self, init=False):

        if init:
            self.episode = 0
            self.curriculum_step_ep = -1
            self.curriculum_step = 0
        else:
            self.episode += 1

        if type(self._level_id_config) is list:
            next_step = self.curriculum_step_ep + 1
            self.curriculum_step_ep = next_step % self._level_id_config[self.curriculum_step]['episodes']
            self.curriculum_step += next_step // self._level_id_config[self.curriculum_step]['episodes']
            self.curriculum_step %= len(self._level_id_config)

            self.current_level_id = self._level_id_config[self.curriculum_step]['level_id']
        else:
            self.current_level_id = self._level_id_config

        logging.info(f"{self.__class__.__name__}[{'train' if self.train else 'eval'}]: "
                     f"init level {self.current_level_id}, episode {self.episode}, "
                     f"curriculum step {self.curriculum_step}, curriculum step ep {self.curriculum_step_ep}")

        self.game = self.simulator.session_create(self.current_level_id, self.level_seed)
        self.board_info = json.loads(self.game["multichannelArrayState"])

        self.clicks_limit = self.game['levelMoveLimit'] * LGEnv.CLICKS_MULTIPLIER
        self.valid_moves_limit = self.game['levelMoveLimit']
        self.collect_goals = self.board_info['collectGoalRemaining']

        self.board_width = self.board_info['boardSize'][0]
        self.board_height = self.board_info['boardSize'][1]
        self.input_channels = self.board_info['boardSize'][2]

        self.board = np.array(self.board_info["board"],
                              dtype=np.uint8).reshape((self.board_width, self.board_height, self.input_channels),
                                                      order='F')

        self.collect_goals_max = self.board[:, :, LGEnv.COLLECT_GOAL_CHANNEL].max()

        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.board_width, self.board_height, self.channels()),
                                            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.board_width, self.board_height])

        self.valid_moves_reward = 0.2 / self.valid_moves_limit
        self.goal_collection_reward = 0.2 / self.collect_goals
        self.completion_reward = 0.4
        self.invalid_action_penalty = -0.1 / self.valid_moves_limit

        self.clicks = 0
        self.valid_moves = 0
        self.goals_collected = 0
        self.cumulative_reward = 0

    def action_masks(self):
        return [np.any(self.action_mask[x, :]) for x in range(self.board_width)] + [np.any(self.action_mask[:, y]) for y
                                                                                    in
                                                                                    range(self.board_height)]

    def simulate_click(self, action):
        x = int(action[0] - (self.board_width // 2))
        y = int(action[1] - (self.board_height // 2))

        reward = 0

        if self.action_mask[action[0], action[1]]:
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
        x = int(action[0] - (self.board_width // 2))
        y = int(action[1] - (self.board_height // 2))
        reward = 0

        self.clicks += 1

        click_successfull = False

        if self.action_mask[action[0], action[1]]:
            try:
                result = self.simulator.session_click(self.game['sessionId'], x, y, False)
                try:
                    self.board_info = json.loads(result["multichannelArrayState"])
                    self.board = np.array(self.board_info["board"],
                                          dtype=np.uint8).reshape(
                        (self.board_width, self.board_height, self.input_channels),
                        order='F')
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

        self.cumulative_reward += reward
        if done and self.log_file:
            with open(self.log_file, 'a+') as f:
                f.write(
                    f"""{datetime.now().strftime('%Y%m%d%H%M%S')},{self.level_seed},{self.spec.id},{self.current_level_id},{self.valid_moves_limit},{self.clicks_limit},{self.collect_goals},{self.valid_moves},{self.clicks},{self.goals_collected},{self.cumulative_reward}\n""")
                f.close()

        return self.processed_observation_space(), reward, done, {'x': x, 'y': y, 'click_successful': click_successfull,
                                                                  'completed': self.board_info[
                                                                                   'collectGoalRemaining'] < 1,
                                                                  'failed': self.clicks_limit < 1 or self.valid_moves_limit < 1}

    def reset(self):
        try:
            self.simulator.session_destroy(self.game['sessionId'])
        except Exception as e:
            logging.error(f"reset: {e}")

        self._reset()

        return self.processed_observation_space()

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
