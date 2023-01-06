import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.dict import Dict
import numpy as np
import json
import logging
from lg_gym.simulator import Simulator
from datetime import datetime
from abc import *


class LGFlat(gym.Env, ABC):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    reward_range = (-1, 1)

    # COLOUR_CHANNELS = [1, 2, 3, 4, 5, 6]
    COLLECT_GOAL_CHANNELS = [9]

    # BOMB_CHANNELS = [11]
    # FLASK_CHANNELS = [12]
    # ROCKET_CHANNELS = [13, 14]
    # GRAVITY_CHANNELS = [15]
    # SPREADABLE_CHANNELS = [16]
    # HEALABLE_CHANNELS = [17]
    # HITTABLE_BY_NEIGHBOUR_CHANNELS = [21]
    #
    PADDED_BOARD_SIZE = 36

    #
    # OBSERVATION_SPACE_CHANNELS = COLOUR_CHANNELS + COLLECT_GOAL_CHANNELS + BOMB_CHANNELS + FLASK_CHANNELS + ROCKET_CHANNELS + GRAVITY_CHANNELS + SPREADABLE_CHANNELS + HEALABLE_CHANNELS + HITTABLE_BY_NEIGHBOUR_CHANNELS

    @property
    def level_seed(self):
        return np.random.randint(100000) if self._level_seed is None else self._level_seed

    def channels(self):
        return 1

    def processed_observation_space(self):
        board = (255 * self.board / self.board_max).astype(np.uint8)
        return np.pad(board, ((11, 12), (13, 14), (0, 0))).T

    def __init__(self, level_id, host="localhost", port=8080, seed=None, log_file=None, extra_moves=0, train=True):
        super().__init__()

        self.log_file = log_file
        self._level_seed = seed
        self.extra_moves = extra_moves
        self.simulator = Simulator(host, port)
        self.level_id = level_id
        self.train = train
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

        self._reset(init=True)

    def _reset(self, init=False):

        if init:
            self.episode = 0
        else:
            self.episode += 1

        logging.info(f"{self.__class__.__name__}[{'train' if self.train else 'eval'}]: "
                     f"init level {self.level_id}, episode {self.episode}")

        self.game = self.simulator.session_create(self.level_id, self.level_seed)
        self.board_info = json.loads(self.game["multichannelArrayState"])

        self.moves_limit = self.game['levelMoveLimit']
        self.level_collect_goals = self.board_info['collectGoalRemaining']

        self.board_width = self.board_info['boardSize'][0]
        self.board_height = self.board_info['boardSize'][1]
        self.input_channels = self.board_info['boardSize'][2]

        self.valid_action_list = [(x, y) for x in range(self.board_width) for y in range(self.board_height)]

        self.board = np.array(self.board_info["board"],
                              dtype=np.uint8).reshape((self.board_width, self.board_height, self.input_channels),
                                                      order='F')

        self.board_max = np.array([max(1, self.board[:, :, i].max()) for i in range(self.input_channels)])

        self.collect_goals_max = self.board[:, :, LGFlat.COLLECT_GOAL_CHANNELS].max()

        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(self.input_channels, LGFlat.PADDED_BOARD_SIZE,
                                                       LGFlat.PADDED_BOARD_SIZE),
                                                dtype=np.uint8)

        self.action_space = Discrete(self.board_width * self.board_height)

        self.update_valid_actions(self.game['validActionPositions'])

        self.valid_moves_reward = 0
        self.goal_collection_reward = 0.4 / self.level_collect_goals
        self.completion_reward = 0.6
        self.loss_reward = -0.6
        self.invalid_action_reward = -0.1

        self.total_clicks_performed = 0
        self.total_valid_moves_used = 0
        self.total_goals_collected = 0
        self.cumulative_reward = 0

    def action_masks(self):
        return self._action_mask

    def update_valid_actions(self, valid_actions_list):
        try:
            val = json.loads(valid_actions_list)
            if len(val) == 0:
                logging.warning("Empty action mask returned from simulator")
            else:
                self.valid_action_list = [self.click_to_action(va[0], va[1]) for
                                          va in val]
        except:
            pass

        self._action_mask = np.zeros(self.board_width * self.board_height, dtype=bool)
        for va in self.valid_action_list:
            self._action_mask[va] = 1

    def click_to_action(self, x, y):
        b_x = int(x + (self.board_width // 2))
        b_y = int(y + (self.board_height // 2))
        return int(b_x + b_y * self.board_width)

    def action_to_click(self, action):
        b_x = int(action % self.board_width)
        b_y = int(action // self.board_width)
        return int(b_x - (self.board_width // 2)), int(b_y - (self.board_height // 2))

    def is_valid_action(self, action):
        return self._action_mask[action]

    def step(self, action):
        x, y = self.action_to_click(action)
        reward = 0

        self.total_clicks_performed += 1

        click_successfull = False
        goals_collected_now = 0

        if self.is_valid_action(action):
            try:
                result = self.simulator.session_click(self.game['sessionId'], x, y, False)
                try:
                    self.update_valid_actions(result['validActionPositions'])
                    self.board_info = json.loads(result["multichannelArrayState"])
                    self.board = np.array(self.board_info["board"],
                                          dtype=np.uint8).reshape(
                        (self.board_width, self.board_height, self.input_channels),
                        order='F')
                except Exception as e:
                    logging.error(f"click:parse: {e}")

                if result['clickSuccessful']:
                    click_successfull = True
                    self.total_valid_moves_used += 1

                    reward = self.valid_moves_reward

                    new_total_goals_collected = self.level_collect_goals - self.board_info['collectGoalRemaining']

                    goals_collected_now = max(0, new_total_goals_collected - self.total_goals_collected)

                    reward += self.goal_collection_reward * goals_collected_now

                    self.total_goals_collected = new_total_goals_collected
                else:
                    reward = self.invalid_action_reward

            except Exception as e:
                logging.error(f"click: {e}")

            if self.board_info['collectGoalRemaining'] <= 0:
                extra_moves_used = max(0, self.total_valid_moves_used - self.moves_limit)
                extra_moves_penalty = 0 if self.extra_moves == 0 else extra_moves_used / self.extra_moves
                reward = self.completion_reward * (1 - extra_moves_penalty / 2)
            elif self.total_valid_moves_used >= self.moves_limit + self.extra_moves or len(
                    self.valid_action_list) == 0:
                reward = self.loss_reward

        else:
            reward = self.invalid_action_reward

        done = self.board_info['collectGoalRemaining'] <= 0 or \
               self.total_valid_moves_used >= self.moves_limit + self.extra_moves or \
               len(self.valid_action_list) == 0

        self.cumulative_reward += reward
        if done and self.log_file:
            with open(self.log_file, 'a+') as f:
                f.write(
                    f"""{datetime.now().strftime('%Y%m%d%H%M%S')},{self.level_seed},{self.spec.id},{self.level_id},{self.moves_limit},{self.level_collect_goals},{self.total_valid_moves_used},{self.total_clicks_performed},{self.total_goals_collected},{self.cumulative_reward}\n""")
                f.close()

        victory = self.board_info['collectGoalRemaining'] <= 0 and self.total_valid_moves_used <= self.moves_limit

        return self.processed_observation_space(), reward, done, {'x': x, 'y': y,
                                                                  'is_success': victory,
                                                                  'valid_action': click_successfull,
                                                                  'goals_collected': goals_collected_now}

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

    def render(self, mode='human', close=False):
        pass
