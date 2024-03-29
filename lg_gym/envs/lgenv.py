import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.multi_discrete import MultiDiscrete
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
    BOMB_CHANNEL = 11
    HITTABLE_BY_NEIGHBOUR = 21

    AS_DISCRETE = "discrete"
    AS_MULTI_DISCRETE = "multi_discrete"

    @property
    def level_seed(self):
        return np.random.randint(100000) if self._level_seed is None else self._level_seed

    @property
    def action_mask(self):
        return self._action_mask

    @abstractmethod
    def channels(self):
        return NotImplemented

    @abstractmethod
    def processed_observation_space(self):
        return NotImplemented

    def __init__(self, level_id, host="localhost", port=8080, seed=None, log_file=None,
                 action_space_type="discrete",
                 extra_moves=0, train=True):
        super().__init__()

        self.log_file = log_file
        self._level_seed = seed
        self.extra_moves = extra_moves
        self.action_space_type = action_space_type

        if self.action_space_type not in ["discrete", "multi_discrete"]:
            raise Exception(
                f"Invalid action space type {self.action_space_type}. It can be either {LGEnv.AS_DISCRETE} or {LGEnv.AS_MULTI_DISCRETE}")

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

        self.clicks_limit = self.game['levelMoveLimit'] * LGEnv.CLICKS_MULTIPLIER
        self.valid_moves_limit = self.game['levelMoveLimit']
        self.level_collect_goals = self.board_info['collectGoalRemaining']

        self.board_width = self.board_info['boardSize'][0]
        self.board_height = self.board_info['boardSize'][1]
        self.input_channels = self.board_info['boardSize'][2]

        self.valid_action_list = [(x, y) for x in range(self.board_width) for y in range(self.board_height)]

        self.board = np.array(self.board_info["board"],
                              dtype=np.uint8).reshape((self.board_width, self.board_height, self.input_channels),
                                                      order='F')

        self.collect_goals_max = self.board[:, :, LGEnv.COLLECT_GOAL_CHANNEL].max()

        self.observation_space = Box(low=0.0, high=1.0,
                                            shape=(self.board_width, self.board_height, self.channels()),
                                            dtype=np.float32)

        if self.action_space_type == LGEnv.AS_MULTI_DISCRETE:
            self.action_space = MultiDiscrete([self.board_width, self.board_height])
        elif self.action_space_type == LGEnv.AS_DISCRETE:
            self.action_space = Discrete(self.board_width * self.board_height)

        self.update_valid_actions(self.game['validActionPositions'])

        self.valid_moves_reward = 0
        self.goal_collection_reward = 0.4 / self.level_collect_goals
        self.completion_reward = 0.6
        self.loss_reward = -0.6
        self.invalid_action_penalty = -1 / self.clicks_limit

        self.total_clicks_performed = 0
        self.total_valid_moves_used = 0
        self.total_goals_collected = 0
        self.cumulative_reward = 0

    def action_masks(self):
        if self.action_space_type == LGEnv.AS_MULTI_DISCRETE:
            return [np.any(self.action_mask[x, :]) for x in range(self.board_width)] + [np.any(self.action_mask[:, y])
                                                                                        for y in
                                                                                        range(self.board_height)]
        elif self.action_space_type == LGEnv.AS_DISCRETE:
            return self.action_mask

    def update_valid_actions(self, valid_actions_list):
        try:
            val = json.loads(valid_actions_list)
            if len(val) == 0:
                logging.warning("Empty action mask returned from simulator")
            self.valid_action_list = [self.click_to_action(va[0], va[1]) for
                                      va in val]
        except:
            pass

        if self.action_space_type == LGEnv.AS_MULTI_DISCRETE:
            self._action_mask = np.zeros([self.board_width, self.board_height], dtype=bool)
            for va in self.valid_action_list:
                self._action_mask[va[0], va[1]] = 1
        elif self.action_space_type == LGEnv.AS_DISCRETE:
            self._action_mask = np.zeros(self.board_width * self.board_height, dtype=bool)
            for va in self.valid_action_list:
                self._action_mask[va] = 1

    def board_index_to_action(self, x, y):
        if self.action_space_type == LGEnv.AS_DISCRETE:
            return int(x + y * self.board_width)
        elif self.action_space_type == LGEnv.AS_MULTI_DISCRETE:
            return x, y

    def click_to_action(self, x, y):
        return self.board_index_to_action(int(x + (self.board_width // 2)), int(y + (self.board_height // 2)))

    def action_to_board_index(self, action):
        if self.action_space_type == LGEnv.AS_MULTI_DISCRETE:
            return action[0], action[1]
        elif self.action_space_type == LGEnv.AS_DISCRETE:
            return int(action % self.board_width), int(action // self.board_width)

    def action_to_click(self, action):
        x, y = self.action_to_board_index(action)
        return int(x - (self.board_width // 2)), int(y - (self.board_height // 2))

    def is_valid_action(self, action):
        if self.action_space_type == LGEnv.AS_MULTI_DISCRETE:
            return self.action_mask[action[0], action[1]]
        elif self.action_space_type == LGEnv.AS_DISCRETE:
            return self.action_mask[action]

    def simulate_click(self, action):
        x, y = self.action_to_click(action)

        reward = 0

        if self.is_valid_action(action):
            try:
                result = self.simulator.session_click(self.game['sessionId'], x, y, dry_run=True)
                simulated_board_info = json.loads(result["multichannelArrayState"])
                if result['clickSuccessful']:

                    reward = self.valid_moves_reward

                    new_total_goals_collected = self.level_collect_goals - simulated_board_info['collectGoalRemaining']

                    goals_collected_now = max(0, new_total_goals_collected - self.total_goals_collected)

                    reward += self.goal_collection_reward * goals_collected_now

                else:
                    reward = self.invalid_action_penalty

                if simulated_board_info['collectGoalRemaining'] == 0:
                    reward = max(0.1,
                                 self.completion_reward + 0.05 * (
                                         self.valid_moves_limit - (self.total_valid_moves_used + 1)))
                elif (self.total_clicks_performed + 1) >= self.clicks_limit or (
                        self.total_valid_moves_used + 1) > self.valid_moves_limit + self.extra_moves or len(
                    self.valid_action_list) == 0:
                    reward = self.loss_reward

            except Exception as e:
                logging.error(f"simulate_click: {e}")
        else:
            reward = self.invalid_action_penalty

        return reward

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
                    reward = self.invalid_action_penalty

            except Exception as e:
                logging.error(f"click: {e}")

            if self.board_info['collectGoalRemaining'] <= 0:
                extra_moves_used = max(0, self.total_valid_moves_used - self.valid_moves_limit)
                extra_moves_penalty = 0 if self.extra_moves == 0 else extra_moves_used / self.extra_moves
                reward = self.completion_reward * (1 - extra_moves_penalty / 2)
            elif self.total_clicks_performed >= self.clicks_limit or self.total_valid_moves_used >= self.valid_moves_limit + self.extra_moves or len(
                    self.valid_action_list) == 0:
                reward = self.loss_reward

        else:
            reward = self.invalid_action_penalty

        done = self.board_info['collectGoalRemaining'] <= 0 or \
               self.total_clicks_performed >= self.clicks_limit or \
               self.total_valid_moves_used >= self.valid_moves_limit + self.extra_moves or \
               len(self.valid_action_list) == 0

        self.cumulative_reward += reward
        if done and self.log_file:
            with open(self.log_file, 'a+') as f:
                f.write(
                    f"""{datetime.now().strftime('%Y%m%d%H%M%S')},{self.level_seed},{self.spec.id},{self.level_id},{self.valid_moves_limit},{self.clicks_limit},{self.level_collect_goals},{self.total_valid_moves_used},{self.total_clicks_performed},{self.total_goals_collected},{self.cumulative_reward}\n""")
                f.close()

        victory = self.board_info['collectGoalRemaining'] <= 0 and \
                  self.total_valid_moves_used <= self.valid_moves_limit

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
