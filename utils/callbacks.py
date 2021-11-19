from stable_baselines3.common.callbacks import *

class StopTrainingOnEpisodeLengthThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose:
    """

    def __init__(self, episode_length_threshold: float, verbose: int = 0):
        super(StopTrainingOnEpisodeLengthThreshold, self).__init__(verbose=verbose)
        self.episode_length_threshold = episode_length_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used " "with an ``EvalCallback``"
        # Convert np.bool_ to bool, otherwise callback() is False won't work
        self.mean_ep_length_last = np.mean(self.parent.evaluations_length[-1])
        continue_training = bool(self.mean_ep_length_last > self.episode_length_threshold)
        if self.verbose > 0 and not continue_training:
            print(
                f"Stopping training because the mean episode length {self.mean_ep_length_last:.1f} "
                f" is equal or below the threshold {self.episode_length_threshold}"
            )
        return continue_training