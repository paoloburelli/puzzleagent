from gym.envs.registration import register

register(
    id='lgenv-v0',
    entry_point='lg_gym.envs:LGEnv',
)

register(
    id='lgenv_small-v0',
    entry_point='lg_gym.envs:LGEnvSmall',
)

register(
    id='lgenv_medium-v0',
    entry_point='lg_gym.envs:LGEnvMedium',
)
