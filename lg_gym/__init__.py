from gym.envs.registration import register

register(
    id='lgenv_full-v0',
    entry_point='lg_gym.envs:LGEnvFull',
)

register(
    id='lgenv_flat-v0',
    entry_point='lg_gym.envs:LGFlat',
)
