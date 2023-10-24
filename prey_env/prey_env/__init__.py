from gymnasium.envs.registration import register

register(
		id='prey_d_1',
		entry_point='prey_env.envs:Environment',
)