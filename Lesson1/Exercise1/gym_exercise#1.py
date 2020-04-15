import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(200):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
print(gym.envs.registry.spec('CartPole-v1').max_episode_steps)
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
env.close()