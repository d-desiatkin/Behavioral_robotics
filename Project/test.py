import gym
from gym import spaces
import pybullet
import pybullet_envs


env = gym.make('AntBulletEnv-v0')

print(env.action_space.low)

env.reset()
for _ in range(200):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
env.close()