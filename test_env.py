import gymnasium as gym
import shimmy

env = gym.make("dm_control/pendulum-swingup-v0")
print("high:", env.action_space.high)
print("low:", env.action_space.low)