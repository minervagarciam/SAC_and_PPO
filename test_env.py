import gymnasium as gym
import shimmy

env = gym.make("dm_control/pendulum-swingup-v0")
obs, _ = env.reset()
for i in range(1100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        print(f"Episode ended at step {i}, terminated={terminated}, truncated={truncated}")
        print(f"info keys: {info.keys()}")
        break