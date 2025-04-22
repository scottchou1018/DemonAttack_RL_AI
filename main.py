from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import ale_py
import numpy as np
import torch


from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.logger import configure
import os
import time


DIR_NAME = "collect_data3"
if __name__ == '__main__':
    os.makedirs(DIR_NAME, exist_ok=True)
    os.makedirs(os.path.join(DIR_NAME, "monitor"), exist_ok=True)
    env = make_atari_env("ALE/DemonAttack-v5", n_envs=8, env_kwargs={"difficulty": 1}
                        ,monitor_dir=os.path.join(DIR_NAME, "monitor"))
    
    model = DQN("CnnPolicy", env, verbose=1, device="cuda", buffer_size=100_000,
                exploration_final_eps=0.01, exploration_fraction=0.1, exploration_initial_eps=1.0,)
    
    logger = configure(folder=DIR_NAME, format_strings=["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    
    for i in range(100):
        model.learn(total_timesteps=100_000, tb_log_name=os.path.join(DIR_NAME), reset_num_timesteps=True)
        model.save(os.path.join(DIR_NAME, "model"))

    env = make_atari_env("ALE/DemonAttack-v5"
                        ,monitor_dir=os.path.join(DIR_NAME, "monitor"))
    
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render('human')
        time.sleep(0.01)
    
