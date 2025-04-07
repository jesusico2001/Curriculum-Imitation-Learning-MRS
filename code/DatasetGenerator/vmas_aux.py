import sys, os, time
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

import gymnasium, vmas
import torch
import matplotlib.pyplot as plt
from dataclasses import asdict

from benchmarl.models.mlp import MlpConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments import VmasTask
from benchmarl.algorithms import MappoConfig

from benchmarl.hydra_config import reload_experiment_from_file
# restore_file = "/mnt/hdd/JesusRoche/Curriculum-Imitation-Learning-MRS/code/saves/datasets/balance_VMAS/4_100/mappo_balance_mlp__70b5938a_25_01_12-22_56_59/checkpoints/checkpoint_300000.pt"
# model_config = MlpConfig.get_from_yaml()
# experiment = reload_experiment_from_file(str(restore_file))

# agent = experiment.policy

obs_history = []
env = vmas.make_env(scenario="navigation", n_agents=4, num_envs=1, max_steps=500, terminated_truncated=True, wrapper="gymnasium")
print(dir(env.env))
input()
print(env.observation_space)
input()
for episode in range(50):
    obs, info = env.reset()

    totalReward, step = 0, 0
    done, truncated = False, False
    while not (done or truncated):
        env.render()
        action = agent.forward(obs)[3]
        obs, reward, done, truncated, info = env.step(action)
        
        obs_history.append(obs)
        totalReward += sum(reward)
        step += 1
        time.sleep(0.15)
    
    print("Episode Reward:", totalReward, " - Steps:", step)
