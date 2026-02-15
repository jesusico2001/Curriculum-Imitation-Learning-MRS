import sys, os, yaml, argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../..")
from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer
from Evaluation.EvalAgent.TrajectoryVisualizer import TrajectoryVisualizer
from pathlib import Path
import torch, gc
import numpy as np
import matplotlib.pyplot as plt

from DatasetGenerator.GeneratorBuilder import GeneratorBuilder
from TrainEvalConfig import TrainEval


def loadBestEpoch(agent):
        na = agent.learn_system.task.numAgents
        epochs = agent.__loadFromHistory__("val_epochs")
        val_loss = agent.__loadFromHistory__(f"loss_val_{agent.learn_system.task.episode_difficulty}_{na}robots")
        idx_min_loss = torch.argmin(torch.tensor( val_loss)).item()
        best_epoch = epochs[idx_min_loss]
        agent.load_checkpoint(best_epoch)
        print("Loaded checkpoint for epoch: ", best_epoch)

def getLosses(real_trajectories, simulated_trajectories, task):
        real_states = task.getRobotStates(real_trajectories)
        sim_states = task.getRobotStates(simulated_trajectories)
        losses = (real_states - sim_states).pow(2).mean(dim=(0, 2))
        print("losses shape:", losses.shape)
        
        return losses

def main(gpu):
    path_configs = os.path.dirname(os.path.abspath(__file__))+"/configs/"
    path_config = path_configs + "VMAS_navigation.yaml"
    path_saves = os.path.dirname(os.path.abspath(__file__))+"/results/qualitative/"
    os.makedirs(path_saves, exist_ok=True)

    num_agents = [12]
    noise_levels = [0, 0.25]
    for na in num_agents:
        real_trajectories = None

        for noise in noise_levels:
            with open(path_configs+"noCL.yaml", "r") as file:
                config_changes = yaml.safe_load(file)
                config_changes["task.robot_obs_noise"] = noise
                config_changes["task.num_agents"] = na
                config_changes["general.device"] = "cuda:"+gpu
                agent_noCL = TrajectoryVisualizer(path_config, config_changes)
                loadBestEpoch(agent_noCL)
            with open(path_configs+"CL.yaml", "r") as file:
                config_changes = yaml.safe_load(file)
                config_changes["task.robot_obs_noise"] = noise
                config_changes["task.num_agents"] = na
                config_changes["general.device"] = "cuda:"+gpu
                agent_CL = TrajectoryVisualizer(path_config, config_changes)
                loadBestEpoch(agent_CL)

            if real_trajectories is None:
                data = agent_CL.dataset_builder.BuildArbitraryNumAgents("test", na)
                real_trajectories = data[:, :500, :] 
                init_states = real_trajectories[0, :, :]  # Usar el primer estado como inicial

            difficulty = agent_CL.learn_system.task.episode_difficulty
            with torch.no_grad():
                traj_noCL, _, _ = agent_noCL.learn_system.forward(init_states, difficulty)
                traj_CL, _, _ = agent_CL.learn_system.forward(init_states, difficulty)
            print("Traectories loaded for noise:", noise)

            losses_noCL = getLosses(real_trajectories, traj_noCL, agent_noCL.learn_system.task)
            losses_CL = getLosses(real_trajectories, traj_CL, agent_noCL.learn_system.task)
            Loss_diff =  losses_CL - losses_noCL
            sorted_idx = torch.argsort(Loss_diff)
            for idx in sorted_idx[:10]:
                # idx = idx.item()  # Convert from tensor to int if needed
                print(f"Index: {idx}, Loss_noCL: {losses_noCL[idx]:.4f}, Loss_CL: {losses_CL[idx]:.4f}, Diff: {Loss_diff[idx]:.4f}")
                plt.figure(figsize=(12, 8))
                plt.rcParams.update({'font.size': 20})
                agent_CL.plotTrajectories(real_trajectories[:,idx,:], 'solid', "Expert", linewidth=4)
                agent_CL.plotTrajectories(traj_noCL[:,idx,:], 'dashed', "NoCL", linewidth=4)
                # agent_CL.plotTrajectories(traj_CL[:,idx,:], 'dotted', "CL")
                agent_CL.plotTrajectoriesClean(traj_CL[:,idx,:], "CL")
                agent_CL.plotFinalPos(real_trajectories[:,idx,:])
                plt.xlabel('x $[$m$]$', fontsize=23)
                plt.ylabel('y $[$m$]$', fontsize=23)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.legend(fontsize=25)
                plt.savefig(path_saves+str(na)+"agents_noise"+str(noise)+"_"+str(idx.item())+".png")   
                plt.cla()  

                # plt.show()

            del agent_noCL, agent_CL
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=True)
    args = parser.parse_args()
    
main(args.gpu)
