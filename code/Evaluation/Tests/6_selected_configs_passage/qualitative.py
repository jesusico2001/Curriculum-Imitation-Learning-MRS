import sys, os, yaml, argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../..")
from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer
from Evaluation.EvalAgent.TrajectoryVisualizer import TrajectoryVisualizer
from pathlib import Path
import torch, gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def plotPassage(agent, obs):
    t = agent.learn_system.task
    agent_pos = obs[t.feature_index["robot_positions"]].reshape(t.numAgents, 2)
    passage_pos_aux = obs[t.feature_index["pos_passage_for_agent"]].reshape(t.numAgents, 2)
    passage_pos = agent_pos[0] + passage_pos_aux[0]
    px = passage_pos[0].cpu().item()
    
    gap = 0.5
    rect_height = 0.5

    # Calculate widths of the two rectangles
    left_limit = -2.5
    right_limit = 2.5

    # Rectangle 1: left of the passage
    rect1_x = left_limit
    rect1_width = min(px - gap / 2, right_limit) - left_limit
    rect1_y = -rect_height / 2
    rect1 = patches.Rectangle((rect1_x, rect1_y), rect1_width, rect_height, linewidth=2, edgecolor='k', facecolor='gray', alpha=0.7)
    plt.gca().add_patch(rect1)

    # Rectangle 2: right of the passage
    rect2_x = max(px + gap / 2, left_limit)
    rect2_width = right_limit - rect2_x
    rect2_y = -rect_height / 2
    rect2 = patches.Rectangle((rect2_x, rect2_y), rect2_width, rect_height, linewidth=2, edgecolor='k', facecolor='gray', alpha=0.7)
    plt.gca().add_patch(rect2)

def figure(expert_traj, pred_traj, idx, agent, teacher):
    na = agent.learn_system.task.numAgents
    noise = agent.learn_system.task.robot_obs_noise
    # for i,f in [[0,150], [150,300]]:
    for i,f in [[0,300]]:
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({'font.size': 20})
        plotPassage(agent, expert_traj[0,idx,:])
        agent.plotTrajectories(expert_traj[:,idx,:], 'solid', "Expert trajectory", linewidth=4)
        agent.plotTrajectoriesClean(pred_traj[i:f,idx,:], teacher, step=2)
        agent.plotFinalPos(expert_traj[i:f,idx,:])
        agent.plotFinalPos(pred_traj[i:f,idx,:], marker='o')
        plt.xlim(-2.5, 2.5)
        # plt.xlabel('x $[$m$]$', fontsize=25)
        # plt.ylabel('y $[$m$]$', fontsize=25)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        plt.xticks([])
        plt.yticks([])
        plt.legend(fontsize=20)
        
        plt.savefig(path_saves+str(na)+"agents_noise"+str(noise)+"_"+str(idx.item())+f"_{teacher}_{f}.png")   
        plt.cla()  

        # plt.show()

path_configs = os.path.dirname(os.path.abspath(__file__))+"/configs/"
path_config = path_configs + "VMAS_passage.yaml"
path_saves = os.path.dirname(os.path.abspath(__file__))+"/results/qualitative/"

def main(gpu):
    os.makedirs(path_saves, exist_ok=True)

    num_agents = [6]
    noise_levels = [0.25]
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
            for idx in sorted_idx[:250:5]:
                # idx = idx.item()  # Convert from tensor to int if needed
                print(f"Index: {idx}, Loss_noCL: {losses_noCL[idx]:.4f}, Loss_CL: {losses_CL[idx]:.4f}, Diff: {Loss_diff[idx]:.4f}")
                figure(real_trajectories, traj_CL, idx, agent_CL, "CL")
                figure(real_trajectories, traj_noCL, idx, agent_noCL, "noCL")

            del agent_noCL, agent_CL
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=True)
    args = parser.parse_args()
    
main(args.gpu)
