import sys, os, yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../..")

from pathlib import Path
import matplotlib.pyplot as plt

from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer

def plotAllNAgents(teacher, noise, colormap, num_agents=None):
    path_configs = os.path.dirname(os.path.abspath(__file__))+"/configs/"
    path_config = path_configs + "VMAS_navigation.yaml"

    if num_agents is None:
        numAgents = [8, 10]
    else:
        numAgents = [num_agents]
    
    colors = plt.cm.get_cmap(colormap, len(numAgents)+1)
    for i, na in enumerate(numAgents):
        with open(path_configs+teacher, "r") as file:
            config_changes = yaml.safe_load(file)
            config_changes["task.robot_obs_noise"] = noise
            config_changes["task.num_agents"] = na

            agent = HistoryVisualizer(path_config, config_changes)

            difficulty = agent.learn_system.task.episode_difficulty
            epochs = agent.__loadFromHistory__("val_epochs")
            loss_val = agent.__loadFromHistory__("loss_val_"+str(difficulty)+"_"+str(agent.learn_system.task.numAgents)+"robots")
            loss_val_formatted = [t.cpu().tolist() for t in loss_val]
            
            label = teacher.split(".")[0]+str(na)+"agents"
            plt.plot(epochs, loss_val_formatted, linestyle='solid', alpha=1, color=colors(i), label=label)       
   
    return difficulty

def main():
    path_saves = os.path.dirname(os.path.abspath(__file__))+"/results/loss_evo/"
    os.makedirs(path_saves, exist_ok=True)
    numAgents = [8, 12]
    noise_levels = [0, 0.05, 0.1, 0.25]
    for na in numAgents:
        for noise in noise_levels:
            plt.figure(figsize=(10, 6)) 

            difficulty = plotAllNAgents("noCL.yaml", noise, "autumn", num_agents=na)
            difficulty = plotAllNAgents("CL.yaml", noise, "winter", num_agents=na)
            plt.title("Loss Evolution - noise: "+str(noise)+" | Difficulty:"+str(difficulty), fontsize=25)
            plt.yscale('log')
            plt.xlabel('Iterations', fontsize=22)
            plt.ylabel('Error', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)
            plt.grid(True)
            plt.savefig(path_saves+str(na)+"robots_"+str(noise)+".png")   
            plt.cla()  




main()