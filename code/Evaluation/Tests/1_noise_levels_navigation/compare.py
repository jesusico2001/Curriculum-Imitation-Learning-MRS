import sys, os, yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../..")

from pathlib import Path
import matplotlib.pyplot as plt

from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer

def plotAllNoises(teacher, colormap):
    path_configs = os.path.dirname(os.path.abspath(__file__))+"/configs/"
    path_config = path_configs + "VMAS_navigation.yaml"

    noise_levels = [0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1]
    colors = plt.cm.get_cmap(colormap, len(noise_levels)+1)
    for i, noise in enumerate(noise_levels):
        with open(path_configs+teacher, "r") as file:
            config_changes = yaml.safe_load(file)
            config_changes["task.robot_obs_noise"] = noise
            
            agent = HistoryVisualizer(path_config, config_changes)

            difficulty = agent.learn_system.task.episode_difficulty
            epochs = agent.__loadFromHistory__("val_epochs")
            loss_val = agent.__loadFromHistory__("loss_val_"+str(difficulty)+"_"+str(agent.learn_system.task.numAgents)+"robots")
            loss_val_formatted = [t.cpu().tolist() for t in loss_val]
            
            label = teacher.split(".")[0]+str(noise)
            plt.plot(epochs, loss_val_formatted, linestyle='solid', alpha=1, color=colors(i), label=label)

    return difficulty

def main():
    
    difficulty = plotAllNoises("noCL.yaml", "autumn")
    plotAllNoises("CL.yaml", "winter")
    
    plt.title('Loss Evolution | Difficulty:'+str(difficulty), fontsize=25)
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=22)
    plt.ylabel('Error', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()
        

main()