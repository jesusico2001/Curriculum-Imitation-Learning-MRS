import sys, os, yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../..")

from pathlib import Path
import matplotlib.pyplot as plt

from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer

def plotMetric(metric, teacher, noise, color, num_agents, plotExperts=True, linestyle='solid', long_label=False):
    path_configs = os.path.dirname(os.path.abspath(__file__))+"/configs/"
    path_config = path_configs + "VMAS_passage.yaml"
    na = num_agents
    if num_agents is None:
        numAgents = [8, 10]
    else:
        numAgents = [num_agents]
    
    with open(path_configs+teacher, "r") as file:
        config_changes = yaml.safe_load(file)
        config_changes["task.robot_obs_noise"] = noise
        config_changes["task.num_agents"] = na
        config_changes["general.device"] = "cpu"  # Use CPU for plotting

        agent = HistoryVisualizer(path_config, config_changes)

        difficulty = agent.learn_system.task.episode_difficulty
        epochs = agent.__loadFromHistory__("val_epochs")
        
        simNoise = 0.25 if noise!=0.25 else None
        metrics = agent.__loadEvalMetrics__("test", simulated_noise=simNoise)
        values = metrics[metric]

        if plotExperts and metric in metrics["real_metrics"]:
            print("Plotting expert metric")
            expert_metric = metrics["real_metrics"][metric]
            if hasattr(expert_metric, "cpu"):
                expert_metric = expert_metric.cpu()

            expert_values = [expert_metric] * len(epochs)
            plt.plot(epochs, expert_values, linestyle='solid', linewidth=3, alpha=0.7, color=color, label="Expert"+str(na)+"agents")

        if hasattr(values[0], "cpu"):
            values_formatted = [t.cpu().tolist() for t in values]
        else:
            values_formatted = values
        label = teacher.split(".")[0] + f" with $\sigma={noise}$"
        if long_label:
            label = metric+"_"+label
        plt.plot(epochs, values_formatted, linestyle=linestyle, linewidth=3, alpha=1, color=color, label=label)       

    return difficulty

def plotL2_in_out(noise, num_agents):
    na= num_agents
    plt.figure(figsize=(12, 8)) 
    difficulty = plotMetric("L2_loss", "noCL.yaml", noise, "blue", num_agents=na, plotExperts=False, linestyle='dashed', long_label=True)
    difficulty = plotMetric("L2_loss", "CL.yaml", noise, "blue", num_agents=na, plotExperts=False, long_label=True)
    difficulty = plotMetric("L2_inliers", "noCL.yaml", noise, "green", num_agents=na, plotExperts=False, linestyle='dashed', long_label=True)
    difficulty = plotMetric("L2_inliers", "CL.yaml", noise, "green", num_agents=na, plotExperts=False, long_label=True)
    difficulty = plotMetric("L2_outliers", "noCL.yaml", noise, "red", num_agents=na, plotExperts=False, linestyle='dashed', long_label=True)
    difficulty = plotMetric("L2_outliers", "CL.yaml", noise, "red", num_agents=na, plotExperts=False, long_label=True)
    plt.title("L2 evolution - noise="+str(noise)+" | Difficulty:"+str(difficulty), fontsize=25)
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=22)
    plt.ylabel("L2_losss", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)

metric_name = {
    "L2_loss": "$\mathcal{L}$",
    "mean_pos_error": "$\mathcal{E}_{pos}$",
    "frechet_distance": "$\mathcal{F}$",
    "smoothness": "Smoothness",
    "mean_reward": "Mean Reward",
    "n_task_completed": "$n_{comp}$",
    "num_collisions": "Number of Collisions"
}   
def main():
    path_saves = os.path.dirname(os.path.abspath(__file__))+"/results/metrics_evo/noisy/"
    os.makedirs(path_saves, exist_ok=True)


    metrics = ["L2_loss", "mean_pos_error", "frechet_distance", "smoothness",
            "mean_reward", "n_task_completed", "num_collisions"]
    metrics = ["L2_loss"]
    numAgents = [6, 12]
    noise_levels = [0, 0.1, 0.25]

    # colors = plt.get_cmap("viridis", len(noise_levels)+1)
    colors = ['#0072B2', '#E69F00', '#009E73']
    for metric in metrics:
        for na in numAgents:
            plt.figure(figsize=(12, 8)) 
            for i, noise in enumerate(noise_levels):
                # difficulty = plotMetric(metric, "noCL.yaml", noise, colors(i), num_agents=na, plotExperts=True, linestyle='dashed')
                difficulty = plotMetric(metric, "CL.yaml", noise, colors[i], num_agents=na, plotExperts=False)

            # plt.title(metric+" evolution - "+str(na)+" agents | Difficulty:"+str(difficulty), fontsize=25)
            plt.yscale('log')
            plt.xlabel('Iterations', fontsize=30)
            plt.ylabel(metric_name[metric], fontsize=40)
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.legend(fontsize=25)
            plt.grid(True)
            plt.savefig(path_saves+metric+str(na)+"agents.png")   
            plt.cla()  
            # plt.show() 


main()