import sys, os, yaml, argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../..")
from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer
from pathlib import Path
import torch, gc
import numpy as np

from DatasetGenerator.GeneratorBuilder import GeneratorBuilder
from TrainEvalConfig import TrainEval


def main(gpu):
    path_configs = os.path.dirname(os.path.abspath(__file__))+"/configs/"
    path_config = path_configs + "VMAS_passage.yaml"

    text = "\hline\nTask+nRobots & Noise $\sigma$ & CL & $\mathcal{L}$ & $\mathcal{E}_\\text{pos}$ & $\mathcal{F}$ & $n_\\text{comp}$ \\\\\n\hline\n"
    
    task = "passage"
    table_metrics = ["L2_loss", "mean_pos_error", "frechet_distance", "n_task_completed"]
    num_agents = [6, 12]
    noise_levels = [0, 0.1, 0.25]
    teachers = ["noCL.yaml", "CL.yaml"]
    for na in num_agents:
        for noise in noise_levels:
            for teacher in teachers:
                with open(path_configs+teacher, "r") as file:
                    config_changes = yaml.safe_load(file)
                    config_changes["task.robot_obs_noise"] = noise
                    config_changes["task.num_agents"] = na
                    config_changes["general.device"] = "cuda:"+gpu
                agent = HistoryVisualizer(path_config, config_changes)

                val_loss = agent.__loadFromHistory__(f"loss_val_{agent.learn_system.task.episode_difficulty}_{na}robots")
                idx_min_loss = torch.argmin(torch.tensor( val_loss)).item()
                print(idx_min_loss)
                
                teacher_txt = "Yes" if teacher == "CL.yaml" else "No"
                text += f"{task}+{na} & {noise} & {teacher_txt}"
                metrics = agent.__loadEvalMetrics__("test")
                for metric in table_metrics:
                    text += f" & ${metrics[metric][idx_min_loss]:.3f}$"
                
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                text += f"\\\\\n"
            text += f"\\hline\n"
        text += f"\\hline\n"
    print(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=True)
    args = parser.parse_args()
    
main(args.gpu)
