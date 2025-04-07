import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")
import yaml
from Evaluation.EvalAgent.TrajectoryVisualizer import TrajectoryVisualizer

path_config = "Training/configs/VMAS_balance.yaml"
teachers_path = "Training/configs/teachers/"

# for filename in ["noCL_easy.yaml", "noCL_hard.yaml", "baby_steps.yaml", "online.yaml"]:
for filename in ["noCL_easy.yaml"]:
    with open(teachers_path+filename, "r") as file:
        config_changes = yaml.safe_load(file)

    agent = TrajectoryVisualizer(path_config, config_changes)
    agent.plotTrajectoriesEpoch(agent.epochs-1,3,"test")
    agent.videoEvolution(2,"test")
    del agent