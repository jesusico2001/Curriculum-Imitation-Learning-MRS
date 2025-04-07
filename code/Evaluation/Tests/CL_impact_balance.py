import sys, os, yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")
import torch
from TrainEvalConfig import TrainEval

path_config = "Training/configs/VMAS_balance.yaml"
teachers_path = "Training/configs/teachers/"
device = "cuda:0"

# for filename in ["noCL_easy.yaml", "noCL_hard.yaml", "baby_steps.yaml", "online.yaml"]:
for filename in ["baby_steps_basic.yaml"]:
    with open(teachers_path+filename, "r") as file:
        config_changes = yaml.safe_load(file)
        config_changes["general.device"] = device

    TrainEval(path_config, config_changes)
