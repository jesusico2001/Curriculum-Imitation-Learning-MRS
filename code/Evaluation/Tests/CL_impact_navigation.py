import sys, os, yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")

from TrainEvalConfig import TrainEval

path_config = "Training/configs/VMAS_navigation.yaml"
teachers_path = "Training/configs/teachers/"
device = "cuda:1"

for filename in ["noCL_easy.yaml", "baby_steps_basic.yaml"]:

    with open(teachers_path+filename, "r") as file:
        config_changes = yaml.safe_load(file)
        config_changes["general.device"] = device

    TrainEval(path_config, config_changes)
