import sys, os, yaml, argparse, shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../..")
from pathlib import Path

from DatasetGenerator.GeneratorBuilder import GeneratorBuilder
from TrainEvalConfig import TrainEval

def main(teacher, gpu):
    path_configs = os.path.dirname(os.path.abspath(__file__))+"/configs/"
    path_results = os.path.dirname(os.path.abspath(__file__))+"/results/"
    path_config = path_configs + "VMAS_navigation.yaml"

    num_agents = [8, 12]
    noise_levels = [0, 0.05, 0.1, 0.25]
    for na in num_agents:
        for noise in noise_levels:
            path = path_results+str(na)+"robots/"+str(noise)+"noise/"+teacher+"/"

            with open(path_configs+teacher, "r") as file:
                config_changes = yaml.safe_load(file)
                config_changes["task.robot_obs_noise"] = noise
                config_changes["task.num_agents"] = na
                config_changes["general.device"] = "cuda:"+gpu
            
            
            gen = GeneratorBuilder(path_config, config_changes)
            path_origin = gen.path_manager.getPathEvaluation()
            if not os.path.isdir(path_origin) or not os.listdir(path_origin):
                print(gen.path_manager.getPathDatasets()+" is not an existing path.")
                exit()
            shutil.copytree(path_origin, path, dirs_exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True, help="Filename for teacher config in \"configs\" folder.")
    parser.add_argument("--gpu", type=str, required=True)
    args = parser.parse_args()
    
    main(args.teacher, args.gpu)
