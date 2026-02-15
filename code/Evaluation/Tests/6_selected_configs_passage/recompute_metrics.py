import sys, os, yaml, argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../..")
from Evaluation.EvalAgent.MetricComputer import MetricComputer
from pathlib import Path
import torch, gc

from DatasetGenerator.GeneratorBuilder import GeneratorBuilder
from TrainEvalConfig import TrainEval

def main(teacher, gpu):
    path_configs = os.path.dirname(os.path.abspath(__file__))+"/configs/"
    path_config = path_configs + "VMAS_passage.yaml"

    num_agents = [6, 12]
    
    noise_levels = [0, 0.1]

    for na in num_agents:
        for noise in noise_levels:
            with open(path_configs+teacher, "r") as file:
                config_changes = yaml.safe_load(file)
                config_changes["task.robot_obs_noise"] = noise
                config_changes["task.num_agents"] = na
                config_changes["general.device"] = "cuda:"+gpu
            
            
            gen = GeneratorBuilder(path_config, config_changes)
            if not os.path.isdir(gen.path_manager.getPathDatasets()) or len(os.listdir(gen.path_manager.getPathDatasets())) < 2:
                print(gen.path_manager.getPathDatasets()+" is not an existing path.")
                gen.generateTrainValTest()
            del gen
            
            agent = MetricComputer(path_config, config_changes)
            agent.metricsEvolution(checkExisting=False, simulated_noise=0.25)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True, help="Filename for teacher config in \"configs\" folder.")
    parser.add_argument("--gpu", type=str, required=True)
    args = parser.parse_args()
    
    main(args.teacher, args.gpu)
