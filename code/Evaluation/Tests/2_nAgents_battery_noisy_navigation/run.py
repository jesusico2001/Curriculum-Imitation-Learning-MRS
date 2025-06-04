import sys, os, yaml, argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../..")

import torch, gc
from DatasetGenerator.GeneratorBuilder import GeneratorBuilder
from TrainEvalConfig import TrainEval

def main(teacher, gpu):
    path_configs = os.path.dirname(os.path.abspath(__file__))+"/configs/"
    path_config = path_configs + "VMAS_navigation.yaml"

    nAgents_list = range(12,13,2)
    for nAgents in nAgents_list:
        with open(path_configs+teacher, "r") as file:
            config_changes = yaml.safe_load(file)
            config_changes["task.num_agents"] = nAgents
            config_changes["general.device"] = "cuda:"+gpu
        
        
        gen = GeneratorBuilder(path_config, config_changes)
        if not os.path.isdir(gen.path_manager.getPathDatasets()) or not os.listdir(gen.path_manager.getPathDatasets()):
            gen.generateTrainValTest()
        del gen
        
        TrainEval(path_config, config_changes)
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True, help="Filename for teacher config in \"configs\" folder.")
    parser.add_argument("--gpu", type=str, required=True, help="Filename for teacher config in \"configs\" folder.")
    args = parser.parse_args()
    
    main(args.teacher, args.gpu)
