
from enum import Enum
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from DatasetGenerator.Generator import Generator 
from DatasetGenerator.RealSystem.RealSystemGenerator import RealSystemGenerator
from DatasetGenerator.VMAS.VMASGenerator import VMASGenerator
from DatasetGenerator.VMAS.PassageGenerator import PassageGenerator
import yaml

class Generators(Enum):
    real_system = "RealSystem"
    vmas = "VMAS"

def GeneratorBuilder(path_config, config_changes):
    if isinstance(path_config, str):
        with open(path_config, "r") as file:
            config = yaml.safe_load(file)
        print("Config loaded...")

    if config_changes:
        for key, value in config_changes.items():
            keys = key.split('.')
            cfg = config
            for k in keys[:-1]:
                cfg = cfg[k]
            cfg[keys[-1]] = value
            print("Config parameter updated: ", key, "=", value)

    if config["task"]["lib"] == Generators.real_system.value:
        return RealSystemGenerator(config)
    if config["task"]["lib"] == Generators.vmas.value:
        if config["task"]["type"] == "passage":
            return PassageGenerator(config)
        
        return VMASGenerator(config)
    else:
        print("Unknown task library: ", config["task"]["lib"])
        exit(0)