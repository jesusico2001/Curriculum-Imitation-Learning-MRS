import torch, yaml
from Training.PathManager import PathManager
from LearnSystem import LearnSystemBuilder
from Training.TrainingAgent import TrainingAgent

class EvalAgent():
    def __init__(self, path_config_train, path_config_eval):
        with open(path_config_train, "r" ) as file:
            self.config_train = yaml.safe_load(file)

        with open(path_config_eval, "r" ) as file:
            config_eval = yaml.safe_load(file)
        
        self.tests = []
        for key, value in config_eval.items():
             
             pass
        
    
    def trainingLoop(self):
          raise Exception("EvalAgent cannot call trainingLoop(), only usable for evaluation.")

    
    def validationLossEvo(self, numAgen):
         

    