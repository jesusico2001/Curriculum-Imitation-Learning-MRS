import torch, yaml
from abc import ABC
from Training.PathManager import PathManager
from LearnSystem import LearnSystemBuilder
from Training.TrainingAgent import TrainingAgent

class EvalAgent(TrainingAgent, ABC):
    def __init__(self, path_config_train, config_changes=None):
        super().__init__(path_config_train, config_changes)
        self.learn_system.eval()

    def trainingLoop(self):
          raise Exception("EvalAgent cannot call trainingLoop().")
    