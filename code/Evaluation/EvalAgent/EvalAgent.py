import torch, yaml
from abc import ABC
from Training.PathManager import PathManager
from LearnSystem import LearnSystemBuilder
from Training.TrainingAgent import TrainingAgent

class EvalAgent(TrainingAgent, ABC):
    def __init__(self, path_config_train, config_changes=None):
        super().__init__(path_config_train, config_changes)
        self.learn_system.eval()
        self.loaded_checkpoint_epoch = -1

        self.learn_system.return_noisy_obs = False
        self.learn_system.action_loss = False
        
    def trainingLoop(self):
          raise Exception("EvalAgent cannot call trainingLoop().")

    def load_checkpoint(self, epoch):
        if self.loaded_checkpoint_epoch != epoch:
            self.learn_system.load_state_dict(torch.load(self.path_manager.getPathCheckpoints()+"/epoch_"+str(epoch)+".pth", map_location=self.device, weights_only=True),)
            self.loaded_checkpoint_epoch = epoch
