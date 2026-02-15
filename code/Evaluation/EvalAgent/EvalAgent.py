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


    # ==========================================
    
    def __loadFromHistory__(self, filename):
        return torch.load(self.path_manager.getPathHistory()+"/"+filename+".pth")
    
    def __loadEvalMetrics__(self, dataset="val", numAgents=None, simulated_noise=None):
        na = self.learn_system.task.numAgents if numAgents is None else numAgents

        if simulated_noise is not None:
            path = self.path_manager.getPathEvaluation() + f"/metrics_{dataset}_{na}agents_{simulated_noise}noise.pth"
        else:
            path = self.path_manager.getPathEvaluation() + f"/metrics_{dataset}_{na}agents.pth"
            
        return torch.load(path)
