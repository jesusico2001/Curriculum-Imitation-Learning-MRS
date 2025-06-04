import torch
from Training.PathManager import PathManager

class DatasetBuilder():
    def __init__(self, training_config, device):
        self.config=training_config
        self.numTrain = int(training_config["general"]["train_size"])
        self.numVal = int(training_config["general"]["val_size"])
        self.numTest = int(training_config["general"]["test_size"])

        self.seed_data = int(training_config["general"]["seed_data"])

        self.path_manager = PathManager(training_config)

        self.device = device
        
    def BuildDatasets(self, load_actions):
        # Observations
        train_data = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train"), weights_only=True).to(self.device)
        val_data = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val"), weights_only=True).to(self.device)
        
        # Actions
        if load_actions:
            train_actions = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", data="actions"), weights_only=True).to(self.device)
            val_actions = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", data="actions"), weights_only=True).to(self.device)
        else:
            train_actions = None
            val_actions = None

        return train_data, train_actions,  val_data, val_actions
    
    def BuildArbitraryNumAgents(self, split, nAgents, noisy=False, data="trajectories"):
        aux_cfg = self.config.copy()
        aux_cfg["task"]["num_agents"] = str(nAgents)
        aux_pm = PathManager(aux_cfg)

        return torch.load(aux_pm.getPathDatasets()+aux_pm.getDatasetFilename(split, data=data, noisy=noisy), weights_only=True).to(self.device)
    