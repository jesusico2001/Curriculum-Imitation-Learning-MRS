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
        
    def BuildDatasets(self, numSamples):
        # Train
        train_data = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", True), weights_only=True).to(self.device)
        
        # Validation
        val_data = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", True), weights_only=True).to(self.device)
        
        return train_data, val_data
    
    def BuildArbitraryNumAgents(self, split, nAgents, noisyObs):
        aux_cfg = self.config.copy()
        aux_cfg["task"]["num_agents"] = str(nAgents)
        aux_pm = PathManager(aux_cfg)
        data = torch.load(aux_pm.getPathDatasets()+aux_pm.getDatasetFilename(split, noisyObs), weights_only=True).to(self.device)
        
        return data
    