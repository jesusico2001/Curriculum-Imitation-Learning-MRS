import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from abc import ABC, abstractmethod
import torch
from Training.PathManager import PathManager
from Task.TaskBuilder import TaskBuilder

class Generator(ABC):
    def __init__(self, config):
        self.config = config
                
        # Path manager
        self.path_manager = PathManager(self.config)
        
        self.na = int(self.config["task"]["num_agents"])
        self.episode_difficulty = int(self.config["task"]["episode_difficulty"])
        
        self.numTrain = int(self.config["general"]["train_size"])
        self.numVal = int(self.config["general"]["val_size"])
        self.numTest = int(self.config["general"]["test_size"])
        
        self.task = TaskBuilder(config)

        self.seed = int(self.config["general"]["seed_data"])
        torch.manual_seed(self.seed)
        
        try:
            os.makedirs(self.path_manager.getPathDatasets())
        except FileExistsError:
            pass 

        self.device = torch.device(config["general"]["device"] if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def generateDataset(self, numData):
        pass
    

    def generateTrainValTest(self):
        if self.task.observation_noise_factor != 0:
            self.generateNoisyTrainValTest()
            return
        
        train_data = self.generateDataset(self.numTrain)
        torch.save(train_data, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train"))

        val_data = self.generateDataset(self.numVal)
        torch.save(val_data, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val"))
        
        test_data = self.generateDataset(self.numTest)
        torch.save(test_data, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test"))
                
    def generateValidation(self):
        val_data = self.generateDataset(self.numVal)
        torch.save(val_data, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val"))

    # Noisy observations
    def generateNoisyTrainValTest(self):
        # Temporally update config to access or generate data without noise
        print("Generating noisy dataset, looking for an existing one to denoise...")
        old_noise_factor = self.task.observation_noise_factor
        self.task.observation_noise_factor = 0
        self.config["task"]["observation_noise_factor"] = 0
        
        if not os.path.exists(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train")):
            print("Datasets without noise not found. Generating them...")
            self.generateTrainValTest()

        train_noNoise = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train")).to(self.device)
        val_noNoise = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val")).to(self.device)
        test_noNoise = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test")).to(self.device)
        print("Datasets without noise lodaded. Adding noise to observations...")



        # Return to original config and add noise
        self.task.observation_noise_factor = old_noise_factor
        self.config["task"]["observation_noise_factor"] =old_noise_factor

        # Copy of noiseless data to new config (for evaluation with GT)
        torch.save(train_noNoise, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", False))
        torch.save(val_noNoise, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", False))
        torch.save(test_noNoise, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", False))

        # Noisy datasets

        train_data = self.task.addNoise(train_noNoise)
        torch.save(train_data, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train"))
        
        val_data =  self.task.addNoise(val_noNoise)
        torch.save(val_data, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val"))
        
        test_data = self.task.addNoise(test_noNoise)
        torch.save(test_data, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test"))