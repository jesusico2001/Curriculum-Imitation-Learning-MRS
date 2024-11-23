from abc import ABC, abstractmethod

class TeacherAgent(ABC):
    def __init__(self, config):
        self.maxDifficulty = config["max_difficulty"]
    
    @abstractmethod
    def getDifficulties(self, batch_size):
        pass
    
    @abstractmethod
    def getDifficultyDistribution(self):
        pass

    @abstractmethod
    def updateDifficulties(self, epoch, losses, isValEpoch):
        pass
