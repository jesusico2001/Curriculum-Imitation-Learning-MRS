import numpy
import torch
from Training.Teacher.TeacherAgent import TeacherAgent

class OnlineTeacher(TeacherAgent):
    def __init__(self, config):        
        super().__init__(config)
        
        self.difficulty_distr = torch.ones([self.maxDifficulty]) / self.maxDifficulty
        self.old_loss_val_distr = torch.ones([self.maxDifficulty]) * 10000
    
    def getDifficulties(self, batch_size):
        indices = numpy.arange(self.maxDifficulty) + 1
        samples = numpy.random.choice(indices, size=batch_size, p=self.difficulty_distr.numpy())
        return torch.tensor(samples)


    def getDifficultyDistribution(self):
        return self.difficulty_distr

    def updateDifficulties(self, epoch, losses, isValEpoch):
        if not isValEpoch:
            return
        
        self.difficulty_distr += losses
        self.difficulty_distr /= sum(self.difficulty_distr)

