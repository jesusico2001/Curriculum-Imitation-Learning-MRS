import numpy
from abc import ABC, abstractmethod

from Training.Teacher.TeacherAgent import TeacherAgent

class OnlineTeacher(TeacherAgent):
    def __init__(self, config):        
        super().__init__(config)
        
        self.learning_rate = config["learning_rate"]
        self.reward = BuildReward(config["reward"])

    def updateGroupedDifficulties(self, epoch, losses, isValEpoch):
        if not isValEpoch:
            return
        
        self.difficulty_distr += self.learning_rate * self.reward.lossToReward(losses)
        self.difficulty_distr /= sum(self.difficulty_distr)

# ==============
# ||| Reward |||
# ==============
def BuildReward(type):
    if type == "l2loss":
        return L2Reward()
    elif type == "l2gain":
        return L2GainReward()
    elif type == "l2totalgain":
        return L2TotalGainReward()
    
class Reward(ABC):
    @abstractmethod
    def lossToReward(self, loss_distr):
        pass
        
class L2Reward(Reward):    
    def lossToReward(self, loss_distr):
        return loss_distr

class L2GainReward(Reward):
    def lossToReward(self, loss_distr):
        try:
            self.previousLosses
        except AttributeError:
            self.previousLosses = loss_distr
        
        rewards = self.previousLosses - loss_distr

        # Ensure non-negative rewards to avoid 
        # negative probabilities at difficulties
        rewards[:] = numpy.clip(rewards, 0, None)

        # Update previous losses
        self.previousLosses = loss_distr
        return rewards

class L2TotalGainReward(Reward):
    def lossToReward(self, loss_distr):
        try:
            self.previousLosses
        except AttributeError:
            self.previousLosses = loss_distr
        
        rewards = self.previousLosses - loss_distr

        # Ensure non-negative rewards to avoid 
        # negative probabilities at difficulties
        rewards[:] = numpy.clip(rewards, 0, None)

        return rewards