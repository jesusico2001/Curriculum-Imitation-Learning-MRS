from abc import ABC, abstractmethod
import torch, numpy

class TeacherAgent(ABC):
    def __init__(self, config):
        self.maxDifficulty = config["max_difficulty"]
        self.difficultyResolution = config["difficulty_resolution"]
        if(self.maxDifficulty % self.difficultyResolution != 0):
            Exception("The maximum difficulty of a teacher must be divisible by it's resolution.")
        
        self.nDifficulties = int(self.maxDifficulty/self.difficultyResolution)
        self.difficulty_distr = torch.ones([self.nDifficulties]) / (self.nDifficulties)
        self.difficulty_grouper = BuilDifficultyGrouper(config["difficulty_grouper"], self.maxDifficulty, self.difficultyResolution)
    
    def getDifficultyDistribution(self):
        return self.difficulty_distr

    def getDifficulties(self, batch_size):
        teacher_difficulties = self.__getDifficulties(batch_size)
        return self.difficulty_grouper.transformDifficulties(teacher_difficulties)

    def __getDifficulties(self, batch_size):
        indices = numpy.arange(self.nDifficulties) + 1
        samples = numpy.random.choice(indices, size=batch_size, p=self.difficulty_distr.numpy())
        return torch.tensor(samples)


    def updateDifficulties(self, epoch, losses, isValEpoch):
        teacher_losses = self.transformValLoss(losses)

        return self.updateGroupedDifficulties(epoch, teacher_losses, isValEpoch)
    
    @abstractmethod
    def updateGroupedDifficulties(self, epoch, losses, isValEpoch):
        pass

    # Validation sampling
    # ===================
    def transformValLoss(self, student_loss_distr):
        teacher_loss = student_loss_distr[self.difficultyResolution-1::self.difficultyResolution]
        return teacher_loss

    def getValidationDifficulties(self, batch_size):
        valDifficulties =  self.sampleUniformDeterministic(self.nDifficulties, batch_size) * self.difficultyResolution
        # valDifficulties =  torch.ones([batch_size], dtype=int) * self.difficultyResolution * self.nDifficulties
        return valDifficulties

    def sampleUniformDeterministic(self, maxValue, numSamples):
        if numSamples % maxValue != 0:
            print("sampleUniformDeterministic: Validation batch size (", numSamples,") cannot be divided by the max difficulty (",maxValue,").")
            exit(0)
        samples_per_value =  int(numSamples / maxValue)
        difficulties = torch.arange(1,maxValue+1, dtype=int).unsqueeze(1)
        difficulties = torch.kron(difficulties, torch.ones((1,samples_per_value), dtype=int)).reshape(-1)

        return difficulties

# ===============================
# ||| Difficulty Downsampling |||
# ===============================
def BuilDifficultyGrouper(type, maxDifficulty, resolution):
    if type == "uniform":
        return UniformGrouper(maxDifficulty, resolution)
    elif type == "steps":
        return StepsGrouper(maxDifficulty, resolution)

class DifficultyGrouper(ABC):
    def __init__(self, maxDifficulty, resolution):
        self.maxDifficulty = maxDifficulty
        self.resolution = resolution
    
    @abstractmethod
    def transformDifficulties(self, teacherDifficulties):
        pass
        
class UniformGrouper(DifficultyGrouper):
    def transformDifficulties(self, teacherDifficulties):
        min_mapped_difficulties = (teacherDifficulties-1) * self.resolution + 1 
        max_mapped_difficulties = teacherDifficulties * self.resolution  
        student_difficulties = torch.tensor([torch.randint(min_mapped_difficulties[i].item(), max_mapped_difficulties[i].item(), (1,)) for i in range(len(min_mapped_difficulties))])

        return student_difficulties

class StepsGrouper(DifficultyGrouper):
    def transformDifficulties(self, teacherDifficulties):
        return teacherDifficulties * self.resolution
    
    

