from Training.Teacher.TeacherAgent import TeacherAgent
import torch

MIN_INTERVAL = 350
MIN_INCREMENT = 1

class BabyStepsTeacher(TeacherAgent):
    def __init__(self, config):        
        super().__init__(config)
        
        self.interval_function = CuadraticFunction.ModelInterval(config["interval_policy"], config["interval_parameter"])
        self.increment_function = CuadraticFunction.ModelIncrement(config["increment_policy"], config["increment_parameter"])
        
        self.difficulty = 4
        self.epochNextIncrement = 0
        self.old_task_quota = config["old_task_quota"]
        
        # Initialize distribution
        self.updateDifficulties(0, torch.ones(self.nDifficulties), None)

    
    def updateGroupedDifficulties(self, epoch, losses, isValEpoch):
        if epoch >= self.epochNextIncrement and self.difficulty < self.maxDifficulty:
            # First Increase numSamples
            self.difficulty += self.increment_function.compute(self.difficulty)
            if self.difficulty > self.maxDifficulty:
                self.difficulty = self.maxDifficulty
            
            # Then update next interval
            self.epochNextIncrement += self.interval_function.compute(self.difficulty)

            # Set the distriubtion
            fullres_difficulty_distr = torch.zeros(self.maxDifficulty) 
            for i in range(self.difficulty-1):
                fullres_difficulty_distr[i] = self.old_task_quota / (self.difficulty-1)
            
            fullres_difficulty_distr[self.difficulty-1] = 1 - self.old_task_quota

            # Group diificulties to match resolution
            self.difficulty_distr = torch.zeros(self.nDifficulties)
            for fullres_difficulty, prob in enumerate(fullres_difficulty_distr):
                self.difficulty_distr[int((fullres_difficulty-1)/self.difficultyResolution)] += prob
                
class CuadraticFunction():
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def compute(self, x):
        return int(self.a * pow(x, 2) + self.b * x + self.c)

    @staticmethod
    def modelConstant(c):
        return CuadraticFunction(0, 0, c)
    
    @staticmethod
    def modelLinear(minVal, maxVal):
        b = (maxVal - minVal) / 245
        c = minVal - 5*b
        return CuadraticFunction(0, b, c)
    
    @staticmethod
    def modelLinearModulated(minVal, maxVal):
        a = (minVal - maxVal) / 60025
        b = ( 500 * maxVal - minVal) / 60025
        c = (62500 * minVal - 2475 *  maxVal) / 60025
        return CuadraticFunction(a, b, c)
    
    @staticmethod
    def ModelInterval(policy, parameter):
        if policy == "fixed":
            return CuadraticFunction.modelConstant(parameter)
        elif policy == "linear":
            return CuadraticFunction.modelLinear(MIN_INTERVAL, parameter)
        elif policy == "modulated":
            return CuadraticFunction.modelLinearModulated(MIN_INTERVAL, parameter)
        else:
            print("Interval Function - "+ policy  +" is not a policy...\n")
            exit(0)
    @staticmethod
    def ModelIncrement(policy, parameter):
        if policy == "fixed":
            return CuadraticFunction.modelConstant(parameter)
        elif policy == "linear":
            return CuadraticFunction.modelLinear(MIN_INCREMENT, parameter)
        elif policy == "modulated":
            return CuadraticFunction.modelLinearModulated(MIN_INCREMENT, parameter)
        else:
            print("Increment Function - "+ policy  +" is not a policy...\n")
            exit(0)
        
