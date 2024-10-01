from abc import ABC, abstractmethod

import torch

class controlPolicy(ABC):

    def __init__(self, r, epsilon, inputSize):        
        self.r = r
        self.epsilon = epsilon
        self.input_size = inputSize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
    def inputSize(self):
        return self.input_size
    
    @abstractmethod
    def laplacian(self, q_agents):
        pass

    @abstractmethod
    def shapeInputs(self, inputs, state_d, pos, vel, L):
        pass