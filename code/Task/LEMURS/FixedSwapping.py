from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable

from Task.LEMURS.TaskLEMURS import TaskLEMURS

class FixedSwapping(TaskLEMURS):

    def __init__(self, config):
        r = torch.as_tensor(1.2*2.0) 
        epsilon = 1e-5
        inputSize = 6

        super().__init__(config, inputSize, epsilon, r)


    def laplacian(self, positions):
        nSamples = int(positions.shape[0])

        na = self.numAgents
        L = torch.eye(na, device=self.device) - torch.diag(torch.ones(na - 1, device=self.device), diagonal=1) - torch.diag(torch.ones(na - 1, device=self.device), diagonal=-1)
        if na > 1:
            L[0, -1] = -1.0
            L[-1, 0] = -1.0

        L = L.unsqueeze(0).repeat(nSamples,1,1)
        return L

    