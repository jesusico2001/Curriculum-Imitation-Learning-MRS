from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable


from Task.LEMURS.TaskLEMURS import TaskLEMURS

class TimeVaryingSwapping(TaskLEMURS):

    def __init__(self, config):
        r = torch.as_tensor(1.2*2.0) 
        epsilon = 1e-5
        inputSize = 6
        super().__init__(config, inputSize, epsilon, r)

