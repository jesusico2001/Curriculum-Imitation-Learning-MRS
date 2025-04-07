from abc import ABC, abstractmethod
from torch import torch, nn
from torch.autograd import Variable


from Task.LEMURS.TaskLEMURS import TaskLEMURS

class Flocking(TaskLEMURS):

    def __init__(self, config):
        epsilon = 1e-12
        inputSize = 14
        r = torch.as_tensor(1.2*1.0) 
        super().__init__(config, inputSize, epsilon, r)

    
    def buildInputVariables(self, inputs):
        na = self.numAgents
        i = int(self.agent_input_size)

        state = torch.zeros((inputs.shape[0], i * na), device=self.device)
        
        pos, vel = self.getPosVel(inputs)
        posj = pos.repeat(1, na, 1).reshape(-1, na, na, 2)
        posi = torch.kron(pos, torch.ones((1, na, 1), device=self.device)).reshape(-1, na, na, 2)
        posij  = (posi - posj)
        posij[:,range(na), range(na), :] = 0.0
        
        normij  = posij.norm(p=2,dim=3).unsqueeze(3)
        normij[:,range(na), range(na), :] = 1.0

        # Mask non neighbours positions to force 0's in sum()
        L = self.laplacian(pos)
        L_discrete = (L > 0).float()
        LL = L_discrete.unsqueeze(3).repeat([1,1,1,2])
        posij = posij * LL

        # States of robots relative to their respective leaders
        state_d = self.getDiffGoal(inputs)
        state[:, 0::i] = state_d[:, :, 0]
        state[:, 1::i] = state_d[:, :, 1]
        state[:, 2::i] = state_d[:, :, 2]
        state[:, 3::i] = state_d[:, :, 3]
        
        # Absolute states of every robot
        state[:, 4::i] = pos[:, :, 0]
        state[:, 5::i] = pos[:, :, 1]
        state[:, 6::i] = vel[:, :, 0]
        state[:, 7::i] = vel[:, :, 1]

        # Dist / norm ^ [2,4]
        sum_normPosij_2 = self.__sum_normPosij(posij, normij, 2)
        state[:, 8::i] = sum_normPosij_2[:, :, 0]
        state[:, 9::i] = sum_normPosij_2[:, :, 1]

        sum_normPosij_4 = self.__sum_normPosij(posij, normij, 4)
        state[:, 10::i] = sum_normPosij_4[:, :, 0]
        state[:, 11::i] = sum_normPosij_4[:, :, 1]


        # Norm between agents and inverse
        state[:, 12::i] = ((pos.reshape(-1, na, 2)).norm(p=2, dim=2) + self.epsilon_norm).pow(-1).unsqueeze(2).reshape(-1, na)
        state[:, 13::i] = (pos.reshape(-1, na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, na)

        inputs_l = Variable(state.reshape(-1, na, i).transpose(1, 2).data, requires_grad=True)
        
        del state
        return inputs_l
    

    def __sum_normPosij(self, posij, normij, pow):
        
        normpow = (normij ** pow) + 1e-5 # Adds to avoid underflown 0's in norm
        normPosij = posij / normpow

        return torch.sum(normPosij, dim=2)
