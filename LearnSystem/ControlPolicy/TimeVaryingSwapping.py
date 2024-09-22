from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable


from .ControlPolicy import controlPolicy

class TimeVaryingSwapping(controlPolicy):

    def __init__(self):
        r = torch.as_tensor(1.2*2.0) 
        epsilon = 1e-5
        inputSize = 6
        super().__init__(r, epsilon, inputSize)

    
    def laplacian(self, q_agents):
        na = int(q_agents.size()[1]/2)

        Q1 = q_agents.reshape(q_agents.shape[0], -1, 2).repeat(1, na, 1)
        Q2 = torch.kron(q_agents.reshape(q_agents.shape[0], -1, 2), torch.ones((1, na, 1), device=self.device))
        Q  = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], na, na)
        L  = Q.le(self.r).float()
        L = L * torch.sigmoid(-(2.0)*(Q - self.r))
        return L
    
        
    def shapeInputs(self, inputs, state_d, pos, vel, L):
        na = pos.size(1)
        i = int(self.input_size)

        state = torch.zeros((state_d.shape[0], i * na), device=self.device)
        
        # States of robots relative to their respective leaders
        state[:, 0::i] = state_d[:, :, 0]
        state[:, 1::i] = state_d[:, :, 1]
        state[:, 2::i] = state_d[:, :, 2]
        state[:, 3::i] = state_d[:, :, 3]

        # Norm between agents and inverse
        state[:, 4::i] = ((pos.reshape(-1, na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, na)
        state[:, 5::i] = (pos.reshape(-1, na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, na)

        inputs_l = Variable(state.reshape(-1, na, i).transpose(1, 2).data, requires_grad=True)
        
        del state
        
        return inputs_l