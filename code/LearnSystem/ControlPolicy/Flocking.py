from abc import ABC, abstractmethod
from torch import torch, nn
from torch.autograd import Variable


from .ControlPolicy import controlPolicy

class Flocking(controlPolicy):

    def __init__(self):
        r = torch.as_tensor(1.2*1.0) 
        epsilon = 1e-12
        inputSize = 14
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
                
        posj = pos.repeat(1, na, 1).reshape(-1, na, na, 2)
        posi = torch.kron(pos, torch.ones((1, na, 1), device=self.device)).reshape(-1, na, na, 2)
        posij  = (posi - posj)
        posij[:,range(na), range(na), :] = 0.0
        
        normij  = posij.norm(p=2,dim=3).unsqueeze(3)
        normij[:,range(na), range(na), :] = 1.0

        # Mask non neighbours positions to force 0's in sum()
        L_discrete = (L > 0).float()
        LL = L_discrete.unsqueeze(3).repeat([1,1,1,2])
        posij = posij * LL

        # print("pos: ", pos[0])
        # print("posi: ", posi[0])
        # print("posj: ", posj[0])
        # print("posij: ", posij[0])
        # print("normij: ", normij[0])
        # print()

        # input("Next iteration...")
        # States of robots relative to their respective leaders
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
        sum_normPosij_2 = self.__sum_normPosij(posij, normij, 2, na)
        state[:, 8::i] = sum_normPosij_2[:, :, 0]
        state[:, 9::i] = sum_normPosij_2[:, :, 1]

        sum_normPosij_4 = self.__sum_normPosij(posij, normij, 4, na)
        state[:, 10::i] = sum_normPosij_4[:, :, 1]
        state[:, 11::i] = sum_normPosij_4[:, :, 1]


        # Norm between agents and inverse
        state[:, 12::i] = ((pos.reshape(-1, na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, na)
        state[:, 13::i] = (pos.reshape(-1, na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, na)

        inputs_l = Variable(state.reshape(-1, na, i).transpose(1, 2).data, requires_grad=True)
        
        del state
        return inputs_l
    

    def __sum_normPosij(self, posij, normij, pow, na):
        
        # print("posij_",pow, " - ", posij.size())
        # print("\tmin", posij.min().item())
        # print("\tmax", posij.max().item())

        
        # print("normij", normij.size())
        # print("\tmin",normij.min().item())
        # print("\tmax", normij.max().item())

        normpow = (normij ** pow) + 1e-5 # Adds to avoid underflown 0's in norm

        # print("normij^pow", normpow.size())
        # print("\tmin",normpow.min().item())
        # print("\tmax", normpow.max().item())

        normPosij = posij / normpow

        # print("normPosij - ", normPosij.size())
        # print("\tmin", normPosij.min().item())
        # print("\tmax", normPosij.max().item())
        # print("=====")

        return torch.sum(normPosij, dim=2)
