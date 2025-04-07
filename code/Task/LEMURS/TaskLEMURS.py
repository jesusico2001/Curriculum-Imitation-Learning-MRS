from abc import ABC, abstractmethod
from enum import Enum
import torch
from torch.autograd import Variable

from Task.Task import Task


class TaskLEMURS(Task):

    def __init__(self, config, agent_input_size, epsilon_norm, communication_radius): 
        super().__init__(config, agent_input_size, communication_radius)        
        self.epsilon_norm = epsilon_norm
        self.simulation_step = 0.04

    def buildFeatureIndex(self):
        feature_index = {
            "robot_positions": slice(0, 2*self.numAgents),
            "robot_velocities": slice(2*self.numAgents, 4*self.numAgents)
        }   
        return feature_index    

    def buildInputVariables(self, inputs):
        na = self.numAgents
        i = int(self.agent_input_size)
        
        state = torch.zeros((inputs.shape[0], i * na), device=self.device)

        # States of robots relative to their respective leaders
        state_d = self.getDiffGoal(inputs)
        state[:, 0::i] = state_d[:, :, 0]
        state[:, 1::i] = state_d[:, :, 1]
        state[:, 2::i] = state_d[:, :, 2]
        state[:, 3::i] = state_d[:, :, 3]

        # Norm between agents and inverse
        pos, _ = self.getPosVel(inputs)
        state[:, 4::i] = ((pos.reshape(-1, na, 2)).norm(p=2, dim=2) + self.epsilon_norm).pow(-1).unsqueeze(2).reshape(-1, na)
        state[:, 5::i] = (pos.reshape(-1, na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, na)

        inputs_l = Variable(state.reshape(-1, na, i).transpose(1, 2).data, requires_grad=True)
        
        del state
        
        return inputs_l
        
    def getDiffGoal(self, inputs):
        na = self.numAgents
        inputs_d         = (inputs[:, :4 * na] - inputs[:, 4 * na:]) #difference between agents and leaders

        state_d          = torch.zeros((inputs.shape[0], na, 4), device=self.device) #(nInputs x nAgents x 4)
        state_d[:, :, 0] = inputs_d[:, 0:2 * na:2] #positions_dif(1)
        state_d[:, :, 1] = inputs_d[:, 1:2 * na:2] #positions_dif(2)
        state_d[:, :, 2] = inputs_d[:, 2 * na + 0::2] #movement_dif(1)
        state_d[:, :, 3] = inputs_d[:, 2 * na + 1::2] #movement_dif(2)

        del inputs_d
        return state_d