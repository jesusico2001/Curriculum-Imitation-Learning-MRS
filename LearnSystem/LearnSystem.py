from abc import ABC, abstractmethod
from torch import nn, torch
from torchdiffeq import odeint

class learnSystem(nn.Module, ABC):

    def __init__(self, parameters):
        super().__init__()
        self.device  = parameters["device"]
        self.na      = parameters['na']
        self.controlPolicy = parameters["control_policy"]

    @abstractmethod
    def flocking_dynamics(self, t, inputs):
       pass
    
    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics(self, t, inputs):
        dd = self.leader_dynamics(t, inputs)
        da = self.flocking_dynamics(t, inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def forward(self, inputs, simulation_time, step_size):
        outputs_2 = odeint(self.overall_dynamics, inputs, simulation_time.to(self.device), method='euler', options={'step_size': step_size})
        return outputs_2
  

    # ===============
    # Private methods
    # ===============

    def getStateDiffs(self, inputs):
        inputs_d         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:]) #difference between agents and leaders

        state_d          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device) #(nInputs x nAgents x 4)
        state_d[:, :, 0] = inputs_d[:, 0:2 * self.na:2] #positions_dif(1)
        state_d[:, :, 1] = inputs_d[:, 1:2 * self.na:2] #positions_dif(2)
        state_d[:, :, 2] = inputs_d[:, 2 * self.na + 0::2] #movement_dif(1)
        state_d[:, :, 3] = inputs_d[:, 2 * self.na + 1::2] #movement_dif(2)

        del inputs_d
        return state_d
    
    def getRelativeStates(self, inputs):
        # Obtain relative states
        inputs_l         = inputs[:, :4 * self.na] #eliminates leaders

        pos = inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2)
        vel = inputs_l[:, 2 * self.na:6 * self.na].reshape(inputs_l.shape[0], -1, 2)
        return pos, vel

    # def getRelativeStates(self, inputs):
    #     # Obtain relative states
    #     inputs_l         = inputs[:, :4 * self.na] #eliminates leaders

    #     pos1 = inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1) # positions repeated (p1p2p3 p1p2p3)
    #     pos2 = torch.kron(inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device)) # positions repeated (p1p1 p2p2 p3p3)
    #     pos  = (pos1 - pos2).reshape(inputs_l.shape[0] * self.na, -1) #relative positions to every other agent

    #     vel1 = inputs_l[:, 2 * self.na:6 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
    #     vel2 = torch.kron(inputs_l[:, 2 * self.na:4 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
    #     vel  = (vel1 - vel2).reshape(inputs_l.shape[0] * self.na, -1) #relative velocities to every other agent

    #     del pos1, pos2, vel1, vel2
    #     return pos, vel
