import torch
from torchdiffeq import odeint
from abc import ABC, abstractmethod

class realSystem(ABC):
    def __init__(self, parameters):
        self.na     = parameters['na']
        self.d      = parameters['d']
        self.e      = parameters['e']
        self.a      = parameters['a']
        self.b      = parameters['b']
        self.c      = parameters['c']
        self.c1     = parameters['c1']
        self.c2     = parameters['c2']

    # Functions 
    def sigma_norm(self, z):
        return (torch.sqrt(1 + self.e * z.norm(p=2) ** 2) - 1) / self.e

    def sigmoid(self, z):
        return z / (torch.sqrt(1 + z ** 2))

    def phi_function(self, z):
        return ((self.a + self.b) * self.sigmoid(z + self.c) + (self.a - self.b)) / 2

    def f_control(self, q_agents, p_agents, q_dynamic, p_dynamic):
        return -self.c1 * (q_agents - q_dynamic) - self.c2 * (p_agents - p_dynamic)

    # Topology
    @abstractmethod
    def laplacian(self, q_agents):
        pass

    def augmented_laplacian(self, q_agents):
        return torch.kron(self.laplacian(q_agents), torch.eye(2))

    # Dynamics
    @abstractmethod
    def grad_V(self, L, q_agents):
        pass

    @abstractmethod
    def flocking_dynamics(self, inputs):
        pass

    def leader_dynamics(self, inputs):
        return inputs[6 * self.na:], torch.zeros(2 * self.na)

    def overall_dynamics(self, t, inputs):
        da = self.flocking_dynamics(inputs)
        dd = self.leader_dynamics(inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]))

    
    def sample(self, inputs, simulation_time, step_size):
        targets = odeint(self.overall_dynamics, inputs, simulation_time, method='euler', options={'step_size': step_size})
        return targets

    # Agent generation
    @abstractmethod
    def generate_agents(self, na):
        pass

    @abstractmethod
    def generate_leader(self, na):
        pass
