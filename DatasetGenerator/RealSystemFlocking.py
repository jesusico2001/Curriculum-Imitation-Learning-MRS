from . import RealSystem
import torch

class realSystemFlocking(RealSystem.realSystem):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.r = parameters['r']
        self.ha     = parameters['ha']


    # Topology
    def __rho_function(self, z, h):
        if 0 <= z < h:
            return 1
        elif h <= z <= 1:
            pi = torch.acos(torch.zeros(1)).item() * 2
            return (1 + torch.cos(pi * ((z - h) / (1 - h)))) / 2
        else:
            return 0

    def laplacian(self, q_agents):
        L       = torch.zeros(self.na, self.na)
        r_sigma = self.sigma_norm(self.r)
        for i in range(self.na):
            for j in range(i + 1, self.na):
                z_sigma = self.sigma_norm(q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2])
                L[i, j] = self.__rho_function(z_sigma / r_sigma, self.ha)
        L = L + L.T
        return torch.diag(torch.sum(L, 1)) - L

    # Dynamics
    def __phi_alpha_function(self, z, r, h, d):
        return self.__rho_function(z / r, h) * self.phi_function(z - d)

    def grad_V(self, L, q_agents):
        grad_V  = torch.zeros(2 * self.na)
        r_sigma = self.sigma_norm(self.r)
        d_sigma = self.sigma_norm(self.d)
        for i in range(self.na):
            for j in range(self.na):
                if i != j and L[2 * i, 2 * j] != 0:
                    z_sigma = self.sigma_norm(q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2])
                    n_ij    = (q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]) / (torch.sqrt(1 + self.e * (q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]).norm(p=2) ** 2))
                    grad_V[2 * i:2 * i + 2] -= self.__phi_alpha_function(z_sigma, r_sigma, self.ha, d_sigma) * n_ij
        return grad_V

    def flocking_dynamics(self, inputs):
        q_agents = inputs[:2 * self.na]
        p_agents = inputs[2 * self.na:4 * self.na]
        q_dynamic = inputs[4 * self.na:6 * self.na]
        p_dynamic = inputs[6 * self.na:]
        L         = self.augmented_laplacian(q_agents)
        DeltaV    = self.grad_V(L, q_agents)

        dq        = p_agents
        dp        = -DeltaV - L @ p_agents + self.f_control(q_agents, p_agents, q_dynamic, p_dynamic)

        return dq, dp
    
    # Agent generation
    def generate_agents(self, na):
        return 4.0 * torch.rand(2 * na), torch.zeros(2 * na)

    def generate_leader(self, na):
        return torch.zeros(2 * na), torch.zeros(2 * na)