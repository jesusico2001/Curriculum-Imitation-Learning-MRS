from . import RealSystem
import torch

class realSystemTVS(RealSystem.realSystem):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.r = parameters['r']


    # Topology
    def laplacian(self, q_agents):
        L = torch.zeros(self.na, self.na)
        for i in range(self.na):
            for j in range(i + 1, self.na):
                L[i, j] = torch.le((q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]).norm(p=2), self.r).float()
        L = L + L.T
        return torch.diag(torch.sum(L, 1)) - L

    # Dynamics
    def grad_V(self, L, q_agents):
        grad_V  = torch.zeros(2 * self.na)
        d_sigma = self.sigma_norm(self.d)
        for i in range(self.na):
            for j in range(self.na):
                if i != j and L[2 * i, 2 * j] != 0:
                    z_sigma = q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]
                    if self.sigma_norm(z_sigma) < d_sigma:
                        n_ij = (q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]) / (torch.sqrt(1 + self.e * (q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]).norm(p=2) ** 2))
                        grad_V[2 * i:2 * i + 2] -= z_sigma / self.sigma_norm(z_sigma) * n_ij
        return grad_V

    def flocking_dynamics(self, inputs):
        q_agents = inputs[:2 * self.na]
        p_agents = inputs[2 * self.na:4 * self.na]
        q_dynamic = inputs[4 * self.na:6 * self.na]
        p_dynamic = inputs[6 * self.na:]
        L = self.augmented_laplacian(q_agents)
        
        dq        = p_agents - p_dynamic
        dp        = -self.grad_V(L, q_agents) + self.f_control(q_agents, p_agents, q_dynamic, p_dynamic)
        return dq, dp
    

    #Agent generation
    def generate_agents(self, na):
        q_agents       = torch.zeros(2 * na)
        q_agents[0::2] = torch.cat((3.0 * torch.ones(int(na/2)), -3.0 * torch.ones(int(na/2))))
        q_agents[1::2] = torch.cat((torch.linspace(-3.0 * (int(na/4)-1) - 1.5, 3.0 * (int(na/4)-1) + 1.5, int(na/2)),
                                    torch.linspace(3.0 * (int(na/4)-1) + 1.5, -3.0 * (int(na/4)-1) - 1.5, int(na/2))))

        return q_agents + 1.0 * torch.rand(2 * na), torch.zeros(2 * na)

    def generate_leader(self, na):
        q_agents       = torch.zeros(2 * na)
        q_agents[0::2] = torch.cat((-3.0 * torch.ones(int(na/2)), 3.0 * torch.ones(int(na/2))))
        q_agents[1::2] = torch.cat((torch.linspace(3.0 * (int(na/4)-1) + 1.5, -3.0 * (int(na/4)-1) - 1.5, int(na/2)),
                                    torch.linspace(-3.0 * (int(na/4)-1) - 1.5, 3.0 * (int(na/4)-1) + 1.5, int(na/2))))

        return q_agents + 1.0 * torch.rand(2 * na), torch.zeros(2 * na)