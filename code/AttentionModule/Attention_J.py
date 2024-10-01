from torch import torch
from AttentionModule import AttentionModule

class Att_J(AttentionModule.AttentionModule):
    

    def __init__(self, r, d, h, nLayers, device):
        super().__init__(r, d, h, nLayers, device, True, 1.0)
        
    def postProcess(self, x, res_attention):
        na = x.shape[2]

        # Reshape, kronecker and post-processing to ensure skew-symmetry
        J11 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J22 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)

        j12 = res_attention.sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        j21 = -res_attention.sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)

        J11 = J11.reshape(int(x.shape[0] / x.shape[2]), na, na)
        J12 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J21 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J22 = J22.reshape(int(x.shape[0] / x.shape[2]), na, na)
        J12[:, range(na), range(na)] = j12
        J21[:, range(na), range(na)] = j21
        J11 = torch.kron(J11, torch.eye(2, device=self.device).unsqueeze(0))
        J12 = torch.kron(J12, torch.eye(2, device=self.device).unsqueeze(0))
        J21 = torch.kron(J21, torch.eye(2, device=self.device).unsqueeze(0))
        J22 = torch.kron(J22, torch.eye(2, device=self.device).unsqueeze(0))

        J   = torch.cat((torch.cat((J11, J21), dim=1), torch.cat((J12, J22), dim=1)), dim=2)
        del J11, J12, J21, J22, j12, j21

        return J