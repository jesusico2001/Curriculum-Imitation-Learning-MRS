from torch import  torch
from AttentionModule import AttentionModule

class Att_R(AttentionModule.AttentionModule):


    def __init__(self, r, d, h, nLayers, device):
        super().__init__(r, d, h, nLayers, device, True, 1.0)
        
    def postProcess(self, x, res_attention):
        na = x.shape[2]

        # Reshape and kronecker before post-processing
        R11 = res_attention[:, 0:4,   :].sum(1)
        R12 = res_attention[:, 4:8,   :].sum(1)
        R21 = res_attention[:, 8:12,  :].sum(1)
        R22 = res_attention[:, 12:16, :].sum(1)
        
        R11 = R11.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R12 = R12.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R21 = R21.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R22 = R22.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R11 = torch.kron(R11, torch.eye(2, device=self.device).unsqueeze(0))
        R12 = torch.kron(R12, torch.eye(2, device=self.device).unsqueeze(0))
        R21 = torch.kron(R21, torch.eye(2, device=self.device).unsqueeze(0))
        R22 = torch.kron(R22, torch.eye(2, device=self.device).unsqueeze(0))
        R   = torch.cat((torch.cat((R11, R21), dim=1), torch.cat((R12, R22), dim=1)), dim=2)
        del R11, R12, R21, R22
        R   = R ** 2

        # Operations to ensure sparsity and positive semidefiniteness
        Rupper = R + R.transpose(1, 2)
        Rdiag  = torch.clone(Rupper)

        l = int(torch.sqrt(torch.as_tensor(self.h)))
        Rdiag[:, range(l * na), range(l * na)] = torch.zeros(l * na, device=self.device)

        Rout = torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * Rupper

        R = Rout + torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        return R