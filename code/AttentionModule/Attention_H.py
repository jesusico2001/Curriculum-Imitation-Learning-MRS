from torch import nn, torch
from AttentionModule import AttentionModule

class Att_H(AttentionModule.AttentionModule):
    
    def __init__(self, r, d, h, nLayers, device):
        super().__init__(r, d, h, nLayers, device, True, 1.0)

    def postProcess(self, x, res_attention):
        l = 2
        na = x.shape[2]

        # Reshape, kronecker and post-processing
        M11 = torch.kron((res_attention[:, 0:5, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((res_attention[:, 5:10, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((res_attention[:, 10:15, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((res_attention[:, 15:20, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (res_attention[:, 20:25, :] ** 2).sum(1)

        Mupper11 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper12 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper21 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper22 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)

        Mupper11[:, range(l * na), range(l * na)] = M11
        Mupper12[:, range(l * na), range(l * na)] = M12
        Mupper21[:, range(l * na), range(l * na)] = M21
        Mupper22[:, range(l * na), range(l * na)] = M22

        del M11, M12, M21, M22
        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)
        del Mupper11, Mupper12, Mupper21, Mupper22
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        q = x[:, :4, :].transpose(1, 2).reshape(-1, 1, 4 * na)

        return torch.bmm(q, torch.bmm(M, q.transpose(1, 2))).sum(2) + Mpp.sum(1).unsqueeze(1)
