from torch import torch
from AttentionModule import AttentionModule

class Att_GNN(AttentionModule.AttentionModule):
    
    def __init__(self, r, d, h, nLayers, device):
        super().__init__(r, d, h, nLayers, device, False, 1)

    def _attentionLayers(self, x, L):
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

        L = L[:,0,:].unsqueeze(1).repeat(1,self.r,1)

        inputs = x
        # print("Inicio ", self.__class__.__name__, "-  L:", L.size(), " - input: ", input.size())
        for layer in range(self.nLayers):

            
            Q = self._calcQKV(inputs, self.weights_q, layer).transpose(1, 2)
            K = self._calcQKV(inputs, self.weights_k, layer)
            V = self._calcQKV(inputs, self.weights_v, layer).transpose(1, 2)
            
            # print("Q.", Q[0])
            # input("enter...")
            # print("K.", K[0])
            # input("enter...")
            # print("V.", V[0])
            # input("enter...")

            o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
            # print("o.", o[0])
            # input("enter...")

            inputs = torch.bmm(self.weights_o[layer].unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o)
            if layer != self.nLayers-1:
                inputs = self.activation_swish(inputs)

        del Q, K, V, o
        return inputs
        
    def postProcess(self, x, res_attention):
        return res_attention
