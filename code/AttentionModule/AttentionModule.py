from abc import ABC, abstractmethod
import time
from torch import nn, torch

class AttentionModule(nn.Module, ABC):
    
    def __init__(self, r, d, h, nLayers, device, remask_bias, init_weight):
        super().__init__()
        
        self.device           = device
        self.activation_soft  = nn.Softmax(dim=2)
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()

        self.nLayers = nLayers
        self.d = d #input dimensions
        self.r = r #intermediate dimensions
        self.h = h #output dimensions

        self.remask_bias = remask_bias

        # Initialize weights to avoid unstable training
        self.weights_q  = nn.ParameterList([ nn.Parameter(init_weight*torch.randn(r, d)) ])
        for _ in range(nLayers-1):
            self.weights_q.append( nn.Parameter(init_weight*torch.randn(r, r)) )       
                                              
        self.weights_k  = nn.ParameterList([ nn.Parameter(init_weight*torch.randn(r, d)) ])
        for _ in range(nLayers-1):
            self.weights_k.append( nn.Parameter(init_weight*torch.randn(r, r)) )       

        self.weights_v  = nn.ParameterList([ nn.Parameter(init_weight*torch.randn(r, d)) ])
        for _ in range(nLayers-1):
            self.weights_v.append( nn.Parameter(init_weight*torch.randn(r, r)) )       

        self.weights_o  = nn.ParameterList([])
        for _ in range(nLayers-1):
            self.weights_o.append( nn.Parameter(init_weight*torch.randn(r, r)) )
        self.weights_o.append( nn.Parameter(init_weight*torch.randn(h, r)) )

    def _calcQKV(self, input, weights, layer):
        return self.activation_sigmo(torch.bmm(weights[layer].unsqueeze(dim=0).repeat(input.shape[0], 1, 1), input))

    def _attentionLayers(self, x, L):
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1
        L = L[:,0,:].unsqueeze(1).repeat(1,self.r,1)
        
        input = x
        # print("Inicio ", self.__class__.__name__, "-  L:", L.size(), " - input: ", input.size())
        for layer in range(self.nLayers):

            Q = self._calcQKV(input, self.weights_q, layer).transpose(1, 2)
            K = self._calcQKV(input, self.weights_k, layer)
            V = self._calcQKV(input, self.weights_v, layer).transpose(1, 2)
            
            o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
            input = self.activation_swish(torch.bmm(self.weights_o[layer].unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

            # Remask to recreate the perception bias
            # self.remask_bias = False #TODO: NO TE OLVIDES DE QUITARLO
            
            if self.remask_bias and layer != self.nLayers-1:
                #print("\tatt - Lmid:", L.size(), " - input: ", input.size()
                input = L * input

        del Q, K, V, o
        return input

    @abstractmethod
    def postProcess(self, x, res_attention):
        pass
    
    def forward(self, x, L):
        att = self._attentionLayers(x, L)
        return self.postProcess(x, att)
        
    