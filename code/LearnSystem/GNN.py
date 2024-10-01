from torch import torch, nn
from torch.autograd import Variable

from LearnSystem.LearnSystem import learnSystem
from AttentionModule import Attention_R, Attention_J, Attention_H


# NO CENTROIDS
class GNN(learnSystem):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.learning_rate = 1e-3

        self.nLayers = int(parameters["nAttLayers"])
        self.d = self.controlPolicy.input_size #input dimensions
        self.r = 16 #intermediate dimensions
        self.h = 4 #output dimensions

        self.activation_swish = nn.SiLU()

        # Initialize weights to avoid unstable training
        self.MLP_weights  = nn.ParameterList([ nn.Parameter(0.1*torch.randn(self.r, self.d)) ]).to(self.device)
        self.GNN_weights  = nn.ParameterList([ nn.Parameter(0.1*torch.randn(self.r, self.r)) ]).to(self.device)
        for _ in range(self.nLayers-2):
            self.MLP_weights.append( nn.Parameter(0.1*torch.randn(self.r, self.r)) ).to(self.device)    
            self.GNN_weights.append( nn.Parameter(0.1*torch.randn(self.r, self.r)) ).to(self.device)    

        self.MLP_weights.append( nn.Parameter(0.1*torch.randn(self.r, self.r)) ).to(self.device)    
        self.GNN_weights.append( nn.Parameter(0.1*torch.randn(self.h, self.r)) ).to(self.device)    


    def flocking_dynamics(self, t, inputs):
        # Obtain Laplacians
        L = self.controlPolicy.laplacian(inputs[:, :2 * self.na]).to(self.device)
        L_GNN = self.__buildLaplacianGNN(L)

        # Obtain policy inputs
        inputs  = nn.functional.normalize(inputs, p=2, dim=1)
        state_d = self.getStateDiffs(inputs)
        pos, vel = self.getRelativeStates(inputs) 

        policy_inputs = self.controlPolicy.shapeInputs(inputs, state_d, pos, vel, L)
        
        for layer in range(self.nLayers):
            wx_MLP = torch.bmm(self.MLP_weights[layer].unsqueeze(dim=0).repeat(policy_inputs.shape[0], 1, 1), policy_inputs)
            wx = torch.bmm(self.GNN_weights[layer].unsqueeze(dim=0).repeat(policy_inputs.shape[0], 1, 1), wx_MLP)
            policy_inputs = torch.bmm(L_GNN, wx.transpose(1,2)).transpose(1,2)
            
            if layer != self.nLayers-1:
                policy_inputs = self.activation_swish(policy_inputs)

        dpos =   policy_inputs[:,:2,:].reshape(-1, 2*self.na)  
        dvel =   policy_inputs[:,2:,:].reshape(-1, 2*self.na)  
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return dpos, dvel
    

    def __buildLaplacianGNN(self, L):
        Lsum_diag = torch.diag_embed(torch.sum(L,2))

        L_noDiag = L.clone()
        L_noDiag[:,range(self.na),range(self.na)] = 0 

        L_GNN = Lsum_diag - L_noDiag

        return L_GNN