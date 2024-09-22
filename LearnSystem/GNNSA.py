from torch import torch, nn

from LearnSystem.LearnSystem import learnSystem
from AttentionModule import Attention_GNN


# NO CENTROIDS
class GNNSA(learnSystem):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.learning_rate = 1e-3
        self.nLayers = int(parameters["nAttLayers"])

        self.d = self.controlPolicy.input_size #input dimensions
        self.r = 8 #intermediate dimensions
        self.h = 4 #output dimensions

        self.activation_swish = nn.SiLU()
        self.activation_soft  = nn.Softmax(dim=2)
        self.activation_sigmo = nn.Sigmoid()

        self.attModules = nn.ParameterList([Attention_GNN.Att_GNN(self.r, self.d, self.r, 1, device=self.device).to(self.device)])
        for _ in range(self.nLayers-2):
            self.attModules.append(Attention_GNN.Att_GNN(self.r, self.r, self.r, 1, device=self.device).to(self.device))
        # self.attModules = []
        self.attModules.append(Attention_GNN.Att_GNN(self.r, self.r, self.h, 1, device=self.device).to(self.device))
        

        self.MLP_weights_q  = nn.ParameterList([ nn.Parameter(0.1*torch.randn(self.d, self.d)) ]).to(self.device)
        for _ in range(self.nLayers-2):
            self.MLP_weights_q.append( nn.Parameter(0.1*torch.randn(self.r, self.r)) ).to(self.device)    

        self.MLP_weights_q.append( nn.Parameter(0.1*torch.randn(self.r, self.r)) ).to(self.device) 

        self.MLP_weights_k  = nn.ParameterList([ nn.Parameter(0.1*torch.randn(self.d, self.d)) ]).to(self.device)
        for _ in range(self.nLayers-2):
            self.MLP_weights_k.append( nn.Parameter(0.1*torch.randn(self.r, self.r)) ).to(self.device)    

        self.MLP_weights_k.append( nn.Parameter(0.1*torch.randn(self.r, self.r)) ).to(self.device)

        self.weights_o  = nn.ParameterList([ nn.Parameter(1*torch.randn(self.r, self.d)) ]).to(self.device)
        for _ in range(self.nLayers-2):
            self.weights_o.append( nn.Parameter(1*torch.randn(self.r, self.r)) ).to(self.device)    

        self.weights_o.append( nn.Parameter(0.1*torch.randn(self.h, self.r)) ).to(self.device) 

    def flocking_dynamics(self, t, inputs):
        # Obtain Laplacians
        L = self.controlPolicy.laplacian(inputs[:, :2 * self.na]).to(self.device)
        L_GNN = self.__buildLaplacianGNN(L)

        # Obtain policy inputs
        inputs  = nn.functional.normalize(inputs, p=2, dim=1)
        state_d = self.getStateDiffs(inputs)
        pos, vel = self.getRelativeStates(inputs) 

        policy_inputs = self.controlPolicy.shapeInputs(inputs, state_d, pos, vel, L)

        # print(policy_inputs[0])
        # input("enter...")
    
        for layer in range(self.nLayers):
            # print("policy_inputs", policy_inputs[0])
            Q = self.activation_sigmo(torch.bmm(self.MLP_weights_q[layer].unsqueeze(dim=0).repeat(policy_inputs.shape[0], 1, 1), policy_inputs))
            K = self.activation_sigmo(torch.bmm(self.MLP_weights_k[layer].unsqueeze(dim=0).repeat(policy_inputs.shape[0], 1, 1), policy_inputs)).transpose(1, 2)
            # print(torch.bmm(Q, K)[0])
            # print(self.activation_soft(torch.bmm(Q, K))[0])
            # input("enter....")

            wx_ = torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), policy_inputs)

            wx = torch.bmm(self.weights_o[layer].unsqueeze(dim=0).repeat(wx_.shape[0], 1, 1), wx_)
            
            # wx = self.attModules[layer].forward(wx_MLP, L)

            policy_inputs = torch.bmm(L_GNN, wx.transpose(1,2)).transpose(1,2)
            if layer != self.nLayers-1:
                policy_inputs = self.activation_swish(policy_inputs)

            # print(policy_inputs[0])
            # input("enter....")

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