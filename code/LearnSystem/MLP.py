from torch import torch, nn

from LearnSystem.LearnSystem import learnSystem


# NO CENTROIDS
class MLP(learnSystem):

    def __init__(self, parameters):
        super().__init__(parameters)

        self.nLayers = int(parameters["nAttLayers"])
        self.d = self.controlPolicy.input_size * self.na #input dimensions
        self.r = 8 #intermediate dimensions
        self.h = 4 * self.na #output dimensions

        self.activation_swish = nn.SiLU()


        # Initialize weights to avoid unstable training
        self.weights  = nn.ParameterList([ nn.Parameter(0.01*torch.randn(self.r, self.d)) ]).to(self.device)
        self.bias  = nn.ParameterList([ nn.Parameter(0.01*torch.randn(self.r)) ]).to(self.device)

        for _ in range(self.nLayers-2):
            self.weights.append( nn.Parameter(0.01*torch.randn(self.r, self.r)) ).to(self.device)                                              
            self.bias.append( nn.Parameter(0.01*torch.randn(self.r)) ).to(self.device)

        self.weights.append( nn.Parameter(0.01*torch.randn(self.h, self.r)) ).to(self.device)    
        self.bias.append( nn.Parameter(0.01*torch.randn(self.h)) ).to(self.device)
        

    def flocking_dynamics(self, t, inputs):
        # Obtain Laplacians
        L = self.controlPolicy.laplacian(inputs[:, :2 * self.na]).to(self.device)
        # print("inputs_MLP:", inputs[0])
        
        # Obtain policy inputs
        inputs  = nn.functional.normalize(inputs, p=2, dim=1)
        
        state_d = self.getStateDiffs(inputs)
        pos, vel = self.getRelativeStates(inputs) 

        # print("inputs_init:", inputs[0])
        # print("state_d_MLP:", state_d[0])
        # print("pos_MLP:", pos[0])
        # print("vel_MLP:", vel[0])
        inputs_policy = self.controlPolicy.shapeInputs(inputs, state_d, pos, vel, L)

        # print("inputs_policy:", inputs_policy[0])

        MLP_inputs = inputs_policy.reshape(-1, self.d).to(self.device) #On MLP the neighbours' states aren't tokenized
        
        for layer in range(self.nLayers):

            wx = torch.bmm(self.weights[layer].unsqueeze(dim=0).repeat(MLP_inputs.shape[0], 1, 1), MLP_inputs.unsqueeze(2)).squeeze(2)
            MLP_inputs = wx + self.bias[layer].unsqueeze(dim=0).repeat(MLP_inputs.shape[0], 1)
            
            if layer != self.nLayers-1:
                MLP_inputs = self.activation_swish(MLP_inputs)


        dx =   MLP_inputs.reshape(-1, 4*self.na)
        # print("dx:", dx[0])
        # print()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:]