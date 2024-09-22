from torch import torch, nn

from LearnSystem.LearnSystem import learnSystem
from AttentionModule import Attention_R, Attention_J, Attention_H


# NO CENTROIDS
class LEMURS(learnSystem):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.learning_rate = 1e-3
        self.R       = Attention_R.Att_R(8, self.controlPolicy.input_size, 4 * 4, parameters["nAttLayers"], device=self.device).to(self.device)
        self.J       = Attention_J.Att_J(8, self.controlPolicy.input_size, 1, parameters["nAttLayers"], device=self.device).to(self.device) #TODO: Decidir si mantener el 1 de TVS o hacer casos
        self.H       = Attention_H.Att_H(8, self.controlPolicy.input_size, 5 * 5, parameters["nAttLayers"], device=self.device).to(self.device)

    def flocking_dynamics(self, t, inputs):
        # Obtain Laplacians
        L = self.controlPolicy.laplacian(inputs[:, :2 * self.na]).to(self.device)

        # Get inputs for the self-attention modules
        inputs  = nn.functional.normalize(inputs, p=2, dim=1)
        
        state_d = self.getStateDiffs(inputs)
        pos, vel = self.getRelativeStates(inputs) 
        
        # Self attention modules
        R, J = self.__forwardRJ(L, state_d.clone(), pos, vel, inputs)
        # print("R = ",R.size())
        # print("J = ",J.size())
        with torch.enable_grad():
            H, inputs_l = self.__forwardH(L, state_d, pos, vel, inputs)
            #print("H = ", H[0])
            del L
            dHdx = self.__structureGradients(H, inputs_l)

        # Closed-loop dynamics
        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)
        del R, J, H, inputs_l, dHdx
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        
        # print("DX: ", dx.size())
        return dx[:, :2 * self.na], dx[:, 2 * self.na:]


    def __forwardRJ(self, L, state_d, pos, vel, inputs):
        # Input for R and J
        inputs_att, LLL = self.__getAttentionInputs(L, state_d, pos, vel, inputs)

        # R and J
        R = self.R.forward(inputs_att.to(torch.float32), LLL)
        J = self.J.forward(inputs_att.to(torch.float32), LLL)

        return R, J
    
    def __forwardH(self, L, state_d, pos, vel, inputs):
        # Input for H
        inputs_att, LLL = self.__getAttentionInputs(L, state_d, pos, vel, inputs)

        H       = self.H.forward(inputs_att.to(torch.float32), LLL)
        
        return H, inputs_att
        

    # For LEMURS' Self-Attention, each element of the batch is a robot's POV 
    def __getAttentionInputs(self, L, state_d, pos, vel, inputs):

        inputs_policy = self.controlPolicy.shapeInputs(inputs, state_d, pos, vel, L)
        inputs_att = torch.kron(inputs_policy, torch.ones((self.na, 1, 1), device=self.device))

        # Masks to create perception bias
        i = self.controlPolicy.input_size
        LLL         = torch.kron(L, torch.ones((1, 1, i), device=self.device))
        LLL[:, :, i-2::i]   = 1
        LLL[:, :, i-1::i]   = 1
        LLL = LLL.reshape(-1, self.na, i).transpose(1, 2)

        inputs_att           = LLL * inputs_att

        return inputs_att, LLL

    def __structureGradients(self, H, inputs_l):
        Hgrad   = torch.autograd.grad(H.sum(), inputs_l, only_inputs=True, create_graph=True)
        dH      = Hgrad[0]

        dHq     = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq     = dHq[:, range(self.na), range(self.na), :]
        
        dHp     = dH[:, 2:4, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp     = dHp[:, range(self.na), range(self.na), :]

        dHdx    = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)
        del Hgrad, dH, dHq, dHp
        
        return dHdx

