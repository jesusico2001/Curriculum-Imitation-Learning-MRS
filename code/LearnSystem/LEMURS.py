from torch import torch, nn

from LearnSystem.LearnSystem import learnSystem
from AttentionModule import Attention_R, Attention_J, Attention_H


# NO CENTROIDS
class LEMURS(learnSystem):

    def __init__(self, config):
        super().__init__(config)
        self.depth=config["learn_system"]["depth"]
        self.R       = Attention_R.Att_R(8, self.task.agent_input_size, 4 * 4, self.depth, device=self.device).to(self.device)
        self.J       = Attention_J.Att_J(8, self.task.agent_input_size, 1, self.depth, device=self.device).to(self.device) #TODO: Decidir si mantener el 1 de TVS o hacer casos
        self.H       = Attention_H.Att_H(8, self.task.agent_input_size, 5 * 5, self.depth, device=self.device).to(self.device)

    def flocking_dynamics(self, t, inputs):
        # Get inputs for the self-attention modules
        inputs  = nn.functional.normalize(inputs, p=2, dim=1)
                
        # Self attention modules
        R, J = self.__forwardRJ(inputs)
        # print("R = ",R.size())
        # print("J = ",J.size())
        with torch.enable_grad():
            H, inputs_l = self.__forwardH(inputs)
            #print("H = ", H[0])
            dHdx = self.__structureGradients(H, inputs_l)

        # Closed-loop dynamics
        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)
        del R, J, H, inputs_l, dHdx
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


        # print("DX: ", dx.size())
        na = self.task.numAgents
        return dx[:, :2 * na], dx[:, 2 * na:]


    def __forwardRJ(self, inputs):
        # Input for R and J
        inputs_att, LLL = self.__getAttentionInputs(inputs)
        # R and J
        R = self.R.forward(inputs_att.to(torch.float32), LLL)
        J = self.J.forward(inputs_att.to(torch.float32), LLL)

        del inputs_att, LLL
        return R, J
    
    def __forwardH(self, inputs):
        # Input for H
        inputs_att, LLL = self.__getAttentionInputs(inputs)

        H       = self.H.forward(inputs_att.to(torch.float32), LLL)

        del LLL
        return H, inputs_att
        

    # For LEMURS' Self-Attention, each element of the batch is a robot's POV 
    def __getAttentionInputs(self, inputs):
        na = self.task.numAgents
        inputs_policy = self.task.buildInputVariables(inputs)
        inputs_att = torch.kron(inputs_policy, torch.ones((na, 1, 1), device=self.device))

        # Masks to create perception bias
        pos, _ = self.task.getPosVel(inputs)
        L = self.task.laplacian(pos)
        i = self.task.agent_input_size
        LLL         = torch.kron(L, torch.ones((1, 1, i), device=self.device))
        # LLL[:, :, i-2::i]   = 1
        # LLL[:, :, i-1::i]   = 1
        LLL = LLL.reshape(-1, na, i).transpose(1, 2)
        inputs_att           = LLL * inputs_att

        del pos, L 
        return inputs_att, LLL

    def __structureGradients(self, H, inputs_l):
        # LEMURS expects the feature vector to contain
        # robot pos and vel in the first 4 components
        na = self.task.numAgents

        Hgrad   = torch.autograd.grad(H.sum(), inputs_l, only_inputs=True, create_graph=True)
        dH      = Hgrad[0]

        dHq     = dH[:, :2, :].reshape(-1, na, 2, na).transpose(2, 3)
        dHq     = dHq[:, range(na), range(na), :]
        
        dHp     = dH[:, 2:4, :].reshape(-1, na, 2, na).transpose(2, 3)
        dHp     = dHp[:, range(na), range(na), :]

        dHdx    = torch.cat((dHq.reshape(-1, 2 * na), dHp.reshape(-1, 2 * na)), dim=1)
        del Hgrad, dH, dHq, dHp
        
        return dHdx

