from collections import namedtuple

from DatasetGenerator.VMAS.VMASGenerator import VMASGenerator
import torch

class PassageGenerator(VMASGenerator):
    def __init__(self, config):
        super().__init__(config)        
        
        self.experiment = self.initExperiment()
        self.agent = self.experiment.policy
                


    def initExperiment(self):
        Experiment = namedtuple("PolicyData", ["policy"])
        expert = self.PotentialFieldsAgent(self.task)
        return Experiment(policy=expert)

    class PotentialFieldsAgent():
        def __init__(self, task):
            self.task = task

        def forward(self, obs):
            obs_tensor = torch.cat([torch.tensor(array, dtype=torch.float32) for array in obs]).unsqueeze(0)
            pos, _ = self.task.getPosVel(obs_tensor)
            pos_goal_for_agent = obs_tensor[:,self.task.feature_index["pos_goal_for_agent"]].reshape(-1,2)
            pos_passage_for_agent = obs_tensor[:,self.task.feature_index["pos_passage_for_agent"]].reshape(-1,2)

            attraction = self.getAttractiveTerm(pos_goal_for_agent, pos_passage_for_agent)
            repulsion_agents  = self.getAgentRepulsion(pos)
            repulsion_wall  = self.getWallRepulsion(pos_passage_for_agent)

            forces = 2*attraction + 0.01*repulsion_agents + 0.01*repulsion_wall
            return [None, None, None, forces]

        def getAttractiveTerm(self, goal, passage):
            overshoot = 0.5
            hasPassed = (passage[:,1] < -overshoot).unsqueeze(-1).repeat(1,2)
            return goal * hasPassed + (passage + torch.tensor([0,overshoot])) * ~hasPassed
            
        def getAgentRepulsion(self, pos):
            na = self.task.numAgents
            p1 = pos.reshape(pos.shape[0], -1, 2).repeat(1, na, 1).reshape(na,na,2)
            p2 = torch.kron(pos.reshape(pos.shape[0], -1, 2), torch.ones((1, na, 1), device=pos.device)).reshape(na,na,2)
            pos_rel = p2-p1
            
            norm = torch.linalg.vector_norm(pos_rel, dim=-1) + 0.001
            mask = (norm < 0.2).unsqueeze(-1).repeat(1,1,2)
            k = 1/norm.unsqueeze(-1).repeat(1,1,2)   
            
            return torch.sum(k*mask*pos_rel,1)

        def getWallRepulsion(self, passage):
            f = torch.zeros_like(passage)
            mask = torch.linalg.norm(passage[:,0]) > 0.2
            
            y_dist = passage[:,1] + 1e-5
            f[:,1] = mask*pow(1/y_dist, 3)         
            return -f