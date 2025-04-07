from abc import ABC, abstractmethod
from enum import Enum
import torch
from torch.autograd import Variable

from Task.Task import Task
from vmas import make_env

class TaskVMAS(Task):

    def __init__(self, config, agent_input_size, communication_radius):     
        self.action_dim_per_agent = 2
        self.obs_dim_per_agent = agent_input_size
   
        super().__init__(config, agent_input_size, communication_radius)

        self.vmas_scenario =  config["task"]["type"] 
        self.__build_real_dynamics()
        
    
    @abstractmethod
    def randomInitialState(self):
        pass

    def buildFeatureIndex(self):
        pos_idx = []
        vel_idx = []
        for i in range(0, self.numAgents*self.obs_dim_per_agent, self.obs_dim_per_agent):
            pos_idx.append(i)
            pos_idx.append(i+1)
            vel_idx.append(i+2)
            vel_idx.append(i+3)

        feature_index = {
            "robot_positions": pos_idx,
            "robot_velocities": vel_idx,
        }   
        return feature_index   

    def buildInputVariables(self, inputs):
        na = self.numAgents
        i = int(inputs.shape[1] / na)
        
        return Variable(inputs.reshape(-1, na, i).transpose(1, 2).data, requires_grad=True)
    
    def reshapeObservation(self, obs):
        batch_size = obs[0].shape[0]

        i = self.obs_dim_per_agent

        res = torch.zeros([batch_size, self.numAgents*i], device=self.device)
        for a, obs_a in enumerate(obs):
            res[:,a*i:(a+1)*i] = obs_a
        return res
    
    def computeActions(self, learned_dynamics):
        na = self.numAgents
        batch_size = learned_dynamics.shape[0]
        
        # Replicate real PHS for batch
        F_sys_pinv = self.F_sys_pinv.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        R_sys = self.R_sys.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        J_sys = self.J_sys.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        # Real dynamics
        dHdx_sys = torch.cat((torch.zeros(learned_dynamics.shape[0], int(learned_dynamics.shape[1]/2), device=self.device),
                                   learned_dynamics[:, :self.action_dim_per_agent * na]), dim=1).unsqueeze(2)
        real_dyn = torch.bmm(J_sys - R_sys, dHdx_sys)

        action_batch = torch.bmm(F_sys_pinv, learned_dynamics.unsqueeze(2) - real_dyn).reshape(batch_size, na, self.action_dim_per_agent)
        action = []
        for i in range(na):
            action.append(action_batch[:,i,:])
        return action
    
    def __build_real_dynamics(self):
        na = self.numAgents
        drag = 1 # TODO: Find appropiate value

        self.F_sys_pinv = torch.cat((torch.zeros(self.action_dim_per_agent * na,
                                                 self.action_dim_per_agent * na,
                                                 device=self.device),
                                 torch.eye(self.action_dim_per_agent * na, device=self.device)), dim=1)

        self.J_sys = torch.cat((torch.cat((torch.zeros(self.action_dim_per_agent * na,
                                                       self.action_dim_per_agent * na,
                                                       device=self.device),
                                 torch.eye(self.action_dim_per_agent * na, device=self.device)), dim=1),
                                torch.cat((-torch.eye(self.action_dim_per_agent * na, device=self.device),
                                torch.zeros(self.action_dim_per_agent * na,
                                            self.action_dim_per_agent * na, device=self.device)), dim=1)
                                ), dim=0)
        self.R_sys = torch.cat((torch.cat((torch.zeros(self.action_dim_per_agent * na,
                                                       self.action_dim_per_agent * na,
                                                       device=self.device),
                                 torch.zeros(self.action_dim_per_agent * na,
                                             self.action_dim_per_agent * na,
                                             device=self.device)), dim=1),
                                torch.cat((torch.zeros(self.action_dim_per_agent * na,
                                                       self.action_dim_per_agent * na,
                                                       device=self.device),
                                drag*torch.eye(self.action_dim_per_agent * na, device=self.device)), dim=1)
                                ), dim=0)

    # Environment management
    # ======================
    @abstractmethod 
    def setWorldStates(inputs):
        pass

    def setupEnvs(self, inputs):
        num_envs = inputs.shape[0]

        self.env = make_env(
            scenario= self.vmas_scenario,
            num_envs=num_envs,
            device=self.device,
            continuous_actions=True,
            clamp_actions=True,
            grad_enabled=True,
            terminated_truncated=True,
            # Environment specific variables
            n_agents=self.numAgents,
            max_steps = self.episode_difficulty-1
        )

        for env_idx in range(self.env.num_envs):
            self.setWorldStates(self.env, env_idx, inputs[env_idx])
        return self.env
    
    # Returns all zeros. Implement your own in 
    # the inherited class to set 
    def randomInitialState(self):
        i = self.agent_input_size
        na = self.numAgents
        obs = []
        for agent in range(na):
            agent_obs = torch.zeros(i)
            obs.append(agent_obs.unsqueeze(0))
        return tuple(obs)

    def gif(self, trajectory):
            frames = []
            for i in range(trajectory.shape[0]):
                obs = trajectory[i,:].unsqueeze(0)
                # print(obs)
                env = self.setupEnvs(obs)
                frame = env.render(mode="rgb_array")
                frames.append(frame)
            return frames