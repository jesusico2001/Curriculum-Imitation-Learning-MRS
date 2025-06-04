from abc import ABC, abstractmethod
from enum import Enum

import torch


class Task(ABC):

    def __init__(self, config, agent_input_size, communication_radius):        
        self.device = torch.device(config["general"]["device"] if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        self.numAgents = config["task"]["num_agents"]
        self.episode_difficulty = config["task"]["episode_difficulty"]
        self.robot_obs_noise = config["task"]["robot_obs_noise"]
        self.action_noise_factor = config["task"]["action_noise_factor"]

        self.agent_input_size = agent_input_size
        self.communication_radius = communication_radius
        self.feature_index = self.buildFeatureIndex()

        self.simulation_step = 0.1 # Default temporal step (assumed for RL envs)

    @abstractmethod
    def buildFeatureIndex(self):
        pass

    # Recieves a batch of initial states [batch, num_agents * robot_state_obs]
    # whose second dimension's structure is undefined (format of the dataset).
    # Returns a tensor t [bath, robot_state_obs] of torch Variables such
    # that the second dimension concatenates each robot's feature vector. The first
    # 4 components of each feature vector represent its position and velocity.
    @abstractmethod
    def buildInputVariables(self, inputs):
        pass

    def laplacian(self, positions):
        # Distance based laplacian
        na = self.numAgents
        pos = positions.reshape(positions.shape[0], int(2*na))
        
        Q1 = pos.reshape(pos.shape[0], -1, 2).repeat(1, na, 1)
        Q2 = torch.kron(pos.reshape(pos.shape[0], -1, 2), torch.ones((1, na, 1), device=Q1.device))
        Q  = (Q1 - Q2).norm(p=2, dim=2).reshape(pos.shape[0], na, na)
        L  = Q.le(self.communication_radius).float()
        L = L * torch.sigmoid(-(2.0)*(Q - self.communication_radius))
        
        return L
    
    def getRobotStates(self, trajectories):
        positions = trajectories[:,:,self.feature_index["robot_positions"]]
        velocities = trajectories[:,:,self.feature_index["robot_velocities"]]
        return torch.cat([positions, velocities], 2)
        
    def getPosVel(self, inputs):
        pos = inputs[:, self.feature_index["robot_positions"]].reshape(inputs.shape[0], -1, 2)
        vel = inputs[:, self.feature_index["robot_velocities"]].reshape(inputs.shape[0], -1, 2)
        return pos, vel

    # Noise in observations
    def addNoise(self, data, actions=False):
        factor = self.action_noise_factor if actions else self.robot_obs_noise
        noise = torch.randn_like(data)  * factor
        return data + noise
    
    # For evaluation
    @abstractmethod
    def numCompletedTasks(self, trajectory):
        pass
