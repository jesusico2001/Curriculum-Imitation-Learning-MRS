import torch

from Task.VMAS.TaskVMAS import TaskVMAS
from Task.VMAS.Mall.MallScenario import Scenario as MallScenario

class Mall(TaskVMAS):

    def __init__(self, config):
        inputSize = 4 # Raw observations as input for the model
        r = torch.as_tensor(0.5) 
        super().__init__(config, inputSize, r)
        self.vmas_scenario =  MallScenario()
        self.simulation_step = 0.1  


    def setWorldStates(self, env, env_index, obs):
        agent_pos = obs[self.feature_index["robot_positions"]].reshape(self.numAgents, 2)
        agent_vel = obs[self.feature_index["robot_velocities"]].reshape(self.numAgents, 2)

        scenario = env.scenario
        for i, agent in enumerate(scenario.world.agents):
            agent.state.pos[env_index] = agent_pos[i]
            agent.state.vel[env_index] = agent_vel[i]


    def numCompletedTasks(self, trajectory):
        return 0

        