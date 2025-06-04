import torch

from Task.VMAS.TaskVMAS import TaskVMAS

class Navigation(TaskVMAS):

    def __init__(self, config):
        inputSize = 18 # Raw observations as input for the model
        r = torch.as_tensor(0.5) 
        super().__init__(config, inputSize, r)
        self.map_size = config["task"]["map_size"]

        pos_goal_idx = []
        for i in range(0, self.numAgents*self.obs_dim_per_agent, self.obs_dim_per_agent):
            pos_goal_idx.append(i+4)
            pos_goal_idx.append(i+5)

        self.feature_index["goal_rel_positions"] = pos_goal_idx
        self.simulation_step = 0.1  

    def reduceObservability(self, inputs):
        input_vars = inputs.clone()
        obs_mask = torch.ones(input_vars.shape[1], dtype=bool)
        obs_mask[self.feature_index["robot_positions"]] = False
        obs_mask[self.feature_index["robot_velocities"]] = False
        obs_mask[self.feature_index["goal_rel_positions"]] = False

        input_vars[:,obs_mask] = 0

        return input_vars

    def setWorldStates(self, env, env_index, obs):
        agent_pos = obs[self.feature_index["robot_positions"]].reshape(self.numAgents, 2)
        agent_vel = obs[self.feature_index["robot_velocities"]].reshape(self.numAgents, 2)
        goal_pos_aux = obs[self.feature_index["goal_rel_positions"]].reshape(self.numAgents, 2)
        goal_pos = agent_pos - goal_pos_aux
        scenario = env.scenario

        for i, agent in enumerate(scenario.world.agents):
            agent.state.pos[env_index] = agent_pos[i]
            agent.state.vel[env_index] = agent_vel[i]

        # Asignar metas a los agentes
        for i, agent in enumerate(scenario.world.agents):
            agent.goal.state.pos[env_index] = goal_pos[i]

        # Calcular la métrica pos_shaping
        for i, agent in enumerate(scenario.world.agents):
            agent.pos_shaping[env_index] = (
                torch.linalg.vector_norm(
                    agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                )
                * scenario.pos_shaping_factor
            )

    def randomInitialState(self, batch_size=1):
        na = self.numAgents

        d = 1
        d2 = 1
        agents_x = torch.zeros(batch_size, na)
        agents_y = torch.zeros(batch_size, na)
        goals_x = torch.zeros(batch_size, na)
        goals_y = torch.zeros(batch_size, na)
        for i in range(batch_size):
            # Randomly generate initial positions for agents and goals
            agents_x[i, :] = torch.cat((d * torch.ones(int(na/2)), -d * torch.ones(int(na/2)))) + 0.6*(torch.rand(na)-0.5)
            agents_y[i, :] = torch.cat((torch.linspace(-d * (int(na/4)-1) - d2, d * (int(na/4)-1) + d2, int(na/2)),
                                        torch.linspace(d * (int(na/4)-1) + d2, -d * (int(na/4)-1) - d2, int(na/2)))) + 0.6*(torch.rand(na)-0.5)

            goals_x[i, :] = torch.cat((-d * torch.ones(int(na/2)), d * torch.ones(int(na/2)))) + 0.6*(torch.rand(na)-0.5)
            goals_y[i, :] = torch.cat((torch.linspace(d * (int(na/4)-1) + d2, -d * (int(na/4)-1) - d2, int(na/2)),
                                        torch.linspace(-d * (int(na/4)-1) - d2, d * (int(na/4)-1) + d2, int(na/2))))+ 0.6*(torch.rand(na)-0.5)
        
        obs = []
        for agent in range(na):
            agent_obs = torch.zeros(batch_size, self.agent_input_size)
            agent_obs[:,0] = agents_x[:, agent]
            agent_obs[:,1] = agents_y[:,agent]
            agent_obs[:,4] = agents_x[:,agent] - goals_x[:,agent] 
            agent_obs[:,5] = agents_y[:,agent] - goals_y[:,agent]
            obs.append(agent_obs)

        return tuple(obs)

    def numCompletedTasks(self, trajectory):
        final_state = trajectory[-1,:]
        goal_pos = final_state[self.feature_index["goal_rel_positions"]].reshape(-1,2)
        dists = torch.norm(goal_pos, dim=1)
        completed = dists < 0.075
        return completed.sum()

        
    def getVMASConfig(self, expert_robots=False):
        conf = {    "world_spawning_x": self.map_size[0] / 2,
                    "world_spawning_y": self.map_size[1] / 2
                }
        return conf