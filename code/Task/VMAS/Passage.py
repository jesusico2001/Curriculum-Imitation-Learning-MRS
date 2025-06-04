import torch

from Task.VMAS.TaskVMAS import TaskVMAS
from torch.autograd import Variable

class Passage(TaskVMAS):
    
    def __init__(self, config):
        inputSize = 8
        r = torch.as_tensor(0.5) 
        super().__init__(config, inputSize, r)
        self.__featureIndexPassage()
        self.simulation_step = 0.1

        assert(self.numAgents == 5)

    def setWorldStates(self, env, env_index, obs):
        # Extract data from observation
        agent_pos = obs[self.feature_index["robot_positions"]].reshape(self.numAgents, 2)
        agent_vel = obs[self.feature_index["robot_velocities"]].reshape(self.numAgents, 2)
        
        goal_pos_aux = obs[self.feature_index["pos_goal_for_agent"]].reshape(self.numAgents, 2)
        goal_pos = agent_pos + goal_pos_aux

        passage_pos_aux = obs[self.feature_index["pos_passage_for_agent"]].reshape(self.numAgents, 2)
        passage_pos = agent_pos[0] + passage_pos_aux[0]

        # Agent states
        scenario = env.scenario
        for i, agent in enumerate(scenario.world.agents):
            agent.state.pos[env_index] = agent_pos[i]
            agent.state.vel[env_index] = agent_vel[i]
            agent.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * scenario.shaping_factor
                )

        # Goals
        goals = scenario.world.landmarks[:self.numAgents]
        for i, goal in enumerate(goals):
            goal.set_pos(goal_pos[i], batch_index=env_index)

        passages = scenario.world.landmarks[self.numAgents:]
        old_idx = -1
        closest = 1e9
        for i, passage in enumerate(passages):
            passage_dist = abs(passage.state.pos[0,0] - passage_pos[0])
            if ( passage_dist < closest):
                old_idx = i
                closest = passage_dist
                
        assert(old_idx!=-1)
        passages[old_idx].set_pos(passages[0].state.pos[env_index], batch_index=env_index)
        passages[0].set_pos(passage_pos.unsqueeze(0), batch_index=env_index)

    def __featureIndexPassage(self):
        pos_goal_idx = []
        pos_passage_idx = []
        for i in range(0, self.numAgents*self.obs_dim_per_agent, self.obs_dim_per_agent):
            pos_goal_idx.append(i+4)
            pos_goal_idx.append(i+5)
            pos_passage_idx.append(i+6)
            pos_passage_idx.append(i+7)

        self.feature_index["pos_goal_for_agent"] = pos_goal_idx
        self.feature_index["pos_passage_for_agent"] = pos_passage_idx

    def numCompletedTasks(self, trajectory):
        final_state = trajectory[-1,:]
        goal_pos = final_state[self.feature_index["goal_rel_positions"]].reshape(-1,2)
        dists = torch.norm(goal_pos, dim=1)
        completed = dists < 0.075
        return completed.sum()
