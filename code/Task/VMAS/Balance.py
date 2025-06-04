import torch

from Task.VMAS.TaskVMAS import TaskVMAS
from torch.autograd import Variable

class Balance(TaskVMAS):
    
    def __init__(self, config):
        self.package_is_visible = config["task"]["package_is_visible"]

        inputSize = 16
        r = torch.as_tensor(0.5) 
        super().__init__(config, inputSize, r)
        self.__featureIndexBalance()
        self.simulation_step = 0.1

    # Implement to constrain observability
    # ======================================
    def reduceObservability(self, inputs):
        input_vars = inputs.clone()
        if not self.package_is_visible:
            obs_mask = torch.zeros(input_vars.shape[1], dtype=bool)
            obs_mask[self.feature_index["pos_robot_for_package"]] = True
            obs_mask[self.feature_index["package_velocities"]] = True

            pos_robot_for_goal = input_vars[:,self.feature_index["pos_robot_for_package"]] + input_vars[:,self.feature_index["pos_package_for_goal"]]

            input_vars[:,obs_mask] = 0
            input_vars[:,self.feature_index["pos_package_for_goal"]] = pos_robot_for_goal

        return input_vars
    # ======================================

    def setWorldStates(self, env, env_index, obs):
        # Extract data from observation
        agent_pos = obs[self.feature_index["robot_positions"]].reshape(self.numAgents, 2)
        agent_vel = obs[self.feature_index["robot_velocities"]].reshape(self.numAgents, 2)
        
        package_pos_aux = obs[self.feature_index["pos_robot_for_package"]].reshape(self.numAgents, 2)
        package_pos = agent_pos[0] - package_pos_aux[0]
        
        line_pos_aux = obs[self.feature_index["pos_robot_for_line"]].reshape(self.numAgents, 2)
        line_pos = agent_pos[0] - line_pos_aux[0]
        
        goal_pos_aux = obs[self.feature_index["pos_package_for_goal"]].reshape(self.numAgents, 2)
        goal_pos = package_pos - goal_pos_aux[0]

        package_vel = obs[self.feature_index["package_velocities"]].reshape(self.numAgents, 2)[0]
        line_vel = obs[self.feature_index["line_velocities"]].reshape(self.numAgents, 2)[0]

        line_angvel = obs[self.feature_index["line_angular_velocities"]][0]
        line_rotation = obs[self.feature_index["line_rotation"]][0]
        # Set state
        scenario = env.scenario
        for i, agent in enumerate(scenario.world.agents):
            agent.state.pos[env_index] = agent_pos[i]
            agent.state.vel[env_index] = agent_vel[i]

        scenario.package.set_pos(package_pos, batch_index=env_index)
        scenario.package.set_vel(package_vel, batch_index=env_index )
        scenario.package.goal.set_pos(goal_pos, batch_index=env_index)

        scenario.line.set_pos(line_pos, batch_index=env_index)
        scenario.line.set_vel(line_vel, batch_index=env_index)
        scenario.line.set_rot(line_rotation, batch_index=env_index)
        scenario.line.state.ang_vel[env_index] = line_angvel

        # As in VMAS reset_world
        scenario.floor.set_pos(
            torch.tensor(
                [
                    0,
                    -scenario.world.y_semidim
                    - scenario.floor.shape.width / 2
                    - scenario.agent_radius,
                ],
                device=scenario.world.device,
            ),
            batch_index=env_index,
        )
        scenario.compute_on_the_ground()
        scenario.global_shaping[env_index] = (
            torch.linalg.vector_norm(
                scenario.package.state.pos[env_index]
                - scenario.package.goal.state.pos[env_index]
            )
            * scenario.shaping_factor
        )

    def __featureIndexBalance(self):
        pos_package_idx = []
        pos_line_idx = []
        pos_goal_idx = []
        vel_package_idx = []
        vel_line_idx = []
        angvel_line_idx = []
        rotation_line_idx = []
        for i in range(0, self.numAgents*self.obs_dim_per_agent, self.obs_dim_per_agent):
            pos_package_idx.append(i+4)
            pos_package_idx.append(i+5)
            pos_line_idx.append(i+6)
            pos_line_idx.append(i+7)
            pos_goal_idx.append(i+8)
            pos_goal_idx.append(i+9)

            vel_package_idx.append(i+10)
            vel_package_idx.append(i+11)
            vel_line_idx.append(i+12)
            vel_line_idx.append(i+13)

            angvel_line_idx.append(i+14)
            rotation_line_idx.append(i+15)

        self.feature_index["pos_robot_for_package"] = pos_package_idx
        self.feature_index["pos_robot_for_line"] = pos_line_idx
        self.feature_index["pos_package_for_goal"] = pos_goal_idx

        self.feature_index["package_velocities"] = vel_package_idx
        self.feature_index["line_velocities"] = vel_line_idx

        self.feature_index["line_angular_velocities"] = angvel_line_idx
        self.feature_index["line_rotation"] = rotation_line_idx

    def numCompletedTasks(self, trajectory):
        final_state = trajectory[-1,:]
        goal_pos = final_state[self.feature_index["pos_package_for_goal"]].reshape(-1,2)
        dists = torch.norm(goal_pos, dim=1)
        completed = dists < 0.075
        return completed.sum()

