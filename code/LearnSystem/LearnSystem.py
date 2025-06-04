import time, imageio
from pathlib import Path
from abc import ABC, abstractmethod
from torch import nn, torch
from torchdiffeq import odeint
from Task.TaskBuilder import TaskBuilder
class learnSystem(nn.Module, ABC):

    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config["general"]["device"] if torch.cuda.is_available() else 'cpu')

        self.open_loop = config["learn_system"]["open_loop"]
        self.action_loss = config["learn_system"]["action_loss"]
        self.learning_rate = config["learn_system"]["learning_rate"]
        self.task = TaskBuilder(config)
        self.return_noisy_obs = False

        if(self.open_loop):
            # if not hasattr(self.task, "RL_env"):
            #     print("Error (LearnSystem): In order to simulate open-loop dynamics",
            #         type(self.task),
            #         "should provide a \"RL_env\".")
            #     exit()
                
            # if not (hasattr(self.task, "computeActions") or
            #         callable(getattr(self.task, "computeActions"))):
            #     print("Error (LearnSystem): In order to simulate open-loop dynamics",
            #         type(self.task),
            #         "should provide the method \"computeActions\".")
            #     exit()
            pass
        else:
            if not hasattr(self.task, "simulation_step"):
                print("Error (LearnSystem): In order to manage closed-loop dynamics",
                    type(self.task),
                    "requires the attribute \"simulation_step\".")
                exit()

    @abstractmethod
    def flocking_dynamics(self, t, inputs):
       pass
    
    def forward(self, inputs, nFrames=10):
        if(self.open_loop):
            trajectories, actions, rewards = self.solve_open_loop(inputs, nFrames)
        else:
            time = nFrames * self.task.simulation_step
            simulation_time = torch.linspace(0, time - self.task.simulation_step, nFrames)

            trajectories = odeint(self.closed_loop_dynamics, inputs, simulation_time.to(self.device), 
                            method='euler', options={'step_size': self.task.simulation_step})
            actions = None
            rewards = None
        return trajectories, actions, rewards
    
    # This discretization assumes that the only dynamic elements  
    # in the feature vectore are robot states
    def closed_loop_dynamics(self, t, inputs):
        robot_dynamics = self.flocking_dynamics(t, inputs)
        # print("pos_dyn", robot_dynamics[0].shape)
        # print("vel_dyn", robot_dynamics[1].shape)
        # input()

        global_dynamics = torch.zeros(inputs.shape).to(self.device)
        global_dynamics[:,self.task.feature_index["robot_positions"]] = robot_dynamics[0]
        global_dynamics[:,self.task.feature_index["robot_velocities"]] = robot_dynamics[1]
        return global_dynamics
    
    def solve_open_loop(self, inputs, nFrames):
        env = self.task.setupEnvs(inputs)

        trajectories = torch.zeros([nFrames, inputs.shape[0], inputs.shape[1]]).to(self.device)
        action_traj = torch.zeros([nFrames, inputs.shape[0], self.task.numAgents*self.task.action_dim_per_agent]).to(self.device)
        rewards = torch.zeros([nFrames, inputs.shape[0], self.task.numAgents]).to(self.device)

        trajectories[0, :, :] = inputs
        inputs_frame_noiseLess = inputs
        for i in range(1,nFrames):
            # print("Frame ", i)
            # env.render()

            # Filter data: Add noise and reduce observability
            inputs_frame = self.task.addNoise(inputs_frame_noiseLess)
            if hasattr(self.task, "reduceObservability") and callable(getattr(self.task, "reduceObservability")):
                inputs_frame = self.task.reduceObservability(inputs_frame)

            # Compute actions    
            closed_loop_dynamics = torch.cat(self.flocking_dynamics(0, inputs_frame), dim=1)
            actions_noiseless = self.task.computeActions(closed_loop_dynamics)
            actions = self.task.addNoise(actions_noiseless, True)
            act_list = []
            for j in range(self.task.numAgents):
                act_list.append(actions[:,j,:])

            # Step
            obs, rews, done, truncated, info = env.step(act_list)
            rewards[i,:,:] = torch.stack(rews).transpose(0,1)
            inputs_frame_noiseLess = self.task.reshapeObservation(obs)

            if self.return_noisy_obs:
                trajectories[i, :, :] = inputs_frame
                action_traj[i-1, :, :] = actions.reshape(-1,self.task.numAgents*self.task.action_dim_per_agent)
            else:
                trajectories[i, :, :] = inputs_frame_noiseLess
                action_traj[i-1, :, :] = actions_noiseless.reshape(-1,self.task.numAgents*self.task.action_dim_per_agent)

        # Get last action
        if hasattr(self.task, "reduceObservability") and callable(getattr(self.task, "reduceObservability")):
            inputs_frame = self.task.reduceObservability(inputs_frame)

        closed_loop_dynamics = torch.cat(self.flocking_dynamics(0, inputs_frame), dim=1)
        actions_noiseless = self.task.computeActions(closed_loop_dynamics)
        actions = self.task.addNoise(actions_noiseless, True)
        aux_act = actions if self.return_noisy_obs else actions_noiseless
        aux_act = aux_act.reshape(-1,self.task.numAgents*self.task.action_dim_per_agent)
        action_traj[-1,:,:] = aux_act
        # ----------------

        return trajectories, action_traj, rewards
    
    
    def next_filename(self, base_name, extension):
        index = 0
        while Path(f"{base_name}_{index}{extension}").exists():
            index += 1
        return f"{base_name}_{index}{extension}"