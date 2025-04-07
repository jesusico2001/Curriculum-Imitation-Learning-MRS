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
        self.learning_rate = config["learn_system"]["learning_rate"]
        self.task = TaskBuilder(config)
        self.return_noisy_obs = True

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
            outputs = self.solve_open_loop(inputs, nFrames)
        else:
            time = nFrames * self.task.simulation_step
            simulation_time = torch.linspace(0, time - self.task.simulation_step, nFrames)

            outputs = odeint(self.closed_loop_dynamics, inputs, simulation_time.to(self.device), 
                            method='euler', options={'step_size': self.task.simulation_step})
        return outputs
    
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
        trajectories[0, :, :] = inputs
        inputs_frame = inputs
        for i in range(1,nFrames):
            # print("Frame ", i)
            # env.render()
            
            if hasattr(self.task, "reduceObservability") and callable(getattr(self.task, "reduceObservability")):
                inputs_frame = self.task.reduceObservability(inputs_frame)
                
            closed_loop_dynamics = torch.cat(self.flocking_dynamics(0, inputs_frame), dim=1)
            actions = self.task.computeActions(closed_loop_dynamics)

            obs, rews, done, truncated, info = env.step(actions)
            inputs_frame_noiseLess = self.task.reshapeObservation(obs)
            inputs_frame = self.task.addNoise(inputs_frame_noiseLess)


            if self.return_noisy_obs:
                trajectories[i, :, :] = inputs_frame
            else:
                trajectories[i, :, :] = inputs_frame_noiseLess

        return trajectories
    
    def next_filename(self, base_name, extension):
        index = 0
        while Path(f"{base_name}_{index}{extension}").exists():
            index += 1
        return f"{base_name}_{index}{extension}"