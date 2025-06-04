from DatasetGenerator.RealSystem import RealSystemBuilder, RealSystem
from DatasetGenerator.Generator import Generator
import torch

class RealSystemGenerator(Generator):
    def __init__(self, config):
        super().__init__(config)
        self.time = 10
        self.step_size = float(self.time) / self.episode_difficulty
        
        # Initialize the realSystem
        self.task = self.config["task"]["type"]
        rsParams = RealSystemBuilder.buildParameters(self.task, self.na)
        self.realSys = RealSystemBuilder.buildRealSystem(self.task, rsParams)
        print("Real System built... - Type= ", type(self.realSys), "\n")

    def generateDataset(self, numData):
        simulation_time = torch.linspace(0, self.time-self.step_size, int(self.time/self.step_size))
        num_trajectories  = int(numData/self.time*self.step_size*self.episode_difficulty)

        demonstrations = torch.zeros(self.episode_difficulty, numData, 8 * self.na)
        for k in range(num_trajectories):
            q_agents, p_agents               = self.realSys.generate_agents(self.na)
            q_dynamic, p_dynamic             = self.realSys.generate_leader(self.na)
            input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
            trajectory                       = self.realSys.sample(input, simulation_time, self.step_size)
            l                                = int(self.time/self.step_size/self.episode_difficulty)

            # Avoid strange examples due to numerical or initial configuration issues
            while torch.isnan(trajectory).any():
                q_agents, p_agents               = self.realSys.generate_agents(self.na)
                q_dynamic, p_dynamic             = self.realSys.generate_leader(self.na)
                input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
                trajectory                       = self.realSys.sample(input, simulation_time, self.step_size)
                l                                = int(self.time/self.step_size/self.episode_difficulty)
            
            demonstrations[:, l*k:l*(k+1), :] = (trajectory.reshape(-1, self.episode_difficulty, 8 * self.na).transpose(0, 1))
            print('\tInstance '+str(k)+'.')

        return demonstrations, None, None
