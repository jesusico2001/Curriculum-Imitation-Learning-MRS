from abc import abstractmethod
import os, re
from DatasetGenerator.Generator import Generator
import torch

from benchmarl.models.mlp import MlpConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments import VmasTask
from benchmarl.algorithms import MappoConfig
from Task.VMAS.Mall.MallScenario import Scenario as MallScenario

BENCHMARL_task_map = {
    "navigation": VmasTask.NAVIGATION, 
    "balance": VmasTask.BALANCE,
    "passage": VmasTask.PASSAGE,
    "mall": MallScenario,
}

class VMASGenerator(Generator):
    def __init__(self, config):
        super().__init__(config)                
        self.initial_states_from_vmas = True


    def generateDataset(self, numData):
            if not (hasattr(self, "experiment") and hasattr(self, "agent")):
                self.experiment = self.initExperiment()
                self.agent = self.experiment.policy

            agent_obs_size = self.task.obs_dim_per_agent
            agent_act_size = self.task.action_dim_per_agent
            
            demonstrations = torch.zeros(numData, self.episode_difficulty, self.na, agent_obs_size).to(self.device)
            actions = torch.zeros(numData, self.episode_difficulty, self.na, agent_act_size).to(self.device)
            rewards = torch.zeros(numData, self.episode_difficulty, self.na).to(self.device)
            
            for i in range(numData):
                j=0
                obs = self.task.randomInitialState()
                env = self.task.setupEnvs(self.task.reshapeObservation(obs))
                if self.initial_states_from_vmas:
                    obs = env.reset()
                
                for a, l in enumerate(obs):
                    demonstrations[i, j, a, :] = l.clone().detach()
                obs = [tensor.squeeze(0).cpu().numpy() for tensor in obs]
                
                done, truncated = False, False
                while not (truncated):
                    # env.render()
                    with torch.no_grad():
                        action = self.agent.forward(obs)[3]
                    action = [ [a.tolist()] for a in action]
                    for a, l in enumerate(action):
                        actions[i, j, a, :] = torch.tensor(l)
                    
                    j += 1

                    obs, reward, done, truncated, info = env.step(action)
                    obs = [tensor.squeeze(0).cpu().numpy() for tensor in obs]
                    for a, l in enumerate(obs):
                        demonstrations[i, j, a, :] = torch.tensor(l)
                    rewards[i,j,:] = torch.stack(reward).squeeze(1)
                
                with torch.no_grad():
                    action = self.agent.forward(obs)[3]
                    action = [ [a.tolist()] for a in action]
                    for a, l in enumerate(action):
                        actions[i, j, a, :] = torch.tensor(l)

                print('\tInstance '+str(i)+'.')

            demonstrations = demonstrations.reshape(numData, self.episode_difficulty, self.na*agent_obs_size)
            demonstrations = demonstrations.transpose(0,1)
            actions = actions.reshape(numData, self.episode_difficulty, self.na*agent_act_size)
            actions = actions.transpose(0,1)

            return demonstrations, actions, rewards.transpose(0,1)
    
    def initExperiment(self):
        self.path_config_experiment = "DatasetGenerator/config_benchmarl/experiment_VMAS.yaml"
        expId = self.getExperimentFolder()
        while(expId is None):
            print("No RL agent found for this config. Training one...")
            self.trainRLAgent()
            print("RL agent finished training!")
            expId = self.getExperimentFolder()
                
        folder = self.path_manager.getPathDatasets()+expId+"/checkpoints/"
        files = os.listdir(folder)
        
        if len(files) == 0:
            print("Error. Checkpoint not found in " + folder)
            exit()
        elif len(files) != 1:
            print("Warning: Multiple BenchMARL experiment ckecpoints at "+folder+". \nUsing arbitrary one.")
            
        return self.loadExperiment(folder+str(files[0]))

    def loadExperiment(self, path_checkpoint):
        # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
        experiment_config = ExperimentConfig.get_from_yaml(self.path_config_experiment)
        # experiment_config.train_device = self.device
        experiment_config.restore_file = path_checkpoint

        task = BENCHMARL_task_map[self.config["task"]["type"]].get_from_yaml()
        task.config["n_agents"] = self.na
        task.config["max_steps"] = self.task.episode_difficulty
        specific_cfg = self.task.getVMASConfig()
        task.config.update(specific_cfg)


        algorithm_config = MappoConfig.get_from_yaml()
        model_config = MlpConfig.get_from_yaml()
        critic_model_config = MlpConfig.get_from_yaml()

        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,

        )
        print("BenchMARL agent loaded!")
        return experiment
    

    def getExperimentFolder(self):
        directory_path = "./" + self.path_manager.getPathDatasets()

        if not os.path.isdir(directory_path):
            return None
        
        folders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]
        pattern = "^mappo_"
        matching_folders = [f for f in folders if re.match(pattern, f)]
        
        if len(matching_folders) == 0:
            return None
        elif len(matching_folders) == 1:
            return matching_folders[0]
        else:
            print("Warning: Multiple BenchMARL experiment folders. Using arbitrary one.")
            return matching_folders[0]


    def trainRLAgent(self):
        # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
        experiment_config = ExperimentConfig.get_from_yaml(self.path_config_experiment)
        # experiment_config.train_device = self.device
        experiment_config.render = False
        experiment_config.checkpoint_at_end = True
        experiment_config.sampling_device= self.device
        experiment_config.train_device= self.device
        experiment_config.buffer_device= self.device
        experiment_config.save_folder = "./" + self.path_manager.getPathDatasets()
        os.makedirs(experiment_config.save_folder, exist_ok=True)
        
        # Some basic other configs
        task = BENCHMARL_task_map[self.config["task"]["type"]].get_from_yaml()
        task.config["n_agents"] = self.na
        task.config["max_steps"] = self.task.episode_difficulty
        specific_cfg = self.task.getVMASConfig()
        task.config.update(specific_cfg)

        algorithm_config = MappoConfig.get_from_yaml()
        model_config = MlpConfig.get_from_yaml()
        critic_model_config = MlpConfig.get_from_yaml()

        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
        )
        experiment.run()



    # TODO:
    def generateDatasetBatch(self, numData):
        if not (hasattr(self, "experiment") and hasattr(self, "agent")):
            self.experiment = self.initExperiment()
            self.agent = self.experiment.policy

        agent_obs_size = self.task.obs_dim_per_agent
        agent_act_size = self.task.action_dim_per_agent
        
        demonstrations = torch.zeros(numData, self.episode_difficulty, self.na * agent_obs_size).to(self.device)
        actions = torch.zeros(numData, self.episode_difficulty, self.na * agent_act_size).to(self.device)
        rewards = torch.zeros(numData, self.episode_difficulty, self.na).to(self.device)
        
        batch_size = 100
        n_iters = numData // batch_size
        for i in range(n_iters):
            j=0
            obs = self.task.randomInitialState(batch_size)
            init_state = self.task.reshapeObservation(obs)
            env = self.task.setupEnvs(init_state)
            if self.initial_states_from_vmas:
                obs = env.reset()
                init_state = self.task.reshapeObservation(obs)
                
            demonstrations[i:i+batch_size, j, :] = init_state
            
            done, truncated = False, False
            while not (truncated):
                # TODO: Figure out how to use batch with BenchMARL
                # # env.render()
                # with torch.no_grad():
                #         # obs = [tensor.squeeze(0).cpu().numpy() for tensor in obs]
                #         action = self.agent.forward(obs)[3]

                # action = [ [a.tolist()] for a in action]
                # for a, l in enumerate(action):
                #     actions[i:i+batch_size, j, :] = torch.tensor(l)
                
                j += 1

                obs, reward, done, truncated, info = env.step(action)
                obs = [tensor.squeeze(0).cpu().numpy() for tensor in obs]
                for a, l in enumerate(obs):
                    demonstrations[i, j, a, :] = torch.tensor(l)
                rewards[i,j,:] = torch.stack(reward).squeeze(1)

            with torch.no_grad():
                action = self.agent.forward(obs)[3]
                action = [ [a.tolist()] for a in action]
                for a, l in enumerate(action):
                    actions[i, j, a, :] = torch.tensor(l)

            print('\tInstance '+str(i)+'.')

        demonstrations = demonstrations.reshape(numData, self.episode_difficulty, self.na*agent_obs_size)
        demonstrations = demonstrations.transpose(0,1)
        actions = actions.reshape(numData, self.episode_difficulty, self.na*agent_act_size)
        actions = actions.transpose(0,1)

        return demonstrations, actions, rewards.transpose(0,1)
