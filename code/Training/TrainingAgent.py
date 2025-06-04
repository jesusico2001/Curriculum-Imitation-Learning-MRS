import os
import yaml
import torch

from Training.PathManager import PathManager
from Training.Teacher.TeacherBuilder import TeacherBuilder
from LearnSystem import LearnSystemBuilder
from Training.DatasetBuilder import DatasetBuilder

import torchlens
from functools import partial

VAL_PERIOD = 50
class TrainingAgent():

    def __init__(self, path_config, config_changes=None):
        if isinstance(path_config, str):
            with open(path_config, "r") as file:
                config = yaml.safe_load(file)
                self.config = config
            print("Config loaded...")

        if config_changes:
            for key, value in config_changes.items():
                keys = key.split('.')
                cfg = config
                for k in keys[:-1]:
                    cfg = cfg[k]
                cfg[keys[-1]] = value
                print("Config parameter updated: ", key, "=", value)
        
        self.seed_train = config["general"]["seed_train"]
        torch.manual_seed(self.seed_train)
        
        # Path manager
        self.path_manager = PathManager(config)

        # Student agent
        self.learn_system = LearnSystemBuilder.buildLearnSystem(config) 
        print("Student created...")

        # Student optimizer
        learning_rate   = self.learn_system.learning_rate
        self.optimizer = torch.optim.Adam(self.learn_system.parameters(), lr=learning_rate, weight_decay=0.0)

        # Teacher agent
        self.teacher = TeacherBuilder(config["teacher"])
        print("Teacher created...")

        # Dataset manager
        self.dataset_builder = DatasetBuilder(config, self.learn_system.device)

        self.device = self.learn_system.device
        self.epochs = config["general"]["epochs"]
        self.perform_early_stopping = config["general"]["early_stopping"]
        self.history = {
            "loss_train" : [],
            "loss_val_distr" : [],
            "difficulty_distr" : [],
            "val_epochs" : []
        }
        self.numSamples_dataset = config["task"]["episode_difficulty"]

        # Batch sizes
        self.train_batch_size      = 100
        self.validation_batch_size = 40 * self.teacher.nDifficulties


    def checkExistingTraining(self):
        try:
            os.makedirs(self.path_manager.getPathCheckpoints())
            os.makedirs(self.path_manager.getPathHistory())
        except FileExistsError:
            
            act = input("There is already data for this configuration. Do you want to overwrite it? (Y/N)\n").lower()
            if act == "y":
                pass
            else:
                print("Aborting training...\n")
                exit(0)

    def trainingLoop(self):
        self.checkExistingTraining()
        # torch.autograd.set_detect_anomaly(True)
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Datasets
        train_data, train_actions, val_data, val_actions = self.dataset_builder.BuildDatasets(self.learn_system.action_loss)

        nextEpochVal = 0
        for epoch in range(0, self.epochs):
            # Run and compute loss
            difficulties = self.teacher.getDifficulties(self.train_batch_size)
            inputs_train, target_train, top_difficulty = self.buildInputsTargets(train_data, train_actions, self.train_batch_size, difficulties)
            
            loss_train = self.runEpochLoss(inputs_train, target_train, difficulties, top_difficulty)

            # Update weights
            self.optimizer.zero_grad()            
            loss_train.backward()
            self.optimizer.step()
         
            # Validate
            isValEpoch = epoch == nextEpochVal  or epoch == self.epochs-1 
            if isValEpoch:
                nextEpochVal += VAL_PERIOD
                
                print('Epoch = %d' % (epoch)  ) 
                print('- - - - - - - - -')
                print("|Training|\n  --Loss = ", loss_train.detach().cpu().numpy(), "\n  --Avg. Difficulty = ", difficulties.detach().cpu().numpy().mean())

                student_loss_distr = self.validate(val_data, val_actions, self.validation_batch_size)
                print('===============================================\n')
            
                # Store checkpoint 
                self.history["val_epochs"].append(epoch)
                self.history["loss_train"].append(loss_train)
                self.history["difficulty_distr"].append(self.teacher.getDifficultyDistribution())
                
                torch.save(self.learn_system.state_dict(), self.path_manager.getPathCheckpoints()+"/epoch_"+str(epoch)+'.pth')
                
                # Early stopping
                if self.perform_early_stopping:
                    print("Early stopping unavailable at the moment...")

            self.teacher.updateDifficulties(epoch, student_loss_distr, isValEpoch)
        print("Training Finished!")   

        self.saveHistory()     
        return
    
    def buildInputsTargets(self, trajectories, actions, batch_size, difficulties):
        # print("trajectories", trajectories.shape)

        # Select batch
        chosen_batch  = torch.randperm(trajectories.size(1))[:batch_size]

        batch = trajectories[:,chosen_batch,:]
        if self.learn_system.action_loss:
            batch_actions = actions[:,chosen_batch,:]

        # Select initial states for traj. of numSamples
        top_difficulty = max(difficulties)
        realNS = trajectories.size()[0]
        chosen_initial_state  = torch.tensor([torch.randint(0, max(1, int(realNS-k)), [1]) for k in difficulties])

        chosen_states = chosen_initial_state.unsqueeze(1) + torch.arange(top_difficulty)
        chosen_states = [row[:size] for row, size in zip(chosen_states, difficulties)]

        # Inputs
        inputs = torch.zeros([batch_size, batch.shape[2]]).to(self.device)
        for i in range(batch_size):
            inputs[i, :] = batch[chosen_initial_state[i], i, :]

        # Targets
        numAgents = self.learn_system.task.numAgents
        if self.learn_system.action_loss:
            # Action targets
            targets = torch.zeros([top_difficulty, batch_size, self.learn_system.task.action_dim_per_agent*numAgents]).to(self.device)
            for i in range(batch_size):
                targets[:difficulties[i], i, :] = batch_actions[chosen_states[i], i, :]
        else:
            # Observation targets
            targets = torch.zeros([top_difficulty, batch_size, 4*numAgents]).to(self.device)
            batch_states = self.learn_system.task.getRobotStates(batch)
            for i in range(batch_size):
                targets[:difficulties[i], i, :] = batch_states[chosen_states[i], i, :]

        return inputs, targets, top_difficulty

    def runEpochLoss(self, inputs, target, difficulties, max_difficulty):
        action_loss = self.learn_system.action_loss

        # Run epoch
        obs, actions, _ = self.learn_system.forward(inputs, max_difficulty)
        assert torch.isfinite(obs).all()

        # Mask lower difficulties with 0's at the end
        for i, ns in enumerate(difficulties):
            if action_loss:
                actions[ns:,i,:] = torch.zeros([max_difficulty-ns, actions.shape[2]])    
            else:
                obs[ns:,i,:] = torch.zeros([max_difficulty-ns, obs.shape[2]])    

        # Compute loss
        prediction = self.learn_system.task.getRobotStates(obs) if not action_loss else actions

        L = self.L2_loss_train(prediction, target, difficulties)
        return L

    def L2_loss_train(self, u, v, ns_distr):
        sumErrors = torch.sum((u - v).pow(2)) 
        numComparedValues = (torch.sum(ns_distr) * u.shape[2])
        return sumErrors / numComparedValues
    
    # ============
    # |Validation|
    # ============
    def validate(self, val_data, val_actions, validation_size):
        # Build targets and validate with maxNumSamples (normalized Loss)          
        difficulties = self.teacher.getValidationDifficulties(validation_size)
        inputs_val, target_val, top_difficulty = self.buildInputsTargets(val_data, val_actions, validation_size, difficulties)
        val_loss_distr, avg_loss = self.validation_loss_distr(inputs_val, target_val, difficulties, top_difficulty)

        self.history["loss_val_distr"].append(self.teacher.transformValLoss(val_loss_distr))
        print("|Validation|\n  --Loss = ", avg_loss, "\n  --Avg. Difficulty = ", difficulties.detach().cpu().numpy().mean())
        # print("  --Loss distr. = ", val_loss_distr)
        return val_loss_distr
    
    def validation_loss_distr(self, inputs_val, target_val, difficulties, max_difficulty):
        # Set evaluation mode
        self.learn_system.eval()                 
        
        # Compute trajectories
        with torch.no_grad():
            obs, actions, _ = self.learn_system.forward(inputs_val, max_difficulty)

        # Mask lower difficulties with 0's at the end
        for i, ns in enumerate(difficulties):
            obs[ns:,i,:] = torch.zeros([max_difficulty-ns, obs.shape[2]])    

        # Raw losses
        prediction = self.learn_system.task.getRobotStates(obs) if not self.learn_system.action_loss else actions

        losses = (prediction - target_val).pow(2)
        losses = losses.sum(dim=(0,2)) / target_val.shape[2]

        # Sum of losses for each difficulty
        loss_accumulated = torch.zeros(max_difficulty).to(self.device)
        for loss, difficulty in zip(losses, difficulties):
            loss_accumulated[difficulty-1] += loss
        
        difficulty_count = torch.bincount(difficulties-1).clone()

        # Mean loss for each difficulty
        loss_distr = torch.zeros(max_difficulty).to(self.device)
        for index_difficulty, loss in enumerate(loss_accumulated):
            difficulty = index_difficulty + 1
            loss_distr[index_difficulty] = loss / max(1,difficulty_count[index_difficulty] * difficulty)
        
        # Average losses (not consdering unsampled difficulties)
        avg_loss = loss_distr.sum() / torch.count_nonzero(difficulty_count)

        # Go back to training mode
        self.learn_system.train()  

        del difficulty_count, loss_accumulated, losses, obs
    
        return loss_distr.detach().cpu().numpy(), avg_loss.detach().cpu().numpy()
    
    
    def saveHistory(self):
        path = self.path_manager.getPathHistory()
        
        for key, value in self.history.items():
            torch.save(value, path+'/'+key+'.pth')

        print("History Saved!")
