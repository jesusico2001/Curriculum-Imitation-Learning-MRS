import argparse
import torch, yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Evaluation.EvalAgent.EvalAgent import EvalAgent
from Evaluation.trajectory_analysis import *

class CheckpointValidator(EvalAgent):    

    def validateLossUniformDifficulty(self, numAgents):
        oldNa = self.learn_system.na 
        self.learn_system.na  = numAgents

        # Load dataset
        valData = self.dataset_builder.BuildValidation(self.numSamples_dataset, numAgents)
        
        # Iterate all checkpoints computing losses
        epochs = torch.load(self.path_manager.getPathHistory()+"/val_epochs.pth")
        for epoch in epochs:
            self.learn_system.load_state_dict(torch.load(self.path_manager.getPathCheckpoints()+"/epoch_"+str(epoch)+".pth", map_location=self.device))
            with torch.no_grad():
                self.validate(valData, self.validation_size)
            self.learn_system.eval() # Ensure evaluation mode
            print("Epoch", epoch, "done.")
            print("====================")

        # Save losses
        torch.save(self.history["loss_val_distr"], self.path_manager.getPathHistory()+'/loss_val_distr_'+str(numAgents)+'.pth')
        
        # Avoid GPU mem. leaks
        for x in self.history["loss_val_distr"]:
            del x
        torch.cuda.empty_cache()
        self.history["loss_val_distr"] = []
        
        self.learn_system.na = oldNa

    # ==========================================

    def validateScalability(self, nAgents_list):
        for na in nAgents_list:
            print("====================")
            print("Num Agents:", na)
            print("====================")
            loss = self.validateLossFixedDifficulty(self.teacher.maxDifficulty, na)
            torch.save(loss, self.path_manager.getPathHistory()+'/loss_val_'+str(self.teacher.maxDifficulty)+"_"+str(na)+'robots.pth')

    def validateLossMaxDifficulty(self, numAgents):
        losses = self.validateLossFixedDifficulty(self.numSamples_dataset, numAgents)
        torch.save(losses, self.path_manager.getPathHistory()+'/loss_val_'+str(self.numSamples_dataset)+"_"+str(numAgents)+'robots.pth')


        
    def validateLossFixedDifficulty(self, difficulty, numAgents):
        oldNa = self.learn_system.na 
        self.learn_system.na  = numAgents

        # Load dataset
        valData = self.dataset_builder.BuildValidation(self.numSamples_dataset, numAgents)
        difficulties = torch.ones(self.validation_size, dtype=int) * difficulty
        inputs_val, target_val, top_difficulty = self.buildInputsTargets(valData, self.validation_size, difficulties)
        
        losses = []
        epochs = torch.load(self.path_manager.getPathHistory()+"/val_epochs.pth")
        for epoch in epochs:
            self.learn_system.load_state_dict(torch.load(self.path_manager.getPathCheckpoints()+"/epoch_"+str(epoch)+".pth", map_location=self.device))
            with torch.no_grad():
                loss_val = self.runEpochLoss(inputs_val, target_val, difficulties, top_difficulty)
            losses.append(loss_val)
            print("Epoch ", epoch, " validated. - L2:",float(loss_val))

        self.learn_system.na = oldNa
        return losses

    # ========================================
    def videoEvolution(self, numAgents):
        oldNa = self.learn_system.na 
        self.learn_system.na  = numAgents
        
        # Load dataset
        valData = self.dataset_builder.BuildValidation(self.numSamples_dataset, numAgents)
        epochs = torch.load(self.path_manager.getPathHistory()+"/val_epochs.pth")
        nEpochs = epochs[-1]
        
        
        initial_states = valData[0,:,:]
        real_trajectories = valData
        time = self.numSamples_dataset * self.step_size
        simulation_time = torch.linspace(0, time, int(time/self.step_size))

        fig = plt.figure(figsize=(14,12))
        myFrames = range(int(nEpochs/EPOCHS_PER_FRAME))
        print(myFrames)
        ani = FuncAnimation(fig, updateFrame, frames=myFrames, interval=60,
                    fargs=(self.learn_system, initial_states[0].to(self.device), real_trajectories[:,0,:].to(self.device), 
                            epochs, self.numSamples_dataset, numAgents,self.device, simulation_time, self.step_size,
                            self.path_manager.getPathCheckpoints()))
        with torch.no_grad():
            ani.save(self.path_manager.getPathEvaluation()+"/video_trajectories.mp4", writer='ffmpeg')
        torch.cuda.empty_cache()
        del ani
        print("Video generated successfully")
        
        self.learn_system.na  = oldNa
