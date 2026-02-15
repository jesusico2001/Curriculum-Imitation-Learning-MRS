import argparse
import torch, yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Evaluation.EvalAgent.EvalAgent import EvalAgent
from Evaluation.trajectory_analysis import *

class LossValidator(EvalAgent):    

    def validateLossUniformDifficulty(self, numAgents):
        if numAgents is None:
            numAgents = self.learn_system.task.numAgents
            
        oldNa = self.learn_system.task.numAgents 
        self.learn_system.task.numAgents  = numAgents

        # Load dataset
        valData = self.dataset_builder.BuildArbitraryNumAgents("val", numAgents, self.learn_system.return_noisy_obs)
        
        # Iterate all checkpoints computing losses
        epochs = torch.load(self.path_manager.getPathHistory()+"/val_epochs.pth")
        for epoch in epochs:
            self.load_checkpoint(epoch)
            with torch.no_grad():
                self.validate(valData, self.validation_batch_size)
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
        
        self.learn_system.task.numAgents = oldNa

    # ==========================================

    def validateLossTeacherDifficulty(self, numAgents=None):
        na = self.learn_system.task.numAgents  if numAgents is None else numAgents
        loss = self.validateLossFixedDifficulty(self.teacher.maxDifficulty, na)
        torch.save(loss, self.path_manager.getPathHistory()+'/loss_val_'+str(self.teacher.maxDifficulty)+"_"+str(na)+'robots.pth')

    def validateLossEpisodeDifficulty(self, numAgents=None):
        na = self.learn_system.task.numAgents  if numAgents is None else numAgents
        loss = self.validateLossFixedDifficulty(self.numSamples_dataset, na)
        torch.save(loss, self.path_manager.getPathHistory()+'/loss_val_'+str(self.numSamples_dataset)+"_"+str(na)+'robots.pth')


    def validateLossFixedDifficulty(self, difficulty, numAgents):
        oldNa = self.learn_system.task.numAgents 
        self.learn_system.task.numAgents  = numAgents

        # Load dataset
        valData = self.dataset_builder.BuildArbitraryNumAgents("val", numAgents, self.learn_system.return_noisy_obs)
        difficulties = torch.ones(self.validation_batch_size, dtype=int) * difficulty
        inputs_val, target_val, top_difficulty = self.buildInputsTargets(valData, None, self.validation_batch_size, difficulties)
        
        losses = []
        epochs = torch.load(self.path_manager.getPathHistory()+"/val_epochs.pth")
        for epoch in epochs:
            self.load_checkpoint(epoch)
            with torch.no_grad():
                loss_val = self.runEpochLoss(inputs_val, target_val, difficulties, top_difficulty)
            losses.append(loss_val)
            print("Epoch ", epoch, " validated. - L2:",float(loss_val))

        self.learn_system.task.numAgents = oldNa
        return losses
