import argparse
import os
import torch, yaml
import matplotlib.pyplot as plt
from Evaluation.EvalAgent.EvalAgent import EvalAgent
from Evaluation.trajectory_analysis import *
 
class PerformanceMeasurer(EvalAgent):    
    
    def trainingPerformance(self, numAgents):
        self.checkExistingEvaluation()

        oldNa = self.learn_system.na 
        self.learn_system.na  = numAgents

        # Load test data
        test_data = self.dataset_builder.BuildTest(self.numSamples_dataset, numAgents)
        initial_states = test_data[0,:,:]
        real_trajectories = test_data
        print("Test trajectories loaded successfully")

        # Load best checkpoint
        bestEpoch = self.getBestEpoch(numAgents)
        self.learn_system.load_state_dict(torch.load(self.path_manager.getPathCheckpoints()+"/epoch_"+str(bestEpoch)+".pth", map_location=self.device))
        
        time = self.numSamples_dataset * self.step_size
        simulation_time = torch.linspace(0, time, int(time/self.step_size))

        with torch.no_grad():
            learned_trajectories = self.learn_system.forward(initial_states.to(self.device), simulation_time, self.step_size)

        # Plot qualitative
        plt.clf()
        plt.grid(False)
        plotFrame(learned_trajectories[:,0,:].to(self.device), real_trajectories[:,0,:].to(self.device), self.epochs, bestEpoch, self.numSamples_dataset, numAgents)
        plt.savefig(self.path_manager.getPathEvaluation()+"/best_trajectories.png")

        # Quantitative analysis
        learn_avg_dist = 0
        learn_min_dist = 0
        real_avg_dist  = 0
        real_min_dist  = 0
        learn_smoothness = 0
        real_smoothness = 0
        area_trajectories = 0
        numTests = 100
        for i in range(numTests):
            
            learn_avg_dist += avgAgentDist(learned_trajectories[:,i,:], numAgents)
            learn_min_dist += minAgentDist(learned_trajectories[:,i,:], numAgents)
            learn_smoothness += getSmoothness(learned_trajectories[:,i,:], numAgents, self.step_size)

            real_avg_dist  += avgAgentDist(real_trajectories[:,i,:], numAgents)
            real_min_dist  += minAgentDist(real_trajectories[:,i,:], numAgents)
            real_smoothness += getSmoothness(real_trajectories[:,i,:], numAgents, self.step_size)
            area_trajectories += getAreaBetweenCurves(learned_trajectories[:,i,:].detach().cpu().numpy(), real_trajectories[:,i,:].detach().cpu().numpy(), numAgents)
        learn_avg_dist /= numTests
        learn_min_dist /= numTests
        real_avg_dist /= numTests
        real_min_dist /= numTests
        learn_smoothness /= numTests
        real_smoothness /= numTests
        area_trajectories /= numTests
        loss_test = L2_loss((learned_trajectories[:, :, :4 * numAgents].reshape(-1, 4 * numAgents)).to(self.device), 
                            real_trajectories[:, :, :4 * numAgents].reshape(-1, 4 * numAgents).to(self.device))
        
        print("L2 loss     : ", float(loss_test))
        print("Average Dist: Learned = ", float(learn_avg_dist), " - Real = ", float(real_avg_dist))
        print("Minimal Dist: Learned = ", float(learn_min_dist), " - Real = ", float((real_min_dist)))
        print("Smoothness  : Learned = ", float(learn_smoothness), " - Real = ", float((real_smoothness)))
        print("Area        : ", float(area_trajectories))

        with open(self.path_manager.getPathEvaluation()+"/info.txt", 'w') as file:
            text = "Best epoch: "+str(bestEpoch)+"\n" + "L2 loss: "+str(float(loss_test)) + "\nAvg dist error: "\
                +str(float(learn_avg_dist)-float(real_avg_dist)) + "\nMin dist error: " \
                +str(float(learn_min_dist)-float(real_min_dist)) + "\nSmoothness error: "\
                +str(float(learn_smoothness)-float(real_smoothness)) + "\nArea between curves: "\
                +str(float(area_trajectories))

            file.write(text)

        self.learn_system.na  = oldNa

    def getBestEpoch(self, numAgents):
        epochs = torch.load(self.path_manager.getPathHistory()+"/val_epochs.pth")
        losses = torch.load(self.path_manager.getPathHistory()+'/loss_val_'+str(self.numSamples_dataset)+"_"+str(numAgents)+'robots.pth')
        
        min_loss, best_epoch = min(zip(losses, epochs))
        return best_epoch

    def checkExistingEvaluation(self):
        try:
            os.makedirs(self.path_manager.getPathEvaluation())
        except FileExistsError:
            
            act = input("There is already an evaluation for this configuration. Do you want to overwrite it? (Y/N)\n").lower()
            if act == "y":
                pass
            else:
                print("Aborting evaluation...\n")
                exit(0)