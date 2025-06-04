import os
import torch, yaml
import matplotlib.pyplot as plt
from Evaluation.EvalAgent.EvalAgent import EvalAgent
import similaritymeasures
 
class PerformanceMeasurer(EvalAgent):    
    
    def trainingPerformance(self, numAgents):
        self.checkExistingEvaluation()

        oldNa = self.learn_system.task.numAgents 
        self.learn_system.task.numAgents  = numAgents

        # Load test data
        real_rewards = self.dataset_builder.BuildArbitraryNumAgents("test", numAgents, data="rewards")
        test_data = self.dataset_builder.BuildArbitraryNumAgents("test", numAgents)
        initial_states = test_data[0,:,:]
        real_trajectories = test_data
        print("Test trajectories loaded successfully")

        # Load best checkpoint
        epoch = self.epochs-1
        self.load_checkpoint(epoch)

        with torch.no_grad():
            learned_trajectories, _, rewards = self.learn_system.forward(initial_states, self.learn_system.task.episode_difficulty)

        # Quantitative analysis
        learn_avg_dist = 0
        learn_min_dist = 0
        real_avg_dist  = 0
        real_min_dist  = 0
        learn_smoothness = 0
        real_smoothness = 0
        area_trajectories = 0
        real_task_completed = 0
        learn_task_completed = 0

        numTests = 100
        for i in range(numTests):
            
            learn_avg_dist += self.avgAgentDist(learned_trajectories[:,i,:])
            learn_min_dist += self.minAgentDist(learned_trajectories[:,i,:])
            learn_smoothness += self.getSmoothness(learned_trajectories[:,i,:])

            real_avg_dist  += self.avgAgentDist(real_trajectories[:,i,:])
            real_min_dist  += self.minAgentDist(real_trajectories[:,i,:])
            real_smoothness += self.getSmoothness(real_trajectories[:,i,:])

            area_trajectories += self.getAreaBetweenCurves(learned_trajectories[:,i,:].detach().cpu(), real_trajectories[:,i,:].detach().cpu())

            learn_task_completed += self.learn_system.task.numCompletedTasks(learned_trajectories[:,i,:]).detach().cpu()
            real_task_completed += self.learn_system.task.numCompletedTasks(real_trajectories[:,i,:]).detach().cpu()
            print(".", end="", flush=True)
        learn_avg_dist /= numTests
        learn_min_dist /= numTests
        real_avg_dist /= numTests
        real_min_dist /= numTests
        learn_smoothness /= numTests
        real_smoothness /= numTests
        area_trajectories /= numTests
        loss_test = self.L2_loss(learned_trajectories.to(self.device), real_trajectories.to(self.device))
        reward_error = torch.mean(real_rewards - rewards)
        learn_task_completed = float(learn_task_completed) / numTests
        real_task_completed = float(real_task_completed) / numTests
        task_completed_error = real_task_completed - learn_task_completed

        print("\nL2 loss     : ", float(loss_test))
        print("Average Dist: Learned = ", float(learn_avg_dist), " - Real = ", float(real_avg_dist))
        print("Minimal Dist: Learned = ", float(learn_min_dist), " - Real = ", float((real_min_dist)))
        print("Smoothness  : Learned = ", float(learn_smoothness), " - Real = ", float((real_smoothness)))
        print("Area        : ", float(area_trajectories))
        print("Reward Error        : ", float(reward_error))
        print("Avg complete tasks : Learned = ", float(learn_task_completed), " - Real = ", float((real_task_completed)))

        with open(self.path_manager.getPathEvaluation()+"/info.txt", 'w') as file:
            text = "L2 loss: "+str(float(loss_test)) + "\nAvg dist error: "\
                +str(float(learn_avg_dist)-float(real_avg_dist)) + "\nMin dist error: " \
                +str(float(learn_min_dist)-float(real_min_dist)) + "\nSmoothness error: "\
                +str(float(learn_smoothness)-float(real_smoothness)) + "\nArea between curves: "\
                +str(float(area_trajectories))+ "\n Reward error: " + str(float(reward_error))\
                +"\nTask completed error: "+str(float(task_completed_error))

            file.write(text)

        self.learn_system.na  = oldNa

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

    
    # Smoothness
    def getSmoothness(self, trajectory):
        ax, ay = self.getAccelerations(trajectory)
    
        return  (torch.sum(ax**2) + torch.sum(ay**2) ) / self.learn_system.task.numAgents

    def getAccelerations(self, trajectory):
        # Extract speeds
        na=self.learn_system.task.numAgents
        step_size = self.learn_system.task.simulation_step
        states = self.learn_system.task.getRobotStates(trajectory.unsqueeze(0)).squeeze(0)
        
        vx = states[:, 2*na::2]
        vy = states[:, 2*na+1::2]
        
        # Aceleracionrd crentrales con diferencias finas
        ax_central = (vx[2:, :] - vx[:-2, :]) / (2 * step_size)
        ay_central = (vy[2:, :] - vy[:-2, :]) / (2 * step_size)
        
        # Aceleraciones en los extremos con diferencias hacia adelante/atrás
        ax_start = (vx[1] - vx[0]) / step_size
        ax_end = (vx[-1] - vx[-2]) / step_size
        ay_start = (vy[1] - vy[0]) / step_size
        ay_end = (vy[-1] - vy[-2]) / step_size
        
        # Combinar todos los valores de aceleración
        ax = torch.cat((ax_start.unsqueeze(0), ax_central, ax_end.unsqueeze(0)))
        ay = torch.cat((ay_start.unsqueeze(0), ay_central, ay_end.unsqueeze(0)))
        return ax, ay
    

    # Distances between robots       
    def getDistances(self, trajectory):
        na=self.learn_system.task.numAgents
        states = self.learn_system.task.getRobotStates(trajectory.unsqueeze(0)).squeeze(0)
        
        positions = states[:, :2*na].reshape(-1, na, 2)

        x = positions[:,:, 0]
        x1 = torch.kron(x, torch.ones((1,na), device=trajectory.device ))
        x2 = x.repeat(1,na)

        y = positions[:,:, 1]
        y1 = torch.kron(y, torch.ones((1,na), device=trajectory.device ))
        y2 = y.repeat(1,na)

        x_diff = abs(x1-x2).reshape(-1,na,na)
        y_diff = abs(y1-y2).reshape(-1,na,na)

        return torch.sqrt(pow(x_diff, 2) + pow(y_diff, 2))

    def avgAgentDist(self, trajectory):
        na=self.learn_system.task.numAgents

        dist = self.getDistances(trajectory)

        avgDist_instant = (torch.sum(dist,(1,2)) / (pow(na,2)-na))
        avgDist_general = torch.sum(avgDist_instant,0) / avgDist_instant.size()[0]

        return avgDist_general

    def minAgentDist(self, trajectory):
        na=self.learn_system.task.numAgents
        
        dist = self.getDistances(trajectory)
        dist[:, range(na), range(na)] = float('inf')

        minDist_instant, _ = torch.min(dist,2)
        minDist_instant, _ = torch.min(minDist_instant,1)

        minDist_general, _ = torch.min(minDist_instant,0)
        
        return minDist_general

    def getAreaBetweenCurves(self, trajectory1, trajectory2):
        na=self.learn_system.task.numAgents

        states1 = self.learn_system.task.getRobotStates(trajectory1.unsqueeze(0)).squeeze(0)
        states2 = self.learn_system.task.getRobotStates(trajectory2.unsqueeze(0)).squeeze(0)
        
        positions1 =  states1[:, :2*na].reshape(-1, na, 2)
        positions2 =  states2[:, :2*na].reshape(-1, na, 2)

        total = 0.0
        for i in range(na):
            total += similaritymeasures.area_between_two_curves(positions1[:,i,:], positions2[:,i,:])
        return total / na

    def L2_loss(self, trajectory1, trajectory2):
        states1 = self.learn_system.task.getRobotStates(trajectory1.unsqueeze(0)).squeeze(0)
        states2 = self.learn_system.task.getRobotStates(trajectory2.unsqueeze(0)).squeeze(0)

        return (states1 - states2).pow(2).mean()
