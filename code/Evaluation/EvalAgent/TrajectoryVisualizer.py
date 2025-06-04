import argparse, os
import torch, yaml, imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Evaluation.EvalAgent.EvalAgent import EvalAgent


class TrajectoryVisualizer(EvalAgent):    
    def __init__(self, path_config_train, config_changes=None):
        super().__init__(path_config_train, config_changes)
        self.colors = plt.cm.get_cmap('hsv', self.learn_system.task.numAgents+1)
    

    def plotTrajectoriesEpoch(self, epoch, numExamples, split_dataset):
        dataset = self.dataset_builder.BuildArbitraryNumAgents(split_dataset, self.learn_system.task.numAgents, self.learn_system.return_noisy_obs)
        save_folder = self.path_manager.getPathEvaluation()+"/qualitative/epoch_"+str(epoch)
        os.makedirs(save_folder, exist_ok=True)

        for i in range(numExamples):
            plt.clf()
            real_trajectory = dataset[:,i,:]
            learned_trajectory = self.__plotFrame(epoch, real_trajectory)
            plt.savefig(save_folder+"/"+split_dataset+"_"+str(i)+".png")

            if hasattr(self.learn_system.task, "gif") and callable(getattr(self.learn_system.task, "gif")):
                frames = self.learn_system.task.gif(learned_trajectory)
                imageio.mimsave(save_folder+"/"+split_dataset+"_"+str(i)+".gif", frames, duration=15)


    def __plotFrame(self, epoch, real_trajectory):
        self.load_checkpoint(epoch)
        with torch.no_grad():
            learned_trajectory, _, _ = self.learn_system.forward(real_trajectory[0,:].unsqueeze(0), real_trajectory.shape[0])
            learned_trajectory = learned_trajectory.squeeze(1)
        # Plotting
        plt.rcParams.update({'font.size': 20})
        plt.figtext(0.12, 0.9, "Iteration: " + str(epoch) + " of " + str(self.epochs),fontsize="15")
        self.plotTrajectories(real_trajectory, 'dotted', "Expert trajectory")
        self.plotTrajectoriesClean(learned_trajectory, "Learned trajectory")
        plt.xlabel('x $[$m$]$', fontsize=25)
        plt.ylabel('y $[$m$]$', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        return learned_trajectory
    
    def plotTrajectories(self, trajectory, linestyle, description):
        trajectory = trajectory.detach().cpu().numpy()
        pos, _ = self.learn_system.task.getPosVel(trajectory)
        
        for i in range(self.learn_system.task.numAgents):
                plt.plot(pos[:, i, 0], pos[:, i, 1], color=self.colors(i), linewidth=2, linestyle=linestyle, label=description if i==0 else '')

    def plotTrajectoriesClean(self, trajectory, description):
        trajectory = trajectory.detach().cpu().numpy()
        pos, _ = self.learn_system.task.getPosVel(trajectory)

        alphas = torch.linspace(0.15, 0.9, steps=trajectory.shape[0]).numpy()
        for i in range(self.learn_system.task.numAgents):
            x = pos[:, i, 0]
            y = pos[:, i, 1]
            
            for j in range(len(x)):
                plt.scatter(x[j], y[j], color=self.colors(i), alpha=alphas[j], s=200)

    # Video
    # =====
    EPOCHS_PER_FRAME = 100
    def videoEvolution(self, numExamples, split_dataset):
        dataset = self.dataset_builder.BuildArbitraryNumAgents(split_dataset, self.learn_system.task.numAgents, self.learn_system.return_noisy_obs)
        save_folder = self.path_manager.getPathEvaluation()+"/qualitative/evolution/"
        os.makedirs(save_folder, exist_ok=True)
        
        nFrames = range(int(self.epochs/self.EPOCHS_PER_FRAME)+1)
        for i in range(numExamples):
            real_trajectory = dataset[:,i,:]

            self.first_updateFrame = True
            fig = plt.figure(figsize=(14,12))
            ani = FuncAnimation(fig, self.updateFrame, frames=nFrames, interval=100, fargs=([real_trajectory]))
            ani.save(save_folder+split_dataset+"_"+str(i)+".mp4", writer='ffmpeg')
            torch.cuda.empty_cache()
            del ani
            print("Video" + str(i) + "generated successfully")
        
    def updateFrame(self, frame, real_trajectory):
        if self.first_updateFrame:
            # Avoids setup
            self.first_updateFrame = False
            return

        epoch_save = self.EPOCHS_PER_FRAME * frame
        if epoch_save == self.epochs:
            epoch_save-=1

        print("Frame "+str(frame)+": epoch= "+str(epoch_save))
        plt.clf()
        self.__plotFrame(epoch_save, real_trajectory)
