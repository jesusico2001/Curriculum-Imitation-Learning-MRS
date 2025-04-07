import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Training.TrainingAgent import TrainingAgent, VAL_PERIOD
from Evaluation.EvalAgent.EvalAgent import EvalAgent


class HistoryVisualizer(EvalAgent):    

    def plotScalabiltyLoss(self, nRobots_list):    
        colors = plt.cm.get_cmap('hsv', len(nRobots_list)+1)
        plt.figure(figsize=(10,6))
        epochs = self.__loadFromHistory__("val_epochs")
        i = 0
        for numRobots in nRobots_list:
            loss = self.__loadFromHistory__("loss_val_"+str(self.teacher.maxDifficulty)+"_"+str(numRobots)+"robots")
            loss_formatted = [t.cpu().tolist() for t in loss]

            plt.plot(epochs, loss_formatted, linestyle='solid', alpha=1, color=colors(i+1), label=str(numRobots)+" robots")

            i += 1
            
        plt.title('Scalability evo', fontsize=25)
        plt.yscale('log')
        plt.xlabel('Iteraciones', fontsize=22)
        plt.ylabel('Error', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True)
        plt.show()
        plt.close()

    def plotLossMaxDifficulty(self):
        plt.figure(figsize=(10,6))
        
        epochs = self.__loadFromHistory__("val_epochs")

        loss_val = self.__loadFromHistory__("loss_val_"+str((self.learn_system.task.episode_difficulty))+"_"+str(int( self.learn_system.task.numAgents))+"robots")
        loss_val_formatted = [t.cpu().tolist() for t in loss_val]

        plt.plot(epochs, loss_val_formatted, linestyle='solid', alpha=1, color="green", label="Validation")

        plt.title('Episode Loss Evolution | Difficulty:'+str(self.learn_system.task.episode_difficulty), fontsize=25)
        plt.yscale('log')
        plt.xlabel('Iterations', fontsize=22)
        plt.ylabel('Error', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True)
        # plt.show()
        plt.savefig(self.path_manager.getPathEvaluation()+"/val_loss_max_difficulty.png")
        plt.close()

    def plotTrainValidationLosses(self):
        plt.figure(figsize=(10,6))
        
        epochs = self.__loadFromHistory__("val_epochs")

        loss_train = self.__loadFromHistory__("loss_train")
        loss_train_formatted = [t.cpu().tolist() for t in loss_train]

        loss_val = self.__loadFromHistory__("loss_val_"+str((self.teacher.maxDifficulty))+"_"+str(int( self.learn_system.task.numAgents))+"robots")
        loss_val_formatted = [t.cpu().tolist() for t in loss_val]


        plt.plot(epochs, loss_train_formatted, linestyle='solid', alpha=1, color="blue", label="Train")
        plt.plot(epochs, loss_val_formatted, linestyle='solid', alpha=1, color="green", label="Validation")

        plt.title('Loss Evolution | Difficulty:'+str(self.teacher.maxDifficulty), fontsize=25)
        plt.yscale('log')
        plt.xlabel('Iterations', fontsize=22)
        plt.ylabel('Error', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True)
        # plt.show()
        plt.savefig(self.path_manager.getPathEvaluation()+"/train_val_losses.png")
        plt.close()

    # ==========================================

    def plotEvoDifficultyDistribution(self):
        difficulties = self.__loadFromHistory__("difficulty_distr")
        difficulties_formatted = torch.tensor([t.cpu().tolist() for t in difficulties])

        self.plotDistributionEvo_heatmap(difficulties_formatted, "Task difficulty evolution")
        plt.savefig(self.path_manager.getPathEvaluation()+"/teacher_difficulties_heatmap.png")
        plt.close()

        self.plotDistributionEvo_lines(difficulties_formatted, "Sampling probability", False)
        plt.savefig(self.path_manager.getPathEvaluation()+"/teacher_difficulties_lines.png")
        plt.close()

    def plotEvoLossDistribution(self):
        loss = self.__loadFromHistory__("loss_val_distr")
        loss_formatted = torch.tensor(np.array(loss))

        self.plotDistributionEvo_heatmap(loss_formatted, "Loss distribution evolution")
        plt.savefig(self.path_manager.getPathEvaluation()+"/teacher_loss_evo_heatmap.png")
        plt.close()
        
        self.plotDistributionEvo_lines(loss_formatted, "Loss", True)
        plt.savefig(self.path_manager.getPathEvaluation()+"/teacher_loss_evo_lines.png")
        plt.close()

    def plotDistributionEvo_heatmap(self, distribution, label):
        matrix = distribution.numpy().transpose()

        plt.figure(figsize=(8, 6))
        
        vmin = matrix[matrix > 0].min()
        plt.imshow(matrix, cmap='viridis', aspect='auto', origin='lower', norm=LogNorm(vmin=vmin, vmax=matrix.max()))

        xtick_interval = int((matrix.shape[1]-1)*VAL_PERIOD/4)
        validation_iterations = np.arange(0, (matrix.shape[1]+1)*VAL_PERIOD, xtick_interval)
        plt.xticks(ticks=np.arange(matrix.shape[1])[::int(xtick_interval/VAL_PERIOD)], labels=validation_iterations)
        
        diff_labels = np.arange(self.teacher.difficultyResolution, (matrix.shape[0]+1)*self.teacher.difficultyResolution, self.teacher.difficultyResolution)
        plt.yticks(ticks=np.arange(matrix.shape[0]), labels=diff_labels)

        plt.colorbar()

        plt.title(label, fontsize=25)
        plt.xlabel("Time (Iterations)", fontsize=22)
        plt.ylabel("Difficulty", fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.show()

    def plotDistributionEvo_lines(self, distribution, label, logscale):
        colors = plt.cm.get_cmap('hsv', distribution.shape[1]+1)
        plt.figure(figsize=(10,6))
        
        epochs = self.__loadFromHistory__("val_epochs")

        
        for i in range(distribution.shape[1]):
            plt.plot(epochs, distribution[:,i], linestyle='solid', alpha=1, color=colors(i), label="Difficulty "+str((i+1)*self.teacher.difficultyResolution))

        if logscale:
            plt.yscale('log')

        plt.title(label+' Evolution:', fontsize=25)
        plt.xlabel('Iterations', fontsize=22)
        plt.ylabel(label, fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20, loc='upper right')
        plt.grid(True)
        # plt.show()


    # ==========================================
    
    def __loadFromHistory__(self, filename):
        return torch.load(self.path_manager.getPathHistory()+"/"+filename+".pth")
