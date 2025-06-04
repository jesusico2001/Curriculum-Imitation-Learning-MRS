import sys, os
import yaml
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer

# trainings = [["Training/configs/VMAS_navigation.yaml", "Training/configs/teachers/noCL_easy.yaml"], 
#              ["Training/configs/VMAS_navigation.yaml", "Training/configs/teachers/noCL_hard.yaml"], 
#              ["Training/configs/VMAS_navigation.yaml", "Training/configs/teachers/baby_steps_basic.yaml"]]

# trainings = [["Training/configs/VMAS_balance.yaml", "Training/configs/teachers/noCL_easy.yaml"], 
#              ["Training/configs/VMAS_balance.yaml", "Training/configs/teachers/baby_steps.yaml"],
#              ["Training/configs/VMAS_balance.yaml", "Training/configs/teachers/baby_steps_basic.yaml"]]
             
trainings = [["Training/configs/VMAS_balance.yaml", "Training/configs/teachers/noCL_easy_40.yaml"], 
             ["Training/configs/VMAS_balance.yaml", "Training/configs/teachers/noCL_hard_40.yaml"],
             ["Training/configs/VMAS_balance.yaml", "Training/configs/teachers/baby_steps_basic_40.yaml"]]

# labels = ["noCL_easy", "noCL_hard", "CL"]
# labels = ["noCL_easy", "linear","fixed"]
labels = ["noCL_easy", "noCL_hard", "babySteps"]
# labels = ["noCL_easy", "babySteps"]
colors = plt.cm.get_cmap('hsv', len(trainings)+1)

plt.figure(figsize=(10,6))

difficulty = -1
for i, t in enumerate(trainings):
    with open(t[1], "r") as file:
        config_changes = yaml.safe_load(file)
    agent = HistoryVisualizer(t[0], config_changes)
    if difficulty == -1:
        difficulty = agent.learn_system.task.episode_difficulty

    epochs = agent.__loadFromHistory__("val_epochs")
    loss_val = agent.__loadFromHistory__("loss_val_"+str((difficulty))+"_"+str(agent.learn_system.task.numAgents)+"robots")
    loss_val_formatted = [t.cpu().tolist() for t in loss_val]
    plt.plot(epochs, loss_val_formatted, linestyle='solid', alpha=1, color=colors(i), label=labels[i])

plt.title('Loss Evolution | Difficulty:'+str(difficulty), fontsize=25)
plt.yscale('log')
plt.xlabel('Iterations', fontsize=22)
plt.ylabel('Error', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.show()

# plt.savefig("saves/trainComparisons/test/train_val_losses.png")
plt.close()
