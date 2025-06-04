import sys, os, time
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
import torch, imageio
import gymnasium, vmas
from LearnSystem import LearnSystemBuilder
from Training.TrainingAgent import TrainingAgent
import yaml

def gif(trajectories, env_index, task):
    frames = []
    for i in range(trajectories.shape[0]):
        obs = trajectories[i,env_index,:].unsqueeze(0)
        # print(obs)
        env = task.setupEnvs(obs)
        frame = env.render(mode="rgb_array")
        frames.append(frame)
    return frames

teachers_path = "Training/configs/teachers/"
# for filename in ["noCL_easy.yaml", "noCL_hard.yaml", "baby_steps.yaml", "online.yaml"]:
for filename in ["noCL_easy.yaml",]:
    with open(teachers_path+filename, "r") as file:
        config_changes = yaml.safe_load(file)
    
    # config = "Training/configs/VMAS_balance.yaml"
    config = "/mnt/hdd/JesusRoche/Curriculum-Imitation-Learning-MRS/code/Evaluation/Tests/2_nAgents_battery_noisy_navigation/configs/VMAS_navigation.yaml"
    # t_agent = TrainingAgent(config, config_changes)
    t_agent = TrainingAgent(config, {"task.num_agents": 10})
    model = t_agent.learn_system
    model.eval()


    trajectories  = t_agent.dataset_builder.BuildArbitraryNumAgents("test", model.task.numAgents, False)
    path_save = t_agent.path_manager.getPathDatasets()+"qualitative/"
    os.makedirs(path_save, exist_ok=True)
    for i in range(5):
        print("Sample:  ", i)
        frames = gif(trajectories, i, model.task)
        imageio.mimsave(path_save+"sample_"+str(i)+".gif", frames, duration=4)

    # model.load_state_dict(torch.load(t_agent.path_manager.getPathCheckpoints()+"/epoch_"+str(epoch_checkpoint)+".pth", map_location=model.device))
    # inputs = trajectories[0,:,:]
    # with torch.no_grad():
    #     trajectories_learned = model.forward(inputs[:5,:], trajectories.shape[0])

    # for i in range(5):
    #     frames = gif(trajectories_learned, i, model.task)
    #     imageio.mimsave(folder_video+"ep"+str(epoch_checkpoint)+"_sample_"+str(i)+".gif", frames, duration=15)
    #     print("Saved:", i)