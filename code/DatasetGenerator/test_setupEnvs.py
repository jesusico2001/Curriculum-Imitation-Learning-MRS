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


def load_and_compare_batches(dataset_builder, num_agents):
    # Cargar datos
    train = dataset_builder.BuildArbitraryNumAgents("train", num_agents)
    val = dataset_builder.BuildArbitraryNumAgents("val", num_agents)
    test = dataset_builder.BuildArbitraryNumAgents("test", num_agents)

    # Seleccionar las primeras 5 muestras del batch (dimensión 1)
    train_first5 = train[:, :1000, :]
    val_first5 = val[:, :1000, :]
    test_first5 = test[:, :1000, :]

    # Comprobar si son iguales (en todos los pasos y features)
    train_val_equal = torch.allclose(train_first5, val_first5)
    train_test_equal = torch.allclose(train_first5, test_first5)
    val_test_equal = torch.allclose(val_first5, test_first5)

    print("¿Train y Val (primeros 5) son iguales?:", train_val_equal)
    print("¿Train y Test (primeros 5) son iguales?:", train_test_equal)
    print("¿Val y Test (primeros 5) son iguales?:", val_test_equal)

    if train_val_equal or train_test_equal or val_test_equal:
        print("¡Alerta! Hay coincidencias en las primeras 5 muestras del batch entre splits.")
    else:
        print("OK: Las primeras 5 muestras del batch son diferentes entre todos los splits.")

teachers_path = "Training/configs/teachers/"
# for filename in ["noCL_easy.yaml", "noCL_hard.yaml", "baby_steps.yaml", "online.yaml"]:
for filename in ["noCL_easy.yaml",]:
    with open(teachers_path+filename, "r") as file:
        config_changes = yaml.safe_load(file)
    
    # config = "Training/configs/VMAS_balance.yaml"
    config = "/mnt/hdd/JesusRoche/Curriculum-Imitation-Learning-MRS/code/Evaluation/Tests/6_selected_configs_passage/configs/VMAS_passage.yaml"
    t_agent = TrainingAgent(config, config_changes)
    model = t_agent.learn_system
    model.eval()

    load_and_compare_batches(t_agent.dataset_builder, model.task.numAgents)

    # trajectories  = t_agent.dataset_builder.BuildArbitraryNumAgents("test", model.task.numAgents, False)
    # path_save = t_agent.path_manager.getPathDatasets()+"qualitative/"
    # os.makedirs(path_save, exist_ok=True)
    # for i in range(5):
    #     print("Sample:  ", i)
    #     frames = gif(trajectories, i, model.task)
    #     imageio.mimsave(path_save+"sample_"+str(i)+".gif", frames, duration=4)
