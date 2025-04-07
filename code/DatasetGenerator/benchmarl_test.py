from pathlib import Path

from benchmarl.hydra_config import reload_experiment_from_file
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig, Mlp
from dataclasses import asdict
from torchrl.data import CompositeSpec, TensorSpec
import torch
from collections import OrderedDict

restore_file = "/mnt/hdd/JesusRoche/Curriculum-Imitation-Learning-MRS/code/outputs/2024-12-19/18-04-23/mappo_balance_mlp__3c6802de_24_12_19-18_04_23/checkpoints/checkpoint_3000000.pt"

# def walk_dict(d, depth):
#     for k,v in d.items():
#         tabs = ""
#         for i in range(depth):
#             tabs += "  "
#         print (tabs + str(k))
        
#         if isinstance(v, dict):
#             walk_dict(v, depth+1)

# checkpoint = OrderedDict(torch.load(restore_file))
# walk_dict(checkpoint, 0)
# print(checkpoint["loss_agents"]["actor_network_params.module.0.module.0.mlp.params.__batch_size"].shape)
# exit()

model_config = MlpConfig.get_from_yaml()
experiment = reload_experiment_from_file(str(restore_file))
alg = experiment.algorithm

agent = alg._get_policy_for_loss("agents", model_config, True)
print(agent)
# print("===============")
# print(dir(experiment))
# print(experiment.model_config)
# print("===============")

# print(model_config)
# print("===============")

# input_spec = CompositeSpec(
#     position=TensorSpec(torch.Size([3]), dtype=torch.float32),
#     velocity=TensorSpec(torch.Size([2]), dtype=torch.float32),
# )

# actor_output_spec = Composite(
#     {
#         group: Composite(
#             {"logits": Unbounded(shape=logits_shape)},
#             shape=(n_agents,),
#         )
#     }
# )
# actor_module = model_config.get_model(
#     input_spec=actor_input_spec,
#     output_spec=actor_output_spec,
#     agent_group=group,
#     input_has_agent_dim=True,
#     n_agents=n_agents,
#     centralised=False,
#     share_params=self.experiment_config.share_policy_params,
#     device=self.device,
#     action_spec=self.action_spec,
# )


# print(model)
# exit()
