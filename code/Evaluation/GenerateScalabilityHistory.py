import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from Evaluation.EvalAgent.CheckpointValidator import CheckpointValidator

numRobots_list = [4]

agent = CheckpointValidator("Training/configs/FS_LEMURS_Online.yaml")
agent.validateScalability(numRobots_list)
del agent

agent = CheckpointValidator("Training/configs/TVS_LEMURS_Online.yaml")
agent.validateScalability(numRobots_list)
del agent

agent = CheckpointValidator("Training/configs/Flocking_LEMURS_Online.yaml")
agent.validateScalability(numRobots_list)
del agent