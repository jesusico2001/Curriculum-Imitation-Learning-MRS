import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from Evaluation.EvalAgent.LossValidator import LossValidator

numRobots_list = [4]

agent = LossValidator("Training/configs/FS_LEMURS_Online.yaml")
agent.validateScalability(numRobots_list)
del agent

agent = LossValidator("Training/configs/TVS_LEMURS_Online.yaml")
agent.validateScalability(numRobots_list)
del agent

agent = LossValidator("Training/configs/Flocking_LEMURS_Online.yaml")
agent.validateScalability(numRobots_list)
del agent