import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer

numRobots_list = [4]

agent = HistoryVisualizer("Training/configs/FS_LEMURS_Online.yaml")
agent.plotScalabiltyLoss(numRobots_list)
del agent

agent = HistoryVisualizer("Training/configs/TVS_LEMURS_Online.yaml")
agent.plotScalabiltyLoss(numRobots_list)
del agent
