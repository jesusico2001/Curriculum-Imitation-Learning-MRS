import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer

agent = HistoryVisualizer("Training/configs/FS_LEMURS_Online.yaml")
agent.plotEvoDifficultyDistribution()
agent.plotEvoLossDistribution()
del agent
