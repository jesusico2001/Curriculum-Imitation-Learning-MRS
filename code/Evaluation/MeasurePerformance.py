import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from Evaluation.EvalAgent.PerformanceMeasurer import PerformanceMeasurer
from Evaluation.EvalAgent.CheckpointValidator import CheckpointValidator

agent = CheckpointValidator("Training/configs/TVS_LEMURS_Online.yaml")
agent.validateLossMaxDifficulty()
del agent

agent = PerformanceMeasurer("Training/configs/TVS_LEMURS_Online.yaml")
agent.trainingPerformance(4)
del agent

