import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")

from Training.TrainingAgent import TrainingAgent
from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer
from Evaluation.EvalAgent.CheckpointValidator import CheckpointValidator
from Evaluation.EvalAgent.PerformanceMeasurer import PerformanceMeasurer


def TrainEval(path_config, config_changes):
    # trainAgent = TrainingAgent(path_config, config_changes )
    # trainAgent.trainingLoop()
    # del trainAgent

    # agent = CheckpointValidator(path_config, config_changes)
    # agent.validateLossMaxDifficulty(4)
    # agent.validateScalability([4])
    # del agent

    # agent = PerformanceMeasurer(path_config, config_changes)
    # agent.trainingPerformance(4)
    # del agent

    # agent = CheckpointValidator(path_config, config_changes)
    # agent.videoEvolution(4)
    # del agent

    agent = HistoryVisualizer(path_config, config_changes)
    # agent.plotTrainValidationLosses()
    agent.plotEvoDifficultyDistribution()
    agent.plotEvoLossDistribution()
    del agent

path = "Training/configs/FS_LEMURS_Online.yaml"
for reward in ["l2loss", "l2gain", "l2totalgain"]:
    for learning_rate in [0.001, 0.01, 0.1]:
        changes = {"teacher.learning_rate": learning_rate,"teacher.reward": reward}
        TrainEval(path, changes)