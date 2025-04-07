import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")

from Training.TrainingAgent import TrainingAgent
from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer
from Evaluation.EvalAgent.CheckpointValidator import CheckpointValidator
from Evaluation.EvalAgent.PerformanceMeasurer import PerformanceMeasurer
from Evaluation.EvalAgent.TrajectoryVisualizer import TrajectoryVisualizer


def TrainEval(path_config, config_changes):
    trainAgent = TrainingAgent(path_config, config_changes )
    trainAgent.trainingLoop()
    del trainAgent

    agent = CheckpointValidator(path_config, config_changes)
    # agent.aux_run()
    agent.validateLossMaxDifficulty(agent.learn_system.task.numAgents)
    agent.validateScalability([agent.learn_system.task.numAgents])
    del agent

    agent = PerformanceMeasurer(path_config, config_changes)
    # agent.trainingPerformance(4)
    agent.checkExistingEvaluation()
    del agent
    
    agent = HistoryVisualizer(path_config, config_changes)
    agent.plotLossMaxDifficulty()
    agent.plotTrainValidationLosses()
    agent.plotEvoDifficultyDistribution()
    agent.plotEvoLossDistribution()
    del agent

    agent = TrajectoryVisualizer(path_config, config_changes)
    agent.plotTrajectoriesEpoch(agent.epochs-1,3,"test")
    agent.videoEvolution(1,"test")
    del agent
