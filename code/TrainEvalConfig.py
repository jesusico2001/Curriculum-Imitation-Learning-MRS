import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")
import gc, torch
from Training.TrainingAgent import TrainingAgent
from Evaluation.EvalAgent.HistoryVisualizer import HistoryVisualizer
from Evaluation.EvalAgent.LossValidator import LossValidator
from Evaluation.EvalAgent.PerformanceMeasurer import PerformanceMeasurer
from Evaluation.EvalAgent.TrajectoryVisualizer import TrajectoryVisualizer
from Evaluation.EvalAgent.MetricComputer import MetricComputer

def clearGPU():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def TrainEval(path_config, config_changes):
    trainAgent = TrainingAgent(path_config, config_changes )
    trainAgent.trainingLoop()
    del trainAgent
    clearGPU()

    agent = LossValidator(path_config, config_changes)
    agent.validateLossTeacherDifficulty()
    agent.validateLossEpisodeDifficulty()
    del agent
    clearGPU()
    
    agent = MetricComputer(path_config, config_changes)
    agent.metricsEvolution()
    clearGPU()

    agent = HistoryVisualizer(path_config, config_changes)
    agent.plotLossMaxDifficulty()
    agent.plotTrainValidationLosses()
    agent.plotEvoDifficultyDistribution()
    agent.plotEvoLossDistribution()
    del agent
    clearGPU()

    agent = TrajectoryVisualizer(path_config, config_changes)
    agent.plotTrajectoriesEpoch(agent.epochs-1,3,"test")
    agent.videoEvolution(1,"test")
    del agent
    clearGPU()
