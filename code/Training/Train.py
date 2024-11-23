import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from Training.TrainingAgent import TrainingAgent


trainAgent = TrainingAgent("Training/configs/FS_LEMURS_Online.yaml")
trainAgent.trainingLoop()
del trainAgent

trainAgent = TrainingAgent("Training/configs/TVS_LEMURS_Online.yaml")
trainAgent.trainingLoop()
del trainAgent

trainAgent = TrainingAgent("Training/configs/Flocking_LEMURS_Online.yaml")
trainAgent.trainingLoop()
del trainAgent