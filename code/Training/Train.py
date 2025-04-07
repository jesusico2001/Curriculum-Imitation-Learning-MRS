import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from Training.TrainingAgent import TrainingAgent


trainAgent = TrainingAgent("Training/configs/TOY_FS_LEMURS_BabySteps.yaml")
trainAgent.trainingLoop()
del trainAgent
