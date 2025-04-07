import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from DatasetGenerator.GeneratorBuilder import GeneratorBuilder


# gen = GeneratorBuilder("Training/configs/TOY_VMAS_LEMURS_BabySteps.yaml", {})
gen = GeneratorBuilder("Training/configs/VMAS_navigation.yaml", {})
gen.generateTrainValTest()
del gen