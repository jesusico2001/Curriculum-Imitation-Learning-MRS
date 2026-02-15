import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from DatasetGenerator.GeneratorBuilder import GeneratorBuilder


# gen = GeneratorBuilder("Training/configs/VMAS_navigation.yaml", {})
gen = GeneratorBuilder("Evaluation/Tests/6_selected_configs_passage/configs/VMAS_passage.yaml", {})
gen.generateTrainValTest()
del gen

