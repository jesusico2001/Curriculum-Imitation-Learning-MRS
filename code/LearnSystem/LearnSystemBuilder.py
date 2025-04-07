from enum import Enum
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from LearnSystem import LEMURS, MLP, GNN, GNNSA

class learn_systems(Enum):
    LEMURS = "LEMURS"
    MLP = "MLP"
    GNN = "GNN"    
    GNNSA = "GNNSA"    

def buildLearnSystem(config):
    architecture = config["learn_system"]["type"]
    if architecture == learn_systems.LEMURS.value:
        return LEMURS.LEMURS(config)
    elif architecture == learn_systems.MLP.value:
        return MLP.MLP(config)
    elif architecture == learn_systems.GNN.value:
        return GNN.GNN(config)
    elif architecture == learn_systems.GNNSA.value:
        return GNNSA.GNNSA(config)

    # ADD NEW MODELS HERE

    else :
        print("buildLearnSystem ERROR: \"", architecture, "\" is not a valid architecture.")
        exit(0)
