from enum import Enum
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from LearnSystem import LEMURS, MLP, GNN, GNNSA
from LearnSystem.ControlPolicy import FixedSwapping, TimeVaryingSwapping, Flocking

import torch

class control_policies(Enum):
    FS = "FS"
    TVS = "TVS"
    Flocking = "Flocking"

class learn_systems(Enum):
    LEMURS = "LEMURS"
    MLP = "MLP"
    GNN = "GNN"    
    GNNSA = "GNNSA"    
 
def buildParameters(control_policy, numAgents, nAttLayers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    na     = torch.as_tensor(numAgents)  # Number of agents
    parameters = {"nAttLayers": nAttLayers, "na": na, "device": device}

    if control_policy == control_policies.FS.value:
        parameters["control_policy"] = FixedSwapping.FixedSwapping()

    elif control_policy == control_policies.TVS.value:
        parameters["control_policy"] = TimeVaryingSwapping.TimeVaryingSwapping()
    
    elif control_policy == control_policies.Flocking.value:
        parameters["control_policy"] = Flocking.Flocking()
        
    # ADD NEW POLICIES HERE
    else :
        print("buildParameters ERROR: \"", control_policy, "\" is not a valid policy.")
        exit(0)
    
    return parameters


def buildLearnSystem(architecture, parameters):
    if architecture == learn_systems.LEMURS.value:
        return LEMURS.LEMURS(parameters)
    
    elif architecture == learn_systems.MLP.value:
        return MLP.MLP(parameters)
    
    elif architecture == learn_systems.GNN.value:
        return GNN.GNN(parameters)
    
    elif architecture == learn_systems.GNNSA.value:
        return GNNSA.GNNSA(parameters)
    
    # ADD NEW SYSTEMS HERE

    else :
        print("buildLearnSystem ERROR: \"", architecture, "\" is not a valid architecture.")
        exit(0)