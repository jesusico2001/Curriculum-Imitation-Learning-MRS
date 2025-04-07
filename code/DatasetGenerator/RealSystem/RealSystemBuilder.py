from enum import Enum
import torch
from . import RealSystemFS, RealSystemTVS, RealSystemFlocking

class control_policies(Enum):
    FS = "FS"
    TVS = "TVS"
    Flocking = "Flocking"

def buildParameters(policy, numAgents):
    na = torch.as_tensor(numAgents)  # Number of agents

    e = torch.as_tensor(0.1)  # For the sigma-norm
    a = torch.as_tensor(5.0)  # For the sigmoid
    b = torch.as_tensor(5.0)  # For the sigmoid
    c = torch.abs(a - b) / torch.sqrt(4 * a * b)  # For the sigmoid

    parameters = {"na": na, "e": e, "a": a, "b": b, "c": c}

    if policy == control_policies.FS.value :
        d = torch.as_tensor(2.0)   # Communication distance
        c1 = torch.as_tensor(0.8)  # Tracking gain 1
        c2 = torch.as_tensor(1.0)  # Tracking gain 2
        parameters["d"] = d
        parameters["c1"] = c1
        parameters["c2"] = c2

    elif policy == control_policies.TVS.value :
        d = torch.as_tensor(2.0)     # Desired flocking distance
        c1 = torch.as_tensor(0.8)    # Tracking gain 1
        c2 = torch.as_tensor(1.0)    # Tracking gain 2
        r = torch.as_tensor(1.2*2.0) # Communication radius
        parameters["d"] = d
        parameters["c1"] = c1
        parameters["c2"] = c2
        parameters["r"] = r

    elif policy == control_policies.Flocking.value :
        d = torch.as_tensor(1.0)     # Desired flocking distance
        c1 = torch.as_tensor(0.4)    # Tracking gain 1
        c2 = torch.as_tensor(0.8)    # Tracking gain 2
        r = torch.as_tensor(1.2) * d # Radius of influence
        ha = torch.as_tensor(0.2)    # For the sigmoid
        parameters["d"] = d
        parameters["c1"] = c1
        parameters["c2"] = c2
        parameters["r"] = r
        parameters["ha"] = ha

    # ADD NEW POLICIES HERE
    else:
        print("buildParameters ERROR: \"", policy, "\" is not a valid policy.")
        exit(0)

    return parameters


def buildRealSystem(policy, parameters):
    if policy == control_policies.FS.value:
        return RealSystemFS.realSystemFS(parameters)
    
    elif policy == control_policies.TVS.value:
        return RealSystemTVS.realSystemTVS(parameters)
    
    elif policy == control_policies.Flocking.value:
        return RealSystemFlocking.realSystemFlocking(parameters)
    
    # ADD NEW POLICIES HERE
    else :
        print("buildRealSystem ERROR: \"", policy, "\" is not a valid policy.")
        exit(0)
