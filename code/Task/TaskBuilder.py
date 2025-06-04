
from enum import Enum
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from Task.LEMURS.Flocking import Flocking
from Task.LEMURS.TimeVaryingSwapping import TimeVaryingSwapping
from Task.LEMURS.FixedSwapping import FixedSwapping
from Task.VMAS.Navigation import Navigation
from Task.VMAS.Balance import Balance
from Task.VMAS.Passage import Passage
from Task.VMAS.Mall.Mall import Mall

import yaml

class Tasks(Enum):
    FS = "FS"
    TVS = "TVS"
    Flocking = "Flocking"
    Navigation = "navigation"
    Balance = "balance"
    Passage = "passage"
    Mall = "mall"

def TaskBuilder(config):
    if config["task"]["type"] == Tasks.FS.value:
        return FixedSwapping(config)
    if config["task"]["type"] == Tasks.TVS.value:
        return TimeVaryingSwapping(config)
    if config["task"]["type"] == Tasks.Flocking.value:
        return Flocking(config)
    if config["task"]["type"] == Tasks.Navigation.value:
        return Navigation(config)
    if config["task"]["type"] == Tasks.Balance.value:
        return Balance(config)
    if config["task"]["type"] == Tasks.Passage.value:
        return Passage(config)
    if config["task"]["type"] == Tasks.Mall.value:
        return Mall(config)

    else:
        print("Unknown task: ", config["task"]["type"])
        exit(0)