
from enum import Enum
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from Training.Teacher.BabySteps import BabyStepsTeacher
from Training.Teacher.Online import OnlineTeacher

class teachers(Enum):
    babysteps = "babysteps"
    online = "online"


def TeacherBuilder(config):
    if config["type"] == teachers.babysteps.value:
        return BabyStepsTeacher(config)
    if config["type"] == teachers.online.value:
        return OnlineTeacher(config)

    else:
        print("Unknown teacher: ", config["type"])
        exit(0)