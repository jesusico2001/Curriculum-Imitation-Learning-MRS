class PathManager():
    def __init__(self, config):
        self.trainConfig = config
        
    def getPathCheckpoints(self):
        return "saves/checkpoints/" + self.trainInfo()
    
    def getPathHistory(self):
        return "saves/history/" + self.trainInfo()
    
    def getPathEvaluation(self):
        return "saves/evaluation/" + self.trainInfo()

    def trainInfo(self):
        train_info = self.teacherInfo() + self.studentInfo() + self.generalInfo() 
        
        return train_info
    
    def teacherInfo(self):
        conf = self.trainConfig["teacher"]

        info = str(conf["type"]) + "/"
        try:
            info += str(conf["reward"]) + "Reward/"
        except:
            pass
        info += str(conf["max_difficulty"]) + "_"
        info += str(conf["difficulty_grouper"]) + "_"
        info += str(conf["difficulty_resolution"]) + "_"

        usedParams = ["type", "reward", "max_difficulty", "difficulty_grouper", "difficulty_resolution"]
        for param, value in conf.items():
            if not (param in usedParams):
                info += str(value) + "_"
        info = info[:-1] + "/"

        return info
    
    def studentInfo(self):
        conf = self.trainConfig["learn_system"]

        info = str(conf["type"]) + "_"
        info += str(conf["num_agents"]) + "_"
        info += str(conf["policy"]) + "_"
        info += str(conf["depth"]) + "/"

        return info
    
    def generalInfo(self):
        conf = self.trainConfig["general"]

        info = str(conf["epochs"]) + "_"
        info += str(conf["train_size"]) + "_"
        info += str(conf["val_size"]) + "_"
        info += str(conf["seed_data"]) + "_"
        info += str(conf["seed_train"]) + "_"
        info += str(conf["early_stopping"])

        return info