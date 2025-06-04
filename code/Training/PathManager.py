class PathManager():
    def __init__(self, config):
        self.trainConfig = config
    
    def getPathDatasets(self):
        return "saves/datasets/" + self.taskInfo(True)

    def getDatasetFilename(self, split, data="", noisy=True):
        conf = self.trainConfig["general"]
        seed = conf["seed_data"]
        
        if split=="train":
            size = conf["train_size"]
        elif split=="val":
            size = conf["val_size"]
        elif split=="test":
            size = conf["test_size"]
        else:
            print("ERROR (getDatasetFilename): \""+ split + "\" is not a valid parameter.")
            exit(0)

        name = split+'_'+str(size)+'_'+str(seed)
        if data=="actions":
            name = "actions_" + name
        elif data=="rewards":
            name = "rewards_" + name

        if not noisy:
            name += "_noiseless"

        return name+'.pth'
        
    # ===========================================
    def getPathCheckpoints(self):
        return "saves/checkpoints/" + self.trainInfo()
    
    def getPathHistory(self):
        return "saves/history/" + self.trainInfo()
    
    def getPathEvaluation(self):
        return "saves/evaluation/" + self.trainInfo()

    def trainInfo(self):
        train_info = self.teacherInfo() + self.studentInfo() + self.taskInfo(False) + self.generalTrainInfo() 
        
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
        
        usedParams = ["type"]
        for param, value in conf.items():
            if not (param in usedParams):
                info += str(value) + "_"
        info = info[:-1] + "/"
        return info
    
    def taskInfo(self, used_for_dataset=False):
        conf = self.trainConfig["task"]
        
        info = str(conf["type"]) + "_" + conf["lib"] + "/"
        
        usedParams = ["type", "lib"]
        if used_for_dataset:
            usedParams.append("robot_obs_noise")
            usedParams.append("action_noise_factor")

        for param, value in conf.items():
            if not (param in usedParams):
                info += str(value) + "_"
        info = info[:-1] + "/"
        return info

    def generalTrainInfo(self):
        conf = self.trainConfig["general"]

        info = str(conf["epochs"]) + "_"
        info += str(conf["train_size"]) + "_"
        info += str(conf["val_size"]) + "_"
        info += str(conf["seed_data"]) + "_"
        info += str(conf["seed_train"]) + "_"
        info += str(conf["early_stopping"])

        return info