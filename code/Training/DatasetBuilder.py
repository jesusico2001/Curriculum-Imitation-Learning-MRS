import torch

class DatasetBuilder():
    def __init__(self, policy, numAgents, numTrain, numValidation, seed_data, device):
        self.policy = policy
        self.numAgents = numAgents
        self.numTrain = numTrain
        self.numValidation = numValidation
        self. seed_data = seed_data
        self.device = device
        
    def BuildDatasets(self, numSamples):
        # Train
        #'datasets/'+policy+str(numAgents)+'_inputsTrain_'+str(numTrain)+'_'+str(numValidation)+'_'+str(numSamples)+'_'+str(seed)+'.pth'
        train_data = torch.load('saves/datasets/'+self.policy+str(self.numAgents)+'_trainData_'+str(self.numTrain)+'_'+str(numSamples)+'_'+str(self.seed_data)+'.pth').to(self.device)
        # Validation
        val_data = torch.load('saves/datasets/'+self.policy+str(self.numAgents)+'_valData_'+str(self.numValidation)+'_'+str(numSamples)+'_'+str(self.seed_data)+'.pth').to(self.device)
        
        return train_data, val_data

    def BuildValidation(self, numSamples, valAgents):
        val_data = torch.load('saves/datasets/'+self.policy+str(valAgents)+'_valData_'+str(self.numValidation)+'_'+str(numSamples)+'_'+str(self.seed_data)+'.pth').to(self.device)
        
        return val_data