import torch

MIN_INTERVAL = 350
MIN_INCREMENT = 1

def L2_loss(u, v):
    return (u - v).pow(2).mean()

class DatasetBuilder():
    def __init__(self, policy, numAgents,numTrain, numTests, seed_data, device):
        self.policy = policy
        self.numAgents = numAgents
        self.numTrain = numTrain
        self.numTests = numTests
        self. seed_data = seed_data
        self.device = device
        
    def BuildDatasets(self, numSamples):
        # Train
        #'datasets/'+policy+str(numAgents)+'_inputsTrain_'+str(numTrain)+'_'+str(numTests)+'_'+str(numSamples)+'_'+str(seed)+'.pth'
        train_data = torch.load('saves/datasets/'+self.policy+str(self.numAgents)+'_trainData_'+str(self.numTrain)+'_'+str(self.numTests)+'_'+str(numSamples)+'_'+str(self.seed_data)+'.pth').to(self.device)
        # Validation
        val_data = torch.load('saves/datasets/'+self.policy+str(self.numAgents)+'_testData_'+str(self.numTrain)+'_'+str(self.numTests)+'_'+str(numSamples)+'_'+str(self.seed_data)+'.pth').to(self.device)
        
        return train_data, val_data

class CuadraticFunction():
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def compute(self, x):
        return int(self.a * pow(x, 2) + self.b * x + self.c)

    @staticmethod
    def modelConstant(c):
        return CuadraticFunction(0, 0, c)
    
    @staticmethod
    def modelLinear(minVal, maxVal):
        b = (maxVal - minVal) / 245
        c = minVal - 5*b
        return CuadraticFunction(0, b, c)
    
    @staticmethod
    def modelLinearModulated(minVal, maxVal):
        a = (minVal - maxVal) / 60025
        b = ( 500 * maxVal - minVal) / 60025
        c = (62500 * minVal - 2475 *  maxVal) / 60025
        return CuadraticFunction(a, b, c)
    
    @staticmethod
    def ModelInterval(policy, parameter):
        if policy == "fixed":
            return CuadraticFunction.modelConstant(parameter)
        elif policy == "linear":
            return CuadraticFunction.modelLinear(MIN_INTERVAL, parameter)
        elif policy == "modulated":
            return CuadraticFunction.modelLinearModulated(MIN_INTERVAL, parameter)
        else:
            print("Interval Function - "+ policy  +" is not a policy...\n")
            exit(0)
    @staticmethod
    def ModelIncrement(policy, parameter):
        if policy == "fixed":
            return CuadraticFunction.modelConstant(parameter)
        elif policy == "linear":
            return CuadraticFunction.modelLinear(MIN_INCREMENT, parameter)
        elif policy == "modulated":
            return CuadraticFunction.modelLinearModulated(MIN_INCREMENT, parameter)
        else:
            print("Increment Function - "+ policy  +" is not a policy...\n")
            exit(0)
        

# TRAINING LOOP
# =====================================================
def runEpochLoss(learn_system, inputs, target, numAgents, step_size, numSamples):
    # Run epoch
    #print("Training with "+str(numSamples)+" samples:\n")
    time            = step_size * numSamples
    simulation_time = torch.linspace(0, time - step_size, numSamples)

    output = learn_system.forward(inputs, simulation_time, step_size)

    # Compute loss
    L = L2_loss((output[:, :, :4 * numAgents].reshape(-1, 4 * numAgents)), target.reshape(-1, 4 * numAgents))
    return L


def validate(learn_system, inputs_val, target_val, numAgents, step_size, numSamples):

    learn_system.eval()                 # Set evaluation mode
    loss_val = runEpochLoss(learn_system, inputs_val, target_val, numAgents, step_size, numSamples) # Run and compute loss
    learn_system.train()                # Go back to training mode

    return loss_val.detach().cpu().numpy()

def buildInputsTargets(trajectories, numSamples, batch_size, device):
    # Select batch
    chosen_batch  = torch.randperm(trajectories.size(1))[:batch_size]
    batch = trajectories[:,chosen_batch,:]

    # Select initial states for traj. of numSamples
    realNS = trajectories.size()[0]
    chosen_initial_state  = torch.randint(0, realNS-numSamples, [batch_size])
    chosen_states = chosen_initial_state.unsqueeze(1) + torch.arange(numSamples)

    # Build uniform unbiased batch
    numAgents = int(trajectories.size(2) / 8)
    inputs = torch.zeros([batch_size, 8*numAgents]).to(device)
    targets = torch.zeros([numSamples, batch_size, 4*numAgents]).to(device)
    for i in range(batch_size):
        inputs[i, :] = batch[chosen_initial_state[i], i, :]
        targets[:, i, :] = batch[chosen_states[i], i, :4*numAgents]

    return inputs, targets

def trainingLoop(learn_system, datasetBuilder, optimizer, initial_epoch, epochs, numAgents, numTrain, numTests, 
                 initial_numSamples, maxNumSamples, path_checkpoint, interval_function:CuadraticFunction , increment_function:CuadraticFunction):
    # Hyperparameters, the typical ones
    step_size       = 0.04

    # Numsamples management
    # NS_datasets =  [5, 10, 25, 50, 125, 250]
    NS_datasets =  [250]

    numSamples = initial_numSamples
    nextEpochNS = initial_epoch + interval_function.compute(numSamples)
    
    actual_dataset = 0
    while numSamples > NS_datasets[actual_dataset]:
        actual_dataset += 1

    # Validation management
    VAL_PERIOD = 50
    nextEpochVal = initial_epoch

    train_size      = 100
    tests_size      = 100
    
    # Log train and evaluation loss
    TrainLosses = []
    ValLosses = []
    ValEpochs = []
    numSamplesLog = []

    train_data, val_data = datasetBuilder.BuildDatasets(NS_datasets[actual_dataset])

    for epoch in range(initial_epoch, epochs):

        #if epoch%NS_INCR_PERIOD == 0 and numSamples < maxNumSamples:
        if epoch == nextEpochNS and numSamples < maxNumSamples:

            # First Increase numSamples
            numSamples += increment_function.compute(numSamples)
            if numSamples > maxNumSamples:
                numSamples = maxNumSamples
            
            # Then update next interval
            nextEpochNS += interval_function.compute(numSamples)

            numSamplesLog.append([epoch, numSamples])
            print("epoch "+str(epoch)+" - now training with "+str(numSamples)+" samples...\n")

        
        # Run and compute loss
        inputs_train, target_train = buildInputsTargets(train_data, numSamples, train_size, learn_system.device)
        loss_train = runEpochLoss(learn_system, inputs_train, target_train, numAgents, step_size, numSamples)
        
        # Update weights
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()


        # Validate
        if epoch == nextEpochVal or epoch == nextEpochNS or epoch+1 == nextEpochNS or epoch == epochs-1 :

            if epoch == nextEpochVal:
                nextEpochVal += VAL_PERIOD

            # Build targets and validate
            inputs_val, target_val = buildInputsTargets(val_data, numSamples, tests_size, learn_system.device)
            loss_val = validate(learn_system, inputs_val, target_val, numAgents, step_size, numSamples)
            
            # Print and store (in RAM) eval info
            print('Epoch = %d' % (epoch)  ) 
            print('Training Loss            = %.12f' % loss_train.detach().cpu().numpy()) 
            print('Validat. Loss (ns:%d/%d) = %.12f' % (numSamples, maxNumSamples, loss_val))
            print('===============================================\n')
            TrainLosses.append(loss_train.detach().cpu().numpy())
            ValLosses.append(loss_val)
            ValEpochs.append(epoch)

            # Store model
            torch.save(learn_system.state_dict(), path_checkpoint+"/epoch_"+str(epoch)+'.pth')
            
    return TrainLosses, ValLosses, ValEpochs, numSamplesLog



