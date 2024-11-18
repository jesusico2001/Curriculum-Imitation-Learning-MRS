import torch

MIN_INTERVAL = 350
MIN_INCREMENT = 1

def L2_loss(u, v, ns_distr):
    sumErrors = torch.sum((u - v).pow(2)) 
    numComparedValues = (torch.sum(ns_distr) * u.shape[1])

    return sumErrors / numComparedValues
    
class DatasetBuilder():
    def __init__(self, policy, numAgents,numTrain, numValidation, seed_data, device):
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
        val_data_list = []
        val_data_list.append(torch.load('saves/datasets/'+self.policy+str(self.numAgents)+'_valData_'+str(self.numValidation)+'_'+str(numSamples)+'_'+str(self.seed_data)+'.pth').to(self.device))
        val_data_list.append(torch.load('saves/datasets/'+self.policy+str(2*self.numAgents)+'_valData_'+str(self.numValidation)+'_'+str(numSamples)+'_'+str(self.seed_data)+'.pth').to(self.device))
        val_data_list.append(torch.load('saves/datasets/'+self.policy+str(3*self.numAgents)+'_valData_'+str(self.numValidation)+'_'+str(numSamples)+'_'+str(self.seed_data)+'.pth').to(self.device))
        
        return train_data, val_data_list

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
def runEpochLoss(learn_system, inputs, target, nS_distr, numAgents, step_size, numSamples):
    # Run epoch
    #print("Training with "+str(numSamples)+" samples:\n")
    time            = step_size * numSamples
    simulation_time = torch.linspace(0, time - step_size, numSamples)

    output = learn_system.forward(inputs, simulation_time, step_size)
    for i, ns in enumerate(nS_distr):
        output[ns:] = torch.zeros([numSamples-ns, output.shape[1], output.shape[2]])    

    # Compute loss
    L = L2_loss((output[:, :, :4 * numAgents].reshape(-1, 4 * numAgents)), target.reshape(-1, 4 * numAgents), nS_distr)
    return L


def validate(learn_system, inputs_val, target_val, nS_distr, numAgents, step_size, numSamples):
    old_na = learn_system.na

    learn_system.na = numAgents
    learn_system.eval()                 # Set evaluation mode
    loss_val = runEpochLoss(learn_system, inputs_val, target_val, nS_distr, numAgents, step_size, numSamples) # Run and compute loss
    learn_system.train()                # Go back to training mode
    learn_system.na = old_na


    return loss_val.detach().cpu().numpy()

def choose_difficulties_quota(batch_size, old_task_quota, current_difficulty):
    nOldTasks = int(batch_size*old_task_quota)
    nNewTasks = int(batch_size*(1-old_task_quota))
    while(nOldTasks + nNewTasks < batch_size):
        nNewTasks += 1

    # Select sizes of each trajectory
    old_tasks_sizes = torch.randint(1, current_difficulty-1, [nOldTasks])
    new_task_sizes = torch.full([nNewTasks], current_difficulty)
    chosen_sizes  = torch.cat(( old_tasks_sizes, new_task_sizes))
    return chosen_sizes

def buildInputsTargets(trajectories, batch_size, difficulties, device):
    # Select batch
    chosen_batch  = torch.randperm(trajectories.size(1))[:batch_size]
    batch = trajectories[:,chosen_batch,:]

    # Select initial states for traj. of numSamples
    top_difficulty = max(difficulties)
    realNS = trajectories.size()[0]
    chosen_initial_state  = torch.tensor([torch.randint(0, realNS-k, [1]) for k in difficulties])
    chosen_states = chosen_initial_state.unsqueeze(1) + torch.arange(top_difficulty)
    chosen_states = [row[:size] for row, size in zip(chosen_states, difficulties)]

    # Build uniform unbiased batch
    numAgents = int(trajectories.size(2) / 8)
    inputs = torch.zeros([batch_size, 8*numAgents]).to(device)
    targets = torch.zeros([top_difficulty, batch_size, 4*numAgents]).to(device)
    
    for i in range(batch_size):
        inputs[i, :] = batch[chosen_initial_state[i], i, :]
        targets[:difficulties[i], i, :] = batch[chosen_states[i], i, :4*numAgents]
    
    return inputs, targets, top_difficulty

VAL_PERIOD = 50
def trainingLoop(learn_system, datasetBuilder, optimizer, initial_epoch, epochs, numAgents, initial_numSamples, maxNumSamples, path_checkpoint, interval_function:CuadraticFunction , increment_function:CuadraticFunction, old_task_quota, ACTIVATE_EARLY_STOPPING):
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
    nextEpochVal = initial_epoch

    train_size      = 100
    validation_size      = 100
    
    # Log train and evaluation loss
    TrainLosses = []
    ValLosses = []
    ValLosses_scalability = []
    ValEpochs = []
    numSamplesLog = []

    train_data, val_data_list = datasetBuilder.BuildDatasets(NS_datasets[actual_dataset])
    
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
        difficulties = choose_difficulties_quota(train_size, old_task_quota, numSamples)
        inputs_train, target_train, top_difficulty = buildInputsTargets(train_data, train_size, difficulties, learn_system.device)
        loss_train = runEpochLoss(learn_system, inputs_train, target_train, difficulties, numAgents, step_size, numSamples)
        
        # Update weights
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()


        # Validate
        if epoch == nextEpochVal or epoch == nextEpochNS or epoch+1 == nextEpochNS or epoch == epochs-1 :
            if epoch == nextEpochVal:
                nextEpochVal += VAL_PERIOD

            # Build targets and validate with maxNumSamples (normalized Loss)
            loss_val = 0
            loss_val_scalable = 0
            for i, val_data in enumerate(val_data_list):
                
                difficulties = choose_difficulties_quota(validation_size, 0.0, maxNumSamples)
                inputs_val, target_val, top_difficulty = buildInputsTargets(val_data, validation_size, difficulties, learn_system.device)
                
                if i==0:
                    loss_val += validate(learn_system, inputs_val, target_val, difficulties, (1+i)*numAgents, step_size, maxNumSamples)
                else:
                    loss_val_scalable += validate(learn_system, inputs_val, target_val, difficulties, (1+i)*numAgents, step_size, maxNumSamples)
            loss_val_scalable /= len(val_data_list)-1
            # Print and store (in RAM) eval info
            print('Epoch = %d' % (epoch)  ) 
            print('Training Loss (ns:%d/%d) = %.12f' % (numSamples, maxNumSamples, loss_train.detach().cpu().numpy())) 
            print('Validat. Loss (ns:%d/%d) = %.12f' % (maxNumSamples, maxNumSamples, loss_val))
            print('===============================================\n')
            TrainLosses.append(loss_train.detach().cpu().numpy())
            ValLosses.append(loss_val)
            ValLosses_scalability.append(loss_val_scalable)
            ValEpochs.append(epoch)

            # Store model
            torch.save(learn_system.state_dict(), path_checkpoint+"/epoch_"+str(epoch)+'.pth')
            
            # Early stopping
            if ACTIVATE_EARLY_STOPPING and earlyStopping_valDiverge(ValLosses, ValLosses_scalability):
                print("Stopping early at epoch ",epoch, " of ", epochs, "...")
                return TrainLosses, ValLosses, ValEpochs, numSamplesLog

    return TrainLosses, ValLosses, ValLosses_scalability, ValEpochs, numSamplesLog


FRAME_SIZE = int(300 / VAL_PERIOD)
def earlyStopping_valDiverge(TrainLosses, ValLosses):
    if len(TrainLosses) < FRAME_SIZE:
        return False
    
    if len(ValLosses) != len(TrainLosses):
        print("Error earlyStopping.")
        exit(0)
    

    may_stop  = True
    for i in reversed(range(FRAME_SIZE-1)):
        if not may_stop:
            break
        betterTrain = TrainLosses[-(i+1)] > TrainLosses[-i]
        worseVal = ValLosses[-(i+1)] < ValLosses[-i]
        
        may_stop = may_stop and betterTrain and worseVal
    return may_stop

def earlyStopping_lowImprovement(ValLosses, minImprovement):
    if len(ValLosses) < FRAME_SIZE:
        return False

    frameLosses = torch.tensor([a.item() for a in ValLosses[-FRAME_SIZE:]], dtype=torch.float32 )
    frameImprovement = frameLosses[0] - frameLosses[-1]

    fin_diff_1 = frameLosses[1:] - frameLosses[:-1]
    fin_diff_1_stop = all(-dif > minImprovement for dif in fin_diff_1)

    # avgAbsVariation = abs(fin_diff_1).mean()

    # fin_diff_2 = fin_diff_1[1:] - fin_diff_1[:-1]
    # a = fin_diff_2.mean().item() 
    # convexLoss = a >= 0

    # stop = convexLoss and a >= 0 and avgAbsVariation < minImprovement
    # if stop: 
    #     print("frameImprovement: ", frameImprovement, " - convexity: ", a)

    return fin_diff_1_stop