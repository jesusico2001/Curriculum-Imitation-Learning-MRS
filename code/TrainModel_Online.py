import argparse
import os
import torch
import time
import numpy
from TrainingTools import *
from LearnSystem import LearnSystemBuilder

def sampleDifficulties(difficulty_distribution, numSamples):
    numDifficulties = len(difficulty_distribution)
    indices = numpy.arange(numDifficulties) + 1
    samples = numpy.random.choice(indices, size=numSamples, p=difficulty_distribution.numpy())
    return torch.tensor(samples)

def sampleUniformDeterministic(maxValue, numSamples):
    if numSamples % maxValue != 0:
        print("sampleUniformDeterministic: Validation batch size (", numSamples,") cannot be divided by the max difficulty (",maxValue,").")
        exit(0)
    samples_per_value =  int(numSamples / maxValue)
    difficulties = torch.arange(1,maxValue+1, dtype=int).unsqueeze(1)
    difficulties = torch.kron(difficulties, torch.ones((1,samples_per_value), dtype=int)).reshape(-1)

    return difficulties

def validate_and_loss_distr(learn_system, inputs_val, target_val, nS_distr, numAgents, step_size, numSamples):
    print("Validating to obtain loss distribution...")
    old_na = learn_system.na

    learn_system.na = numAgents
    learn_system.eval()                 # Set evaluation mode
    
    time            = step_size * numSamples
    simulation_time = torch.linspace(0, time - step_size, numSamples)

    output = learn_system.forward(inputs_val, simulation_time, step_size)
    for i, ns in enumerate(nS_distr):
        output[ns:] = torch.zeros([numSamples-ns, output.shape[1], output.shape[2]])    
    
    losses = (output[:, :, :4 * numAgents] - target_val).pow(2)
    losses = losses.sum(dim=(0,2))
    
    loss_accumulated = torch.zeros(numSamples).to(learn_system.device)
    for loss, difficulty in zip(losses, nS_distr):
        loss_accumulated[difficulty-1] += loss
    
    difficulty_count = torch.bincount(nS_distr-1).clone()
    
    loss_distr = torch.zeros(numSamples).to(learn_system.device)
    for index_difficulty, loss in enumerate(loss_accumulated):
        difficulty = index_difficulty + 1
        loss_distr[index_difficulty] = loss / (difficulty * max(1,difficulty_count[index_difficulty]))
        
    learn_system.train()                # Go back to training mode
    learn_system.na = old_na

    del difficulty_count, loss_accumulated, losses, output
    
    return loss_distr.detach().cpu().numpy()


def trainingLoopOnline(learn_system, datasetBuilder, optimizer, initial_epoch, epochs, numAgents, maxNumSamples, path_checkpoint, ACTIVATE_EARLY_STOPPING):
    # Hyperparameters, the typical ones
    step_size       = 0.04

    # Batch sizes
    train_size      = 100
    validation_size = 2 * maxNumSamples

    # Numsamples management
    NS_datasets =  [250]
    actual_dataset = 0
    difficulty_distr = torch.ones([maxNumSamples]) / maxNumSamples
    difficulties_val = sampleUniformDeterministic(maxNumSamples, validation_size)
    old_loss_val_distr = torch.ones([maxNumSamples]) * 1000
    
    # Validation management
    nextEpochVal = initial_epoch
    
    # Log train and evaluation loss
    TrainLosses = []
    ValLosses = []
    ValLosses_scalability = []
    ValEpochs = []
    numSamplesLog = []

    train_data, val_data_list = datasetBuilder.BuildDatasets(NS_datasets[actual_dataset])
    
    for epoch in range(initial_epoch, epochs):

        # Run and compute loss
        difficulties = sampleDifficulties(difficulty_distr, train_size)
        inputs_train, target_train, top_difficulty = buildInputsTargets(train_data,  train_size, difficulties, learn_system.device)
        loss_train = runEpochLoss(learn_system, inputs_train, target_train, difficulties, numAgents, step_size, top_difficulty)
        
        # Update weights
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Validate
        if epoch == nextEpochVal or epoch == epochs-1 :
            if epoch == nextEpochVal:
                nextEpochVal += VAL_PERIOD

            # Build targets and validate with maxNumSamples (normalized Loss)
            loss_val = 0
            loss_val_scalable = 0
            for i, val_data in enumerate(val_data_list):
                
                if i==0:
                    inputs_val, target_val, top_difficulty = buildInputsTargets(val_data, validation_size, difficulties_val, learn_system.device)
                    
                    loss_val_distr = validate_and_loss_distr(learn_system, inputs_val, target_val, difficulties_val, (1+i)*numAgents, step_size, top_difficulty)
                    # print("Validation Loss Distribution: \n", loss_val_distr)
                    loss_val += loss_val_distr[-1]
                    
                else:
                    inputs_val, target_val, top_difficulty = buildInputsTargets(val_data, validation_size, difficulties_val, learn_system.device)
                    loss_val_scalable += validate(learn_system, inputs_val, target_val, difficulties_val, (1+i)*numAgents, step_size, top_difficulty)
                    
            loss_val_scalable /= len(val_data_list)-1
            
            # Print and store (in RAM) eval info
            print('Epoch = %d' % (epoch)  ) 
            print('Training Loss = %.12f' % (loss_train.detach().cpu().numpy())) 
            print('Validat. Loss (Uniform NS) = %.12f' % (loss_val))
            print('===============================================\n')
            TrainLosses.append(loss_train.detach().cpu().numpy())
            ValLosses.append(loss_val)
            ValLosses_scalability.append(loss_val_scalable)
            ValEpochs.append(epoch)

            # Update difficulty distribution
            loss_improvement = (old_loss_val_distr - loss_val_distr)
            old_loss_val_distr = loss_val_distr # Save this variation to compare w/ in the next update
            # print("loss_improvement;\n",loss_improvement)

            # print("worst_improvement", min(loss_improvement))
            loss_improvement = loss_improvement - min(loss_improvement)
            # print("loss_improvement (positivized);\n",loss_improvement)

            
            # print("Inverted loss variations:\n",inverse_loss_variation)
            difficulty_distr += loss_improvement
            difficulty_distr /= sum(difficulty_distr)
            # print("Updated difficulties:\n", difficulty_distr)
            # Store model
            torch.save(learn_system.state_dict(), path_checkpoint+"/epoch_"+str(epoch)+'.pth')
            
            # Early stopping
            if ACTIVATE_EARLY_STOPPING and earlyStopping_valDiverge(ValLosses, ValLosses_scalability):
                print("Stopping early at epoch ",epoch, " of ", epochs, "...")
                return TrainLosses, ValLosses, ValEpochs, numSamplesLog

    return TrainLosses, ValLosses, ValLosses_scalability, ValEpochs, numSamplesLog


def main(policy='FS', architecture='LEMURS', nAttLayers=3, epochs=40000, numTrain=20000, numVal=20000, 
         maxNumSamples=5, seed_data=42, seed_train=42, numAgents=4):
    # Set fixed random number seed
    torch.manual_seed(seed_train)
    
    train_info = policy+str(numAgents)+architecture+'_'+str(nAttLayers)+"_"+str(epochs)\
        +"_"+str(numTrain)+"_"+str(numVal)+"_"+str(maxNumSamples)+"_"+str(seed_train)+"_"\
        +str(seed_data)+"_online"
    
    path_checkpoints = "saves/checkpoints/"+train_info
    path_loss = "evaluation/loss_evo/"+train_info
    try:
        os.makedirs(path_checkpoints)
        os.makedirs(path_loss)
    except FileExistsError:
        
        act = input("There is already data for this configuration. Do you want to overwrite it? (Y/N)\n").lower()
        if act == "y":
            pass
        else:
            print("Aborting training...\n")
            exit(0)

    # Build learnSystem and set train mode
    parameters = LearnSystemBuilder.buildParameters(policy, numAgents, nAttLayers)
    learn_system = LearnSystemBuilder.buildLearnSystem(architecture, parameters) 
    learn_system.train()

    # Define the loss function and optimizer
    learning_rate   = learn_system.learning_rate
    optimizer = torch.optim.Adam(learn_system.parameters(), lr=learning_rate, weight_decay=0.0)

    # Build dataset    
    dsBuilder = DatasetBuilder(policy, numAgents, numTrain, numVal, seed_data, parameters["device"])    

    #Train
    startTime = time.time()

    TrainLosses, ValLosses, ValLosses_scalability, ValEpochs, numSamplesLog = trainingLoopOnline(learn_system, dsBuilder, optimizer,                                                                       
        0, epochs, numAgents, maxNumSamples, path_checkpoints, False)
    
    totalTime = time.time() - startTime
    print('Training process has finished in '+str(totalTime/60)+' min  '+str(totalTime%60)+' seg. Saving evaluation info...')

    # Save
    torch.save(TrainLosses, path_loss+'/trainLosses.pth')
    torch.save(ValLosses, path_loss+'/valLosses.pth')
    torch.save(ValLosses_scalability, path_loss+'/valLosses_scalability.pth')
    torch.save(ValEpochs, path_loss+'/valEpochs.pth')
    torch.save(numSamplesLog, path_loss+'/numSamplesLog.pth')
    torch.save(totalTime, path_loss+'/time.pth')

    # Free unused memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Various LearnSystem\'s training process.')
    parser.add_argument('--policy', type=str, nargs=1, help='control policy name')
    parser.add_argument('--architecture', type=str, nargs=1, help='architecture name')
    parser.add_argument('--nAttLayers', type=int, nargs=1, help='number of attention layers used')
    parser.add_argument('--nEpochs', type=int, nargs=1, help='number of epochs during trainings')
    parser.add_argument('--numTrain', type=int, nargs=1, help='number of instances for training')
    parser.add_argument('--numVal', type=int, nargs=1, help='number of instances for validation')
    parser.add_argument('--maxNumSamples', type=int, nargs=1, help='maximum number of samples per instance (min is 5)')
    parser.add_argument('--seed_data', type=int, nargs=1, help='seed used for generating the data')
    parser.add_argument('--seed_train', type=int, nargs=1, help='seed to generate reproducible data')
    parser.add_argument('--numAgents', type=int, nargs=1, help='number of agents')

    args = parser.parse_args()
    main(args.policy[0], args.architecture[0], args.nAttLayers[0], args.nEpochs[0], args.numTrain[0], args.numVal[0], 
         args.maxNumSamples[0], args.seed_data[0], args.seed_train[0], args.numAgents[0])
