import argparse
import os
import torch
import time
from Training.TrainingTools import *
from LearnSystem import LearnSystemBuilder

def main(policy='FS', architecture='LEMURS', nAttLayers=3, epochs=40000, numTrain=20000, numVal=20000, 
         maxNumSamples=5, seed_data=42, seed_train=42, numAgents=4, interval_policy='fixed',
         interval_parameter=500, increment_policy ='fixed', increment_parameter=1):
    # Set fixed random number seed
    torch.manual_seed(seed_train)
    
    train_info = policy+str(numAgents)+architecture+'_'+str(nAttLayers)+"_"+str(epochs)\
        +"_"+str(numTrain)+"_"+str(numVal)+"_"+str(maxNumSamples)+"_"+str(seed_train)+"_"\
        +str(seed_data)+"_"+str(interval_policy)+"_"+str(interval_parameter)+"_"\
        +str(increment_policy)+"_"+str(increment_parameter)
    
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

    # Define numSamples evolution
    interval_function = CuadraticFunction.ModelInterval(interval_policy, interval_parameter)
    increment_function = CuadraticFunction.ModelIncrement(increment_policy, increment_parameter)

    old_task_quota = 0.0 if interval_parameter==0.0 and increment_parameter==0.0 else 0.0
    print("Training with ", old_task_quota*100, "% smaller data")

    #Train
    startTime = time.time()

    TrainLosses, ValLosses, ValLosses_scalability, ValEpochs, numSamplesLog = trainingLoop(learn_system, dsBuilder, optimizer,                                                                       
        0, epochs, numAgents, 5, maxNumSamples, path_checkpoints, interval_function, increment_function, old_task_quota, True)
    
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
    parser.add_argument('--interval_policy', type=str, nargs=1, help='either fixed, linear or modulated')
    parser.add_argument('--interval_parameter', type=float, nargs=1, help='parameter: c, b or maxValue according to the policy')
    parser.add_argument('--increment_policy', type=str, nargs=1, help='either fixed, linear or modulated')
    parser.add_argument('--increment_parameter', type=float, nargs=1, help='parameter: c, b or maxValue according to the policy')

    args = parser.parse_args()
    main(args.policy[0], args.architecture[0], args.nAttLayers[0], args.nEpochs[0], args.numTrain[0], args.numVal[0], 
         args.maxNumSamples[0], args.seed_data[0], args.seed_train[0], args.numAgents[0], args.interval_policy[0],
         args.interval_parameter[0], args.increment_policy[0] ,args.increment_parameter[0])
