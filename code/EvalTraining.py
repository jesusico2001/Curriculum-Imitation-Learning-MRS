import argparse
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from DatasetGenerator import RealSystemBuilder
from LearnSystem import LearnSystemBuilder
from trajectory_analysis import *
from TrainingTools import *

VAL_PERIOD = 100

def getNumSampelesLog(epochs, maxNumSamples, interval_policy, interval_parameter, increment_policy, increment_parameter):
    interval_function = CuadraticFunction.ModelInterval(interval_policy, interval_parameter)
    increment_function = CuadraticFunction.ModelIncrement(increment_policy, increment_parameter)

    epoch = 0
    k = 5
    log = []
    while epoch < epochs:
        log.append([epoch, k])
        aux = interval_function.compute(k)
        if aux == 0 or k >= maxNumSamples: 
            return log
        
        epoch += aux
        k += increment_function.compute(k)

    return log

def main(control_policy='FS', architecture='LEMURS', nAttLayers=3, epochs=40000, numTrain=20000, numVal=20000, numTests=20000, 
         maxNumSamples=5, seed_data=42, seed_train=42, numAgents=4, numAgents_train=4, interval_policy='fixed',
         interval_parameter=500, increment_policy ='fixed', increment_parameter=1):
    torch.manual_seed(505)

    train_info = control_policy+str(numAgents_train)+architecture+'_'+str(nAttLayers)+"_"+str(epochs)\
        +"_"+str(numTrain)+"_"+str(numVal)+"_"+str(maxNumSamples)+"_"+str(seed_train)+"_"\
        +str(seed_data)+"_"+str(interval_policy)+"_"+str(interval_parameter)+"_"\
        +str(increment_policy)+"_"+str(increment_parameter)
    path_checkpoints = "saves/checkpoints/"+train_info
    path_results = "evaluation/results/"+train_info+"/"+str(numAgents)+"_agents"

    path_valData = 'saves/datasets/'+control_policy+str(numAgents)+'_valData_'+str(numVal)+'_'+str(250)+'_'+str(seed_data)+'.pth'
    path_testData = 'saves/datasets/'+control_policy+str(numAgents)+'_testData_'+str(numTests)+'_'+str(250)+'_'+str(seed_data)+'.pth'

    try:
        os.makedirs(path_results)
    except FileExistsError:
        act = input("There is already an evaluation for this configuration.\n Do you want to overwrite it? (Y/N)\n").lower()
        if act == "y":
            pass
        else:
            print("Aborting evaluation...\n")
            exit(0)

    # Hyperparameters   
    step_size       = 0.04
    time            = 10.0
    true_nSamples   = int(time / step_size)
    simulation_time = torch.linspace(0, time, int(time/step_size))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build learn system
    ls_params = LearnSystemBuilder.buildParameters(control_policy, numAgents, nAttLayers)
    myLearnSystem = LearnSystemBuilder.buildLearnSystem(architecture, ls_params)
    with torch.no_grad():
        myLearnSystem.eval()


    # NumSamples evolution
    numSamplesLog = getNumSampelesLog(epochs, maxNumSamples, interval_policy, interval_parameter, increment_policy, increment_parameter)
    print("NumSamples evolution generated successfully")

    # Validation
    # ==========
    val_data = torch.load(path_valData).to(myLearnSystem.device)
    initial_states = val_data[0,:,:]
    real_trajectories = val_data
    print("Validation trajectories loaded successfully")

    # Video evolution
    fig = plt.figure(figsize=(14,12))
    ani = FuncAnimation(fig, updateFrame, frames=int(epochs/EPOCHS_PER_FRAME), interval=60,
                fargs=(myLearnSystem, initial_states[0].to(device), real_trajectories[:,0,:].to(device), 
                        epochs, maxNumSamples, numAgents,device, simulation_time, step_size, path_checkpoints, numSamplesLog))
    ani.save(path_results+"/video.mp4", writer='ffmpeg')
    torch.cuda.empty_cache()
    del ani
    print("Video generated successfully")

    # Best epoch
    print("\nFinding best configuration along the training...")
    learned_trajectories = torch.zeros(true_nSamples, numVal, 8 * numAgents)
    loss_evo = []
    with torch.no_grad():
        for epoch_save in range(0, epochs, VAL_PERIOD):
            myLearnSystem.load_state_dict(torch.load(path_checkpoints+"/epoch_"+str(epoch_save)+".pth", map_location=device))
            learned_trajectories = myLearnSystem.forward(initial_states.to(device), simulation_time, step_size)
            loss = L2_loss((learned_trajectories[:, :, :4 * numAgents].reshape(-1, 4 * numAgents)).to(device), real_trajectories[:, :, :4 * numAgents].reshape(-1, 4 * numAgents).to(device))
            loss_evo.append(loss.detach().cpu().numpy())
            print(".",end="", flush=True)

    bestL2 = min(loss_evo)
    inxedBestEpoch = loss_evo.index(bestL2)
    bestEpoch = inxedBestEpoch*VAL_PERIOD
    print("\nBestL2 = ", bestL2, " in epoch: ", inxedBestEpoch*VAL_PERIOD)

    # Loss evolution
    torch.save(loss_evo, path_results+'/error_evo.pth')
    plt.clf()
    plt.yscale('log')
    plt.plot(range(0, epochs, VAL_PERIOD), loss_evo, marker='o', linestyle='solid', color='blue')
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Loss $\mathcal{L}$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.savefig(path_results+"/error_evo.png")
    
    # Evaluation
    # ==========
    # Load test data
    test_data = torch.load(path_testData).to(myLearnSystem.device)
    initial_states = test_data[0,:,:]
    real_trajectories = test_data
    print("Test trajectories loaded successfully")

    # Load best model from validation
    myLearnSystem.load_state_dict(torch.load(path_checkpoints+"/epoch_"+str(bestEpoch)+".pth", map_location=device))
    with torch.no_grad():
        learned_trajectories = myLearnSystem.forward(initial_states.to(device), simulation_time, step_size)
    print("\nCheckpoint loaded for evaluation: ", bestEpoch)

    # Plot qualitative
    plt.clf()
    plt.grid(False)
    plotFrame(learned_trajectories[:,0,:].to(device), real_trajectories[:,0,:].to(device), epochs, bestEpoch, maxNumSamples, numAgents, numSamplesLog)
    plt.savefig(path_results+"/best_trajectories.png")

    # Quantitative analysis
    learn_avg_dist = 0
    learn_min_dist = 0
    real_avg_dist  = 0
    real_min_dist  = 0
    learn_smoothness = 0
    real_smoothness = 0
    area_trajectories = 0
    for i in range(numTests):
        
        learn_avg_dist += avgAgentDist(learned_trajectories[:,i,:], numAgents)
        learn_min_dist += minAgentDist(learned_trajectories[:,i,:], numAgents)
        learn_smoothness += getSmoothness(learned_trajectories[:,i,:], numAgents, step_size)

        real_avg_dist  += avgAgentDist(real_trajectories[:,i,:], numAgents)
        real_min_dist  += minAgentDist(real_trajectories[:,i,:], numAgents)
        real_smoothness += getSmoothness(real_trajectories[:,i,:], numAgents, step_size)
        area_trajectories += getAreaBetweenCurves(learned_trajectories[:,i,:].detach().cpu().numpy(), real_trajectories[:,i,:].detach().cpu().numpy(), numAgents)
    learn_avg_dist /= numTests
    learn_min_dist /= numTests
    real_avg_dist /= numTests
    real_min_dist /= numTests
    learn_smoothness /= numTests
    real_smoothness /= numTests
    area_trajectories /= numTests
    loss_test = L2_loss((learned_trajectories[:, :, :4 * numAgents].reshape(-1, 4 * numAgents)).to(device), real_trajectories[:, :, :4 * numAgents].reshape(-1, 4 * numAgents).to(device))
    
    print("L2 loss     : ", float(loss_test))
    print("Average Dist: Learned = ", float(learn_avg_dist), " - Real = ", float(real_avg_dist))
    print("Minimal Dist: Learned = ", float(learn_min_dist), " - Real = ", float((real_min_dist)))
    print("Smoothness  : Learned = ", float(learn_smoothness), " - Real = ", float((real_smoothness)))
    print("Area        : ", float(area_trajectories))

    with open(path_results+"/info.txt", 'w') as file:
        text = "Best epoch: "+str(bestEpoch)+"\n" + "L2 loss: "+str(float(loss_test)) + "\nAvg dist error: "\
            +str(float(learn_avg_dist)-float(real_avg_dist)) + "\nMin dist error: " \
            +str(float(learn_min_dist)-float(real_min_dist)) + "\nSmoothness error: "\
            +str(float(learn_smoothness)-float(real_smoothness)) + "\nArea between curves: "\
            +str(float(area_trajectories))

        file.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training evaluation')
    parser.add_argument('--policy', type=str, nargs=1, help='control policy name')
    parser.add_argument('--architecture', type=str, nargs=1, help='architecture name')
    parser.add_argument('--nAttLayers', type=int, nargs=1, help='number of attention layers used')
    parser.add_argument('--nEpochs', type=int, nargs=1, help='number of epochs during trainings')
    parser.add_argument('--numTrain', type=int, nargs=1, help='number of instances for training')
    parser.add_argument('--numVal', type=int, nargs=1, help='number of instances for validation')
    parser.add_argument('--numTests', type=int, nargs=1, help='number of instances for testing')
    parser.add_argument('--maxNumSamples', type=int, nargs=1, help='maximum number of samples per instance (min is 5)')
    parser.add_argument('--seed_data', type=int, nargs=1, help='seed used for generating the data')
    parser.add_argument('--seed_train', type=int, nargs=1, help='seed to generate reproducible data')
    parser.add_argument('--numAgents', type=int, nargs=1, help='number of agents for evaluation')
    parser.add_argument('--numAgents_train', type=int, nargs=1, help='number of agents during training')
    parser.add_argument('--interval_policy', type=str, nargs=1, help='either fixed, linear or modulated')
    parser.add_argument('--interval_parameter', type=float, nargs=1, help='parameter: c, b or maxValue according to the policy')
    parser.add_argument('--increment_policy', type=str, nargs=1, help='either fixed, linear or modulated')
    parser.add_argument('--increment_parameter', type=float, nargs=1, help='parameter: c, b or maxValue according to the policy')

    args = parser.parse_args()
    main(args.policy[0], args.architecture[0], args.nAttLayers[0], args.nEpochs[0], args.numTrain[0], args.numVal[0], args.numTests[0], 
         args.maxNumSamples[0], args.seed_data[0], args.seed_train[0], args.numAgents[0], args.numAgents_train[0], args.interval_policy[0],
         args.interval_parameter[0], args.increment_policy[0] ,args.increment_parameter[0])
