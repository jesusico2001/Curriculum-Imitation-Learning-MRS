import argparse
import torch
import matplotlib.pyplot as plt


def validationPath(trainPath):
    return trainPath.replace("trainLosses", "valLosses")

def valScalPath(trainPath):
    return trainPath.replace("trainLosses", "valLosses_scalability")

def epochsPath(trainPath):
    return trainPath.replace("trainLosses", "valEpochs")

def numSamplesPath(trainPath):
    return trainPath.replace("trainLosses", "numSamplesLog")



def main(trainLosses, descriptions):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    startFrom = 1

    nModels = 0
    lossTrain = []
    lossVal = []
    lossVal_scalability = []
    epochs = []
    numSamplesLog = []
    for path in trainLosses:
        lossTrain.append(torch.load(path))
        lossVal.append(torch.load(validationPath(path)))
        lossVal_scalability.append(torch.load(valScalPath(path)))
        epochs.append(torch.load(epochsPath(path)))

        try:
            numSamplesLog.append(torch.load(numSamplesPath(path)))
        except FileNotFoundError:
            numSamplesLog.append([])

        nModels += 1

    # Crear el gráfico
    colors = plt.cm.get_cmap('hsv', nModels+2)
    plt.figure(figsize=(10,6))

    for i in range(nModels):
        plt.plot(epochs[i][startFrom:], lossTrain[i][startFrom:],  linestyle='dotted', alpha=0.7, color=colors(i+1), label=descriptions[i])
        plt.plot(epochs[i][startFrom:], lossVal[i][startFrom:], linestyle='--', alpha=0.7, color=colors(i+1), label='')
        plt.plot(epochs[i][startFrom:], lossVal_scalability[i][startFrom:], linestyle='solid', alpha=0.6, color=colors(i+1), label='')
        
        # for p in numSamplesLog[i]:
        #     plt.axvline(x=p[0], color=colors(i+1), linestyle='dashdot')
        #     plt.text(p[0], .01*i, "nS="+str(p[1]),color=colors(i+1), ha='center', va='bottom')




    # Free unused memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    plt.yscale('log')
    plt.xlabel('Iteraciones', fontsize=25)
    plt.ylabel('Error', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LEMURS training for the flocking problem')
    parser.add_argument('--trainLosses', type=str, nargs="*", help='list of paths to train losses')
    parser.add_argument('--descriptions', type=str, nargs="*", help='list of descriptios of the compared data')
    
    args = parser.parse_args()
    main(args.trainLosses, args.descriptions)
