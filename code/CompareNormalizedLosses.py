import argparse
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import EvalTraining

def main(trainLosses, descriptions):
    nModels = 0
    loss_evo = []
    epochs = []
    for path in trainLosses:
        model_loss = torch.load(path)
        loss_evo.append(model_loss)
        epochs.append(len(model_loss)*EvalTraining.VAL_PERIOD)
        nModels += 1

    # Crear el gráfico
    colors = plt.cm.get_cmap('hsv', nModels+2)
    plt.figure(figsize=(14,9))
    plt.yscale('log')
    for i in range(nModels):
        plt.plot(range(0, epochs[i], EvalTraining.VAL_PERIOD), loss_evo[i], marker='o', linestyle='solid', color=colors(i+1), label=descriptions[i])
    plt.xlabel('Iterations', fontsize=38)
    plt.ylabel('$\mathcal{L}(\mathcal{D}_K, \overline{\mathcal{D}}_K)$', fontsize=38)

    # plt.gca().get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    # plt.gca().xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    # plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.gca().xaxis.set_major_locator(mtick.MaxNLocator(nbins=7))

    plt.legend(fontsize=28)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.grid(True)
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LEMURS training for the flocking problem')
    parser.add_argument('--trainLosses', type=str, nargs="*", help='list of paths to train losses')
    parser.add_argument('--descriptions', type=str, nargs="*", help='list of descriptios of the compared data')
    
    args = parser.parse_args()
    main(args.trainLosses, args.descriptions)
