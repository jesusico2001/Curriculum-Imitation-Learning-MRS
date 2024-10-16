import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def plotRadars(configurations, path_results):
    metrics = ['$\mathcal{L}$', '$s$', '$d_{\mu}$', '$d_{\min}$', '$\mathcal{A}$']
    num_metrics = len(metrics)
    # Convertir etiquetas a ángulos
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()

    for i in range(num_metrics):
        angle = angles[i]
        angle += np.pi / 2
        if angle >= 2 * np.pi:
            angle -= 2 * np.pi

        angles[i] = angle
        
    angles += angles[:1]  # Cerrar el círculo


    # Dibujar cada configuración
    plotNow = False
    fig, ax = plt.subplots( subplot_kw=dict(polar=True))
    for config_name, stats in configurations.items():
        print(config_name)
        if not plotNow:
            fig, ax = plt.subplots( subplot_kw=dict(polar=True))
            plt.tight_layout()

            stats += stats[:1]  # Cerrar el círculo

            ax.fill(angles, stats, alpha=0.25, color='blue')
            ax.plot(angles, stats, linewidth=2, color='blue')

            plotNow =True

        else:
            stats += stats[:1]  # Cerrar el círculo
            ax.fill(angles, stats, alpha=0.2, color='orange')
            ax.plot(angles, stats, linewidth=1.5, color='orange')
            plt.yticks(fontsize=0)

            # Añadir etiquetas
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=50)

            # Alejar las etiquetas del círculo
            for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
                x, y = label.get_position()
                if x == (np.pi / 2):
                    label.set_position((x, y-.05))  
                else:
                    label.set_position((x, y-.15))  
                plt.savefig(path_results+"/"+config_name+".png")
                plotNow = False


def parseInfo(file):
    file.readline()
    l = file.readline()
    l2 = float(l.split(":")[1])

    l = file.readline()
    avg_dist = float(l.split(":")[1])

    l = file.readline()
    min_dist = float(l.split(":")[1])

    l = file.readline()
    smooth = float(l.split(":")[1])

    l = file.readline()
    area = float(l.split(":")[1])
    
    return [l2, smooth, avg_dist, min_dist, area]

def normalizeMetrics(metrics):
    metrics = np.array(metrics)
    metrics = np.abs(metrics) 
    
    nModels = metrics.shape[0]

    min = np.min(metrics,0)
    max = np.max(metrics,0)
   
    min = np.tile(min, (nModels, 1))
    max = np.tile(max, (nModels, 1))
    
    normalized = (max - metrics) / (max - min)
    scaled = 0.2 + normalized * (1 - 0.2)

    return scaled
def main(numAgents, evaluation_paths):
    eval_id = []
    eval_info = []
    for path in evaluation_paths:
        # Parse name
        eval_id.append(path.split("/")[-1])

        # save eval info
        with open(path+"/"+str(numAgents)+"_agents/info.txt", "r") as file:
            eval_info.append(parseInfo(file))
        print("Evaluation loaded correctly - "+path.split("/")[-1])
    # Normalize eval info 
    eval_info = normalizeMetrics(eval_info).tolist()
    print("\nEvaluation data normalized succesfully.")
    # Plot
    i = 1
    while os.path.exists("scripts/tests/radar_graphs/test"+str(i)):
        i += 1
    configurations = dict(zip(eval_id, eval_info))
    os.makedirs("scripts/tests/radar_graphs/test"+str(i))
    plotRadars(configurations, "scripts/tests/radar_graphs/test"+str(i))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Radar graphs generator that compares CL and classic.')
    parser.add_argument('--numAgents', type=int, nargs=1, help='number of agents during evaluation')
    parser.add_argument('--eval_paths', type=str, nargs="*", help='list of paths of the evaluations')

    args = parser.parse_args()
    main(args.numAgents[0], args.eval_paths)
