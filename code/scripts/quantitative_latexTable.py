import numpy as np

def main():
    architectures = ["MLP", "GNN", "LEMURS"]
    tasks = ["FS", "TVS", "Flocking"]

    pathEnd_noCL = "5_42_42_fixed_0.0_fixed_0.0/4_agents/info.txt"
    pathEnd_CL = "50_42_42_fixed_800.0_fixed_1.0/4_agents/info.txt"

    text = "Task+Policy & CL & $\mathcal{L}$ & $\mathcal{A}$ & $\\tilde{s}$ & $\\tilde{d}_{\mu}$ & $\\tilde{d}_{\min}$\\\\\n\hline\n"
    for architecture in architectures:
        for task in tasks:
            path_init = "evaluation/results/"+task+"4"+architecture+"_3_40000_20000_2000_"
            pathNoCL = path_init + pathEnd_noCL
            pathCL = path_init + pathEnd_CL
            
            with open(pathNoCL, 'r') as fileNoCL, open(pathCL, 'r') as fileCL:
                noCL = fileNoCL.readlines()
                CL = fileCL.readlines()  
                
                l2 = [float(noCL[1].split(":")[1]), float(CL[1].split(":")[1])]
                A =  [float(noCL[5].split(":")[1]), float(CL[5].split(":")[1])]
                s = [float(noCL[4].split(":")[1]), float(CL[4].split(":")[1])]
                dmu = [float(noCL[2].split(":")[1]), float(CL[2].split(":")[1])]
                dmin = [float(noCL[3].split(":")[1]), float(CL[3].split(":")[1])]
                metrics = [l2, A, s, dmu, dmin]
                bolds = []
                for metric in metrics:
                    bolds.append(abs(metric[1]) <= abs(metric[0]))
                    
                textNoCL = task+"+"+architecture+finaltext(False, metrics, bolds)
                textCL = task+"+"+architecture+finaltext(True, metrics, bolds)
                text += textNoCL+textCL+"\hline\n"
    print(text)
    
def finaltext(isCL, metrics, bolds):
    metrics = abs(np.array(metrics))
    
    text = " & "
    if isCL:
        text += "Yes"
        metrics = metrics[:,1]
    else:
        text += "No"
        bolds = np.logical_not(bolds)
        metrics = metrics[:,0]

    i=0
    for metric in metrics:
        text += " & $"
        if bolds[i]:
            text += "\mathbf{%.4f}$" % metric
        else:
            text += "%.4f$" % metric
        i += 1
    text += "\\\\\n"
    return text        

if __name__ == "__main__":
    main()