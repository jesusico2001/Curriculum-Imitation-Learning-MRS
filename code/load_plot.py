import torch
import matplotlib.pyplot as plt

def loadplot(path):
    file_epochs = path.replace(path.split("/")[-1], "val_epochs.pth")  

    # Cargar la lista desde el archivo
    data = torch.load(path) 
    valEpochs = torch.load(file_epochs) 

    # Crear la gráfica
    plt.plot(valEpochs, data)  
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.title("Gráfica de datos cargados")
    plt.grid()
    plt.show()

# Ruta del archivo
file_path = "saves/history/babysteps/50_steps_10_fixed_200_fixed_1_0.2/LEMURS_3_True_0.005/navigation_VMAS/4_100/15000_2000_200_42_42_False/loss_val_50_4robots.pth"  
loadplot(file_path)
