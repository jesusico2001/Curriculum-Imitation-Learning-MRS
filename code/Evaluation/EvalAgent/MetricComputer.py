import os
import torch
import similaritymeasures
import matplotlib.pyplot as plt
from Evaluation.EvalAgent.EvalAgent import EvalAgent
class MetricComputer(EvalAgent):
    """
    EvalAgent que integra validación de checkpoints y cálculo de métricas avanzadas
    sobre datasets de validación y test, permitiendo visualizar la evolución de todas
    las métricas a lo largo de los epochs.
    """
    def __init__(self, path_config, config_changes=None):
        super().__init__(path_config, config_changes)
        self.num_tests = self.validation_batch_size  # Número de trayectorias a evaluar para cada métrica

    def checkExistingEvaluation(self):
        try:
            os.makedirs(self.path_manager.getPathEvaluation())
        except FileExistsError:
            
            act = input("There is already an evaluation for this configuration. Do you want to overwrite it? (Y/N)\n").lower()
            if act == "y":
                pass
            else:
                print("Aborting evaluation...\n")
                exit(0)


    def metricsEvolution(self, dataset_type="test", numAgents=None, checkExisting=True, simulated_noise=None):
        if checkExisting:
            self.checkExistingEvaluation()

        if simulated_noise is not None:
            old_noise = self.learn_system.task.robot_obs_noise
            self.learn_system.task.robot_obs_noise = simulated_noise
            print(f"Simulated noise set to {simulated_noise}")
            
        if numAgents is not None:
            oldNa = self.learn_system.task.numAgents
            self.learn_system.task.numAgents = numAgents
        else:
            oldNa = None
            numAgents = self.learn_system.task.numAgents

        # Cargar datos
        real_rewards = self.dataset_builder.BuildArbitraryNumAgents("test", numAgents, data="rewards")
        data = self.dataset_builder.BuildArbitraryNumAgents(dataset_type, numAgents)
        real_rewards = real_rewards[:, :self.num_tests, :]  # Limitar a las primeras num_tests recompensas
        real_trajectories = data[:, :self.num_tests, :]  # Trajectories reales

        print(f"{dataset_type.capitalize()} trajectories loaded successfully")

        # Calcular métricas de las trayectorias reales solo una vez
        real_metrics = self.compute_trajectory_metrics(real_trajectories, real_rewards)

        # Epochs a evaluar
        epochs = torch.load(self.path_manager.getPathHistory()+f"/val_epochs.pth")

        # Historial de métricas aprendidas
        metrics_history = {
            "L2_loss": [],
            "L2_aux": [],
            "L2_inliers": [],
            "L2_outliers": [],
            "mean_pos_error": [],
            "frechet_distance": [],
            "smoothness": [],
            "mean_reward": [],
            "n_task_completed": [],
            "num_collisions": [],
            "real_metrics": real_metrics  # Guardamos todas las métricas reales aquí
        }

        for epoch in epochs:
            print(f"Evaluating epoch {epoch}...")
            metrics = self.compute_metrics_for_epoch(epoch, real_trajectories)
            for key in metrics:
                metrics_history[key].append(metrics[key])

        if simulated_noise is not None:
            self.learn_system.task.robot_obs_noise = old_noise            
            path = self.path_manager.getPathEvaluation() + f"/metrics_{dataset_type}_{numAgents}agents_{simulated_noise}noise.pth"           
        else:
            path = self.path_manager.getPathEvaluation() + f"/metrics_{dataset_type}_{numAgents}agents.pth"
        torch.save(metrics_history, path)
        print(f"Métricas guardadas en {path}")

        if oldNa is not None:
            self.learn_system.task.numAgents = oldNa
            

    def compute_metrics_for_epoch(self, epoch, real_trajectories):
        self.load_checkpoint(epoch)
        init_states = real_trajectories[0, :, :]  # Usar el primer estado como inicial
        with torch.no_grad():
            trajectories, _, rewards = self.learn_system.forward(
                init_states, self.learn_system.task.episode_difficulty
            )
        metrics_learned = self.compute_trajectory_metrics(trajectories, rewards)
        metrics_learned["frechet_distance"] = self.frechetDistance(trajectories, real_trajectories)  # L2 loss with respect to the initial state
        metrics_learned["mean_pos_error"] = self.mean_pos_error(trajectories, real_trajectories)
        metrics_learned["L2_loss"] = self.L2_loss(trajectories, real_trajectories)  # L2 loss with respect to the initial state
        
        difficulties = torch.ones(self.validation_batch_size, dtype=int) * self.learn_system.task.episode_difficulty
        metrics_learned["L2_aux"] = self.L2_loss_train(self.learn_system.task.getRobotStates(trajectories), self.learn_system.task.getRobotStates(real_trajectories), difficulties)  # L2 loss with respect to the initial state
        print("L2: ", metrics_learned["L2_loss"].item(), " - L2_aux: ", metrics_learned["L2_aux"].item())
        
        L2_inliers, L2_outliers = self.L2_loss_outliers(trajectories, real_trajectories)
        metrics_learned["L2_inliers"] = L2_inliers
        metrics_learned["L2_outliers"] = L2_outliers
        return metrics_learned

    def compute_trajectory_metrics(self, trajectories, rewards):
        numTests = trajectories.shape[1] 
        smoothness = 0
        task_completed = 0
        for i in range(numTests):
            task_completed += self.learn_system.task.numCompletedTasks(trajectories[:, i, :]).detach().cpu()

        n_collisions = self.numberOfCollisions(trajectories)
        task_completed = float(task_completed) / numTests
        smoothness += self.getSmoothness(trajectories)
        mean_reward = torch.mean(rewards)
        return {
            "smoothness": float(smoothness),
            "n_task_completed": float(task_completed),
            "mean_reward": mean_reward,
            "num_collisions": n_collisions
        }

    # ========== MÉTRICAS ==========

    def getSmoothness(self, trajectories):
        ax, ay = self.getAccelerations(trajectories)
        a_sum = (torch.sum(ax ** 2) + torch.sum(ay ** 2)) 
        smoothness = a_sum / (trajectories.shape[0] * trajectories.shape[1] * self.learn_system.task.numAgents)
        return smoothness

    def getAccelerations(self, trajectories):
        na = self.learn_system.task.numAgents
        step_size = self.learn_system.task.simulation_step
        states = self.learn_system.task.getRobotStates(trajectories)
        vx = states[:, :, 2 * na::2]
        vy = states[:, :, 2 * na + 1::2]
        ax_central = (vx[2:, :, :] - vx[:-2, :, :]) / (2 * step_size)
        ay_central = (vy[2:, :, :] - vy[:-2, :, :]) / (2 * step_size)
        ax_start = (vx[1] - vx[0]) / step_size
        ax_end = (vx[-1] - vx[-2]) / step_size
        ay_start = (vy[1] - vy[0]) / step_size
        ay_end = (vy[-1] - vy[-2]) / step_size
        ax = torch.cat((ax_start.unsqueeze(0), ax_central, ax_end.unsqueeze(0)))
        ay = torch.cat((ay_start.unsqueeze(0), ay_central, ay_end.unsqueeze(0)))
        return ax, ay

    def getDistances(self, trajectories):
        na = self.learn_system.task.numAgents
        ns = trajectories.shape[0]
        batch_size = trajectories.shape[1]
        
        states = self.learn_system.task.getRobotStates(trajectories)

        positions = states[:, :, :2 * na].reshape(ns, batch_size, na, 2)
        x = positions[:, :, :, 0]
        x1 = torch.kron(x, torch.ones((1, 1, na), device=trajectories.device))
        x2 = x.repeat(1, 1, na)
        y = positions[:, :, :, 1]
        y1 = torch.kron(y, torch.ones((1, 1, na), device=trajectories.device))
        y2 = y.repeat(1, 1, na)
        x_diff = abs(x1 - x2).reshape(ns, batch_size, na, na)
        y_diff = abs(y1 - y2).reshape(ns, batch_size, na, na)
        return torch.sqrt(pow(x_diff, 2) + pow(y_diff, 2))

    def numberOfCollisions(self, trajectories):
        na = self.learn_system.task.numAgents
        r = self.learn_system.task.robot_radius

        dists = self.getDistances(trajectories)
        dists[:,:, range(na), range(na)] = float('inf')
        # Count collisions (distance < 2*radius)
        collisions = (dists <= 2 * r + 0.005)
        # Each collision is counted twice (i,j) and (j,i), so divide by 2
        nCollisions = collisions.sum().item() // 2
        collisions_per_agent = nCollisions / (trajectories.shape[1] * na)
        return collisions_per_agent

    def frechetDistance(self, trajectories1, trajectories2):
        na = self.learn_system.task.numAgents
        states1 = self.learn_system.task.getRobotStates(trajectories1)
        states2 = self.learn_system.task.getRobotStates(trajectories2)

        total = 0.0
        for b in range(self.num_tests):
            pos1 = states1[:, b, :2 * na].reshape(-1, na, 2).to("cpu")
            pos2 = states2[:, b, :2 * na].reshape(-1, na, 2).to("cpu")
            
            for i in range(na):
                total += similaritymeasures.frechet_dist(pos1[:, i, :], pos2[:, i, :])
        total = total / (self.num_tests * na)
        return total
    
    def getAreaBetweenCurves(self, trajectories1, trajectories2):
        na = self.learn_system.task.numAgents
        states1 = self.learn_system.task.getRobotStates(trajectories1)
        states2 = self.learn_system.task.getRobotStates(trajectories2)

        total = 0.0
        for b in range(self.num_tests):
            pos1 = states1[:, b, :2 * na].reshape(-1, na, 2)
            pos2 = states2[:, b, :2 * na].reshape(-1, na, 2)
            
            for i in range(na):
                total += similaritymeasures.area_between_two_curves(pos1[:, i, :], pos2[:, i, :])
        total = total / (self.num_tests * na)
        return total

    def L2_loss(self, trajectories1, trajectories2):
        states1 = self.learn_system.task.getRobotStates(trajectories1)
        states2 = self.learn_system.task.getRobotStates(trajectories2)

        return (states1 - states2).pow(2).mean()
    
    def L2_loss_outliers(self, trajectories_learn, trajectories_real):
        outliers = self.learn_system.task.flagBadTrajectories(trajectories_learn)
        out_mask = outliers.unsqueeze(0).repeat(trajectories_real.shape[0], 1, 1)
        out_mask = torch.kron(out_mask, torch.ones((1, 1, 2), dtype=torch.bool, device=self.device)).repeat(1, 1, 2)

        states1 = self.learn_system.task.getRobotStates(trajectories_learn)
        states2 = self.learn_system.task.getRobotStates(trajectories_real)

        loss_good = (states1[~out_mask] - states2[~out_mask]).pow(2).mean()
        loss_outliers = (states1[out_mask] - states2[out_mask]).pow(2).mean()
        return loss_good, loss_outliers
    
    def mean_pos_error(self, trajectories1, trajectories2):
        pos1 = trajectories1[:,:,self.learn_system.task.feature_index["robot_positions"]]
        pos1 = pos1.reshape(pos1.shape[0], pos1.shape[1], self.learn_system.task.numAgents, 2)

        pos2 = trajectories2[:,:,self.learn_system.task.feature_index["robot_positions"]]
        pos2 = pos2.reshape(pos2.shape[0], pos2.shape[1], self.learn_system.task.numAgents, 2)

        pos_error = torch.linalg.norm(pos1 - pos2, dim=3)
        
        return pos_error.mean()
    
