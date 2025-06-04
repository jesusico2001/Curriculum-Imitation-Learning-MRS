from collections import namedtuple
import queue

import torch, os
from shapely.geometry import Polygon, box
from DatasetGenerator.VMAS.VMASGenerator import VMASGenerator
from DatasetGenerator.VMAS.PathPlanning.RRTStar.rrt_star import RRTStar

ROIs = {
    # "y1": [0.0, -0.9],
    "y1": [0.0, 0.0],
    "y2": [-0.5, 0.65],
    "y3": [0.5, 0.65],
    "y4": [-0.5, -0.1],
    "y5": [-0.5, -0.6],
    "y6": [0.5, -0.1],
    "y7": [0.5, -0.6],
}

class MallGenerator(VMASGenerator):
    def __init__(self, config):
        super().__init__(config)        
        
        self.experiment = self.initExperiment()
        self.agent = self.experiment.policy
        self.task = self.agent.task

    def initExperiment(self):
        Experiment = namedtuple("PolicyData", ["policy"])
        expert = self.Expert(self.task)
        return Experiment(policy=expert)

    class Expert():
        def __init__(self, task):
            self.task = task

            self.walls, self.corners = self.initMap()

            self.hl_plan = self.load_hl_plan()
            self.potfields = self.PotentialFieldsAgent(self.task, self.walls)
            self.resetEpisode()

        def resetEpisode(self):
            self.hl_step = -1
            self.hl_done = False    

            self.waypoints = {}
            self.wp_idx = {}
            self.potfields.resetGoals()

            self.episode_step = 1

        def forward(self, obs):

            obs_tensor = torch.cat([torch.tensor(array, dtype=torch.float32) for array in obs]).unsqueeze(0)
            pos, _ = self.task.getPosVel(obs_tensor)
            pos = pos.squeeze(0)
        
            # Update the high-level step if every robot has arrived
            if not self.hl_done:
                self.hl_done = self.updateWaypoints(pos)
            
            # Update goals in potential fields
            if not self.hl_done:
                self.updateLowLevelGoals(pos)
                
            # Compute the actions using potential fields
            actions = self.potfields.forward(pos.unsqueeze(0))
            actions = actions.squeeze(0)  # Remove batch dimension

            # Reset memory if enough steps taken
            self.episode_step += 1
            if self.episode_step > self.task.episode_difficulty:
                # print(f"Episode step {self.episode_step} exceeds difficulty {self.task.episode_difficulty}. Ending episode.")
                self.resetEpisode()

            return [None, None, None, actions]
                
        # def initWaypoints(self, pos):
        #     print("Initializing waypoints for all robots. Positions:", pos)
        #     for i, pos in enumerate(pos):
        #         robot_id = i+1
        #         self.waypoints[robot_id] = [pos]
        #         self.wp_idx[robot_id] = 0
        #         self.potfields.updateGoal(robot_id, self.waypoints[robot_id][self.wp_idx[robot_id]])  # Set the first waypoint as the goal for each agent

        def updateWaypoints(self, pos):
            if len(self.waypoints) == 0 or all( [self.close_enough(pos[robot_id-1], goal) for robot_id, goal in self.hl_plan[self.hl_step]]):
                # print(f"All robots reached their goals at step {self.hl_step}. Updating waypoints.")
                self.hl_step += 1
                if self.hl_step >= len(self.hl_plan):
                    return True  # All high-level steps completed
                
                # print(f"Moving to high-level step: {self.hl_step}.")
                for robot_id, goal in self.hl_plan[self.hl_step]:
                    start = pos[robot_id-1]
                    waypoints = None
                    while waypoints is None:
                        waypoints = self.RRTWaypoints(start, goal)
                    waypoints = waypoints[::-1]

                    self.waypoints[robot_id] = waypoints
                    self.wp_idx[robot_id] = 0
                    self.potfields.updateGoal(robot_id, self.waypoints[robot_id][self.wp_idx[robot_id]])  # Set the first waypoint as the goal for each agent
                    
            return False
        
        def updateLowLevelGoals(self, pos):
            for robot_id, _ in self.hl_plan[self.hl_step]:
                goal = self.waypoints[robot_id][self.wp_idx[robot_id]]
                if self.close_enough(pos[robot_id-1], goal) and self.wp_idx[robot_id] + 1  < len(self.waypoints[robot_id]):
                    # print(f"Robot {robot_id} reached waypoint {self.wp_idx[robot_id]} at {goal}. Moving to next waypoint.")
                    self.wp_idx[robot_id] += 1
                    self.potfields.updateGoal(robot_id, self.waypoints[robot_id][self.wp_idx[robot_id]])  # Set the first waypoint as the goal for each agent
                    
        def RRTWaypoints(self, start, goal):
            # print(f"Generating waypoints from {start} to {goal} using RRT*.")
            planner = RRTStar(
                start=start.clone().detach().tolist(),
                goal=goal.clone().detach().tolist(),
                rand_area=[-1, 1],
                obstacle_list=self.corners,
                expand_dis=0.1,
                path_resolution=0.05,
                robot_radius=0.05,
                max_iter=500)

            return planner.planning(animation=False)

        def close_enough(self, pos, goal, threshold=0.1):
            # print(f"Checking if position {pos} is close enough to goal {goal}:")
            d = torch.linalg.norm(pos - torch.tensor(goal))
            # print("\tDistance:", d)
            return d < threshold

        def initMap(self):
            boxes = []
            corners = []
            with open(os.path.dirname(os.path.abspath(__file__))+"/plans/mall_map.txt", "r") as f:
                text = f.read()
                for obstacle_str in text.split("\n"):
                    if len(obstacle_str) == 0:
                        continue
                    x_1, y_1, x_2, y_2 = obstacle_str.split(",")
                    x_min, y_min = min(float(x_1), float(x_2)), min(float(y_1), float(y_2))
                    x_max, y_max = max(float(x_1), float(x_2)), max(float(y_1), float(y_2))
                    boxes.append(box(x_min, y_min, x_max, y_max))
                    corners.append((x_min, y_min, x_max, y_max))
            
            return boxes, corners

        def load_hl_plan(self):
            plan = []
            with open(os.path.dirname(os.path.abspath(__file__))+"/plans/mall.txt", "r") as f:
                text = f.read()

            for step_str in text.split("\n"):
                step = []
                for robot in step_str.split("-"):
                    id = int(robot.split("(")[0][1:])
                    goal = robot.split(",")[1].split("(")[1][:-1]
                    goal_pos = torch.tensor(ROIs[goal], dtype=torch.float32)
                    # goal_pos = goal_pos + torch.randn_like(goal_pos) * 0.2  # Ajusta el factor de dispersión si es necesario
                    step.append([id, goal_pos])
                plan.append(step)

            return plan
        

        class PotentialFieldsAgent():
            def __init__(self, task, walls):
                self.task = task
                self.walls = walls
                self.resetGoals()

            def resetGoals (self):
                self.goals = torch.tensor(ROIs["y1"]).unsqueeze(0).repeat(self.task.numAgents, 1)

            def updateGoal(self, robot_id, goal):
                # print(f"Updating goal for robot {robot_id} to {goal}")
                robot_idx = robot_id - 1
                self.goals[robot_idx, :] = torch.tensor(goal)

            def forward(self, pos):
                attraction = self.getAttractiveTerm(pos)
                repulsion_agents  = self.getAgentRepulsion(pos)
                # repulsion_wall  = self.getWallRepulsion(pos)

                forces = 5 * attraction + 0.25*repulsion_agents #+ 0.01*repulsion_wall
                
                norm = torch.linalg.norm(forces, dim=-1, keepdim=True) + 1e-8
                SQRT2 = torch.tensor(1.4142135623730951)
                for agent in range(self.task.numAgents):
                    if norm[agent] > SQRT2:
                        forces[agent] = forces[agent] / norm[agent] * SQRT2

                return forces

            def getAttractiveTerm(self, pos):
                # print("goals:", self.goals)
                return self.goals - pos.squeeze(0)
                
            def getAgentRepulsion(self, pos):
                na = self.task.numAgents
                p1 = pos.reshape(pos.shape[0], -1, 2).repeat(1, na, 1).reshape(na,na,2)
                p2 = torch.kron(pos.reshape(pos.shape[0], -1, 2), torch.ones((1, na, 1), device=pos.device)).reshape(na,na,2)
                pos_rel = p2-p1
                
                norm = torch.linalg.vector_norm(pos_rel, dim=-1) + 0.001
                mask = (norm < 0.2).unsqueeze(-1).repeat(1,1,2)
                k = 1/norm.unsqueeze(-1).repeat(1,1,2)   
                
                return torch.sum(k*mask*pos_rel,1)

            def getWallRepulsion(self, pos):
                # pos: (num_agents, 2)
                f = torch.zeros_like(pos)
                for i, p in enumerate(pos):
                    min_dist = float('inf')
                    closest_point = None
                    for wall in self.walls:
                        # Find closest point on wall to agent
                        point = wall.exterior.interpolate(wall.exterior.project((p[0].item(), p[1].item())))
                        dist = ((p[0].item() - point.x)**2 + (p[1].item() - point.y)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = point
                    if closest_point is not None and min_dist < 0.3:
                        # Repulsion force: stronger when closer
                        direction = torch.tensor([p[0].item() - closest_point.x, p[1].item() - closest_point.y])
                        norm = torch.linalg.norm(direction) + 1e-6
                        force = direction / norm * (1.0 / (min_dist + 1e-3))**2
                        f[i] = force
                return f