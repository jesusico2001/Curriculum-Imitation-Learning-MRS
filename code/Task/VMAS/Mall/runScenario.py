import torch, os
from vmas import make_env
import imageio
import MallScenario  # importa tu clase personalizada

def run_random_policy():
    device = "cpu"
    # Crea el entorno
    env = make_env(
            scenario= MallScenario.Scenario(),
            num_envs=1,
            device=device,
            continuous_actions=True,
            clamp_actions=True,
            grad_enabled=True,
            terminated_truncated=True,
            # Environment specific variables
        )

    fName = os.path.dirname(os.path.abspath(__file__)) + "/simulation.gif"
    print("==================="+fName+"===================")
    frames = []
    obs = env.reset()
    for step in range(100):
        # Crea acciones aleatorias con la forma correcta
        actions = [
            torch.rand((1, 2), device=device) * 2  # cada acción in [-1, 1], shape (1, 2)
            for _ in range(env.n_agents)
        ]
        obs, rewards, dones, truncated, infos = env.step(actions)
    
        env.render()
        # frame = env.render(mode="rgb_array")
        # frames.append(frame)

        if dones.any():
            print("Done, reseteando")
            obs = env.reset()
    
    imageio.mimsave(fName, frames, duration=4)


if __name__ == "__main__":
    run_random_policy()
