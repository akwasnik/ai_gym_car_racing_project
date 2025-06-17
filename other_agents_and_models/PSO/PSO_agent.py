import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import pyswarms as ps

# Dyskretne akcje: skręt, gaz, hamulec
acts = [
    np.array([0.0, 0.0, 0.0], np.float32),
    np.array([0.0, 1.0, 0.0], np.float32),
    np.array([0.0, 0.0, 0.8], np.float32),
    np.array([-1.0, 0.0, 0.0], np.float32),
    np.array([1.0, 0.0, 0.0], np.float32),
    np.array([-1.0, 0.5, 0.0], np.float32),
    np.array([1.0, 0.5, 0.0], np.float32),
    np.array([-1.0, 1.0, 0.0], np.float32),
    np.array([1.0, 1.0, 0.0], np.float32),
]

# CNN do CarRacing
class DQN(nn.Module):
    def __init__(self, n_actions=len(acts)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 21 * 21, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    def forward(self, x): return self.net(x.float() / 255.)

# Zamiana wag <-> wektor
def flatten(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])

def inject(vec, model):
    idx = 0
    for p in model.parameters():
        size = p.numel()
        reshaped = torch.tensor(vec[idx:idx+size].reshape(p.shape), dtype=torch.float32)
        p.data.copy_(reshaped)
        idx += size

# Ocena wielu cząstek naraz
def fitness(X):
    rewards = []
    for i in range(X.shape[0]):
        net = DQN()
        inject(X[i], net)
        env = gym.make("CarRacing-v3")
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            with torch.no_grad():
                state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)
                act = net(state).argmax().item()
            obs, reward, term, trunc, _ = env.step(acts[act])
            total += reward
            done = term or trunc
        env.close()
        rewards.append(-total)  # PSO minimalizuje, więc dajemy minus
    return np.array(rewards)

# Trenowanie przez PSO
def train():
    model = DQN()
    dim = len(flatten(model))
    lb = -1.0 * np.ones(dim)
    ub =  1.0 * np.ones(dim)

    options = {'c1': 1.5, 'c2': 1, 'w': 1}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=30, dimensions=dim, options=options, bounds=(lb, ub)
    )

    print(f"Start PSO (dim={dim})...")
    best_cost, best_pos = optimizer.optimize(fitness, iters=5)


    inject(best_pos, model)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pso_pyswarms_model.pth")
    print(f"Zapisano model ➜ models/pso_pyswarms_model.pth")

# Odtwarzanie przejazdu
def play(path="models/dqn_carracing_1000.pth"):
    model = DQN()
    model.load_state_dict(torch.load(path))
    model.eval()
    env = gym.make("CarRacing-v3", render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            x = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)
            act = model(x).argmax().item()
        obs, _, term, trunc, _ = env.step(acts[act])
        done = term or trunc
    env.close()

# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        play()

