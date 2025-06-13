import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import argparse

class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 21 * 21, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        x = x.float() / 255.0  # normalize pixel values to [0,1]
        return self.net(x)

acts = [
    np.array([0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.8], dtype=np.float32),
    np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    np.array([-1.0, 0.5, 0.0], dtype=np.float32),
    np.array([1.0, 0.5, 0.0], dtype=np.float32),
    np.array([-1.0, 1.0, 0.0], dtype=np.float32),
    np.array([1.0, 1.0, 0.0], dtype=np.float32)
]

def train(episodes=200, resume=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CarRacing-v3')

    net = DQN(len(acts)).to(device)
    tgt = DQN(len(acts)).to(device)

    # optionally resume from saved model
    if resume:
        if os.path.exists('models/dqn_carracing_200.pth'):
            net.load_state_dict(torch.load('models/dqn_carracing_200.pth', map_location=device))
            print('Resumed training from dqn_carracing_200.pth')
        else:
            print('No checkpoint found at dqn_carracing_200.pth, starting fresh training.')
    
    # initialize target network
    tgt.load_state_dict(net.state_dict())

    opt = optim.Adam(net.parameters(), lr=1e-4)
    buf = deque(maxlen=5000)

    # set starting epsilon
    eps = 0.1 if resume else 1.0

    for ep in range(episodes):
        obs, _ = env.reset()
        state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            # epsilon-greedy action selection
            if random.random() < eps:
                a = random.randrange(len(acts))
            else:
                with torch.no_grad():
                    a = net(state).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(acts[a])
            done = terminated or truncated
            next_state = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).to(device)
            buf.append((state, a, reward, next_state, done))
            state = next_state
            total_reward += reward
            step += 1

            # training step
            if len(buf) >= 1000:
                batch = random.sample(buf, 32)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                actions = torch.tensor(actions, device=device)
                rewards = torch.tensor(rewards, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = tgt(next_states).max(dim=1)[0]
                target = rewards + 0.99 * next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(q_values, target)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if step % 1000 == 0:
                tgt.load_state_dict(net.state_dict())

        eps = max(0.1, eps * 0.995)
        print(f"Episode {ep+1}/{episodes} - Reward: {total_reward:.2f}, Epsilon: {eps:.3f}")

    torch.save(net.state_dict(), f'dqn_carracing.pth')
    print('Model saved to dqn_carracing.pth')
    env.close()

def play():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CarRacing-v3', render_mode='human')
    net = DQN(len(acts)).to(device)
    net.load_state_dict(torch.load('models/dqn_carracing_200.pth', map_location=device))
    net.eval()

    obs, _ = env.reset()
    state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
    done = False

    while not done:
        with torch.no_grad():
            a = net(state).argmax(dim=1).item()
        next_obs, _, terminated, truncated, _ = env.step(acts[a])
        done = terminated or truncated
        env.render()
        state = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).to(device)

    env.close()


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'play', 'resume'], default='train',
                    help="'resume' will continue training from the last saved checkpoint")
parser.add_argument('--episodes', type=int, default=200)
args = parser.parse_args()

if args.mode == 'train':
    train(args.episodes)
elif args.mode == 'resume':
    train(args.episodes, resume=True)
else:
    play()
