import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import imageio

# Dueling Double DQN network
torch.backends.cudnn.benchmark = True  # speed optimization
class DuelingDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # shared convolutional feature extractor
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        # compute size of conv output
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            conv_out_size = self.feature(dummy).shape[1]
        # value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x.float() / 255.0
        feats = self.feature(x)
        values = self.value_stream(feats)
        advars = self.adv_stream(feats)
        # combine to Q-values
        qvals = values + (advars - advars.mean(dim=1, keepdim=True))
        return qvals

# Build full discrete action space: steer x accel x brake combinations
steers = [-1.0, 0.0, 1.0]
gases = [0.0, 0.5, 1.0]
brakes = [0.0, 0.8]
actions = [np.array([s, g, b], dtype=np.float32)
           for s in steers for g in gases for b in brakes]

# Training function with learning & loss curves, checkpointing, timing, and video recording
def train(
    episodes=800,
    batch_size=32,
    gamma=0.99,
    lr=1e-4,
    buffer_size=10000,
    target_update=1000,
    save_threshold=800,
    eval_episodes=50
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CarRacing-v3')
    net = DuelingDQN(len(actions)).to(device)
    tgt = DuelingDQN(len(actions)).to(device)
    tgt.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=lr)
    replay = deque(maxlen=buffer_size)

    eps_start, eps_end, eps_decay = 1.0, 0.1, 0.995
    eps = eps_start

    rewards_log, losses_log = [], []
    start_time = time.time()

    for ep in range(1, episodes + 1):
        ep_start = time.time()
        obs, _ = env.reset()
        state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        done = False
        total_reward = 0.0
        step_count = 0
        ep_losses = []

        while not done:
            # epsilon-greedy
            if random.random() < eps:
                action_idx = random.randrange(len(actions))
            else:
                with torch.no_grad():
                    action_idx = net(state).argmax(1).item()

            next_obs, reward, terminated, truncated, _ = env.step(actions[action_idx])
            done = terminated or truncated
            next_state = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).to(device)
            replay.append((state, action_idx, reward, next_state, done))
            state = next_state
            total_reward += reward
            step_count += 1

            # learn
            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                states, acts, rews, next_states, dones = zip(*batch)
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                acts = torch.tensor(acts, device=device)
                rews = torch.tensor(rews, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                # current Q
                q_values = net(states).gather(1, acts.unsqueeze(1)).squeeze(1)
                # Double DQN target
                with torch.no_grad():
                    next_actions = net(next_states).argmax(1)
                    next_q = tgt(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target = rews + gamma * next_q * (1 - dones)

                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_losses.append(loss.item())

                # update target
                if step_count % target_update == 0:
                    tgt.load_state_dict(net.state_dict())

        # episode end
        eps = max(eps_end, eps * eps_decay)
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        rewards_log.append(total_reward)
        losses_log.append(avg_loss)
        elapsed = time.time() - start_time
        avg_per_ep = elapsed / ep
        remaining = avg_per_ep * (episodes - ep)
        m_rem, s_rem = divmod(int(remaining), 60)
        print(f"Ep {ep}/{episodes} | Reward: {total_reward:.2f} | Loss: {avg_loss:.4f} | Eps: {eps:.3f} | Remaining ~{m_rem}m {s_rem}s")

        # save model checkpoint every 100 episodes
        if ep % 100 == 0:
            os.makedirs('models', exist_ok=True)
            ckpt_path = f'models/dueling_double_dqn_ep{ep}.pth'
            torch.save(net.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Saving final model
    os.makedirs('models', exist_ok=True)
    torch.save(net.state_dict(), 'models/dueling_double_dqn.pth')
    print("Model saved to models/dueling_double_dqn.pth")

    # Plotting
    os.makedirs('plots', exist_ok=True)
    plt.figure()
    plt.plot(rewards_log)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('plots/rewards.png')
    plt.close()

    plt.figure()
    plt.plot(losses_log)
    plt.title('Episode Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig('plots/losses.png')
    plt.close()
    print("Saved learning curves to plots/")

    # Evaluation and video recording
    os.makedirs('videos', exist_ok=True)
    eval_env = gym.make('CarRacing-v3', render_mode='rgb_array')
    for ep in range(eval_episodes):
        obs, _ = eval_env.reset()
        state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        done = False
        total_r = 0.0
        frames = []
        while not done:
            with torch.no_grad():
                a_idx = net(state).argmax(1).item()
            obs, r, terminated, truncated, _ = eval_env.step(actions[a_idx])
            done = terminated or truncated
            frame = eval_env.render()
            frames.append(frame)
            total_r += r
            state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        if total_r > save_threshold:
            path = f"videos/episode_{ep}_{int(total_r)}.mp4"
            imageio.mimwrite(path, frames, fps=30)
            print(f"Saved video: {path}")
    eval_env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=101)
    args = parser.parse_args()
    train(episodes=args.episodes)
