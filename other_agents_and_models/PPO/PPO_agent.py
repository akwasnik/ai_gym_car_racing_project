import os
import argparse
import numpy as np
import gymnasium as gym
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation


class SaveOnEpisodeEndCallback(BaseCallback):
    """
    Custom callback for saving a model every N episodes.
    """
    def __init__(self, save_freq_episodes: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq_episodes
        self.save_path = save_path
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            ep_info = info.get('episode')
            if ep_info is not None:
                self.episode_count += 1
                if self.episode_count % self.save_freq == 0:
                    save_file = os.path.join(self.save_path, f"ppo_carracing_ep{self.episode_count}")
                    os.makedirs(self.save_path, exist_ok=True)
                    self.model.save(save_file)
                    if self.verbose > 0:
                        print(f"Saved model checkpoint at {save_file}.zip (Episode {self.episode_count})")
        return True


def make_env(render_mode=None):
    env = gym.make('CarRacing-v3', render_mode=render_mode)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=True)
    return env


def find_latest_model(model_dir: str, prefix: str = 'ppo_carracing') -> str:
    files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith('.zip')]
    if not files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    latest = sorted(files)[-1]
    return os.path.join(model_dir, latest)

parser = argparse.ArgumentParser(description="PPO training for CarRacing-v3 with periodic saves and resume.")
parser.add_argument('--mode', choices=['train', 'play'], default='train',
                    help="Select 'train' to train or resume training, 'play' to render using a trained model.")
parser.add_argument('--timesteps', type=int, default=1_000_000,
                    help="Total timesteps for training when starting fresh.")
parser.add_argument('--save_freq', type=int, default=30,
                    help="Save the model every N episodes during training.")
parser.add_argument('--model_dir', type=str, default='models',
                    help="Directory to save and load model checkpoints.")
parser.add_argument('--resume', action='store_true',
                    help="Resume training from the latest checkpoint in model_dir.")
parser.add_argument('--model_name', type=str, default=None,
                    help="(play mode) Filename of the model (with or without .zip) to load. If omitted, loads latest.")

args = parser.parse_args()

if args.mode == 'train':
    # Setup vectorized environments
    envs = DummyVecEnv([make_env for _ in range(8)])
    envs = VecMonitor(envs)
    envs = VecFrameStack(envs, n_stack=4)
    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    if args.resume:
        latest_model = find_latest_model(args.model_dir)
        model = PPO.load(latest_model, env=envs, device='auto')
        print(f"Resumed training from {latest_model}")
    else:
        # Initialize new model
        model = PPO(
            policy='CnnPolicy', env=envs,
            verbose=1,
            device='auto',
            n_steps=2048,
            batch_size=256,
            learning_rate=lambda progress: progress * 2.5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
    # Setup callback for periodic saving
    callback = SaveOnEpisodeEndCallback(
        save_freq_episodes=args.save_freq,
        save_path=args.model_dir,
        verbose=1
    )
    # Start or resume learning
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback
    )
    # Final save
    final_path = os.path.join(args.model_dir, 'ppo_carracing_final')
    model.save(final_path)
    print(f"Training complete. Final model saved at {final_path}.zip")

else:  # play mode
    env = DummyVecEnv([lambda: make_env(render_mode='human')])
    env = VecFrameStack(env, n_stack=4)

    if args.model_name:
        name = args.model_name if args.model_name.endswith('.zip') else args.model_name + '.zip'
        model_path = os.path.join(args.model_dir, name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Nie znaleziono pliku modelu: {model_path}")
    else:
        model_path = find_latest_model(args.model_dir)

    model = PPO.load(model_path, device='auto')
    print(f"Loaded model: {model_path}")

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        env.render()

        if (isinstance(done, (list, np.ndarray)) and done[0]) or (isinstance(done, bool) and done):
            obs = env.reset()