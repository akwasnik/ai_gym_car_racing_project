import os
import argparse
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, RecordVideo


def make_env(render_mode=None):
    env = gym.make('CarRacing-v3', render_mode=render_mode)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=True)
    return env


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


class PlotAndVideoCallback(BaseCallback):
    """
    Callback for plotting the learning curve and saving videos of high-reward episodes.
    """
    def __init__(self, save_path: str, reward_threshold: float = 932.0, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.rew_thresh = reward_threshold
        self.episode_rewards = []
        self.episode_count = 0

        # Make video folder
        self.video_folder = os.path.join(self.save_path, 'videos')
        os.makedirs(self.video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            ep_info = info.get('episode')
            if ep_info is not None:
                ep_rew = ep_info['r']
                self.episode_count += 1
                self.episode_rewards.append(ep_rew)

                # If the episode reward exceeds threshold, record a video
                if ep_rew >= self.rew_thresh:
                    vid_env = gym.make('CarRacing-v3', render_mode='rgb_array')
                    vid_env = ResizeObservation(vid_env, (84, 84))
                    vid_env = GrayscaleObservation(vid_env, keep_dim=True)
                    vid_env = RecordVideo(
                        vid_env,
                        self.video_folder,
                        episode_trigger=lambda eid: True,
                        name_prefix=f"ep{self.episode_count}_rew{int(ep_rew)}",
                        video_length=0
                    )

                    obs, _ = vid_env.reset()
                    frame_buf = deque([obs] * 4, maxlen=4)
                    done = False

                    while not done:
                        # stack and transpose to (4,84,84)
                        hw4 = np.concatenate(list(frame_buf), axis=2)
                        obs_stack = np.transpose(hw4, (2, 0, 1))

                        action, _ = self.model.predict(obs_stack, deterministic=True)
                        obs, reward, terminated, truncated, _ = vid_env.step(action)

                        frame_buf.append(obs)
                        done = terminated or truncated

                    vid_env.close()

                    if self.verbose > 0:
                        print(f"Recorded video for episode {self.episode_count} with reward {ep_rew}")
        return True

    def _on_training_end(self) -> None:
        # Plot and save the learning curve
        plt.figure()
        plt.plot(self.episode_rewards, label='Episode Reward')

        # Compute and plot rolling average
        window = max(1, len(self.episode_rewards) // 20)
        rolling = np.convolve(self.episode_rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(self.episode_rewards)), rolling, label='Rolling Avg')

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Learning Curve')
        plt.legend()

        plot_path = os.path.join(self.save_path, 'learning_curve.png')
        plt.savefig(plot_path)
        if self.verbose > 0:
            print(f"Saved learning curve plot at {plot_path}")


class ProgressCallback(BaseCallback):
    """
    Callback for printing training progress every N timesteps.
    """
    def __init__(self, total_timesteps: int, print_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        # self.num_timesteps is updated inside BaseCallback
        if self.num_timesteps % self.print_freq == 0:
            remaining = self.total_timesteps - self.num_timesteps
            if remaining < 0:
                remaining = 0
            print(f"[Progress] Timesteps {self.num_timesteps}/{self.total_timesteps} — {remaining} remaining")
        return True


def find_latest_model(model_dir: str, prefix: str = 'ppo_carracing') -> str:
    files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith('.zip')]
    if not files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    latest = sorted(files)[-1]
    return os.path.join(model_dir, latest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PPO training for CarRacing-v3 with plots, videos, and progress prints."
    )
    parser.add_argument('--mode', choices=['train', 'play'], default='train',
                        help="Select 'train' to train or resume training, 'play' to render using a trained model.")
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                        help="Total timesteps for training when starting fresh.")
    parser.add_argument('--save_freq', type=int, default=500,
                        help="Save the model every N episodes during training.")
    parser.add_argument('--model_dir', type=str, default='models',
                        help="Directory to save and load model checkpoints and outputs.")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from the latest checkpoint in model_dir.")
    parser.add_argument('--model_name', type=str, default=None,
                        help="(play mode) Filename of the model (with or without .zip) to load. If omitted, loads latest.")
    parser.add_argument('--print_freq', type=int, default=10000,
                        help="How many timesteps between progress prints.")

    args = parser.parse_args()

    if args.mode == 'train':
        # Setup vectorized environments
        envs = DummyVecEnv([make_env for _ in range(8)])
        envs = VecMonitor(envs)
        envs = VecFrameStack(envs, n_stack=4)
        os.makedirs(args.model_dir, exist_ok=True)

        save_cb = SaveOnEpisodeEndCallback(
            save_freq_episodes=args.save_freq,
            save_path=args.model_dir
        )
        plot_vid_cb = PlotAndVideoCallback(
            save_path=args.model_dir,
            reward_threshold=932
        )
        prog_cb = ProgressCallback(
            total_timesteps=args.timesteps,
            print_freq=args.print_freq
        )
        callback = CallbackList([save_cb, plot_vid_cb, prog_cb])

        if args.resume:
            latest_model = find_latest_model(args.model_dir)
            model = PPO.load(latest_model, env=envs, device='auto')
            print(f"Resumed training from {latest_model}")
        else:
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

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback
        )
        final_path = os.path.join(args.model_dir, 'ppo_carracing_final')
        model.save(final_path)
        print(f"Training complete. Final model saved at {final_path}.zip")

    else:
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
        done = False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated
            env.render()
            if done:
                obs, _ = env.reset()
