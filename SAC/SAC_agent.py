import os,time
import argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation

def make_env(render_mode=None):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = ResizeObservation(env, shape=(84, 84))
        env = GrayscaleObservation(env, keep_dim=True)  
        env = Monitor(env)
        return env
    return _init

parser = argparse.ArgumentParser(description="Train or play SAC on CarRacing-v3")
parser.add_argument('--mode', choices=['train', 'resume', 'play'], default='train',
                    help="train: start fresh; resume: continue training; play: render trained model")
parser.add_argument('--timesteps', type=int, default=int(1e6), help="number of timesteps for training")
parser.add_argument('--model-path', type=str, default='models/sac_carracing', help="path to save/load model")
parser.add_argument('--log-dir', type=str, default='logs/', help="directory for logs and checkpoints")
args = parser.parse_args()
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
# Prepare vectorized environment
if args.mode in ['train', 'resume']:
    env = DummyVecEnv([make_env(render_mode=None)])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    # Callbacks: save checkpoints and evaluate
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,    # save every 100k steps
        save_path=args.log_dir,
        name_prefix='sac_carracing'
    )
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=900,   # stop if average reward >=900
        verbose=1
    )
    eval_env = DummyVecEnv([make_env(render_mode=None)])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=os.path.join(args.log_dir, 'best_model'),
        log_path=os.path.join(args.log_dir, 'eval_logs'),
        eval_freq=50000,
        deterministic=True,
        render=False
    )
    
if args.mode == 'train':
    # Initialize new model
    model = SAC(
        'CnnPolicy',
        env,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=3e-4,
        buffer_size=int(2e5),
        batch_size=256,
        tau=0.005,
        gamma=0.99,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    model.save(args.model_path)

elif args.mode == 'resume':
    # Load existing model and continue
    model = SAC.load(
        args.model_path,
        env=env,
        tensorboard_log=args.log_dir,
        verbose=1
    )
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=False,
        callback=[checkpoint_callback, eval_callback]
    )
    model.save(args.model_path)

elif args.mode == 'play':
    play_env = DummyVecEnv([make_env(render_mode='human')])
    play_env = VecTransposeImage(play_env)
    play_env = VecFrameStack(play_env, n_stack=4)

    model = SAC.load(args.model_path, device='auto')

    obs = play_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = play_env.step(action)

        # dones jest wektorem (n_envs,). Przy DummyVecEnv n_envs == 1.
        done = dones[0]
        time.sleep(0.02)

    play_env.close()

