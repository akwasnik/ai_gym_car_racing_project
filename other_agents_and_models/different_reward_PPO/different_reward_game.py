import os, argparse, gym, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# monkey-patch dla numpy.bool8 (Gym ≥0.26)
if not hasattr(np, "bool8"): np.bool8 = np.bool_

class SpeedRewardWrapper(gym.Wrapper):
    def __init__(self, env, coef=0.2):
        super().__init__(env); self.coef = coef
    def step(self, a):
        a = [float(x) for x in a]
        o, r, term, trunc, info = self.env.step(a)
        v = self.env.unwrapped.car.hull.linearVelocity
        bonus = self.coef * (v.x**2 + v.y**2)**.5
        return o, r + bonus, term, trunc, info
    def reset(self, **kw): return self.env.reset(**kw)

class EpisodeCheckpointCallback(BaseCallback):
    def __init__(self, freq, path, prefix, verbose=0):
        super().__init__(verbose)
        self.freq, self.path, self.prefix, self.ep = freq, path, prefix, 0
        os.makedirs(path, exist_ok=True)
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep += 1
                if self.ep % self.freq == 0:
                    fn = f"{self.path}/{self.prefix}_ep{self.ep}"
                    self.model.save(fn)
                    if self.verbose: print(f"[CB] saved {fn}")
        return True

def train(path, ts):
    env = DummyVecEnv([lambda: SpeedRewardWrapper(gym.make("CarRacing-v2"))])
    cb  = EpisodeCheckpointCallback(60, "checkpoints", "ppo", verbose=1)
    model = PPO.load(path, env) if os.path.exists(path + ".zip") \
            else PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=ts, callback=cb)
    model.save(path)

def play(path):
    env = SpeedRewardWrapper(gym.make("CarRacing-v2", render_mode="human"))
    model = PPO.load(path)
    obs, done = env.reset(), False
    while not done:
        a, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(a)
        env.render()
    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train","play"], default="train")
    p.add_argument("--model-path", default="ppo_carracing")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    args = p.parse_args()
    if args.mode == "train":
        train(args.model_path, args.timesteps)
    else:
        play(args.model_path)
