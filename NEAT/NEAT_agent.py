import os, pickle, argparse, time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import pygad
import matplotlib.pyplot as plt

# -------- akcje dyskretne (steer, gas, brake) ----------------
acts = [
    np.array([ 0.0, 0.0, 0.0], np.float32),
    np.array([ 0.0, 1.0, 0.0], np.float32),
    np.array([ 0.0, 0.0, 0.8], np.float32),
    np.array([-1.0, 0.0, 0.0], np.float32),
    np.array([ 1.0, 0.0, 0.0], np.float32),
    np.array([-1.0, 0.5, 0.0], np.float32),
    np.array([ 1.0, 0.5, 0.0], np.float32),
    np.array([-1.0, 1.0, 0.0], np.float32),
    np.array([ 1.0, 1.0, 0.0], np.float32),
]

# -------- katalogi / pliki ----------------------------------
MODELS_DIR  = "models"
VIDEOS_DIR  = "videos"
METRICS_FILE = "fitness_history.pkl"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# ----------- sieć CNN → dyskretne akcje ----------------------
class DQN(nn.Module):
    def __init__(self, n_actions: int = len(acts)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 21 * 21, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):           # [B,C,H,W], piksele 0-255
        return self.net(x.float() / 255.0)

# ----------- flatten <--> weights ----------------------------
def to_vec(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])

def to_model(vec: np.ndarray, model: nn.Module) -> None:
    idx = 0
    for p in model.parameters():
        num = p.numel()
        p.data = torch.tensor(vec[idx:idx + num].reshape(p.shape), dtype=torch.float32)
        idx += num

# ----------- pojedynczy przejazd -----------------------------
def play_episode(model: DQN, record_prefix: str | None = None) -> float:
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array" if record_prefix else None
    )
    if record_prefix:
        env = gym.wrappers.RecordVideo(
            env, VIDEOS_DIR,
            episode_trigger=lambda _: True,
            name_prefix=record_prefix
        )

    obs, _ = env.reset()
    done, total = False, 0.0

    while not done:
        state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)  # [1,3,96,96]
        with torch.no_grad():
            action_idx = model(state).argmax().item()
        obs, reward, terminated, truncated, _ = env.step(acts[action_idx])
        total += reward
        done = terminated or truncated

    env.close()
    return total

# ------------- funkcja fitness (3-arg dla PyGAD ≥ 2.20) ------
def fitness_func(ga_instance, solution, solution_idx):
    net = DQN()
    to_model(solution, net)
    return play_episode(net)

# ------------- trening GA -----------------------------------
def train(resume: bool = False):
    template_net = DQN()
    gene_count   = len(to_vec(template_net))

    # ========== inicjalizacja populacji / historii ===========
    if resume and os.path.exists(f"{MODELS_DIR}/population.pkl"):
        population = pickle.load(open(f"{MODELS_DIR}/population.pkl", "rb"))
        history    = pickle.load(open(METRICS_FILE, "rb"))
    else:
        population = np.random.uniform(-1.0, 1.0, (10, gene_count))
        history    = {"best": [], "mean": []}

    gen_times: list[float] = []        # czas każdej generacji
    gen_start = time.time()            # start pierwszej

    # ------------ callback po każdej generacji ---------------
    def on_generation(ga: pygad.GA):
        nonlocal gen_start
        g = ga.generations_completed          # 0-indexed
        total_gens = ga.num_generations

        best_vec, best_fit, _ = ga.best_solution()
        best_net = DQN();  to_model(best_vec, best_net)

        # --- checkpoint sieci + populacji
        torch.save(best_net.state_dict(),
                   f"{MODELS_DIR}/best_gen_{g:03}.pth")
        pickle.dump(ga.population,
                    open(f"{MODELS_DIR}/population.pkl", "wb"))

        # --- nagranie wideo
        play_episode(best_net, record_prefix=f"gen{g:03}")

        # --- metryki
        history["best"].append(best_fit)
        history["mean"].append(np.mean(ga.last_generation_fitness))
        pickle.dump(history, open(METRICS_FILE, "wb"))

        # --- timing & ETA
        gen_time = time.time() - gen_start
        gen_times.append(gen_time)
        avg_time = np.mean(gen_times)
        remaining = (total_gens - g - 1) * avg_time
        eta = timedelta(seconds=int(remaining))

        print(f"[Gen {g+1:>3}/{total_gens}] "
              f"time {gen_time:5.1f}s | ETA {eta} | "
              f"best {best_fit:7.1f} | mean {history['mean'][-1]:7.1f}")

        gen_start = time.time()     # reset na kolejną generację

    # ------------ instancja GA --------------------------------
    ga = pygad.GA(
        num_generations       = 200,        # 🔧 zmień według potrzeb
        sol_per_pop           = len(population),
        num_parents_mating    = 5,
        num_genes             = gene_count,
        fitness_func          = fitness_func,
        initial_population    = population,
        mutation_percent_genes= 10,
        mutation_type         = "random",
        on_generation         = on_generation
    )

    ga.run()

    # ------------------- wykres ------------------------------
    plt.figure()
    plt.plot(history["best"], label="Best fitness")
    plt.plot(history["mean"], label="Mean fitness")
    plt.xlabel("Generation"); plt.ylabel("Fitness")
    plt.title("GA training curve")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("fitness_curve.png")
    print("Wykres zapisany ➜ fitness_curve.png")

# ----------------- odtwarzanie --------------------------------
def play(checkpoint: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DQN().to(device)

    if checkpoint is None:                 # ostatni checkpoint
        files = sorted(f for f in os.listdir(MODELS_DIR) if f.startswith("best_gen"))
        if not files:
            raise FileNotFoundError("Brak checkpointów w katalogu 'models/'.")
        checkpoint = os.path.join(MODELS_DIR, files[-1])

    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.eval()

    env = gym.make("CarRacing-v3", render_mode="human")
    obs, _ = env.reset(); done = False
    while not done:
        state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            action_idx = net(state).argmax().item()
        obs, _, terminated, truncated, _ = env.step(acts[action_idx])
        done = terminated or truncated
    env.close()

# ----------------- CLI ----------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "resume", "play"], default="train")
    ap.add_argument("--checkpoint", help="ścieżka .pth do odtworzenia (tryb play)")
    args = ap.parse_args()

    if args.mode == "train":
        train(resume=False)
    elif args.mode == "resume":
        train(resume=True)
    else:
        play(args.checkpoint)
