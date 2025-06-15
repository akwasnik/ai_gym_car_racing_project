import gymnasium as gym
import numpy as np
import pygad

# długość sekwencji akcji
N_STEPS = 1000
SEED = 42

# 1) Definiujemy listę możliwych akcji [steering, gas, brake]:
action_set = [
    np.array([ 0.0, 1.0, 0.0], dtype=np.float32),   # pełny gaz prosto
    np.array([ 0.0, 0.5, 0.0], dtype=np.float32),   # pół gazu prosto
    np.array([ 0.0, 0.0, 0.8], dtype=np.float32),   # hamulec
    np.array([-1.0, 0.0, 0.0], dtype=np.float32),   # obrót w lewo w miejscu
    np.array([ 1.0, 0.0, 0.0], dtype=np.float32),   # obrót w prawo w miejscu
    np.array([-1.0, 1.0, 0.0], dtype=np.float32),   # skręt w lewo z full gazem
    np.array([ 1.0, 1.0, 0.0], dtype=np.float32),   # skręt w prawo z full gazem
    np.array([-1.0, 0.5, 0.0], dtype=np.float32),   # skręt w lewo z pół gazem
    np.array([ 1.0, 0.5, 0.0], dtype=np.float32),   # skręt w prawo z pół gazem
]

# 2) poprawiona funkcja fitness z trzema argumentami
def fitness_func(ga_instance, solution, solution_idx):
    env = gym.make('CarRacing-v3', render_mode=None)
    obs, _ = env.reset(seed=SEED)
    total_reward = 0.0

    print(solution)

    idxs = np.round(solution).astype(int)
    for idx in idxs:
        action = action_set[idx]
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    return total_reward

# 3) gene_space: dyskretny zakres od 0 do len(action_set)-1
gene_space = {'low': 0, 'high': len(action_set)-1, 'step': 1}

ga = pygad.GA(
    num_generations       = 50,
    sol_per_pop           = 30,
    num_parents_mating    = 15,
    fitness_func          = fitness_func,  # teraz ok
    num_genes             = N_STEPS,
    gene_space            = gene_space,
    parent_selection_type = "sss",
    keep_parents          = 5,
    crossover_type        = "single_point",
    mutation_type         = "random",
    mutation_percent_genes= 5
)

# 4) uruchom ewolucję
ga.run()

# 5) najlepsze rozwiązanie
solution, solution_fitness, _ = ga.best_solution()
print(f"Najlepsze fitness: {solution_fitness}")

# Opcjonalnie: podgląd
best_idxs = np.round(solution).astype(int)
env = gym.make('CarRacing-v3', render_mode='human')
obs, _ = env.reset(seed=SEED)
for idx in best_idxs:
    obs, _, terminated, truncated, _ = env.step(action_set[idx])
    env.render()
    if terminated or truncated:
        break
env.close()
