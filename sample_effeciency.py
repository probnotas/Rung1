import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- 1. COLLECT one big pool of data ----------
def collect_data(n):
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset()
    S, A, S2 = [], [], []
    for _ in range(n):
        a = env.action_space.sample()
        obs2, _, term, trunc, _ = env.step(a)
        S.append(obs); A.append(a); S2.append(obs2)
        obs = obs2
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return np.array(S), np.array(A), np.array(S2)

# ---------- 2. TRAIN a forward model on a given amount of data ----------
def train_model(states, actions, next_states, epochs=1000):
    X = torch.tensor(np.concatenate([states, actions], axis=1), dtype=torch.float32)
    Y = torch.tensor(next_states, dtype=torch.float32)
    model = nn.Sequential(
        nn.Linear(4, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 3),
    )
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

# ---------- 3. Helpers for control ----------
def predict(model, state, action):
    x = torch.tensor(np.concatenate([state, action]), dtype=torch.float32)
    with torch.no_grad():
        return model(x).numpy()

def cost(state):
    target = np.array([1.0, 0.0, 0.0])
    return np.sum((state - target) ** 2)

def choose_action(model, state):
    best_first, best_c = None, float("inf")
    for _ in range(300):
        total, sim = 0, state.copy()
        acts = np.random.uniform(-2, 2, size=(20, 1))
        for a in acts:
            sim = predict(model, sim, a)
            total += cost(sim)
        if total < best_c:
            best_c, best_first = total, acts[0]
    return best_first

# ---------- 4. TEST how well a trained model controls the pendulum ----------
def evaluate(model, steps=200):
    env = gym.make("Pendulum-v1")
    state, _ = env.reset()
    total_cost = 0
    for _ in range(steps):
        action = choose_action(model, state)
        state, _, term, trunc, _ = env.step(action)
        total_cost += cost(state)
        if term or trunc:
            state, _ = env.reset()
    env.close()
    return total_cost / steps   # average cost: LOWER = better control

# ---------- 5. RUN the experiment ----------
print("Collecting data pool...")
S, A, S2 = collect_data(2000)

sample_sizes = [2000, 1000, 500, 200, 100, 50]
results = []

for n in sample_sizes:
    print(f"Training on {n} samples...")
    model = train_model(S[:n], A[:n], S2[:n])
    avg_cost = evaluate(model)
    results.append(avg_cost)
    print(f"   {n} samples -> avg control cost = {avg_cost:.3f}")

# ---------- 6. PLOT the result ----------
plt.figure(figsize=(7,5))
plt.plot(sample_sizes, results, marker="o")
plt.xlabel("Number of training samples")
plt.ylabel("Average control cost (lower = better)")
plt.title("Sample efficiency: control quality vs. training data")
plt.gca().invert_xaxis()
plt.grid(True)
plt.savefig("sample_efficiency.png", dpi=150)
print("Saved graph to sample_efficiency.png")
