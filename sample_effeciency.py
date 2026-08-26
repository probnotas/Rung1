import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- Experiment configuration ----------
POOL_SIZE = 5000            # size of the one big data pool
SAMPLE_SIZES = [5000, 2000, 1000, 500, 200, 100, 50]
N_SEEDS = 3                 # independently trained models per sample size
N_EVAL = 15                 # control rollouts per trained model
EVAL_STEPS = 200            # env steps per rollout
HORIZON = 20                # MPC planning horizon
N_CANDIDATES = 300          # MPC random-shooting candidates
MASTER_SEED = 0

# Prediction-accuracy experiment: a held-out test set that is never trained on.
# The pool is TEST_SIZE larger than POOL_SIZE so that the biggest sample size
# (5000) can still be drawn entirely from outside the 1000-sample test set.
TEST_SIZE = 1000
PRED_POOL_SIZE = POOL_SIZE + TEST_SIZE

torch.set_num_threads(2)

# ---------- 1. COLLECT one big pool of data ----------
def collect_data(n, seed=MASTER_SEED):
    env = gym.make("Pendulum-v1")
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
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
def train_model(states, actions, next_states, epochs=1000, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
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

_TARGET_T = torch.tensor([1.0, 0.0, 0.0])

def choose_action(model, state, acts=None):
    """Random-shooting MPC.

    Identical in behaviour to the original per-candidate Python loop: sample
    N_CANDIDATES action sequences of length HORIZON, roll each one forward
    through the learned model, sum cost(state) over the horizon, and return the
    first action of the lowest-cost sequence. The candidates are simply rolled
    out as one batch instead of one at a time, which is ~120x faster and makes
    the averaged experiment feasible. See test_planner_equivalence.py.
    """
    if acts is None:
        acts = np.random.uniform(-2, 2, size=(N_CANDIDATES, HORIZON, 1)).astype(np.float32)
    n_cand = acts.shape[0]
    sim = torch.from_numpy(np.tile(np.asarray(state, dtype=np.float32), (n_cand, 1)))
    acts_t = torch.from_numpy(np.ascontiguousarray(acts, dtype=np.float32))
    total = torch.zeros(n_cand)
    with torch.no_grad():
        for h in range(HORIZON):
            sim = model(torch.cat([sim, acts_t[:, h, :]], dim=1))
            total += ((sim - _TARGET_T) ** 2).sum(dim=1)
    return acts[int(torch.argmin(total)), 0]

# ---------- 4. TEST how well a trained model controls the pendulum ----------
def evaluate(model, steps=None, n_eval=None, eval_seeds=None):
    """Run the control test n_eval times and return (mean, std, per_run_costs).

    Each run starts from a fresh env.reset(). Passing the same eval_seeds to
    every model means all conditions face an identical set of start states,
    which removes start-state variance from between-condition comparisons.
    """
    # Read config at call time, and let eval_seeds decide the count when given,
    # so the seed array and the loop can never disagree.
    steps = EVAL_STEPS if steps is None else steps
    if n_eval is None:
        n_eval = N_EVAL if eval_seeds is None else len(eval_seeds)
    if eval_seeds is not None and len(eval_seeds) != n_eval:
        raise ValueError(f"eval_seeds has {len(eval_seeds)} entries, need {n_eval}")
    env = gym.make("Pendulum-v1")
    run_costs = []
    for i in range(n_eval):
        if eval_seeds is not None:
            state, _ = env.reset(seed=int(eval_seeds[i]))
        else:
            state, _ = env.reset()
        total_cost = 0.0
        for _ in range(steps):
            action = choose_action(model, state)
            state, _, term, trunc, _ = env.step(action)
            total_cost += cost(state)
            if term or trunc:
                state, _ = env.reset()
        run_costs.append(total_cost / steps)
    env.close()
    run_costs = np.array(run_costs)
    return run_costs.mean(), run_costs.std(), run_costs

# ---------- 5. PREDICTION ACCURACY (no MPC in the loop) ----------
def prediction_mse(model, X_test, Y_test):
    """One-step prediction MSE on a held-out set."""
    with torch.no_grad():
        return float(((model(X_test) - Y_test) ** 2).mean())


def run_prediction_experiment():
    """Measure forward-model quality directly, isolating it from the controller.

    Holds out a fixed TEST_SIZE test set that no model ever trains on, then
    trains fresh models on each sample size and scores one-step MSE on it.
    """
    t_start = time.time()
    rng = np.random.default_rng(MASTER_SEED)

    print(f"Collecting data pool ({PRED_POOL_SIZE} samples: "
          f"{POOL_SIZE} train pool + {TEST_SIZE} held-out test)...")
    S, A, S2 = collect_data(PRED_POOL_SIZE)

    # Fixed disjoint split: test indices are never available for training.
    perm = rng.permutation(PRED_POOL_SIZE)
    test_idx, train_pool_idx = perm[:TEST_SIZE], perm[TEST_SIZE:]
    assert not set(test_idx) & set(train_pool_idx), "test set leaked into train pool"
    assert len(train_pool_idx) >= max(SAMPLE_SIZES), (
        f"train pool has {len(train_pool_idx)} samples but the sweep needs "
        f"{max(SAMPLE_SIZES)}; increase PRED_POOL_SIZE"
    )

    X_test = torch.tensor(
        np.concatenate([S[test_idx], A[test_idx]], axis=1), dtype=torch.float32
    )
    Y_test = torch.tensor(S2[test_idx], dtype=torch.float32)

    # Reference point: how much error remains if you just predict "no change".
    identity_mse = float(((torch.tensor(S[test_idx], dtype=torch.float32) - Y_test) ** 2).mean())

    means, stds = [], []
    for n in SAMPLE_SIZES:
        seed_mses = []
        for s in range(N_SEEDS):
            idx = rng.choice(train_pool_idx, size=n, replace=False)
            model = train_model(S[idx], A[idx], S2[idx], seed=MASTER_SEED + 1000 * s + n)
            mse = prediction_mse(model, X_test, Y_test)
            seed_mses.append(mse)
            print(f"   n={n:>5}  seed {s + 1}/{N_SEEDS}  ->  test MSE = {mse:.6f}")
        means.append(float(np.mean(seed_mses)))
        stds.append(float(np.std(seed_mses, ddof=1)))
        print(f"{n:>5} samples -> test MSE = {means[-1]:.6f} +/- {stds[-1]:.6f}\n")

    # ---------- PLOT ----------
    plt.figure(figsize=(7, 5))
    plt.errorbar(SAMPLE_SIZES, means, yerr=stds, marker="o", capsize=4)
    plt.xlabel("Number of training samples")
    plt.ylabel("One-step prediction MSE on held-out set (lower = better)")
    plt.title("Model quality: prediction error vs. training data")
    if max(means) / max(min(means), 1e-12) > 20:
        plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("prediction_mse.png", dpi=150)
    print("Saved graph to prediction_mse.png")

    # ---------- TABLE ----------
    print(f"\nHeld-out test set: {TEST_SIZE} samples, never trained on")
    print(f"Identity baseline (predict next_state = state): MSE = {identity_mse:.6f}\n")
    print(f"{'samples':>8} {'test MSE':>12} {'std':>12}   {'vs identity':>12}")
    print("-" * 52)
    for n, m, sd in zip(SAMPLE_SIZES, means, stds):
        print(f"{n:>8} {m:>12.6f} {sd:>12.6f}   {identity_mse / m:>11.1f}x")
    print("-" * 52)
    print(f"Best/worst MSE ratio across sample sizes: {max(means) / min(means):.1f}x")
    print(f"\nPrediction experiment runtime: {(time.time() - t_start) / 60:.1f} min")
    return means, stds


# ---------- 6. RUN the control experiment ----------
def main():
    t_start = time.time()
    rng = np.random.default_rng(MASTER_SEED)

    print(f"Collecting data pool ({POOL_SIZE} samples)...")
    S, A, S2 = collect_data(POOL_SIZE)

    # Common start states shared by every model, for variance reduction.
    eval_seeds = rng.integers(0, 2**31 - 1, size=N_EVAL)

    means, stds, sems, per_size_runs = [], [], [], []

    for n in SAMPLE_SIZES:
        seed_means, all_runs = [], []
        for s in range(N_SEEDS):
            # Fresh slice of the pool per seed (whole pool when n == POOL_SIZE).
            idx = rng.choice(POOL_SIZE, size=n, replace=False)
            model = train_model(S[idx], A[idx], S2[idx], seed=MASTER_SEED + 1000 * s + n)
            m, sd, runs = evaluate(model, eval_seeds=eval_seeds)
            seed_means.append(m)
            all_runs.append(runs)
            print(f"   n={n:>5}  seed {s + 1}/{N_SEEDS}  ->  {m:.3f} +/- {sd:.3f}")

        all_runs = np.concatenate(all_runs)
        combined_mean = float(np.mean(seed_means))       # average of the per-model means
        combined_std = float(all_runs.std(ddof=1))       # spread across all rollouts
        combined_sem = combined_std / np.sqrt(all_runs.size)

        means.append(combined_mean)
        stds.append(combined_std)
        sems.append(combined_sem)
        per_size_runs.append(all_runs)
        print(f"{n:>5} samples -> avg control cost = {combined_mean:.3f} +/- {combined_std:.3f}\n")

    # ---------- 6. PLOT the result ----------
    plt.figure(figsize=(7, 5))
    plt.errorbar(SAMPLE_SIZES, means, yerr=stds, marker="o", capsize=4)
    plt.xlabel("Number of training samples")
    plt.ylabel("Average control cost (lower = better)")
    plt.title("Sample efficiency: control quality vs. training data")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.savefig("sample_efficiency.png", dpi=150)
    print("Saved graph to sample_efficiency.png")

    # ---------- 7. RESULTS TABLE ----------
    total_runs = N_SEEDS * N_EVAL
    print(
        f"\nResults  ({N_SEEDS} models x {N_EVAL} rollouts = {total_runs} runs per sample size)"
    )
    print(f"{'samples':>8} {'mean':>9} {'std':>9} {'sem':>8}   {'95% CI':>18}")
    print("-" * 58)
    for n, m, sd, se in zip(SAMPLE_SIZES, means, stds, sems):
        lo, hi = m - 1.96 * se, m + 1.96 * se
        print(f"{n:>8} {m:>9.3f} {sd:>9.3f} {se:>8.3f}   [{lo:>7.3f}, {hi:>7.3f}]")
    print("-" * 58)
    print("Error bars on the plot are std. Use the 95% CI column to judge")
    print("significance: overlapping intervals mean the difference is noise.")

    # Which neighbouring sample sizes actually separate?
    print("\nPairwise check (adjacent sizes, non-overlapping 95% CI = real):")
    for i in range(len(SAMPLE_SIZES) - 1):
        a, b = i, i + 1
        lo_a, hi_a = means[a] - 1.96 * sems[a], means[a] + 1.96 * sems[a]
        lo_b, hi_b = means[b] - 1.96 * sems[b], means[b] + 1.96 * sems[b]
        real = hi_a < lo_b or hi_b < lo_a
        verdict = "REAL " if real else "noise"
        print(
            f"  {SAMPLE_SIZES[a]:>5} vs {SAMPLE_SIZES[b]:>5}: "
            f"{means[a]:.3f} vs {means[b]:.3f}  -> {verdict}"
        )

    print(f"\nTotal runtime: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["control", "prediction", "both"],
        default="both",
        help="control = MPC control cost; prediction = held-out one-step MSE",
    )
    args = parser.parse_args()

    if args.experiment in ("prediction", "both"):
        run_prediction_experiment()
    if args.experiment in ("control", "both"):
        main()
