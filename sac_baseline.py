"""Model-free SAC baseline on Pendulum-v1, scored on the model-based cost metric.

Trains SAC and evaluates it at a series of environment-step checkpoints using
the *same* evaluation loop, cost function, episode length, start-state seeds and
episode count as the model-based experiment in sample_effeciency.py, so the two
sample-efficiency curves are directly comparable.
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC

import sample_effeciency as se
from sample_effeciency import MASTER_SEED, evaluate, get_eval_seeds

CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 20000, 50000]
N_SAC_SEEDS = 3             # independent SAC training runs, matching N_SEEDS
TARGET_COST = 6.0           # the level the model-based CEM controller reached

# The model-based result being compared against (results_cem.json).
MB_BEST_COST = 5.300
MB_BEST_SAMPLES = 2000
MB_RANGE = "5.3-5.9 at 200-5000 samples"

torch.set_num_threads(2)


def sac_action(model, state):
    """Deterministic SAC policy, shaped for the shared evaluate() loop."""
    action, _ = model.predict(state, deterministic=True)
    return action


def run_sac_baseline():
    t_start = time.time()
    eval_seeds = get_eval_seeds()
    print(f"SAC baseline: {N_SAC_SEEDS} seeds, checkpoints {CHECKPOINTS}")
    print(f"Evaluation: {se.N_EVAL} episodes x {se.EVAL_STEPS} steps, "
          f"identical cost/seeds to the model-based run\n")

    # per_seed[s][c] = mean cost for seed s at checkpoint c
    per_seed_means = np.zeros((N_SAC_SEEDS, len(CHECKPOINTS)))
    all_runs = [[] for _ in CHECKPOINTS]

    for s in range(N_SAC_SEEDS):
        model = SAC("MlpPolicy", "Pendulum-v1", seed=MASTER_SEED + s, verbose=0)
        trained = 0
        for c, ckpt in enumerate(CHECKPOINTS):
            # Train incrementally so each checkpoint is the same agent further on.
            delta = ckpt - trained
            model.learn(total_timesteps=delta, reset_num_timesteps=False,
                        log_interval=None)
            trained = ckpt

            m, sd, runs = evaluate(model, eval_seeds=eval_seeds,
                                   action_fn=sac_action)
            per_seed_means[s, c] = m
            all_runs[c].append(runs)
            print(f"   seed {s + 1}/{N_SAC_SEEDS}  {ckpt:>6} steps  ->  "
                  f"{m:.3f} +/- {sd:.3f}")
        print()

    means, stds, sems = [], [], []
    for c, ckpt in enumerate(CHECKPOINTS):
        pooled = np.concatenate(all_runs[c])
        means.append(float(per_seed_means[:, c].mean()))
        stds.append(float(pooled.std(ddof=1)))
        sems.append(float(pooled.std(ddof=1) / np.sqrt(pooled.size)))

    # ---------- PLOT ----------
    plt.figure(figsize=(7, 5))
    plt.errorbar(CHECKPOINTS, means, yerr=stds, marker="o", capsize=4,
                 label="SAC (model-free)")
    plt.axhline(MB_BEST_COST, color="tab:red", linestyle="--",
                label=f"model-based CEM best ({MB_BEST_COST:.1f} @ {MB_BEST_SAMPLES} samples)")
    plt.axhline(TARGET_COST, color="tab:gray", linestyle=":",
                label=f"target cost {TARGET_COST:.1f}")
    plt.xscale("log")
    plt.xlabel("Environment steps of training")
    plt.ylabel("Average control cost (lower = better)")
    plt.title("Model-free SAC: control quality vs. environment steps")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("sac_baseline.png", dpi=150)
    print("Saved graph to sac_baseline.png")

    # ---------- TABLE ----------
    print(f"\nSAC results  ({N_SAC_SEEDS} seeds x {se.N_EVAL} episodes = "
          f"{N_SAC_SEEDS * se.N_EVAL} runs per checkpoint)")
    print(f"{'env steps':>10} {'mean':>9} {'std':>9} {'sem':>8}   {'95% CI':>18}")
    print("-" * 60)
    for ckpt, m, sd, sem in zip(CHECKPOINTS, means, stds, sems):
        print(f"{ckpt:>10} {m:>9.3f} {sd:>9.3f} {sem:>8.3f}   "
              f"[{m - 1.96 * sem:>7.3f}, {m + 1.96 * sem:>7.3f}]")
    print("-" * 60)

    # ---------- CROSSING POINT ----------
    crossing = None
    for i, (ckpt, m) in enumerate(zip(CHECKPOINTS, means)):
        if m <= TARGET_COST:
            if i == 0:
                crossing = float(ckpt)
            else:
                # Log-linear interpolation between the bracketing checkpoints.
                x0, x1 = np.log10(CHECKPOINTS[i - 1]), np.log10(ckpt)
                y0, y1 = means[i - 1], m
                frac = (y0 - TARGET_COST) / (y0 - y1) if y0 != y1 else 1.0
                crossing = float(10 ** (x0 + frac * (x1 - x0)))
            break

    print(f"\nModel-based (CEM) reference: {MB_RANGE}")
    if crossing is None:
        print(f"SAC never reached cost {TARGET_COST:.1f} within "
              f"{max(CHECKPOINTS)} env steps.")
    else:
        print(f"SAC first reaches cost <= {TARGET_COST:.1f} at "
              f"~{crossing:,.0f} environment steps.")
        print(f"Sample ratio vs model-based ({MB_BEST_SAMPLES} samples): "
              f"~{crossing / MB_BEST_SAMPLES:.1f}x more env steps for SAC.")

    with open("results_sac.json", "w") as f:
        json.dump({
            "checkpoints": CHECKPOINTS,
            "means": means, "stds": stds, "sems": sems,
            "n_sac_seeds": N_SAC_SEEDS, "n_eval": se.N_EVAL,
            "eval_steps": se.EVAL_STEPS, "target_cost": TARGET_COST,
            "crossing_steps": crossing,
        }, f, indent=2)
    print("Saved results to results_sac.json")
    print(f"\nTotal runtime: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    run_sac_baseline()
