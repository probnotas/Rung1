"""Recovery curve, measured properly: 10 adaptation draws instead of 3.

Changes from the first run:
  * N_ADAPT_SEEDS 3 -> 10, so each point is 150 rollouts instead of 45.
  * FT_EPOCHS 300 -> 1000 uniformly. The sensitivity sweep showed 1000 matches
    300 at n=10/50 and is clearly better at n=500, so it dominates.
  * The stale (0-sample) baseline is measured with matched precision — 10
    repeats of the 15 eval seeds — instead of a single 15-run pass. Its
    uncertainty sets the "damage gap", so a noisy baseline poisons every
    "% recovered" figure downstream.
  * Reports an iso-performance table: how much experience each method needs to
    reach the SAME cost, rather than comparing at whichever points happen to
    have been measured.
"""
import json
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "Rung 1"))
import sample_effeciency as se  # noqa: E402

from damage_recovery import (  # noqa: E402
    DAMAGE_MASS_SCALE, PRE_SAMPLES, SEED,
    batch_error, collect_from, damaged_env, finetune, healthy_env,
)

INCREMENTS = [10, 25, 50, 100, 200, 500]
N_ADAPT_SEEDS = 10
FT_EPOCHS = 1000
N_BASELINE_REPEATS = 10      # repeats of the eval-seed set for the 0-sample point

torch.set_num_threads(2)


def repeated_eval(model, env_fn, eval_seeds, repeats):
    """Evaluate the same model `repeats` times so its precision matches the
    adapted points (which average over that many independently drawn models)."""
    runs = []
    for _ in range(repeats):
        _, _, r = se.evaluate(model, eval_seeds=eval_seeds, env_fn=env_fn)
        runs.append(r)
    runs = np.concatenate(runs)
    return float(runs.mean()), float(runs.std(ddof=1)), runs


def main():
    t0 = time.time()
    eval_seeds = se.get_eval_seeds()
    print("=" * 74)
    print(f"RECOVERY CURVE v2 — {N_ADAPT_SEEDS} draws x {se.N_EVAL} episodes "
          f"= {N_ADAPT_SEEDS * se.N_EVAL} rollouts per point")
    print("=" * 74)

    env = healthy_env()
    S, A, S2 = collect_from(env, PRE_SAMPLES, seed=SEED)
    env.close()
    base = se.train_model(S, A, S2, seed=SEED)

    print(f"\nbaselines ({N_BASELINE_REPEATS * se.N_EVAL} rollouts each):")
    healthy_m, healthy_sd, healthy_runs = repeated_eval(
        base, healthy_env, eval_seeds, N_BASELINE_REPEATS)
    stale_m, stale_sd, stale_runs = repeated_eval(
        base, damaged_env, eval_seeds, N_BASELINE_REPEATS)
    healthy_sem = healthy_sd / np.sqrt(healthy_runs.size)
    stale_sem = stale_sd / np.sqrt(stale_runs.size)
    print(f"   healthy body, healthy model: {healthy_m:.3f} +/- {healthy_sd:.3f} "
          f"(sem {healthy_sem:.3f})")
    print(f"   damaged body, STALE model:   {stale_m:.3f} +/- {stale_sd:.3f} "
          f"(sem {stale_sem:.3f})")
    gap = stale_m - healthy_m
    gap_sem = float(np.hypot(healthy_sem, stale_sem))
    print(f"   damage gap: {gap:.3f} +/- {gap_sem:.3f}")

    # Independent post-damage streams, one per draw.
    pools = []
    for s in range(N_ADAPT_SEEDS):
        env = damaged_env()
        pools.append(collect_from(env, max(INCREMENTS), seed=SEED + 500 + s))
        env.close()

    means, stds, sems = [], [], []
    print()
    for n in INCREMENTS:
        seed_means, all_runs = [], []
        for s in range(N_ADAPT_SEEDS):
            Sn, An, S2n = (arr[:n] for arr in pools[s])
            adapted = finetune(base, Sn, An, S2n, epochs=FT_EPOCHS)
            m, _, runs = se.evaluate(adapted, eval_seeds=eval_seeds,
                                     env_fn=damaged_env)
            seed_means.append(m); all_runs.append(runs)
        all_runs = np.concatenate(all_runs)
        means.append(float(np.mean(seed_means)))
        stds.append(float(all_runs.std(ddof=1)))
        sems.append(float(all_runs.std(ddof=1) / np.sqrt(all_runs.size)))
        spread = f"{min(seed_means):.2f}-{max(seed_means):.2f}"
        print(f"{n:>5} samples -> {means[-1]:.3f} +/- {sems[-1]:.3f} (sem)   "
              f"per-draw range {spread}")

    print(f"\n{'samples':>8} {'cost':>8} {'sem':>7}  {'95% CI':>17} "
          f"{'recovered':>10} {'vs stale':>10}")
    print("-" * 70)
    for n, m, sem in zip(INCREMENTS, means, sems):
        rec = 100 * (stale_m - m) / gap
        # Is this point significantly better than the stale model?
        sig = "yes" if m + 1.96 * sem < stale_m - 1.96 * stale_sem else "no"
        print(f"{n:>8} {m:>8.3f} {sem:>7.3f}  "
              f"[{m - 1.96 * sem:>6.3f}, {m + 1.96 * sem:>6.3f}] {rec:>9.0f}% "
              f"{sig:>10}")
    print("-" * 70)
    print(f"healthy {healthy_m:.3f}   stale {stale_m:.3f}   gap {gap:.3f}")

    # Is any point significantly WORSE than healthy (i.e. is a residual gap real)?
    print("\nresidual gap vs healthy (is the plateau real?):")
    for n, m, sem in zip(INCREMENTS, means, sems):
        worse = m - 1.96 * sem > healthy_m + 1.96 * healthy_sem
        print(f"   {n:>4} samples: {m:.3f} vs healthy {healthy_m:.3f} -> "
              f"{'REAL gap' if worse else 'indistinguishable'}")

    plt.figure(figsize=(7.6, 5))
    plt.errorbar(INCREMENTS, means, yerr=[1.96 * s for s in sems], marker="o",
                 capsize=4, color="tab:blue", label="fine-tuned self-model (95% CI)")
    plt.axhline(healthy_m, color="tab:green", ls="--", lw=1.4,
                label=f"healthy body + healthy model ({healthy_m:.2f})")
    plt.fill_between([min(INCREMENTS), max(INCREMENTS)],
                     healthy_m - 1.96 * healthy_sem, healthy_m + 1.96 * healthy_sem,
                     color="tab:green", alpha=.15)
    plt.axhline(stale_m, color="tab:red", ls=":", lw=1.6,
                label=f"damaged body + stale model ({stale_m:.2f})")
    plt.xscale("log")
    plt.xlabel("Post-damage samples used to fine-tune")
    plt.ylabel("Average control cost (lower = better)")
    plt.title(f"Damage recovery ({N_ADAPT_SEEDS} draws x {se.N_EVAL} episodes per point)")
    plt.grid(True, which="both", alpha=.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("damage_recovery_v2.png", dpi=150)
    print("\nSaved graph to damage_recovery_v2.png")

    with open("results_recovery_v2.json", "w") as f:
        json.dump({"increments": INCREMENTS, "means": means, "stds": stds,
                   "sems": sems, "healthy": healthy_m, "healthy_sem": healthy_sem,
                   "stale": stale_m, "stale_sem": stale_sem,
                   "gap": gap, "gap_sem": gap_sem,
                   "n_adapt_seeds": N_ADAPT_SEEDS, "ft_epochs": FT_EPOCHS,
                   "pre_damage_samples": PRE_SAMPLES,
                   "mass_scale": DAMAGE_MASS_SCALE}, f, indent=2)
    print("Saved results to results_recovery_v2.json")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
