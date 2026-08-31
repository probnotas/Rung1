"""Is the leftover recovery gap a planning-horizon limit rather than a model limit?

The sensitivity sweep showed the adapted model predicts the damaged body as
accurately as the original model ever predicted the healthy one (MSE 0.00019 vs
0.00020), yet control stays at ~4.6 against a healthy baseline of ~3.76. So the
residual gap is not ignorance about the body.

Hypothesis: damage cut torque authority to 0.667x, so swing-up needs more
energy-pumping swings, but CEM still plans a fixed 20 steps (1.0 s) ahead.

Test: sweep the horizon for the adapted model on the damaged body. The control
is the SAME sweep for the healthy model on the healthy body — if a longer
horizon helps both equally, the effect is generic and says nothing about damage.
The hypothesis predicts a LARGER gain on the damaged body.
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

HORIZONS = [20, 30, 40, 60]
ADAPT_SAMPLES = 500      # the fully-adapted model
ADAPT_EPOCHS = 3000      # best budget found at n=500
N_ADAPT_SEEDS = 3

torch.set_num_threads(2)


def eval_at_horizon(model, env_fn, eval_seeds, horizon):
    original = se.HORIZON
    se.HORIZON = horizon                    # read at call time by the planner
    try:
        return se.evaluate(model, eval_seeds=eval_seeds, env_fn=env_fn)
    finally:
        se.HORIZON = original


def main():
    t0 = time.time()
    eval_seeds = se.get_eval_seeds()
    print("=" * 70)
    print("PLANNING-HORIZON SWEEP")
    print("=" * 70)
    print(f"damaged body: mass x{DAMAGE_MASS_SCALE} -> torque authority "
          f"x{1 / DAMAGE_MASS_SCALE:.3f}")
    print(f"horizon in seconds: {[round(h * 0.05, 2) for h in HORIZONS]} "
          f"(dt = 0.05 s)\n")

    # Pre-damage model, exactly as in the main experiment.
    env = healthy_env()
    S, A, S2 = collect_from(env, PRE_SAMPLES, seed=SEED)
    env.close()
    base = se.train_model(S, A, S2, seed=SEED)

    # Fully-adapted models: fine-tuned on ADAPT_SAMPLES from the damaged body.
    adapted = []
    for s in range(N_ADAPT_SEEDS):
        env = damaged_env()
        Sd, Ad, S2d = collect_from(env, ADAPT_SAMPLES, seed=SEED + 500 + s)
        env.close()
        adapted.append(finetune(base, Sd, Ad, S2d, epochs=ADAPT_EPOCHS))

    env = damaged_env()
    Sh, Ah, S2h = collect_from(env, 1000, seed=SEED + 4242)
    env.close()
    print(f"adapted model prediction MSE on damaged body: "
          f"{np.mean([batch_error(m, Sh, Ah, S2h) for m in adapted]):.6f}")
    print("(the original healthy model scored 0.000198 on the healthy body)\n")

    damaged_means, damaged_stds, healthy_means, healthy_stds = [], [], [], []
    for h in HORIZONS:
        seed_means, runs_all = [], []
        for m in adapted:
            mu, sd, runs = eval_at_horizon(m, damaged_env, eval_seeds, h)
            seed_means.append(mu); runs_all.append(runs)
        runs_all = np.concatenate(runs_all)
        damaged_means.append(float(np.mean(seed_means)))
        damaged_stds.append(float(runs_all.std(ddof=1)))

        hu, hsd, _ = eval_at_horizon(base, healthy_env, eval_seeds, h)
        healthy_means.append(float(hu)); healthy_stds.append(float(hsd))

        print(f"   horizon {h:>3} ({h * 0.05:.2f}s):  "
              f"damaged+adapted {damaged_means[-1]:.3f}   "
              f"healthy+healthy {healthy_means[-1]:.3f}")

    print(f"\n{'horizon':>8} {'seconds':>9} {'damaged (adapted)':>19} "
          f"{'healthy (control)':>19}")
    print("-" * 60)
    for h, dm, hm in zip(HORIZONS, damaged_means, healthy_means):
        print(f"{h:>8} {h * 0.05:>8.2f}s {dm:>19.3f} {hm:>19.3f}")
    print("-" * 60)

    d_gain = 100 * (damaged_means[0] - min(damaged_means)) / damaged_means[0]
    h_gain = 100 * (healthy_means[0] - min(healthy_means)) / healthy_means[0]
    print(f"best improvement from a longer horizon:")
    print(f"   damaged body: {damaged_means[0]:.3f} -> {min(damaged_means):.3f}  "
          f"({d_gain:+.1f}%)")
    print(f"   healthy body: {healthy_means[0]:.3f} -> {min(healthy_means):.3f}  "
          f"({h_gain:+.1f}%)")
    print()
    if d_gain > h_gain + 5:
        print("=> Longer horizons help the DAMAGED body substantially more.")
        print("   The residual gap is a planning limit, not a model limit.")
    elif abs(d_gain - h_gain) <= 5:
        print("=> Longer horizons help both bodies about equally.")
        print("   Horizon is a generic tuning knob; it does NOT explain the")
        print("   damage-specific gap. Hypothesis not supported.")
    else:
        print("=> Longer horizons help the HEALTHY body more. Hypothesis rejected.")

    plt.figure(figsize=(7.4, 5))
    plt.errorbar([h * 0.05 for h in HORIZONS], damaged_means, yerr=damaged_stds,
                 marker="o", capsize=4, color="tab:red",
                 label="damaged body, adapted model")
    plt.errorbar([h * 0.05 for h in HORIZONS], healthy_means, yerr=healthy_stds,
                 marker="s", capsize=4, color="tab:green",
                 label="healthy body, healthy model (control)")
    plt.xlabel("Planning horizon (seconds)")
    plt.ylabel("Average control cost (lower = better)")
    plt.title("Does a longer plan close the damage gap?")
    plt.grid(True, alpha=.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("horizon_sweep.png", dpi=150)
    print("\nSaved graph to horizon_sweep.png")

    with open("results_horizon.json", "w") as f:
        json.dump({"horizons": HORIZONS, "damaged_means": damaged_means,
                   "damaged_stds": damaged_stds, "healthy_means": healthy_means,
                   "healthy_stds": healthy_stds,
                   "adapt_samples": ADAPT_SAMPLES, "adapt_epochs": ADAPT_EPOCHS},
                  f, indent=2)
    print("Saved results to results_horizon.json")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
