"""The missing baseline: SAC that ADAPTS instead of starting over.

The first comparison was confounded. The self-model kept 2000 samples of prior
knowledge about the healthy body and only had to learn what changed; SAC was
made to relearn the whole task from a random initialisation. That contrast is
"adaptation vs. from scratch", not "model-based vs. model-free".

This trains SAC on the healthy body first, then adapts the SAME agent to the
damaged body — the true apples-to-apples control for the recovery experiment.

Two adaptation variants, because the choice genuinely matters:
  * fresh-buffer:  replay buffer cleared at damage time, so SAC learns from new
                   transitions only. This mirrors how the self-model is
                   fine-tuned (new data only) and is the fair comparison.
  * kept-buffer:   old healthy-body transitions retained. More stable early on,
                   but the buffer now contains transitions that are wrong about
                   the current body.
"""
import json
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "Rung 1"))
import sample_effeciency as se  # noqa: E402

from damage_recovery import SEED, damaged_env, healthy_env  # noqa: E402

PRETRAIN_STEPS = 20000                     # healthy-body SAC, converged (~2.3)
ADAPT_CHECKPOINTS = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
N_SEEDS = 3
VARIANTS = ["fresh-buffer", "kept-buffer"]

torch.set_num_threads(2)


def sac_action(model, state):
    return model.predict(state, deterministic=True)[0]


def main():
    t0 = time.time()
    eval_seeds = se.get_eval_seeds()
    print("=" * 74)
    print("SAC ADAPTATION BASELINE — pre-trained on healthy, then damaged")
    print("=" * 74)

    results = {v: {"per_seed": np.zeros((N_SEEDS, len(ADAPT_CHECKPOINTS))),
                   "runs": [[] for _ in ADAPT_CHECKPOINTS]} for v in VARIANTS}
    pre_costs_healthy, pre_costs_damaged = [], []

    for s in range(N_SEEDS):
        print(f"\n--- seed {s + 1}/{N_SEEDS}: pre-training {PRETRAIN_STEPS} steps "
              f"on the healthy body ---")
        agent = SAC("MlpPolicy", healthy_env(), seed=SEED + s, verbose=0)
        agent.learn(total_timesteps=PRETRAIN_STEPS, log_interval=None)

        h, _, _ = se.evaluate(agent, eval_seeds=eval_seeds,
                              action_fn=sac_action, env_fn=healthy_env)
        d, _, _ = se.evaluate(agent, eval_seeds=eval_seeds,
                              action_fn=sac_action, env_fn=damaged_env)
        pre_costs_healthy.append(h); pre_costs_damaged.append(d)
        print(f"    healthy body: {h:.3f}   damaged body (stale policy): {d:.3f}")

        agent.save(f"/tmp/sac_pre_{s}.zip")
        # save() does NOT include the replay buffer; it must be saved separately
        # or "kept-buffer" would silently start empty and duplicate fresh-buffer.
        agent.save_replay_buffer(f"/tmp/sac_pre_buf_{s}.pkl")

        for variant in VARIANTS:
            ag = SAC.load(f"/tmp/sac_pre_{s}.zip", env=damaged_env())
            if variant == "kept-buffer":
                ag.load_replay_buffer(f"/tmp/sac_pre_buf_{s}.pkl")
            # fresh-buffer: load() already starts with an empty buffer
            trained = 0
            for c, ckpt in enumerate(ADAPT_CHECKPOINTS):
                ag.learn(total_timesteps=ckpt - trained,
                         reset_num_timesteps=False, log_interval=None)
                trained = ckpt
                m, _, runs = se.evaluate(ag, eval_seeds=eval_seeds,
                                         action_fn=sac_action, env_fn=damaged_env)
                results[variant]["per_seed"][s, c] = m
                results[variant]["runs"][c].append(runs)
            row = "  ".join(f"{ckpt}:{results[variant]['per_seed'][s, c]:.2f}"
                            for c, ckpt in enumerate(ADAPT_CHECKPOINTS))
            print(f"    {variant:>13}: {row}")

    summary = {}
    for variant in VARIANTS:
        means, sems = [], []
        for c in range(len(ADAPT_CHECKPOINTS)):
            pooled = np.concatenate(results[variant]["runs"][c])
            means.append(float(results[variant]["per_seed"][:, c].mean()))
            sems.append(float(pooled.std(ddof=1) / np.sqrt(pooled.size)))
        summary[variant] = {"means": means, "sems": sems}

    print(f"\npre-damage SAC:  healthy {np.mean(pre_costs_healthy):.3f}   "
          f"damaged with stale policy {np.mean(pre_costs_damaged):.3f}")
    print(f"\n{'new steps':>10}" + "".join(f"{v:>22}" for v in VARIANTS))
    print("-" * (10 + 22 * len(VARIANTS)))
    for c, ckpt in enumerate(ADAPT_CHECKPOINTS):
        row = f"{ckpt:>10}"
        for v in VARIANTS:
            row += f"{summary[v]['means'][c]:>15.3f} +/-{summary[v]['sems'][c]:<4.2f}"
        print(row)
    print("-" * (10 + 22 * len(VARIANTS)))

    plt.figure(figsize=(7.8, 5))
    for v, c in zip(VARIANTS, ("tab:orange", "tab:brown")):
        plt.errorbar(ADAPT_CHECKPOINTS, summary[v]["means"],
                     yerr=[1.96 * s for s in summary[v]["sems"]],
                     marker="o", capsize=4, label=f"SAC adapted ({v})")
    plt.axhline(np.mean(pre_costs_damaged), color="tab:red", ls=":", lw=1.5,
                label=f"stale SAC policy on damaged body "
                      f"({np.mean(pre_costs_damaged):.2f})")
    plt.xscale("log")
    plt.xlabel("New post-damage environment steps")
    plt.ylabel("Average control cost (lower = better)")
    plt.title("Model-free adaptation: SAC pre-trained, then damaged")
    plt.grid(True, which="both", alpha=.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("sac_adaptation.png", dpi=150)
    print("\nSaved graph to sac_adaptation.png")

    with open("results_sac_adapt.json", "w") as f:
        json.dump({"checkpoints": ADAPT_CHECKPOINTS,
                   "pretrain_steps": PRETRAIN_STEPS,
                   "pre_healthy": float(np.mean(pre_costs_healthy)),
                   "pre_damaged_stale": float(np.mean(pre_costs_damaged)),
                   "variants": summary, "n_seeds": N_SEEDS}, f, indent=2)
    print("Saved results to results_sac_adapt.json")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
