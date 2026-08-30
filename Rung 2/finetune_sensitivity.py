"""Is the recovery plateau real, or an artefact of the fine-tuning budget?

The main experiment fine-tunes for FT_EPOCHS=300 and plateaus at ~62% of the
damage gap. Suspicion: 300 epochs is plenty to absorb 10 new samples but far
too few to re-fit 500, leaving stale structure in the weights. The tell is that
Rung 1 reached 4.715 training from scratch on 500 samples, while fine-tuning on
500 damaged samples only reached 6.220 — fine-tuning should not lose to
starting over on the same data unless it is under-adapting.

This sweeps the epoch budget at three sample counts and adds a from-scratch
control at each, reporting both control cost and held-out prediction error on
the damaged body. Prediction error is the cleaner signal: it measures whether
the model actually learned the new dynamics, without the planner's noise.
"""
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "Rung 1"))
import sample_effeciency as se  # noqa: E402

from damage_recovery import (  # noqa: E402
    DAMAGE_MASS_SCALE, PRE_SAMPLES, SEED,
    batch_error, collect_from, damaged_env, finetune, healthy_env,
)

EPOCH_BUDGETS = [300, 1000, 3000]
SAMPLE_COUNTS = [10, 50, 500]
N_ADAPT_SEEDS = 3
HELDOUT = 1000          # fresh damaged transitions, never fine-tuned on

torch.set_num_threads(2)


def main():
    t0 = time.time()
    eval_seeds = se.get_eval_seeds()

    print("=" * 72)
    print("FINE-TUNE BUDGET SENSITIVITY  (is the ~62% plateau an artefact?)")
    print("=" * 72)

    # Same pre-damage model as the main experiment.
    env = healthy_env()
    S, A, S2 = collect_from(env, PRE_SAMPLES, seed=SEED)
    env.close()
    base = se.train_model(S, A, S2, seed=SEED)

    # Held-out damaged data for an honest prediction-error read.
    env = damaged_env()
    Sh, Ah, S2h = collect_from(env, HELDOUT, seed=SEED + 4242)
    env.close()

    # Post-damage streams, one per adaptation seed (nested prefixes).
    pools = []
    for s in range(N_ADAPT_SEEDS):
        env = damaged_env()
        pools.append(collect_from(env, max(SAMPLE_COUNTS), seed=SEED + 500 + s))
        env.close()

    stale_mse = batch_error(base, Sh, Ah, S2h)
    stale_m, _, _ = se.evaluate(base, eval_seeds=eval_seeds, env_fn=damaged_env)
    print(f"\nstale model on damaged body: pred MSE {stale_mse:.6f}, "
          f"control {stale_m:.3f}")

    rows = []
    for n in SAMPLE_COUNTS:
        for ep in EPOCH_BUDGETS + ["scratch"]:
            costs, mses = [], []
            for s in range(N_ADAPT_SEEDS):
                Sn, An, S2n = (arr[:n] for arr in pools[s])
                if ep == "scratch":
                    m = se.train_model(Sn, An, S2n, epochs=1000,
                                       seed=SEED + 1000 * s + n)
                else:
                    m = finetune(base, Sn, An, S2n, epochs=ep)
                c, _, _ = se.evaluate(m, eval_seeds=eval_seeds, env_fn=damaged_env)
                costs.append(c)
                mses.append(batch_error(m, Sh, Ah, S2h))
            rows.append({
                "n": n, "budget": ep,
                "cost": float(np.mean(costs)), "cost_sd": float(np.std(costs, ddof=1)),
                "mse": float(np.mean(mses)),
            })
            print(f"   n={n:>4}  {str(ep):>7} epochs  ->  cost {rows[-1]['cost']:.3f}"
                  f"  pred MSE {rows[-1]['mse']:.6f}")

    print(f"\n{'samples':>8} {'budget':>9} {'control cost':>14} {'pred MSE':>12}"
          f"  {'vs stale MSE':>13}")
    print("-" * 62)
    for r in rows:
        print(f"{r['n']:>8} {str(r['budget']):>9} {r['cost']:>9.3f} "
              f"+/-{r['cost_sd']:<4.2f} {r['mse']:>12.6f}  "
              f"{stale_mse / r['mse']:>12.1f}x")
    print("-" * 62)
    print("'scratch' = trained from random init on the same damaged samples "
          "(1000 epochs).")
    print("A fine-tune that loses to 'scratch' on the same data is "
          "under-adapting, not saturating.")

    with open("results_finetune_sensitivity.json", "w") as f:
        json.dump({"stale_mse": stale_mse, "stale_cost": stale_m,
                   "mass_scale": DAMAGE_MASS_SCALE, "rows": rows}, f, indent=2)
    print("\nSaved results to results_finetune_sensitivity.json")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
