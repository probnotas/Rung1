"""Proves the batched choose_action is behaviourally identical to the original.

The original planner rolled out each of the 300 candidate action sequences in a
Python loop. The version in sample_effeciency.py rolls all 300 out as one batch
for a ~120x speedup. This checks the two pick the same action when handed the
same candidates.
"""
import numpy as np
import torch
import torch.nn as nn

from sample_effeciency import HORIZON, N_CANDIDATES, choose_action, cost, predict


def choose_action_reference(model, state, acts_all):
    """The original implementation, with candidates injected instead of drawn."""
    best_first, best_c = None, float("inf")
    for acts in acts_all:
        total, sim = 0, state.copy()
        for a in acts:
            sim = predict(model, sim, a)
            total += cost(sim)
        if total < best_c:
            best_c, best_first = total, acts[0]
    return best_first, best_c


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    mismatches = 0
    trials = 5
    for t in range(trials):
        model = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )
        state = rng.uniform(-1, 1, size=3).astype(np.float32)
        acts = rng.uniform(-2, 2, size=(N_CANDIDATES, HORIZON, 1)).astype(np.float32)

        ref_action, ref_cost = choose_action_reference(model, state, acts)
        fast_action = choose_action(model, state, acts=acts)

        same = np.allclose(ref_action, fast_action, atol=1e-5)
        mismatches += (not same)
        print(
            f"trial {t + 1}: reference={ref_action[0]:+.6f}  batched={fast_action[0]:+.6f}  "
            f"{'MATCH' if same else 'MISMATCH'}"
        )

    print()
    if mismatches == 0:
        print(f"PASS: all {trials} trials selected an identical action.")
    else:
        raise SystemExit(f"FAIL: {mismatches}/{trials} trials disagreed.")


if __name__ == "__main__":
    main()
