"""Side-by-side comparison of delta vs direct next-state prediction.

Both modes are measured identically: prediction error in next-state space on
the same held-out test set, and CEM control cost on the same start states.
"""
import json
import os


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    pd_, pd_delta = load("results_prediction_direct.json"), load("results_prediction_delta.json")
    cd, cdelta = load("results_cem_direct.json"), load("results_cem_delta.json")

    if pd_ and pd_delta:
        print("=" * 66)
        print("ONE-STEP PREDICTION TEST MSE  (next-state space; lower = better)")
        print("=" * 66)
        print(f"{'samples':>8} {'direct':>13} {'delta':>13} {'improvement':>13}")
        print("-" * 66)
        for i, n in enumerate(pd_["sample_sizes"]):
            a, b = pd_["means"][i], pd_delta["means"][i]
            print(f"{n:>8} {a:>13.6f} {b:>13.6f} {a / b:>12.1f}x")
        print("-" * 66)
        print(f"identity baseline (predict no change): {pd_['identity_mse']:.6f}")

    if cd and cdelta:
        print()
        print("=" * 66)
        print("CEM CONTROL COST  (lower = better)")
        print("=" * 66)
        print(f"{'samples':>8} {'direct':>18} {'delta':>18} {'change':>10}")
        print("-" * 66)
        for i, n in enumerate(cd["sample_sizes"]):
            a, sa = cd["means"][i], cd["stds"][i]
            b, sb = cdelta["means"][i], cdelta["stds"][i]
            print(f"{n:>8} {a:>9.3f} +/-{sa:<6.3f} {b:>9.3f} +/-{sb:<6.3f} "
                  f"{100 * (a - b) / a:>+9.1f}%")
        print("-" * 66)

        # Do the two control curves separate anywhere?
        print("\nWhere the control difference is real (non-overlapping 95% CI):")
        any_real = False
        for i, n in enumerate(cd["sample_sizes"]):
            la, ha = cd["means"][i] - 1.96 * cd["sems"][i], cd["means"][i] + 1.96 * cd["sems"][i]
            lb, hb = cdelta["means"][i] - 1.96 * cdelta["sems"][i], cdelta["means"][i] + 1.96 * cdelta["sems"][i]
            if ha < lb or hb < la:
                any_real = True
                better = "delta" if cdelta["means"][i] < cd["means"][i] else "direct"
                print(f"  {n:>5}: {cd['means'][i]:.3f} vs {cdelta['means'][i]:.3f}  -> {better} better")
        if not any_real:
            print("  (none — control costs overlap at every sample size)")


if __name__ == "__main__":
    main()
