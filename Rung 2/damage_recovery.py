"""Rung 2 — damage detection and recovery for the pendulum self-model.

Takes the Rung 1 delta-prediction model (2000 samples, CEM planner), breaks the
body by increasing the pendulum's mass 50%, and asks three questions:

  1. Does the model notice?          (one-step prediction error across the break)
  2. Does control degrade?           (CEM cost on the damaged body, stale model)
  3. How few new samples to recover? (fine-tune from pre-damage weights)

plus a model-free SAC baseline retrained from scratch on the damaged body.

Physics note: in Pendulum-v1 the update is
    newthdot = thdot + (3g/(2l)·sin(th) + 3/(m·l²)·u)·dt
so mass enters ONLY through the torque term. Raising m to 1.5 cuts actuator
authority to 0.667x and leaves the gravity/free-swing dynamics untouched. The
damage is therefore invisible whenever u = 0 and most visible under large
torque — which is why detection is probed under two different policies below.
"""
import copy
import json
import pathlib
import sys
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "Rung 1"))
import sample_effeciency as se  # noqa: E402

# ---------- configuration ----------
DAMAGE_MASS_SCALE = 1.5      # 1.0 -> 1.5 kg
PRE_SAMPLES       = 2000     # pre-damage training set (matches Rung 1's best point)
INCREMENTS        = [10, 25, 50, 100, 200, 500]
N_ADAPT_SEEDS     = 3        # independent draws of post-damage data
FT_EPOCHS         = 300      # fine-tune budget (vs 1000 to train from scratch)
FT_LR             = 1e-3
TRACE_EPISODES    = 10       # detection traces to average
TRACE_HALF        = 200      # steps before / after the break
SAC_CHECKPOINTS   = [500, 1000, 2000, 5000, 10000, 20000]
N_SAC_SEEDS       = 3
SEED              = 0

torch.set_num_threads(2)
assert se.PREDICT_DELTA, "Rung 2 assumes the delta-prediction model"


# ---------- bodies ----------
def make_env(m_scale=1.0, max_episode_steps=None):
    """A Pendulum whose mass is scaled. reset() does not restore m."""
    kw = {} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}
    env = gym.make("Pendulum-v1", **kw)
    env.unwrapped.m = 1.0 * m_scale
    return env


def healthy_env():
    return make_env(1.0)


def damaged_env():
    return make_env(DAMAGE_MASS_SCALE)


def collect_from(env, n, seed):
    """Random-action transitions from a given body."""
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
    return np.array(S), np.array(A), np.array(S2)


# ---------- adaptation ----------
def finetune(model, S, A, S2, epochs=FT_EPOCHS, lr=FT_LR):
    """Continue training a COPY of the model on new data only.

    Starts from the pre-damage weights (never from scratch) with a fresh Adam
    state. Trains on the post-damage transitions alone: the old body's data is
    now wrong, so there is nothing to rehearse.
    """
    m = copy.deepcopy(model)
    X = torch.tensor(np.concatenate([S, A], axis=1), dtype=torch.float32)
    Y = torch.tensor(S2 - S, dtype=torch.float32)      # delta target
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        loss = loss_fn(m(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def step_error(model, s, a, s_next):
    """One-step squared prediction error, averaged over the 3 state dims."""
    return float(np.mean((se.predict(model, s, a) - s_next) ** 2))


def batch_error(model, S, A, S2):
    X = torch.tensor(np.concatenate([S, A], axis=1), dtype=torch.float32)
    Y = torch.tensor(S2, dtype=torch.float32)
    return se.prediction_mse(model, X, Y)


# ---------- part 2: detection ----------
def detection_trace(model, policy, seed, half=None):
    """One continuous episode; the body breaks at step `half`."""
    half = TRACE_HALF if half is None else half      # read config at call time
    env = make_env(1.0, max_episode_steps=2 * half + 10)
    env.action_space.seed(seed)
    state, _ = env.reset(seed=seed)
    errs = []
    for t in range(2 * half):
        if t == half:
            env.unwrapped.m = 1.0 * DAMAGE_MASS_SCALE      # <- damage
        a = policy(model, state, env)
        nxt, _, term, trunc, _ = env.step(a)
        errs.append(step_error(model, state, a, nxt))
        state = nxt
        if term:
            state, _ = env.reset()
    env.close()
    return np.array(errs)


def policy_random(model, state, env):
    return env.action_space.sample()


def policy_cem(model, state, env):
    return se.choose_action(model, state)


def run_detection(model):
    print("\n" + "=" * 68)
    print("PART 2 — DOES THE MODEL NOTICE?")
    print("=" * 68)
    out = {}
    for name, pol in (("random probe", policy_random), ("CEM control", policy_cem)):
        traces = np.array([
            detection_trace(model, pol, seed=SEED + 100 * i + 1)
            for i in range(TRACE_EPISODES)
        ])
        pre = traces[:, :TRACE_HALF]
        post = traces[:, TRACE_HALF:]
        out[name] = {
            "curve": traces.mean(axis=0),
            "pre_mean": float(pre.mean()), "pre_std": float(pre.mean(axis=1).std(ddof=1)),
            "post_mean": float(post.mean()), "post_std": float(post.mean(axis=1).std(ddof=1)),
        }
        print(f"   {name}: pre={out[name]['pre_mean']:.6f}  "
              f"post={out[name]['post_mean']:.6f}  "
              f"ratio={out[name]['post_mean'] / out[name]['pre_mean']:.1f}x")

    print(f"\n{'policy':>14} {'pre-damage':>14} {'post-damage':>14} {'ratio':>9}")
    print("-" * 56)
    for name, d in out.items():
        print(f"{name:>14} {d['pre_mean']:>14.6f} {d['post_mean']:>14.6f} "
              f"{d['post_mean'] / d['pre_mean']:>8.1f}x")
    print("-" * 56)
    print(f"(mean one-step squared error per state dim, {TRACE_EPISODES} traces "
          f"x {TRACE_HALF} steps each side)")

    # ---------- plot ----------
    plt.figure(figsize=(8, 4.6))
    n_steps = len(next(iter(out.values()))["curve"])
    t = np.arange(n_steps)
    for (name, d), c in zip(out.items(), ("tab:blue", "tab:purple")):
        plt.semilogy(t, np.convolve(d["curve"], np.ones(5) / 5, mode="same"),
                     color=c, lw=1.4, label=name)
    plt.axvline(TRACE_HALF, color="tab:red", ls="--", lw=1.5,
                label=f"damage: m x{DAMAGE_MASS_SCALE}")
    plt.xlabel("Environment step")
    plt.ylabel("One-step prediction error (MSE, log)")
    plt.title("Damage detection: the self-model's error across the break")
    plt.grid(True, which="both", alpha=.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("damage_detection.png", dpi=150)
    print("\nSaved graph to damage_detection.png")
    return out


# ---------- parts 3 & 4: degradation and recovery ----------
def run_recovery(model, eval_seeds):
    print("\n" + "=" * 68)
    print("PART 3 — DOES CONTROL DEGRADE?")
    print("=" * 68)
    healthy_m, healthy_sd, _ = se.evaluate(model, eval_seeds=eval_seeds, env_fn=healthy_env)
    stale_m, stale_sd, stale_runs = se.evaluate(model, eval_seeds=eval_seeds, env_fn=damaged_env)
    print(f"{'condition':>34} {'cost':>9} {'std':>9}")
    print("-" * 56)
    print(f"{'healthy body, healthy model':>34} {healthy_m:>9.3f} {healthy_sd:>9.3f}")
    print(f"{'damaged body, STALE model':>34} {stale_m:>9.3f} {stale_sd:>9.3f}")
    print("-" * 56)
    print(f"degradation: {stale_m - healthy_m:+.3f} "
          f"({100 * (stale_m - healthy_m) / healthy_m:+.1f}%)")

    print("\n" + "=" * 68)
    print("PART 4 — RECOVERY CURVE")
    print("=" * 68)
    # One post-damage stream per adaptation seed; increments are nested prefixes.
    pools = []
    for s in range(N_ADAPT_SEEDS):
        env = damaged_env()
        pools.append(collect_from(env, max(INCREMENTS), seed=SEED + 500 + s))
        env.close()

    means, stds, sems, mses = [], [], [], []
    for n in INCREMENTS:
        seed_means, all_runs, seed_mses = [], [], []
        for s in range(N_ADAPT_SEEDS):
            S, A, S2 = (arr[:n] for arr in pools[s])
            adapted = finetune(model, S, A, S2)
            m, sd, runs = se.evaluate(adapted, eval_seeds=eval_seeds, env_fn=damaged_env)
            seed_means.append(m); all_runs.append(runs)
            seed_mses.append(batch_error(adapted, *pools[s]))
            print(f"   n={n:>4}  seed {s + 1}/{N_ADAPT_SEEDS}  ->  {m:.3f} +/- {sd:.3f}")
        all_runs = np.concatenate(all_runs)
        means.append(float(np.mean(seed_means)))
        stds.append(float(all_runs.std(ddof=1)))
        sems.append(float(all_runs.std(ddof=1) / np.sqrt(all_runs.size)))
        mses.append(float(np.mean(seed_mses)))
        print(f"{n:>5} new samples -> cost = {means[-1]:.3f} +/- {stds[-1]:.3f}\n")

    print(f"{'new samples':>12} {'cost':>9} {'std':>9} {'sem':>8}   {'95% CI':>18}  "
          f"{'recovered':>10}")
    print("-" * 78)
    span = stale_m - healthy_m
    for n, m, sd, sem in zip(INCREMENTS, means, stds, sems):
        rec = 100 * (stale_m - m) / span if span > 0 else float("nan")
        print(f"{n:>12} {m:>9.3f} {sd:>9.3f} {sem:>8.3f}   "
              f"[{m - 1.96 * sem:>7.3f}, {m + 1.96 * sem:>7.3f}]  {rec:>9.0f}%")
    print("-" * 78)
    print(f"reference: healthy {healthy_m:.3f}   stale-on-damaged {stale_m:.3f}")
    print('"recovered" = fraction of the damage gap closed (100% = back to healthy)')

    # First increment that is no longer significantly worse than the healthy
    # baseline. This is an absence-of-difference test, not proof of equality —
    # it says the gap is no longer detectable at this sample count.
    restored = None
    for n, m, sem in zip(INCREMENTS, means, sems):
        if m - 1.96 * sem <= healthy_m:
            restored = n
            break
    if restored is None:
        print(f"\nEvery increment up to {max(INCREMENTS)} samples remained "
              f"significantly worse than the healthy baseline.")
    else:
        print(f"\nAfter {restored} post-damage samples, control is no longer "
              f"significantly worse than the healthy baseline "
              f"(95% CI reaches {healthy_m:.3f}).")

    # ---------- plot ----------
    plt.figure(figsize=(7.6, 5))
    plt.errorbar(INCREMENTS, means, yerr=stds, marker="o", capsize=4,
                 color="tab:blue", label="fine-tuned self-model")
    plt.axhline(healthy_m, color="tab:green", ls="--", lw=1.4,
                label=f"healthy body, healthy model ({healthy_m:.2f})")
    plt.axhline(stale_m, color="tab:red", ls=":", lw=1.6,
                label=f"damaged body, stale model ({stale_m:.2f})")
    plt.xscale("log")
    plt.xlabel("Post-damage samples used to fine-tune")
    plt.ylabel("Average control cost (lower = better)")
    plt.title("Damage recovery: control vs. new experience")
    plt.grid(True, which="both", alpha=.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("damage_recovery.png", dpi=150)
    print("Saved graph to damage_recovery.png")

    return dict(healthy=healthy_m, healthy_std=healthy_sd, stale=stale_m,
                stale_std=stale_sd, increments=INCREMENTS, means=means,
                stds=stds, sems=sems, adapted_mse=mses, restored_at=restored)


# ---------- part 5: SAC from scratch on the damaged body ----------
def run_sac(eval_seeds, target_cost):
    from stable_baselines3 import SAC

    print("\n" + "=" * 68)
    print("PART 5 — SAC RETRAINED FROM SCRATCH ON THE DAMAGED BODY")
    print("=" * 68)

    def sac_action(model, state):
        return model.predict(state, deterministic=True)[0]

    per_seed = np.zeros((N_SAC_SEEDS, len(SAC_CHECKPOINTS)))
    all_runs = [[] for _ in SAC_CHECKPOINTS]
    for s in range(N_SAC_SEEDS):
        agent = SAC("MlpPolicy", damaged_env(), seed=SEED + s, verbose=0)
        trained = 0
        for c, ckpt in enumerate(SAC_CHECKPOINTS):
            agent.learn(total_timesteps=ckpt - trained, reset_num_timesteps=False,
                        log_interval=None)
            trained = ckpt
            m, sd, runs = se.evaluate(agent, eval_seeds=eval_seeds,
                                      action_fn=sac_action, env_fn=damaged_env)
            per_seed[s, c] = m
            all_runs[c].append(runs)
            print(f"   seed {s + 1}/{N_SAC_SEEDS}  {ckpt:>6} steps  ->  {m:.3f} +/- {sd:.3f}")
        print()

    means, stds, sems = [], [], []
    for c in range(len(SAC_CHECKPOINTS)):
        pooled = np.concatenate(all_runs[c])
        means.append(float(per_seed[:, c].mean()))
        stds.append(float(pooled.std(ddof=1)))
        sems.append(float(pooled.std(ddof=1) / np.sqrt(pooled.size)))

    print(f"{'env steps':>10} {'cost':>9} {'std':>9} {'sem':>8}   {'95% CI':>18}")
    print("-" * 60)
    for ckpt, m, sd, sem in zip(SAC_CHECKPOINTS, means, stds, sems):
        print(f"{ckpt:>10} {m:>9.3f} {sd:>9.3f} {sem:>8.3f}   "
              f"[{m - 1.96 * sem:>7.3f}, {m + 1.96 * sem:>7.3f}]")
    print("-" * 60)

    crossing = None
    for i, (ckpt, m) in enumerate(zip(SAC_CHECKPOINTS, means)):
        if m <= target_cost:
            if i == 0:
                crossing = float(ckpt)
            else:
                x0, x1 = np.log10(SAC_CHECKPOINTS[i - 1]), np.log10(ckpt)
                y0, y1 = means[i - 1], m
                frac = (y0 - target_cost) / (y0 - y1) if y0 != y1 else 1.0
                crossing = float(10 ** (x0 + frac * (x1 - x0)))
            break
    if crossing is None:
        print(f"SAC never reached cost {target_cost:.3f} within "
              f"{max(SAC_CHECKPOINTS)} env steps.")
    else:
        print(f"SAC reaches the recovered self-model's cost ({target_cost:.3f}) "
              f"at ~{crossing:,.0f} env steps.")
    return dict(checkpoints=SAC_CHECKPOINTS, means=means, stds=stds, sems=sems,
                crossing=crossing, target=target_cost)


# ---------- main ----------
def main():
    t0 = time.time()
    eval_seeds = se.get_eval_seeds()
    print("=" * 68)
    print(f"RUNG 2 — DAMAGE RECOVERY   (mass x{DAMAGE_MASS_SCALE}, "
          f"torque authority x{1 / DAMAGE_MASS_SCALE:.3f})")
    print("=" * 68)

    print(f"\nPART 1 — training the pre-damage self-model on {PRE_SAMPLES} "
          f"healthy samples...")
    env = healthy_env()
    S, A, S2 = collect_from(env, PRE_SAMPLES, seed=SEED)
    env.close()
    model = se.train_model(S, A, S2, seed=SEED)

    henv, denv = healthy_env(), damaged_env()
    Sh, Ah, S2h = collect_from(henv, 1000, seed=SEED + 91)
    Sd, Ad, S2d = collect_from(denv, 1000, seed=SEED + 92)
    henv.close(); denv.close()
    print(f"   held-out one-step MSE  healthy body: {batch_error(model, Sh, Ah, S2h):.6f}")
    print(f"   held-out one-step MSE  damaged body: {batch_error(model, Sd, Ad, S2d):.6f}")

    det = run_detection(model)
    rec = run_recovery(model, eval_seeds)
    sac = run_sac(eval_seeds, target_cost=min(rec["means"]))

    with open("results_damage.json", "w") as f:
        json.dump({
            "mass_scale": DAMAGE_MASS_SCALE,
            "pre_samples": PRE_SAMPLES,
            "finetune": {"epochs": FT_EPOCHS, "lr": FT_LR, "seeds": N_ADAPT_SEEDS},
            "detection": {k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                          for k, v in det.items()},
            "recovery": rec,
            "sac": sac,
        }, f, indent=2)
    print("\nSaved results to results_damage.json")
    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
