import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Configuration ────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
OUTPUT_PATH = "plots/ppo_learning_curves.png"
ALGORITHM   = "ppo_continuous_action"

ENVS = {
    "dm_control_pendulum-swingup-v0":  "Pendulum Swingup",
    "dm_control_cartpole-swingup-v0":  "Cartpole Swingup",
    "dm_control_reacher-easy-v0":      "Reacher Easy",
    "dm_control_hopper-stand-v0":      "Hopper Stand",
    "dm_control_cheetah-run-v0":       "Cheetah Run",
}

# Number of bins to reduce each raw curve to before plotting
N_BINS = 100

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_seed_curves(env_key, algorithm, results_dir):
    """Load all CSV files for a given environment and algorithm.
    Returns a list of (timesteps, returns) arrays, one per seed.
    """
    pattern = os.path.join(results_dir, f"{env_key}__{algorithm}__*_results.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"WARNING: No files found for {env_key}")
        return []

    curves = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty or len(df) < 2:
            continue
        timesteps = df["timestep"].values.astype(float)
        returns   = df["episodic_return"].values.astype(float)
        curves.append((timesteps, returns))

    return curves


def bin_curve(timesteps, returns, n_bins=N_BINS):
    """Reduce a raw episodic curve to n_bins points by averaging within bins.
    This preserves the actual shape of the curve without interpolating across seeds.
    """
    t_min, t_max = timesteps[0], timesteps[-1]
    edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    binned_returns = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (timesteps >= edges[i]) & (timesteps < edges[i + 1])
        if mask.sum() > 0:
            binned_returns[i] = returns[mask].mean()

    # forward-fill any empty bins (rare, only at edges)
    for i in range(1, n_bins):
        if np.isnan(binned_returns[i]):
            binned_returns[i] = binned_returns[i - 1]

    return bin_centers, binned_returns


def compute_auc(timesteps, returns):
    """Compute area under the curve using the trapezoidal rule."""
    return np.trapezoid(returns, timesteps)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sac(envs, algorithm, results_dir, output_path):
    n_envs = len(envs)
    ncols  = 3
    nrows  = int(np.ceil(n_envs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax_idx, (env_key, env_name) in enumerate(envs.items()):
        ax = axes[ax_idx]

        curves = load_seed_curves(env_key, algorithm, results_dir)
        if not curves:
            ax.set_title(f"{env_name}\n(no data)")
            ax.axis("off")
            continue

        # bin each seed curve independently to N_BINS points
        binned_curves = []
        aucs = []
        for timesteps, returns in curves:
            x_bin, y_bin = bin_curve(timesteps, returns, n_bins=N_BINS)
            binned_curves.append((x_bin, y_bin))
            aucs.append(compute_auc(x_bin, y_bin))

        # find the seed whose AUC is closest to the median AUC
        median_auc = np.median(aucs)
        median_idx = int(np.argmin(np.abs(np.array(aucs) - median_auc)))

        # plot all seeds as thin faint lines
        for i, (x_bin, y_bin) in enumerate(binned_curves):
            if i == median_idx:
                continue  # draw median on top
            ax.plot(x_bin, y_bin, color="steelblue", alpha=0.25, linewidth=0.8)

        # bold the median-AUC seed on top
        x_med, y_med = binned_curves[median_idx]
        ax.plot(x_med, y_med, color="steelblue", linewidth=2.5, label="Median AUC seed")

        ax.set_title(env_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Environment Steps", fontsize=10)
        ax.set_ylabel("Episodic Return", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K"
        ))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # hide any unused subplots
    for ax_idx in range(len(envs), len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle("PPO — DM Control Suite (100 seeds)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_sac(ENVS, ALGORITHM, RESULTS_DIR, OUTPUT_PATH)
