import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ── Configuration ────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
OUTPUT_PATH = "plots/ppo_pendulum_kde_histogram.png"
ALGORITHM   = "ppo_continuous_action"
ENV_KEY     = "dm_control_pendulum-swingup-v0"
ENV_NAME    = "Pendulum Swingup"
N_BINS      = 10          # number of histogram bins
N_KDE_PTS   = 500         # resolution of the KDE curve
PERF_MIN    = 0
PERF_MAX    = 1000

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_final_returns(env_key, algorithm, results_dir):
    """Load the final episodic return (last logged value) for each seed."""
    pattern = os.path.join(results_dir, f"{env_key}__{algorithm}__*_results.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {env_key}")

    final_returns = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        # take the mean of the last 10 episodes as a stable estimate of final performance
        last_returns = df["episodic_return"].values[-10:]
        final_returns.append(last_returns.mean())

    return np.array(final_returns)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_kde_histogram(env_key, env_name, algorithm, results_dir, output_path):
    final_returns = load_final_returns(env_key, algorithm, results_dir)
    print(f"Loaded {len(final_returns)} seeds")

    perf_grid = np.linspace(PERF_MIN, PERF_MAX, N_KDE_PTS)

    # Gaussian KDE with Scott's rule
    kde = gaussian_kde(final_returns, bw_method="scott")
    kde_vals = kde(perf_grid)

    fig, ax = plt.subplots(figsize=(6, 7))

    # horizontal histogram — bars extend rightward, performance on Y axis
    ax.hist(
        final_returns,
        bins=N_BINS,
        range=(PERF_MIN, PERF_MAX),
        orientation="horizontal",
        color="steelblue",
        alpha=0.4,
        density=True,        # normalise to density so KDE is on the same scale
        edgecolor="white",
        linewidth=0.5,
    )

    # KDE line
    ax.plot(kde_vals, perf_grid, color="steelblue", linewidth=2.5)

    ax.set_ylabel("Episodic Return", fontsize=12)
    ax.set_xlabel("Density", fontsize=12)
    ax.set_ylim(PERF_MIN, PERF_MAX)
    ax.set_title(
        f"{env_name} — PPO (100 seeds)\n"
        f"Final performance distribution\n"
        f"Gaussian KDE, Scott's rule ({N_BINS} bins)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_kde_histogram(ENV_KEY, ENV_NAME, ALGORITHM, RESULTS_DIR, OUTPUT_PATH)
