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

def load_mean_returns(env_key, algorithm, results_dir):
    """Compute the sample mean episodic return across the entire run for each seed."""
    pattern = os.path.join(results_dir, f"{env_key}__{algorithm}__*_results.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {env_key}")

    mean_returns = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        mean_returns.append(df["episodic_return"].mean())

    return np.array(mean_returns)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_kde_histogram(env_key, env_name, algorithm, results_dir, output_path):
    mean_returns = load_mean_returns(env_key, algorithm, results_dir)
    print(f"Loaded {len(mean_returns)} seeds")

    perf_grid = np.linspace(PERF_MIN, PERF_MAX, N_KDE_PTS)

    # Gaussian KDE with Scott's rule
    kde = gaussian_kde(mean_returns, bw_method="scott")
    kde_vals = kde(perf_grid)

    # bin width needed to convert density to probability
    bin_width = (PERF_MAX - PERF_MIN) / N_BINS

    fig, ax = plt.subplots(figsize=(6, 7))

    # horizontal histogram with empirical probability (density * bin_width)
    counts, edges = np.histogram(mean_returns, bins=N_BINS, range=(PERF_MIN, PERF_MAX))
    probs = counts / counts.sum()  # empirical probability: each bar sums to 1 total

    for i in range(N_BINS):
        bin_lo = edges[i]
        bin_hi = edges[i + 1]
        bin_mid = (bin_lo + bin_hi) / 2
        ax.barh(
            bin_mid,
            probs[i],
            height=(bin_hi - bin_lo) * 0.9,
            color="steelblue",
            alpha=0.4,
            edgecolor="white",
            linewidth=0.5,
        )

    # KDE line scaled to probability (density * bin_width)
    ax.plot(kde_vals * bin_width, perf_grid, color="steelblue", linewidth=2.5)

    ax.set_ylabel("Sample Mean Return (full run)", fontsize=12)
    ax.set_xlabel("Empirical Probability", fontsize=12)
    ax.set_ylim(PERF_MIN, PERF_MAX)
    ax.set_title(
        f"{env_name} — PPO (100 seeds)\n"
        f"Sample mean return distribution\n"
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
