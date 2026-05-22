import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde


# ── Configuration ────────────────────────────────────────────────────────────

RESULTS_DIR  = "results"
OUTPUT_PATH  = "plots/ppo_pendulum_kde_histogram.png"
ALGORITHM    = "ppo_continuous_action"
ENV_KEY      = "dm_control_pendulum-swingup-v0"
ENV_NAME     = "Pendulum Swingup"
N_BINS       = 100        # bins per seed curve (same as learning curve script)
N_HIST_BINS  = 10         # number of histogram bins along the timestep axis
N_KDE_POINTS = 300        # resolution of the KDE curve


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_seed_curves(env_key, algorithm, results_dir):
    pattern = os.path.join(results_dir, f"{env_key}__{algorithm}__*_results.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {env_key}")

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
    t_min, t_max = timesteps[0], timesteps[-1]
    edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    binned_returns = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (timesteps >= edges[i]) & (timesteps < edges[i + 1])
        if mask.sum() > 0:
            binned_returns[i] = returns[mask].mean()

    for i in range(1, n_bins):
        if np.isnan(binned_returns[i]):
            binned_returns[i] = binned_returns[i - 1]

    return bin_centers, binned_returns


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_kde_histogram(env_key, env_name, algorithm, results_dir, output_path):
    curves = load_seed_curves(env_key, algorithm, results_dir)
    print(f"Loaded {len(curves)} seed curves")

    # bin each seed curve
    binned_curves = []
    for timesteps, returns in curves:
        x_bin, y_bin = bin_curve(timesteps, returns, n_bins=N_BINS)
        binned_curves.append((x_bin, y_bin))

    # interpolate all seeds onto a common timestep grid for histogram bins
    x_min = max(c[0][0]  for c in binned_curves)
    x_max = min(c[0][-1] for c in binned_curves)
    x_common = np.linspace(x_min, x_max, N_BINS)

    # shape: (n_seeds, N_BINS)
    y_matrix = np.array([
        np.interp(x_common, x_bin, y_bin)
        for x_bin, y_bin in binned_curves
    ])

    # define histogram bin edges along the timestep axis
    hist_edges = np.linspace(x_min, x_max, N_HIST_BINS + 1)
    hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    hist_width = hist_edges[1] - hist_edges[0]

    # performance axis for KDE evaluation
    perf_min, perf_max = 0, 1000
    perf_grid = np.linspace(perf_min, perf_max, N_KDE_POINTS)

    fig, ax = plt.subplots(figsize=(10, 6))

    for bin_idx in range(N_HIST_BINS):
        # collect all seed values that fall in this timestep bin
        t_lo, t_hi = hist_edges[bin_idx], hist_edges[bin_idx + 1]
        col_mask = (x_common >= t_lo) & (x_common < t_hi)
        values = y_matrix[:, col_mask].flatten()

        if len(values) < 2:
            continue

        # Gaussian KDE with Scott's rule bandwidth
        kde = gaussian_kde(values, bw_method="scott")
        kde_vals = kde(perf_grid)

        # normalise KDE to fit within the histogram bin width for display
        kde_vals_norm = kde_vals / kde_vals.max() * hist_width * 0.9

        # draw KDE curve centred on the histogram bin
        ax.plot(
            hist_centers[bin_idx] + kde_vals_norm,
            perf_grid,
            color="steelblue",
            linewidth=1.2,
        )
        ax.fill_betweenx(
            perf_grid,
            hist_centers[bin_idx],
            hist_centers[bin_idx] + kde_vals_norm,
            color="steelblue",
            alpha=0.3,
        )

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Episodic Return", fontsize=12)
    ax.set_title(
        f"{env_name} — PPO (100 seeds)\nGaussian KDE per timestep bin (Scott 1992), {N_HIST_BINS} bins",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(perf_min, perf_max)
    ax.set_xlim(x_min - hist_width * 0.5, x_max + hist_width * 0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K"
    ))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_kde_histogram(ENV_KEY, ENV_NAME, ALGORITHM, RESULTS_DIR, OUTPUT_PATH)
