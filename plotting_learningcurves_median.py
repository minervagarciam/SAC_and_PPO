import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Configuration ────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
OUTPUT_PATH = "plots/sac_learning_curves.png"
ALGORITHM   = "sac_continuous_action"

ENVS = {
    "dm_control_pendulum-swingup-v0":  "Pendulum Swingup",
    "dm_control_cartpole-swingup-v0":  "Cartpole Swingup",
    "dm_control_reacher-easy-v0":      "Reacher Easy",
    "dm_control_hopper-stand-v0":      "Hopper Stand",
    "dm_control_cheetah-run-v0":       "Cheetah Run",
}

# Number of points to interpolate each curve onto (for smooth alignment)
N_INTERP = 1000

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


def interpolate_curves(curves, n_points=N_INTERP):
    """Interpolate all seed curves onto a common timestep grid."""
    if not curves:
        return None, None

    # common x axis: from the max of all minimums to the min of all maximums
    x_min = max(c[0][0]  for c in curves)
    x_max = min(c[0][-1] for c in curves)

    if x_min >= x_max:
        # fallback: use the global range
        x_min = min(c[0][0]  for c in curves)
        x_max = max(c[0][-1] for c in curves)

    x_common = np.linspace(x_min, x_max, n_points)

    interpolated = []
    for timesteps, returns in curves:
        y_interp = np.interp(x_common, timesteps, returns)
        interpolated.append(y_interp)

    return x_common, np.array(interpolated)  # shape: (n_seeds, n_points)


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

        x_common, y_matrix = interpolate_curves(curves)

        # plot each seed as a thin, faint line
        for y_seed in y_matrix:
            ax.plot(x_common, y_seed, color="steelblue", alpha=0.25, linewidth=0.8)

        # bold median curve
        y_median = np.median(y_matrix, axis=0)
        ax.plot(x_common, y_median, color="steelblue", linewidth=2.5, label="Median")

        ax.set_title(env_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Environment Steps", fontsize=10)
        ax.set_ylabel("Episodic Return", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # hide any unused subplots
    for ax_idx in range(len(envs), len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle("SAC — DM Control Suite (10 seeds)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_sac(ENVS, ALGORITHM, RESULTS_DIR, OUTPUT_PATH)