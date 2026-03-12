import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import regex as re
from pathlib import Path
from collections import defaultdict


plt.rcParams.update({
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
    "axes.formatter.use_mathtext": True,
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "svg.fonttype":'path',
})

CONFIG_COLORS = {
    "actor_normFalsecritic_normFalse": "#e74c3c",   # red   — baseline
    "actor_normFalsecritic_normTrue":  "#2ecc71",   # green — critic norm only
    "actor_normTruecritic_normFalse":  "#3498db",   # blue  — actor norm only
}

CONFIG_LABELS = {
    "actor_normFalsecritic_normFalse": "No norm (baseline)",
    "actor_normFalsecritic_normTrue":  "Critic norm only",
    "actor_normTruecritic_normFalse":  "Actor norm only",
}

BOX_CONFIG_LABELS = {
    "actor_normFalsecritic_normFalse": "No norm\n (baseline)",
    "actor_normFalsecritic_normTrue":  "Critic norm\n only",
    "actor_normTruecritic_normFalse":  "Actor norm\n only",
}

def plot_side_by_side(envs: list):
    fig, axes = plt.subplots(1, len(envs), figsize=(18, 7))

    for ax, env in zip(axes, envs):
        files_path = Path(f"logs/{env}")
        all_returns = []

        for file in sorted(files_path.iterdir()):
            if not re.search(r"\dactor_normFalsecritic_normTrue_evals\.csv$", str(file)):
                continue
            df = pd.read_csv(file)
            steps   = df["global_step"].to_numpy()
            returns = df["eval_return"].to_numpy()

            ax.plot(steps, returns, alpha=0.25, linewidth=1.2, color="#2ecc71")
            all_returns.append(returns)

        if all_returns:
            mean_returns = np.nanmean(all_returns, axis=0)
            ax.plot(steps, mean_returns,
                    linewidth=2.5,
                    color="#2ecc71",
                    label=f"Mean across {len(all_returns)} seeds")

        ax.set_xlabel("Time steps")
        ax.set_ylabel("Episodic return")
        ax.set_title(f"SAC {env}")
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1)
            ax.spines[spine].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel("Time steps")
        ax.set_ylabel("Episodic return")
        ax.set_title(f"{env}")
        ax.legend().set_visible(False)   # hide on all subplots
        ax.grid(True, alpha=0.3)

        # Deduplicate handles and labels
        handles, labels = axes[0].get_legend_handles_labels()

        legend = fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0),
            ncol=3,                  # one column per condition
            frameon=True,
            edgecolor='lightgray',
            facecolor='white',
            framealpha=1,
        )
        legend.get_frame().set_linewidth(1.5)

        plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave 8% space at bottom for legend
        plt.savefig(Path("logs") / f"eval_returns_critic_combined.svg",
                    format="svg", bbox_inches="tight")
        plt.show()

def extract_config(filename: str,file_suffix:str) -> str:
    # Strip env prefix and _evals suffix, then remove seed number
    after_seed = filename.split("seed")[1]          # "2actor_normFalsecritic_normTrue_evals"
    after_seed = after_seed.lstrip("0123456789")    # "actor_normFalsecritic_normTrue_evals"
    config = after_seed.replace(file_suffix.replace(".csv",""), "")   # "actor_normFalsecritic_normTrue"
    return config


def plot_configs_side_by_side(envs: list, file_suffix: str = "_evals.csv", column: str = "eval_return", log_interval: int = 1):
    fig, axes = plt.subplots(1, len(envs), figsize=(18, 7), dpi=100)


    for ax, env in zip(axes, envs):
        logs_path  = Path(f"logs/{env}")
        config_data = defaultdict(list)

        for file in sorted(logs_path.iterdir()):
            if not str(file).endswith(file_suffix):
                continue
            df = pd.read_csv(file)
            if df.empty:
                continue

            config = extract_config(file.stem, file_suffix)
            values = df[column].to_numpy()

            if "global_step" in df.columns:
                steps = df["global_step"].to_numpy()
            else:
                steps = np.arange(1, len(values) + 1) * log_interval

            config_data[config].append((steps, values))

        for config, seed_runs in config_data.items():
            color = CONFIG_COLORS.get(config, "#888888")
            label = CONFIG_LABELS.get(config, config)

            min_len     = min(len(s) for _, s in seed_runs)
            all_values  = np.array([s[:min_len] for _, s in seed_runs])
            all_steps   = seed_runs[0][0][:min_len]
            mean_values = np.nanmean(all_values, axis=0)

            ax.plot(all_steps, mean_values, color=color, linewidth=2.5, label=label)

        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1)
            ax.spines[spine].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel("Time steps")
        ax.set_ylabel("Episodic return")
        ax.set_title(f"{env}")
        ax.legend().set_visible(False)   # hide on all subplots
        ax.grid(True, alpha=0.3)

    # Deduplicate handles and labels
    handles, labels = axes[0].get_legend_handles_labels()

    legend = fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=3,                  # one column per condition
        frameon=True,
        edgecolor='lightgray',
        facecolor='white',
        framealpha=1,
    )
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave 8% space at bottom for legend
    plt.savefig(Path("logs") / f"layernorm_ablation_{column}_combined.svg",
                format="svg", bbox_inches="tight")
    plt.show()

def plot_configs(files_path: Path, file_suffix: str = "_evals.csv", column: str = "eval_return", log_interval:int=1):
    config_data = defaultdict(list)

    for file in sorted(files_path.iterdir()):
        if not str(file).endswith(file_suffix):
            continue
        df = pd.read_csv(file)
        if df.empty:
            continue
        
        config = extract_config(file.stem,file_suffix)
        values = df[column].to_numpy()

        if "global_step" in df.columns:
            steps = df["global_step"].to_numpy()
        else:
            steps = np.arange(1, len(values) + 1) * log_interval

        config_data[config].append((steps, values))

    fig, ax = plt.subplots(figsize=(12, 6))

    for config, seed_runs in config_data.items():
        color = CONFIG_COLORS.get(config, "#888888")
        label = CONFIG_LABELS.get(config, config)

        min_len    = min(len(s) for _, s in seed_runs)
        all_values = np.array([s[:min_len] for _, s in seed_runs])
        all_steps  = seed_runs[0][0][:min_len]
        mean_values = np.nanmean(all_values, axis=0)

        ax.plot(all_steps, mean_values,
                color=color, linewidth=2.5, label=label)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Episodic return")
    ax.set_title(f"SAC {env}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(files_path / f"{env}_layernorm_ablation_{column}.svg", format='svg')

LAST_20_PCT_START = 800_000  # steps 800k–1M

envs = ["Ant-v4", "Walker2d-v4"]


plot_configs_side_by_side(envs, file_suffix="_evals.csv", column="eval_return")
plot_side_by_side(envs)
