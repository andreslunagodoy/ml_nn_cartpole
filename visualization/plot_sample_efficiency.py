import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import os

LOGS_DIR = "artifacts/logs"
FIGURES_DIR = "artifacts/figures"


def latest_csv(pattern, logs_dir=LOGS_DIR):
    files = sorted(glob.glob(os.path.join(logs_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSV matching {pattern} in {logs_dir}/")
    return files[-1]


def parse_results_from_log(logs_dir=LOGS_DIR):
    log_path = latest_csv("run_*.log", logs_dir)
    with open(log_path) as f:
        for line in f:
            match = re.search(
                r"Results — Model-based: ([\d.]+) \(real interactions: (\d+)\)",
                line,
            )
            if match:
                return float(match.group(1)), int(match.group(2))
    raise ValueError("No results line found in latest log")


def plot_sample_efficiency(window=20, logs_dir=LOGS_DIR, figures_dir=FIGURES_DIR):
    mb_path = latest_csv("agent_modelbased_*.csv", logs_dir)
    mf_path = latest_csv("agent_modelfree_*.csv", logs_dir)

    mb = pd.read_csv(mb_path)
    mf = pd.read_csv(mf_path)

    mb_eval_reward, wm_data_collection_steps = parse_results_from_log(logs_dir)
    mb_train_reward = mb["reward"].rolling(window).mean().iloc[-1]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(mf["total_steps"], mf["reward"].rolling(window).mean(),
            color="tab:orange", label="Model-free (real env)", alpha=0.9)
    ax.axhline(y=mb_train_reward,
               color="tab:blue", linestyle="-", linewidth=2,
               label=f"Model-based training reward (world model): {mb_train_reward:.0f}")
    ax.axhline(y=mb_eval_reward,
               color="tab:blue", linestyle=":", linewidth=2,
               label=f"Model-based eval reward (real env): {mb_eval_reward:.0f}")
    ax.axhline(y=22, color="tab:red", linestyle="--", label="Random (~22)")

    ax.set_xlabel("Real Environment Interactions")
    ax.set_ylabel(f"Reward ({window}-ep rolling avg)")
    ax.set_title("Sample Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "sample_efficiency.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    plot_sample_efficiency()
