import glob
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


def plot_learning_curves(window=20, logs_dir=LOGS_DIR, figures_dir=FIGURES_DIR):
    mb_path = latest_csv("agent_modelbased_*.csv", logs_dir)
    mf_path = latest_csv("agent_modelfree_*.csv", logs_dir)

    mb = pd.read_csv(mb_path)
    mf = pd.read_csv(mf_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(mb["episode"], mb["reward"].rolling(window).mean(),
            label="Model-based", alpha=0.9)
    ax.plot(mf["episode"], mf["reward"].rolling(window).mean(),
            label="Model-free", alpha=0.9)
    ax.axhline(y=22, color="gray", linestyle="--", label="Random (~22)")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-ep rolling avg)")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "learning_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    plot_learning_curves()
