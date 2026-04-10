import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from models.wm import WorldModel
from utils.evaluation import multi_step_rollout

FIGURES_DIR = "artifacts/figures"


def plot_wm_error(model_path="artifacts/models/wm.pt", mean_path="artifacts/models/state_mean.npy",
                  std_path="artifacts/models/state_std.npy", num_rollouts=20, steps=50, figures_dir=FIGURES_DIR):
    model = WorldModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    mean = np.load(mean_path)
    std = np.load(std_path)

    all_errors = []
    for _ in range(num_rollouts):
        env = gym.make("CartPole-v1")
        errors = multi_step_rollout(env, model, mean, std, steps=steps)
        env.close()
        all_errors.append(errors)

    # Pad shorter rollouts with NaN so we can average
    max_len = max(len(e) for e in all_errors)
    padded = np.full((len(all_errors), max_len), np.nan)
    for i, e in enumerate(all_errors):
        padded[i, :len(e)] = e

    mean_error = np.nanmean(padded, axis=0)
    std_error = np.nanstd(padded, axis=0)
    steps_axis = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps_axis, mean_error, color="tab:blue", label="Mean MSE")
    ax.fill_between(steps_axis, mean_error - std_error, mean_error + std_error,
                    color="tab:blue", alpha=0.2, label="± 1 std")

    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("MSE (predicted vs real state)")
    ax.set_title(f"World Model Compounding Error ({num_rollouts} rollouts)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "wm_error.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    plot_wm_error()
