import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from models.wm import WorldModel
from config import STATE_DIM
from utils.preprocessing import one_hot

FIGURES_DIR = "artifacts/figures"

STATE_LABELS = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]


def collect_trajectory(model, mean, std, steps=50):
    env = gym.make("CartPole-v1")
    state, _ = env.reset()

    state_norm = (state - mean) / std
    state_pred = state_norm.copy()

    real_states = [state_norm.copy()]
    pred_states = [state_pred.copy()]

    for _ in range(steps):
        action = env.action_space.sample()

        # World model prediction
        input_vec = np.concatenate([state_pred, one_hot(action)])
        input_tensor = torch.tensor(input_vec, dtype=torch.float32)
        pred = model(input_tensor)
        delta = pred[:STATE_DIM].detach().numpy()
        state_pred = state_pred + delta
        pred_states.append(state_pred.copy())

        # Real env
        next_state, _, done, _, _ = env.step(action)
        state_norm = (next_state - mean) / std
        real_states.append(state_norm.copy())

        if done:
            break

    env.close()
    return np.array(real_states), np.array(pred_states)


def plot_trajectories(model_path="artifacts/models/wm.pt", mean_path="artifacts/models/state_mean.npy",
                      std_path="artifacts/models/state_std.npy", steps=50, figures_dir=FIGURES_DIR):
    model = WorldModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    mean = np.load(mean_path)
    std = np.load(std_path)

    real, pred = collect_trajectory(model, mean, std, steps=steps)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(real[:, i], label="Real", color="tab:orange")
        ax.plot(pred[:, i], label="Predicted", color="tab:blue", linestyle="--")
        ax.set_title(STATE_LABELS[i])
        ax.set_xlabel("Step")
        ax.set_ylabel("Value (normalized)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("World Model: Predicted vs Real Trajectories", fontsize=14)
    fig.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "trajectories.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    plot_trajectories()
