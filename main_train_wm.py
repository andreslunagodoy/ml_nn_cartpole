import numpy as np
import torch
from sklearn.model_selection import train_test_split
import gymnasium as gym

from utils.data_collection import collect_data
from utils.preprocessing import (
    process_data,
    normalize_states,
    create_inputs,
    compute_deltas
)
from models.wm import WorldModel
from training.train_wm import train, evaluate
from utils.evaluation import multi_step_rollout
from utils.logger import logger
from config import REWARD_WEIGHT


def train_world_model(reward_weight=None, model_path=None,
                      mean_path=None, std_path=None, evaluate_rollout=False):
    model_path = model_path or "artifacts/models/wm.pt"
    mean_path = mean_path or "artifacts/models/state_mean.npy"
    std_path = std_path or "artifacts/models/state_std.npy"
    reward_weight = reward_weight if reward_weight is not None else REWARD_WEIGHT

    # 1. Collect data
    data, num_transitions = collect_data(num_episodes=300)
    logger.info(f"Collected {num_transitions} transitions from real environment")

    # 2. Process
    states, actions, next_states, rewards = process_data(data)

    # 3. Normalize
    states, next_states, mean, std = normalize_states(states, next_states)

    # Save stats
    np.save(mean_path, mean)
    np.save(std_path, std)

    # 4. Inputs + targets
    inputs = create_inputs(states, actions)
    deltas = compute_deltas(states, next_states)

    # 5. Train/val split
    X_train, X_val, d_train, d_val, r_train, r_val = train_test_split(
        inputs, deltas, rewards, test_size=0.2, stratify=actions
    )

    # 6. Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    d_train = torch.tensor(d_train, dtype=torch.float32)
    d_val = torch.tensor(d_val, dtype=torch.float32)
    r_train = torch.tensor(r_train, dtype=torch.float32)
    r_val = torch.tensor(r_val, dtype=torch.float32)

    # 7. Model
    model = WorldModel()

    # 8. Train
    train(model, X_train, d_train, r_train, X_val, d_val, r_val, reward_weight=reward_weight)

    # 9. Evaluate model accuracy
    val_metrics = evaluate(model, X_val, d_val, r_val, reward_weight)
    logger.info(f"World model val metrics — State MSE: {val_metrics['state_mse']:.6f} | Reward MSE: {val_metrics['reward_mse']:.6f}")

    # 10. Save model
    torch.save(model.state_dict(), model_path)
    logger.info(f"World model saved to {model_path}")

    # 11. Evaluate rollout
    if evaluate_rollout:
        env = gym.make("CartPole-v1")
        multi_step_rollout(env, model, mean, std)

    return num_transitions


if __name__ == "__main__":
    train_world_model()
