import numpy as np
import torch
from utils.preprocessing import one_hot
from utils.logger import logger


def multi_step_rollout(env, model, mean, std, steps=50):
    state, _ = env.reset()

    state_norm = (state - mean) / std
    state_pred = state_norm.copy()

    step_errors = []

    for step in range(steps):
        action = env.action_space.sample()

        input_vec = np.concatenate([state_pred, one_hot(action)])
        input_tensor = torch.tensor(input_vec, dtype=torch.float32)

        pred = model(input_tensor)

        delta = pred[:4].detach().numpy()
        state_pred = state_pred + delta

        next_state, _, done, _, _ = env.step(action)
        state_norm = (next_state - mean) / std

        error = float(np.mean((state_pred - state_norm) ** 2))
        step_errors.append(error)

        if done:
            break

    logger.info(f"Rollout: {len(step_errors)} steps | "
                f"Step 1 MSE: {step_errors[0]:.6f} | "
                f"Final MSE: {step_errors[-1]:.6f}")

    return step_errors
