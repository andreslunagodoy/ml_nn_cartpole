# project/training/env_wm.py
import torch
import numpy as np
from models.wm import WorldModel
from utils.preprocessing import normalize_state
from config import STATE_DIM, ACTION_DIM, CART_POS_LIMIT, POLE_ANGLE_LIMIT

DEFAULT_MODEL_PATH = "artifacts/models/wm.pt"
DEFAULT_STATE_MEAN_PATH = "artifacts/models/state_mean.npy"
DEFAULT_STATE_STD_PATH = "artifacts/models/state_std.npy"


class WorldModelEnv:
    def __init__(self, model_path=None, state_mean_path=None,
                 state_std_path=None, device="cpu", max_steps=200):
        model_path = model_path or DEFAULT_MODEL_PATH
        state_mean_path = state_mean_path or DEFAULT_STATE_MEAN_PATH
        state_std_path = state_std_path or DEFAULT_STATE_STD_PATH

        self.device = device
        self.model = WorldModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()

        self.state_mean = np.load(state_mean_path)
        self.state_std = np.load(state_std_path)

        self.max_steps = max_steps
        self.step_count = 0
        self.state = None

    def reset(self):
        raw_state = np.random.uniform(-0.05, 0.05, size=self.state_mean.shape)
        self.state = normalize_state(raw_state, self.state_mean, self.state_std)
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        self.step_count += 1
        action_oh = np.eye(ACTION_DIM)[action]
        state_action_np = np.concatenate([self.state, action_oh])
        state_action = torch.tensor(state_action_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(state_action)

        assert pred.shape == (1, STATE_DIM + 1), f"Unexpected model output shape: {pred.shape}"
        delta = pred[0, :STATE_DIM].cpu().numpy()
        reward = pred[0, STATE_DIM].item()

        next_state = self.state + delta
        self.state = next_state

        raw = next_state * self.state_std + self.state_mean
        if not np.isfinite(raw).all():
            return next_state.copy(), 0.0, True, {}
        cart_pos, _, pole_angle, _ = raw
        fell = abs(cart_pos) > CART_POS_LIMIT or abs(pole_angle) > POLE_ANGLE_LIMIT
        if fell:
            reward = 0.0
        done = fell or self.step_count >= self.max_steps

        return next_state.copy(), reward, done, {}
