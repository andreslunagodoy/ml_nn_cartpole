# project/training/env_wm.py
import torch
import numpy as np
from models.wm import WorldModel
from models.wm_cc import WorldModelCC
from utils.preprocessing import normalize_state


MODEL_CLASSES = {
    "standard": WorldModel,
    "cc": WorldModelCC,
}

DEFAULTS = {
    "standard": {
        "model_path": "artifacts/models/wm.pt",
        "state_mean_path": "artifacts/models/state_mean.npy",
        "state_std_path": "artifacts/models/state_std.npy",
    },
    "cc": {
        "model_path": "artifacts/models/wm_cc.pt",
        "state_mean_path": "artifacts/models/state_mean_cc.npy",
        "state_std_path": "artifacts/models/state_std_cc.npy",
    },
}


class WorldModelEnv:
    def __init__(self, variant="standard", model_path=None, state_mean_path=None,
                 state_std_path=None, device="cpu", max_steps=200):
        defaults = DEFAULTS[variant]
        model_path = model_path or defaults["model_path"]
        state_mean_path = state_mean_path or defaults["state_mean_path"]
        state_std_path = state_std_path or defaults["state_std_path"]

        self.device = device
        self.model = MODEL_CLASSES[variant]()
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
        action_oh = np.eye(2)[action]
        state_action_np = np.concatenate([self.state, action_oh])
        state_action = torch.tensor(state_action_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(state_action)

        if isinstance(pred, tuple):
            delta = pred[0].cpu().numpy().squeeze()
            reward = pred[1][0, 0].item()
        else:
            delta = pred[:, :4].cpu().numpy().squeeze()
            reward = pred[0, 4].item()

        next_state = self.state + delta
        self.state = next_state

        raw = next_state * self.state_std + self.state_mean
        cart_pos, _, pole_angle, _ = raw
        fell = abs(cart_pos) > 2.4 or abs(pole_angle) > 0.2094
        done = fell or self.step_count >= self.max_steps

        return next_state.copy(), reward, done, {}
