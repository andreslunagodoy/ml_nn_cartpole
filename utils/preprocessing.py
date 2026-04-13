import numpy as np
from config import ACTION_DIM

def process_data(data):
    states, actions, next_states, rewards = [], [], [], []

    for s, a, ns, r in data:
        states.append(s)
        actions.append(a)
        next_states.append(ns)
        rewards.append(r)

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(next_states, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
    )

def one_hot(actions, num_actions=ACTION_DIM):
    return np.eye(num_actions)[actions]

def normalize_states(states, next_states):
    mean = states.mean(axis=0)
    std = states.std(axis=0) + 1e-8

    states_norm = (states - mean) / std
    next_states_norm = (next_states - mean) / std

    return states_norm, next_states_norm, mean, std

def create_inputs(states, actions):
    actions_oh = one_hot(actions)
    return np.concatenate([states, actions_oh], axis=1)

def compute_deltas(states, next_states):
    return next_states - states


def normalize_state(state, mean, std):
    """
    Normalize a single state using precomputed mean and std.
    
    Args:
        state: np.array of shape (state_dim,)
        mean: np.array of shape (state_dim,)
        std: np.array of shape (state_dim,)
    
    Returns:
        normalized_state: np.array of shape (state_dim,)
    """
    return (state - mean) / std

def denormalize_state(norm_state, mean, std):
    """
    Convert a normalized state back to original scale.
    
    Args:
        norm_state: np.array of shape (state_dim,)
        mean: np.array of shape (state_dim,)
        std: np.array of shape (state_dim,)
    
    Returns:
        original_state: np.array of shape (state_dim,)
    """
    return norm_state * std + mean