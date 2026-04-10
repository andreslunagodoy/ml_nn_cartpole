# project/training/evaluate_agent.py
import gymnasium as gym
import numpy as np
from utils.logger import logger

def evaluate_agent(agent, env_name='CartPole-v1', episodes=15, render=False, verbose=True):
    """
    Evaluate a trained agent in a Gym environment.

    Args:
        agent: RL agent object with a select_action(state) method
        env_name: string, Gym environment name
        episodes: number of evaluation episodes
        render: bool, whether to render the environment
    Returns:
        avg_reward: float, average total reward over episodes
    """
    env = gym.make(env_name)
    total_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if render:
                env.render()
        
        total_rewards.append(ep_reward)

    env.close()
    results = {
        'mean': float(np.mean(total_rewards)),
        'std': float(np.std(total_rewards)),
        'max': float(np.max(total_rewards)),
    }
    if verbose:
        logger.info(f"Evaluation ({episodes} eps): mean={results['mean']:.1f}, std={results['std']:.1f}, max={results['max']:.1f}")
    return results