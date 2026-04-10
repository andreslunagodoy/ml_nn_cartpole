# project/main.py
import os

import gymnasium as gym
from training.env_wm import WorldModelEnv
from training.agent_dqn import DQNAgent
from training.agent_random import RandomAgent
from training.train_agent import train_agent
from training.evaluate_agent import evaluate_agent
from main_train_wm import train_world_model
from config import *
from utils.logger import logger


def save_if_best(agent, avg_reward):
    os.makedirs(BEST_DIR, exist_ok=True)
    best_reward = float("-inf")
    if os.path.exists(BEST_REWARD_PATH):
        best_reward = float(open(BEST_REWARD_PATH).read())
    if avg_reward > best_reward:
        agent.save(BEST_AGENT_PATH)
        with open(BEST_REWARD_PATH, "w") as f:
            f.write(str(avg_reward))
        logger.info(f"New best agent saved with avg reward: {avg_reward:.1f} (previous best: {best_reward:.1f})")


def train_or_load_agent(name, retrain, save_path, load_path, train_fn):
    need_training = retrain or not os.path.exists(load_path)
    if need_training:
        agent, real_interactions = train_fn()
        agent.save(save_path)
        logger.info(f"{name} agent saved to {save_path} | real interactions: {real_interactions}")
    else:
        logger.info(f"Loading {name} agent from {load_path}")
        agent = DQNAgent(4, 2)
        agent.load(load_path)
        real_interactions = 0
    return agent, need_training, real_interactions


def train_modelbased(use_cc, retrain_wm, model_path, mean_path, std_path, reward_weight, model_dir="artifacts/models"):
    variant = "cc" if use_cc else "standard"
    suffix = "_cc" if use_cc else ""
    default_model_path = os.path.join(model_dir, "wm_cc.pt" if use_cc else "wm.pt")
    model_path = model_path or default_model_path
    mean_path = mean_path or os.path.join(model_dir, f"state_mean{suffix}.npy")
    std_path = std_path or os.path.join(model_dir, f"state_std{suffix}.npy")

    real_interactions = 0
    if retrain_wm or not os.path.exists(model_path):
        logger.info("Training world model...")
        real_interactions = train_world_model(
            use_cc=use_cc, reward_weight=reward_weight,
            model_path=model_path, mean_path=mean_path,
            std_path=std_path)
        logger.info(f"World model saved to {model_path}")

    env = WorldModelEnv(variant=variant, model_path=model_path,
                        state_mean_path=mean_path, state_std_path=std_path)
    logger.info(f"Training model-based agent on world model ({variant})")
    agent = DQNAgent(env.state_mean.shape[0], 2)
    train_agent(env, agent, num_episodes=500, target_update_freq=10, gym_env=False, csv_name="agent_modelbased")
    return agent, real_interactions


def train_modelfree():
    env = gym.make("CartPole-v1")
    logger.info("Training model-free agent on real CartPole environment")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    real_interactions = train_agent(env, agent, num_episodes=500, target_update_freq=10, gym_env=True, csv_name="agent_modelfree")
    env.close()
    return agent, real_interactions


def main(use_cc=USE_CC, retrain_agent=RETRAIN_AGENT, retrain_wm=RETRAIN_WM,
         retrain_modelfree=RETRAIN_MODELFREE,
         model_path=None, mean_path=None, std_path=None, reward_weight=None,
         model_dir="artifacts/models"):

    mb_save = os.path.join(model_dir, "agent_modelbased.pt")
    mf_save = os.path.join(model_dir, "agent_modelfree.pt")
    os.makedirs(model_dir, exist_ok=True)

    # 1. Model-based agent
    logger.info("=== Model-based agent ===")
    mb_agent, mb_trained, mb_interactions = train_or_load_agent(
        "model-based", retrain_agent,
        mb_save, AGENT_MODELBASED_LOAD_PATH,
        lambda: train_modelbased(use_cc, retrain_wm, model_path, mean_path, std_path, reward_weight, model_dir),
    )
    mb_results = evaluate_agent(mb_agent, env_name='CartPole-v1', episodes=15, render=False)
    if mb_trained:
        save_if_best(mb_agent, mb_results['mean'])

    # 2. Model-free agent
    logger.info("=== Model-free agent ===")
    mf_agent, _, mf_interactions = train_or_load_agent(
        "model-free", retrain_modelfree,
        mf_save, AGENT_MODELFREE_LOAD_PATH,
        train_modelfree,
    )
    mf_results = evaluate_agent(mf_agent, env_name='CartPole-v1', episodes=15, render=False)

    # 3. Random agent
    logger.info("=== Random agent ===")
    rand_agent = RandomAgent()
    rand_results = evaluate_agent(rand_agent, env_name='CartPole-v1', episodes=15, render=False)

    # Summary
    logger.info(
        f"Results — Model-based: {mb_results['mean']:.1f} (real interactions: {mb_interactions}) "
        f"| Model-free: {mf_results['mean']:.1f} (real interactions: {mf_interactions}) "
        f"| Random: {rand_results['mean']:.1f} (real interactions: 0)"
    )
    return mb_results, mf_results, rand_results


if __name__ == "__main__":
    main()
