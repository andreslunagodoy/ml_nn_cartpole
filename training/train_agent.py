# project/training/train_agent.py
from utils.logger import logger, CSVLogger

def train_agent(env, agent, num_episodes=500, target_update_freq=10, gym_env=False, csv_name="agent_training"):
    total_steps = 0
    csv_logger = CSVLogger(csv_name, ["episode", "reward", "total_steps"])

    for ep in range(num_episodes):
        if gym_env:
            state, _ = env.reset()
        else:
            state = env.reset()

        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)

            if gym_env:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
            total_steps += 1

        if ep % target_update_freq == 0:
            agent.update_target()

        csv_logger.log(ep, total_reward, total_steps)

        if ep % 50 == 0:
            logger.info(f"Agent Episode {ep}/{num_episodes} - Reward: {total_reward:.2f}")

    csv_logger.close()
    return total_steps
