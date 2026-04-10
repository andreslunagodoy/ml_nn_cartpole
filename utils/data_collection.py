import gymnasium as gym

def collect_data(env_name="CartPole-v1", num_episodes=300):
    env = gym.make(env_name)
    data = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            data.append((state, action, next_state, reward))
            state = next_state

    env.close()
    return data, len(data)