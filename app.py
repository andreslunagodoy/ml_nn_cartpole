import streamlit as st
import os

from training.agent_dqn import DQNAgent
from training.agent_random import RandomAgent
from training.evaluate_agent import evaluate_agent
from config import *

st.set_page_config(page_title="CartPole RL", layout="wide")
st.title("Model-Based RL for CartPole")

# --- Sidebar: Agent Selection & Evaluation ---
st.sidebar.header("Evaluate Agent")

agent_choice = st.sidebar.selectbox("Agent", ["Model-based", "Model-free", "Random"])
num_episodes = st.sidebar.slider("Evaluation episodes", 5, 50, 15)

AGENT_PATHS = {
    "Model-based": AGENT_MODELBASED_LOAD_PATH,
    "Model-free": AGENT_MODELFREE_LOAD_PATH,
}

if st.sidebar.button("Run Evaluation"):
    if agent_choice == "Random":
        agent = RandomAgent()
    else:
        path = AGENT_PATHS[agent_choice]
        if not os.path.exists(path):
            st.sidebar.error(f"No saved model at {path}. Train first.")
            st.stop()
        agent = DQNAgent(4, 2)
        agent.load(path)

    with st.spinner(f"Evaluating {agent_choice} agent..."):
        results = evaluate_agent(agent, episodes=num_episodes, verbose=False)

    st.sidebar.success("Done!")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Reward", f"{results['mean']:.1f}")
    col2.metric("Std", f"{results['std']:.1f}")
    col3.metric("Max Reward", f"{results['max']:.1f}")

# --- Plots ---
st.header("Training Analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "Learning Curves", "Sample Efficiency", "Trajectories", "WM Compounding Error"
])

with tab1:
    path = "artifacts/figures/learning_curves.png"
    if os.path.exists(path):
        st.image(path)
    else:
        st.info("Run `python visualization/plot_learning_curves.py` to generate.")

with tab2:
    path = "artifacts/figures/sample_efficiency.png"
    if os.path.exists(path):
        st.image(path)
    else:
        st.info("Run `python visualization/plot_sample_efficiency.py` to generate.")

with tab3:
    path = "artifacts/figures/trajectories.png"
    if os.path.exists(path):
        st.image(path)
    else:
        st.info("Run `python visualization/plot_trajectories.py` to generate.")

with tab4:
    path = "artifacts/figures/wm_error.png"
    if os.path.exists(path):
        st.image(path)
    else:
        st.info("Run `python visualization/plot_wm_error.py` to generate.")
