import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym

from training.agent_dqn import DQNAgent
from training.agent_random import RandomAgent
from training.evaluate_agent import evaluate_agent
from config import STATE_DIM, ACTION_DIM

RUNS_DIR = "runs"

st.set_page_config(page_title="CartPole MBRL", layout="wide")
st.title("Model-Based RL for CartPole")
st.caption(
    "Exploring model-based reinforcement learning: can a learned world model reduce "
    "the real environment interactions needed to train a good CartPole policy?"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_runs():
    if not os.path.isdir(RUNS_DIR):
        return []
    runs = sorted(
        [d for d in os.listdir(RUNS_DIR)
         if os.path.isdir(os.path.join(RUNS_DIR, d))],
        reverse=True,
    )
    return runs


def load_summary(run_dir):
    path = os.path.join(run_dir, "summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def latest_csv(pattern, logs_dir):
    files = sorted(glob.glob(os.path.join(logs_dir, pattern)))
    return files[-1] if files else None


def load_training_csv(pattern, logs_dir):
    path = latest_csv(pattern, logs_dir)
    if path:
        return pd.read_csv(path)
    return None


def run_episode_with_recording(agent):
    """Run one episode and record every state and action."""
    env = gym.make("CartPole-v1")
    state, _ = env.reset()
    states = [state.copy()]
    actions = []
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        actions.append(action)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(state.copy())
        total_reward += reward

    env.close()
    return np.array(states), actions, total_reward


def draw_cartpole(state, step, total_steps, action=None):
    """Draw a cartpole frame from raw state [cart_pos, cart_vel, pole_angle, pole_ang_vel]."""
    cart_pos, _, pole_angle, _ = state

    cart_w, cart_h = 0.4, 0.2
    pole_len = 0.6

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Track
    ax.plot([-2.4, 2.4], [0, 0], "k-", linewidth=2)

    # Cart
    cart_x = cart_pos - cart_w / 2
    cart = patches.FancyBboxPatch(
        (cart_x, 0), cart_w, cart_h,
        boxstyle="round,pad=0.02", facecolor="tab:blue", edgecolor="black", linewidth=1.5,
    )
    ax.add_patch(cart)

    # Wheels
    for wx in [cart_pos - cart_w / 4, cart_pos + cart_w / 4]:
        wheel = plt.Circle((wx, 0), 0.03, color="black")
        ax.add_patch(wheel)

    # Pole
    pole_base = (cart_pos, cart_h)
    pole_tip = (
        cart_pos + pole_len * np.sin(pole_angle),
        cart_h + pole_len * np.cos(pole_angle),
    )
    ax.plot([pole_base[0], pole_tip[0]], [pole_base[1], pole_tip[1]],
            color="tab:orange", linewidth=4, solid_capstyle="round")

    # Pivot dot
    ax.plot(cart_pos, cart_h, "o", color="black", markersize=5, zorder=5)

    # Action arrow
    if action is not None:
        arrow_y = cart_h / 2
        arrow_dx = 0.3 if action == 1 else -0.3
        ax.annotate("", xy=(cart_pos + arrow_dx, arrow_y), xytext=(cart_pos, arrow_y),
                     arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
        label = "Push right" if action == 1 else "Push left"
        ax.text(cart_pos, -0.15, label, ha="center", fontsize=9, color="tab:red")

    # Termination zone shading
    ax.axvspan(-3, -2.4, alpha=0.08, color="red")
    ax.axvspan(2.4, 3, alpha=0.08, color="red")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.3, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"Step {step}/{total_steps - 1}", fontsize=11)
    ax.axis("off")

    return fig


# ---------------------------------------------------------------------------
# Sidebar — run selection
# ---------------------------------------------------------------------------

runs = list_runs()

if not runs:
    st.warning("No experiment runs found in `runs/`. Run `python run_experiments.py` first.")
    st.stop()

st.sidebar.header("Experiment Runs")
selected_run = st.sidebar.selectbox("Select run", runs, format_func=lambda r: f"Run {r}")
run_dir = os.path.join(RUNS_DIR, selected_run)
logs_dir = os.path.join(run_dir, "logs")
models_dir = os.path.join(run_dir, "models")
figures_dir = os.path.join(run_dir, "figures")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_analysis, tab_eval = st.tabs(["Overview", "Run Analysis", "Agent Evaluation"])

# ===========================================================================
# TAB 0: Overview
# ===========================================================================

with tab_overview:

    st.header("How it works")
    st.markdown("""
This project trains a **DQN agent** to solve CartPole using two approaches and compares them:

1. **Collect data** -- a random policy gathers ~7k transitions from the real CartPole environment.
2. **Train a world model** -- an MLP learns to predict the next state and reward from (state, action) pairs.
3. **Train a model-based agent** -- a DQN agent trains entirely inside the learned world model, using zero additional real interactions.
4. **Train a model-free agent** -- a standard DQN trains directly on the real environment (~70-100k interactions) as a baseline.
5. **Evaluate** -- all agents are tested on the real CartPole environment.

The central trade-off: the model-based agent is far more sample-efficient, but the world model's compounding
prediction error creates a sim-to-real gap that limits transfer performance.
""")

    st.subheader("Agents")
    agent_col1, agent_col2, agent_col3 = st.columns(3)
    with agent_col1:
        st.markdown("**Model-based DQN**")
        st.markdown("Trained inside the learned world model. Uses ~10x fewer real interactions.")
    with agent_col2:
        st.markdown("**Model-free DQN**")
        st.markdown("Trained directly on the real environment. The performance baseline.")
    with agent_col3:
        st.markdown("**Random**")
        st.markdown("Selects actions uniformly at random. The floor baseline (~22 reward).")

    st.divider()
    st.markdown(
        "Use the **Run Analysis** tab to explore training data from past experiments, "
        "or the **Agent Evaluation** tab to run a saved agent on the real environment "
        "and watch it play step by step."
    )


# ===========================================================================
# TAB 1: Run Analysis
# ===========================================================================

with tab_analysis:

    st.markdown(
        "Training metrics and plots for a completed experiment run. "
        "Select a run from the sidebar, and optionally compare it against another."
    )

    # --- Run comparison (scoped to this tab) ---
    compare = st.checkbox("Compare with another run")
    compare_run = None
    if compare and len(runs) > 1:
        other_runs = [r for r in runs if r != selected_run]
        compare_run = st.selectbox("Compare run", other_runs, format_func=lambda r: f"Run {r}")

    # --- Summary metrics ---
    st.header("Results Summary")
    summary = load_summary(run_dir)

    if summary is not None:
        cols = st.columns(len(summary))
        for col, (_, row) in zip(cols, summary.iterrows()):
            col.metric(row["agent"], f"{row['mean']} avg", f"\u00b1 {row['std']} std")

        if compare_run:
            compare_dir = os.path.join(RUNS_DIR, compare_run)
            summary_b = load_summary(compare_dir)
            if summary_b is not None:
                merged = summary.merge(summary_b, on="agent", suffixes=(f" ({selected_run})", f" ({compare_run})"))
                st.dataframe(merged, use_container_width=True, hide_index=True)
    else:
        st.info("No summary.csv found for this run.")

    # --- Learning curves & sample efficiency ---
    st.header("Training Analysis")
    window = st.slider("Smoothing window (episodes)", 1, 50, 20, key="lc_window")

    mb_df = load_training_csv("agent_modelbased_*.csv", logs_dir)
    mf_df = load_training_csv("agent_modelfree_*.csv", logs_dir)

    col_lc, col_se = st.columns(2)

    with col_lc:
        st.subheader("Learning Curves")
        if mb_df is not None or mf_df is not None:
            fig, ax = plt.subplots()
            episodes = mb_df["episode"] if mb_df is not None else mf_df["episode"]
            if mb_df is not None:
                ax.plot(episodes, mb_df["reward"].rolling(window, min_periods=1).mean(), label="Model-based")
            if mf_df is not None:
                ax.plot(episodes, mf_df["reward"].rolling(window, min_periods=1).mean(), label="Model-free")

            if compare_run:
                compare_logs = os.path.join(RUNS_DIR, compare_run, "logs")
                mb_b = load_training_csv("agent_modelbased_*.csv", compare_logs)
                mf_b = load_training_csv("agent_modelfree_*.csv", compare_logs)
                if mb_b is not None:
                    ax.plot(mb_b["episode"], mb_b["reward"].rolling(window, min_periods=1).mean(),
                            label=f"Model-based ({compare_run})", linestyle="--")
                if mf_b is not None:
                    ax.plot(mf_b["episode"], mf_b["reward"].rolling(window, min_periods=1).mean(),
                            label=f"Model-free ({compare_run})", linestyle="--")

            ax.set_xlabel("Episode")
            ax.set_ylabel(f"Reward ({window}-ep avg)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No agent training CSVs found for this run.")

    with col_se:
        st.subheader("Sample Efficiency")
        if mb_df is not None and mf_df is not None:
            fig, ax = plt.subplots()
            ax.plot(mf_df["total_steps"], mf_df["reward"].rolling(window, min_periods=1).mean(),
                    color="tab:orange", label="Model-free (real env)")
            ax.set_xlabel("Real Environment Steps")
            ax.set_ylabel(f"Reward ({window}-ep avg)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            if summary is not None:
                mb_row = summary[summary["agent"] == "Model-based"]
                if not mb_row.empty:
                    st.caption(
                        f"Model-based agent used ~7k real interactions "
                        f"and achieved **{mb_row.iloc[0]['mean']}** mean reward."
                    )
        else:
            st.info("Need both agent CSVs to show sample efficiency.")

    # --- World model training ---
    st.header("World Model Training")

    wm_df = load_training_csv("wm_training_*.csv", logs_dir)
    if wm_df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Loss")
            fig, ax = plt.subplots()
            ax.plot(wm_df["epoch"], wm_df["train_loss"], label="Train")
            ax.plot(wm_df["epoch"], wm_df["val_loss"], label="Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Validation Breakdown")
            fig, ax = plt.subplots()
            ax.plot(wm_df["epoch"], wm_df["val_state_mse"], label="State MSE")
            ax.plot(wm_df["epoch"], wm_df["val_reward_mse"], label="Reward MSE")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        final = wm_df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Val Loss", f"{final['val_loss']:.6f}")
        c2.metric("State MSE", f"{final['val_state_mse']:.6f}")
        c3.metric("Reward MSE", f"{final['val_reward_mse']:.6f}")
    else:
        st.info("No world model training CSV found for this run.")

    # --- Pre-generated figures ---
    st.header("World Model Analysis")
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.subheader("Predicted vs Real Trajectories")
        traj_path = os.path.join(figures_dir, "trajectories.png")
        if os.path.exists(traj_path):
            st.image(traj_path)
        else:
            st.info("No trajectories plot found for this run.")

    with fig_col2:
        st.subheader("Compounding Error")
        wm_err_path = os.path.join(figures_dir, "wm_error.png")
        if os.path.exists(wm_err_path):
            st.image(wm_err_path)
        else:
            st.info("No WM error plot found for this run.")


# ===========================================================================
# TAB 2: Agent Evaluation
# ===========================================================================

with tab_eval:

    st.markdown(
        "Run a saved agent on the real CartPole environment and see aggregate statistics. "
        "One episode is recorded so you can replay it step by step below."
    )

    st.header("Evaluate Agent")

    eval_col1, eval_col2 = st.columns([1, 2])

    with eval_col1:
        agent_source = st.radio("Agent source", ["From selected run", "Best saved agent"])
        agent_type = st.selectbox("Agent type", ["Model-based", "Model-free", "Random"])
        num_eval_episodes = st.slider("Episodes", 5, 100, 20)
        run_eval = st.button("Run Evaluation")

    if run_eval:
        if agent_type == "Random":
            agent = RandomAgent()
        else:
            if agent_source == "Best saved agent":
                path = "artifacts/best/agent_best.pt"
            else:
                filename = "agent_modelbased.pt" if agent_type == "Model-based" else "agent_modelfree.pt"
                path = os.path.join(models_dir, filename)

            if not os.path.exists(path):
                st.error(f"Model not found: {path}")
                st.stop()

            agent = DQNAgent(STATE_DIM, ACTION_DIM)
            agent.load(path)

        with st.spinner(f"Evaluating {agent_type}..."):
            results = evaluate_agent(agent, episodes=num_eval_episodes, verbose=False)
            # Record one episode for replay
            replay_states, replay_actions, replay_reward = run_episode_with_recording(agent)

        st.session_state["eval_results"] = results
        st.session_state["replay_states"] = replay_states
        st.session_state["replay_actions"] = replay_actions
        st.session_state["replay_reward"] = replay_reward

    # Show results if we have them
    if "eval_results" in st.session_state:
        results = st.session_state["eval_results"]

        with eval_col2:
            e1, e2, e3 = st.columns(3)
            e1.metric("Mean Reward", f"{results['mean']:.1f}")
            e2.metric("Std", f"{results['std']:.1f}")
            e3.metric("Max Reward", f"{results['max']:.1f}")

    if "replay_states" in st.session_state:
        st.divider()
        st.header("Episode Replay")

        states = st.session_state["replay_states"]
        actions = st.session_state["replay_actions"]
        replay_reward = st.session_state["replay_reward"]
        total_steps = len(states)
        max_step = total_steps - 1

        st.caption(f"Episode length: {max_step} steps | Total reward: {replay_reward:.0f}")

        # Initialize slider state
        if "replay_slider" not in st.session_state:
            st.session_state["replay_slider"] = 0

        # Navigation buttons — callbacks update the slider's own key
        def go_start(): st.session_state["replay_slider"] = 0
        def go_prev():  st.session_state["replay_slider"] = max(0, st.session_state["replay_slider"] - 1)
        def go_next():  st.session_state["replay_slider"] = min(max_step, st.session_state["replay_slider"] + 1)
        def go_end():   st.session_state["replay_slider"] = max_step

        btn_cols = st.columns([1, 1, 1, 1, 6])
        with btn_cols[0]:
            st.button("\u23ee Start", on_click=go_start)
        with btn_cols[1]:
            st.button("\u25c0 Prev", on_click=go_prev)
        with btn_cols[2]:
            st.button("Next \u25b6", on_click=go_next)
        with btn_cols[3]:
            st.button("End \u23ed", on_click=go_end)

        step = st.slider("Step", 0, max_step, key="replay_slider")

        action = actions[step] if step < len(actions) else None
        fig = draw_cartpole(states[step], step, total_steps, action=action)
        st.pyplot(fig)
        plt.close(fig)

        # State details
        with st.expander("State details"):
            labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
            s = states[step]
            detail_cols = st.columns(4)
            for col, label, val in zip(detail_cols, labels, s):
                col.metric(label, f"{val:.4f}")
