import os

USE_CC = False

# Model-based agent
RETRAIN_AGENT = False
RETRAIN_WM = False
AGENT_MODELBASED_SAVE_PATH = "artifacts/models/agent_modelbased.pt"
#AGENT_MODELBASED_LOAD_PATH = "artifacts/models/agent_modelbased.pt"
AGENT_MODELBASED_LOAD_PATH = "artifacts/best/agent_best.pt"

# Model-free agent
RETRAIN_MODELFREE = False
AGENT_MODELFREE_SAVE_PATH = "artifacts/models/agent_modelfree.pt"
AGENT_MODELFREE_LOAD_PATH = "artifacts/models/agent_modelfree.pt"

# Best model tracking
BEST_DIR = "artifacts/best"
BEST_REWARD_PATH = os.path.join(BEST_DIR, "best_reward.txt")
BEST_AGENT_PATH = os.path.join(BEST_DIR, "agent_best.pt")
