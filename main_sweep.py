import os
from main import main

REWARD_WEIGHTS = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
MODEL_PATH = "artifacts/models/wm_sweep.pt"
MEAN_PATH = "artifacts/models/state_mean.npy"
STD_PATH = "artifacts/models/state_std.npy"


def sweep():
    results = []

    for w in REWARD_WEIGHTS:
        print(f"\n===== reward_weight={w} =====")
        mb_results, _, _, _ = main(
            retrain_agent=True,
            retrain_wm=True,
            retrain_modelfree=False,
            reward_weight=w,
            model_path=MODEL_PATH,
            mean_path=MEAN_PATH,
            std_path=STD_PATH,
        )
        results.append((w, mb_results['mean']))

    # Clean up temp model
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    print()
    print(f"{'reward_weight':>15}  {'avg_reward':>10}")
    print(f"{'-'*15}  {'-'*10}")
    for w, r in results:
        print(f"{w:>15}  {r:>10.1f}")


if __name__ == "__main__":
    sweep()
