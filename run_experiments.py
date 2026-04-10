import csv
import os

from main import main
from visualization.plot_learning_curves import plot_learning_curves
from visualization.plot_sample_efficiency import plot_sample_efficiency
from visualization.plot_trajectories import plot_trajectories
from visualization.plot_wm_error import plot_wm_error
from utils.logger import logger, TIMESTAMP, set_log_dir

RUNS_DIR = "runs"


def save_summary(results, run_dir):
    path = os.path.join(run_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "mean", "std", "max"])
        for name, r in results.items():
            writer.writerow([name, f"{r['mean']:.1f}", f"{r['std']:.1f}", f"{r['max']:.1f}"])
    logger.info(f"Summary table saved to {path}")


def log_summary_table(results):
    logger.info(f"{'Agent':<15} {'Mean':>8} {'Std':>8} {'Max':>8}")
    logger.info(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    for name, r in results.items():
        logger.info(f"{name:<15} {r['mean']:>8.1f} {r['std']:>8.1f} {r['max']:>8.1f}")


def run():
    run_dir = os.path.join(RUNS_DIR, TIMESTAMP)
    logs_dir = os.path.join(run_dir, "logs")
    figures_dir = os.path.join(run_dir, "figures")
    models_dir = os.path.join(run_dir, "models")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    set_log_dir(logs_dir)

    # 1. Train and evaluate all agents
    logger.info(f"===== Running all experiments (run dir: {run_dir}) =====")
    mb_results, mf_results, rand_results = main(
        retrain_agent=True,
        retrain_wm=True,
        retrain_modelfree=True,
        model_dir=models_dir,
    )

    # 2. Generate all plots
    logger.info("===== Generating plots =====")
    wm_path = os.path.join(models_dir, "wm.pt")
    mean_path = os.path.join(models_dir, "state_mean.npy")
    std_path = os.path.join(models_dir, "state_std.npy")
    plot_learning_curves(logs_dir=logs_dir, figures_dir=figures_dir)
    plot_sample_efficiency(logs_dir=logs_dir, figures_dir=figures_dir)
    plot_trajectories(model_path=wm_path, mean_path=mean_path, std_path=std_path, figures_dir=figures_dir)
    plot_wm_error(model_path=wm_path, mean_path=mean_path, std_path=std_path, figures_dir=figures_dir)

    # 3. Summary
    logger.info("===== Experiment complete =====")
    results = {
        "Model-based": mb_results,
        "Model-free": mf_results,
        "Random": rand_results,
    }
    log_summary_table(results)
    save_summary(results, run_dir)
    logger.info(f"All artifacts saved to {run_dir}/")


if __name__ == "__main__":
    run()
