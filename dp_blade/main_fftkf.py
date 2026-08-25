import os
import pandas as pd
from src.config import SAVE_DIR, FFTKF_DEFAULT_CONFIG
from src.models.train_models import run_experiment

EXPERIMENTS_TO_RUN = [
    {"name": "FFTKF_Baseline", "method": "FFTKF", "fftkf_config": FFTKF_DEFAULT_CONFIG}
]

if __name__ == "__main__":
    SEEDS = [42]
    results_summaries = []
    all_histories = {}

    for exp in EXPERIMENTS_TO_RUN:
        for seed in SEEDS:
            run_identifier = f"{exp['name']}_S{seed}"
            hist, summ = run_experiment(
                exp_name=exp["name"],
                method=exp["method"],
                iakf_config=exp.get("iakf_config"),
                fftkf_config=exp.get("fftkf_config"),
                seed=seed
            )
            results_summaries.append(summ)
            all_histories[run_identifier] = hist

    df_results = pd.DataFrame(results_summaries)
    df_results.to_csv(os.path.join(SAVE_DIR, "all_runs_summary.csv"), index=False)

    print("\n" + "="*50)
    print("1. PER-RUN SUMMARY TABLE")
    print("="*50)
    print(df_results)

    print("\n" + "="*50)
    print("2. GROUPED SUMMARY TABLE (Aggregated over Seeds)")
    print("="*50)
    df_grouped = df_results.groupby(["method", "exp_name"]).agg({
        "best_val_acc": ["mean", "std"],
        "best_val_f1": ["mean", "std"],
        "final_val_acc": ["mean", "std"],
        "final_epsilon": ["mean"]
    }).reset_index()
    print(df_grouped)

    print("\n" + "="*50)
    print("3. RANKING TABLE (Sorted by Best F1, then Best Acc)")
    print("="*50)
    df_ranked = df_grouped.sort_values(by=[("best_val_f1", "mean"), ("best_val_acc", "mean")], ascending=[False, False])
    print(df_ranked)

    print("\n" + "="*50)
    print("4. PRIMARY RESEARCH GOAL GAP ANALYSIS (Only FFTKF shown during isolated run)")
    print("="*50)
    
    if not df_results[df_results["method"] == "FFTKF"].empty:
        best_fftkf = df_results[df_results["method"] == "FFTKF"]["best_val_f1"].max()
        print(f"Absolute Best FFTKF: {best_fftkf:.4f}")
    else:
        print("Nenhum resultado FFTKF encontrado.")