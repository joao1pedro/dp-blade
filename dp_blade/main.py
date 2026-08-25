import os
import pandas as pd
from src.config import SAVE_DIR, IAKF_DEFAULT_CONFIG, FFTKF_DEFAULT_CONFIG
from src.models.train_models import run_experiment

EXPERIMENTS_TO_RUN = [
    {"name": "NoDP_Baseline", "method": "NO_DP"},
    {"name": "DPAdamW_Baseline", "method": "DP_ADAMW"},
    {"name": "IAKF_Baseline", "method": "IAKF", "iakf_config": IAKF_DEFAULT_CONFIG},
    {"name": "Doppler_Baseline", "method": "DOPPLER"},
    {"name": "DiSK_Baseline", "method": "DISK"},
    {"name": "FFTKF_Baseline", "method": "FFTKF", "fftkf_config": FFTKF_DEFAULT_CONFIG}
]

def generate_iakf_ablations():
    ablations = []
    for a in [0.01, 0.1]:
        conf = IAKF_DEFAULT_CONFIG.copy()
        conf["alpha"] = a
        ablations.append({"name": f"IAKF_alpha_{a}", "method": "IAKF", "iakf_config": conf})
    for q in [1e-3, 1e-1]:
        conf = IAKF_DEFAULT_CONFIG.copy()
        conf["q_scale"] = q
        ablations.append({"name": f"IAKF_qscale_{q}", "method": "IAKF", "iakf_config": conf})
    return ablations

#EXPERIMENTS_TO_RUN.extend(generate_iakf_ablations())

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
    print("4. PRIMARY RESEARCH GOAL GAP ANALYSIS: IAKF vs DOPPLER & DiSK")
    print("="*50)

    best_dop = df_results[df_results["method"] == "DOPPLER"]["best_val_f1"].max()
    best_dsk = df_results[df_results["method"] == "DISK"]["best_val_f1"].max()
    best_iakf = df_results[df_results["method"] == "IAKF"]["best_val_f1"].max()
    best_fftkf = df_results[df_results["method"] == "FFTKF"]["best_val_f1"].max()

    print(f"Absolute Best DOPPLER: {best_dop:.4f}")
    print(f"Absolute Best DiSK:    {best_dsk:.4f}")
    print(f"Absolute Best IAKF:    {best_iakf:.4f}")
    print(f"Absolute Best FFTKF: {best_fftkf:.4f}")
    print("-" * 50)
    print(f"GAP (IAKF - DOPPLER):  {best_iakf - best_dop:+.4f}")
    print(f"GAP (IAKF - DiSK):     {best_iakf - best_dsk:+.4f}")
    print(f"GAP (IAKF - FFTKF):     {best_iakf - best_fftkf:+.4f}")