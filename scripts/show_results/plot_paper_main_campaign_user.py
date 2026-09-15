#!/usr/bin/env python3
"""
Clean plots for the paper_main_campaign TRAIN_USER results (see
modules/federated_generate_labels_trigger_joint/gen_configs_paper_main_campaign.py and
orchestrate_student_machines/orchestrate_runs_paper_main_campaign_user.sh).

Same house style as scripts/show_results/show_results.py (compute_cta_pta_mean_var /
plot_cta_vs_pta / annotate_key_points) -- this script just points that same machinery at the
paper_main_campaign's own directory layout, which none of show_results.py's existing
compute_cta_pta_mean_var_* variants match:

  experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign/
    {model_flag}/{dataset}/{tag}/seed{seed}/
      mean/train_user_{budget}/                       <- single_user branch (1v0/mean)
      federated_3vs7/{agg}/train_user_{budget}/        <- federated branch (3v7/agg)

caccs.npy/paccs.npy (final-epoch value = CTA / ASR) are read and averaged over seeds exactly as
show_results.py's get_final_value / compute_cta_pta_mean_var do.

Usage:
  python scripts/show_results/plot_paper_main_campaign_user.py
  python scripts/show_results/plot_paper_main_campaign_user.py --model r32p --tag baseline \
      --datasets cifar svhn
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Style (Paper-ready) -- identical to show_results.py's own plt.rcParams block.
# =========================
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2.5,
})

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign"
)

BUDGETS = [0, 150, 300, 500, 1000, 2000, 2500, 5000]
SEEDS = list(range(10))
FEDERATED_TAG = "federated_3vs7"
DEPLOY_AGG_METHODS_FEDERATED = ["mean", "krum", "multikrum", "median", "trmean"]
SINGLE_USER_AGG = "mean"

# Same palette as show_results.py's AGG_COLORS, plus single_user (that script's
# FEDERATED_MULTIKRUM_COLORS reserves tab:blue for single_user/tab:purple for the federated
# branch as a whole; here every federated aggregator gets its own AGG_COLORS entry instead, so
# single_user is kept visually distinct as black).
BRANCH_COLORS = {
    "single_user": "black",
    "federated_mean": "tab:blue",
    "federated_median": "tab:orange",
    "federated_krum": "tab:green",
    "federated_trmean": "tab:red",
    "federated_multikrum": "tab:purple",
}

BRANCH_LABELS = {
    "single_user": "single_user (1v0/mean)",
    "federated_mean": "federated_3vs7 (mean)",
    "federated_median": "federated_3vs7 (median)",
    "federated_krum": "federated_3vs7 (krum)",
    "federated_trmean": "federated_3vs7 (trmean)",
    "federated_multikrum": "federated_3vs7 (multikrum)",
}


# =========================
# Utils -- identical to show_results.py's get_final_value.
# =========================
def get_final_value(npy_path):
    try:
        if not os.path.exists(npy_path):
            return np.nan
        if os.path.getsize(npy_path) == 0:
            return np.nan

        data = np.load(npy_path, allow_pickle=True)

        if data.size == 0:
            return np.nan

        if data.ndim > 1:
            return float(data[-1][0])

        return float(data[-1])

    except Exception:
        return np.nan


# =========================
# COMPUTE
# =========================
def _branch_run_dir(model_flag, dataset, tag, seed, branch, agg=None):
    cell_dir = EXP_BASE / model_flag / dataset / tag / f"seed{seed}"
    if branch == "single_user":
        return cell_dir / SINGLE_USER_AGG
    return cell_dir / FEDERATED_TAG / agg


def compute_cta_pta_mean_var(
    model_flag,
    dataset,
    tag,
    branch,
    budgets,
    seeds,
    agg=None,
    cta_file="caccs.npy",
    pta_file="paccs.npy",
):
    """Same shape/semantics as show_results.py's own compute_cta_pta_mean_var: for each budget,
    averages caccs.npy/paccs.npy's final value across `seeds`, skipping any (budget, seed) cell
    whose files don't exist yet (never crashes on a partial/in-progress campaign)."""
    records = []

    for budget in budgets:
        cta_vals, pta_vals = [], []
        for seed in seeds:
            run_dir = _branch_run_dir(model_flag, dataset, tag, seed, branch, agg) / f"train_user_{budget}"
            cta_path = run_dir / cta_file
            pta_path = run_dir / pta_file
            if not (cta_path.exists() and pta_path.exists()):
                continue
            cta_vals.append(get_final_value(cta_path))
            pta_vals.append(get_final_value(pta_path))

        records.append({
            "dataset": dataset,
            "branch": branch,
            "agg": agg,
            "budget": budget,
            "model": model_flag,
            "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
            "cta_var": np.var(cta_vals) if cta_vals else np.nan,
            "pta_mean": np.mean(pta_vals) if pta_vals else np.nan,
            "pta_var": np.var(pta_vals) if pta_vals else np.nan,
        })

    return pd.DataFrame.from_records(records)


def collect_dataset(model_flag, dataset, tag):
    """{branch_key: DataFrame} for single_user + every federated aggregator, one dataset."""
    all_data = {
        "single_user": compute_cta_pta_mean_var(
            model_flag, dataset, tag, "single_user", BUDGETS, SEEDS,
        ),
    }
    for agg in DEPLOY_AGG_METHODS_FEDERATED:
        all_data[f"federated_{agg}"] = compute_cta_pta_mean_var(
            model_flag, dataset, tag, "federated", BUDGETS, SEEDS, agg=agg,
        )
    return all_data


# =========================
# PLOT -- annotate_key_points / plot_cta_vs_pta ported verbatim from show_results.py, just with
# BRANCH_COLORS/BRANCH_LABELS in place of AGG_COLORS/agg_method.
# =========================
def annotate_key_points(df, x, y, score, color):
    budgets = df["budget"].values

    if np.all(np.isnan(score)):
        return

    max_score = np.nanmax(score)
    candidates = np.where(np.isclose(score, max_score, atol=1e-12))[0]
    idx_best = candidates[np.argmin(budgets[candidates])]

    x_range = max(np.nanmax(x) - np.nanmin(x), 1e-6)
    y_range = max(np.nanmax(y) - np.nanmin(y), 1e-6)
    dx = 0.03 * x_range
    dy = 0.03 * y_range

    if np.isnan(x[idx_best]) or np.isnan(y[idx_best]):
        return

    plt.text(
        x[idx_best] + dx,
        y[idx_best] + dy,
        str(int(budgets[idx_best])),
        fontsize=11,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.8),
        zorder=10,
    )


def plot_cta_vs_pta(all_data, dataset, model_flag, tag, save_dir=None, filename=None):
    """One figure, one line per branch (single_user + each federated aggregator). Directly
    modeled on show_results.py's plot_cta_vs_pta (error bars = sqrt(var), best-budget point
    highlighted and annotated, equal-aspect axes, dashed grid)."""
    plt.figure(figsize=(7.5, 6))
    any_data = False

    for branch_key, df in all_data.items():
        df = df.dropna()
        if df.empty:
            continue
        any_data = True

        df = df.sort_values("budget")

        x = df["pta_mean"].values * 100
        y = df["cta_mean"].values * 100

        xerr = np.sqrt(df["pta_var"].values) * 100
        yerr = np.sqrt(df["cta_var"].values) * 100

        color = BRANCH_COLORS.get(branch_key)

        plt.plot(
            x, y, linestyle="-", linewidth=2.2, color=color, alpha=0.85,
            label=BRANCH_LABELS.get(branch_key, branch_key), zorder=3,
        )

        plt.errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color,
            elinewidth=1.2, capsize=3, alpha=0.35, zorder=1,
        )

        plt.scatter(x, y, s=45, color=color, edgecolors="none", zorder=4)

        score = x  # ASR
        max_score = np.nanmax(score)
        candidates = np.where(np.isclose(score, max_score, atol=1e-12))[0]
        budgets = df["budget"].values
        idx_best = candidates[np.argmin(budgets[candidates])]

        plt.scatter(
            x[idx_best], y[idx_best], s=95, color=color,
            edgecolor="black", linewidth=1.2, zorder=6,
        )

        annotate_key_points(df, x, y, score, color)

    if not any_data:
        plt.close()
        return False

    plt.xlabel("ASR (%)")
    plt.ylabel("CTA (%)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(frameon=True, fontsize=10, loc="lower left")
    plt.title(f"{model_flag} / {dataset} / tag={tag}")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename or f"{dataset}.png")
        plt.savefig(path, dpi=300)
        print(f"[INFO] Saved plot: {path}")

    plt.close()
    return True


def plot_metric_vs_budget(all_data, dataset, model_flag, tag, metric, ylabel, save_dir=None, filename=None):
    """Companion view not present in show_results.py (that script only plots the CTA-vs-ASR
    trade-off) but kept in the same house style -- dashed grid, error bars via fill_between,
    same color/label mapping -- since a budget sweep is otherwise hard to read off the
    trade-off curve alone."""
    plt.figure(figsize=(7.5, 6))
    any_data = False

    for branch_key, df in all_data.items():
        df = df.dropna(subset=[f"{metric}_mean"])
        if df.empty:
            continue
        any_data = True

        df = df.sort_values("budget")
        budgets = df["budget"].values
        means = df[f"{metric}_mean"].values * 100
        stds = np.sqrt(df[f"{metric}_var"].values) * 100
        color = BRANCH_COLORS.get(branch_key)

        plt.plot(
            budgets, means, "-o", color=color, alpha=0.85, markersize=5,
            label=BRANCH_LABELS.get(branch_key, branch_key), zorder=3,
        )
        plt.fill_between(budgets, means - stds, means + stds, color=color, alpha=0.15, zorder=1)

    if not any_data:
        plt.close()
        return False

    plt.xlabel("Deployment budget (poisoned samples)")
    plt.ylabel(ylabel)
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.title(f"{ylabel} vs budget -- {model_flag} / {dataset} / tag={tag}")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename or f"{metric}_vs_budget.png")
        plt.savefig(path, dpi=300)
        print(f"[INFO] Saved plot: {path}")

    plt.close()
    return True


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="r32p")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--datasets", nargs="+", default=["cifar", "svhn"])
    parser.add_argument("--out-dir", default="./plots_paper_main_campaign_user")
    args = parser.parse_args()

    out_root = Path(args.out_dir) / f"{args.model}_{args.tag}"

    for dataset in args.datasets:
        print(f"\n=== [paper_main_campaign/user] {args.model} / {dataset} / tag={args.tag} ===")
        all_data = collect_dataset(args.model, dataset, args.tag)

        for branch_key, df in all_data.items():
            n_cta = df["cta_mean"].notna().sum()
            n_asr = df["pta_mean"].notna().sum()
            print(f"   -> {branch_key:24s} budgets with data: cta={n_cta}/{len(BUDGETS)} asr={n_asr}/{len(BUDGETS)}")

        save_dir = str(out_root / dataset)
        ok_tradeoff = plot_cta_vs_pta(all_data, dataset, args.model, args.tag, save_dir=save_dir)
        ok_cta = plot_metric_vs_budget(
            all_data, dataset, args.model, args.tag,
            metric="cta", ylabel="CTA (%)", save_dir=save_dir,
        )
        ok_asr = plot_metric_vs_budget(
            all_data, dataset, args.model, args.tag,
            metric="pta", ylabel="ASR (%)", save_dir=save_dir,
        )

        if not (ok_tradeoff or ok_cta or ok_asr):
            print(f"[WARNING] no data found under {EXP_BASE / args.model / dataset / args.tag} -- "
                  f"nothing plotted for {dataset}.")

    print("\nDone.")


if __name__ == "__main__":
    main()
