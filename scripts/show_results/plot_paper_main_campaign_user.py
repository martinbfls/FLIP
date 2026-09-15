#!/usr/bin/env python3
"""
Clean plots for the paper_main_campaign TRAIN_USER results (see
modules/federated_generate_labels_trigger_joint/gen_configs_paper_main_campaign.py and
orchestrate_student_machines/orchestrate_runs_paper_main_campaign_user.sh).

Reads caccs.npy/paccs.npy (final-epoch value = CTA / ASR) written by federated_train_user
under:

  experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign/
    {model_flag}/{dataset}/{tag}/seed{seed}/
      mean/train_user_{budget}/                       <- single_user branch
      federated_3vs7/{agg}/train_user_{budget}/        <- federated branch

and aggregates over seeds (mean +/- std) for a given (model_flag, tag), one figure set per
dataset:

  1. CTA and ASR vs deployment budget, one line per branch (single_user + each federated
     aggregator), shaded std band across seeds.
  2. CTA vs ASR trade-off curve, same branches/colors, budget-ordered.

Usage:
  python scripts/show_results/plot_paper_main_campaign_user.py
  python scripts/show_results/plot_paper_main_campaign_user.py --model r32p --tag baseline \
      --datasets cifar svhn
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
})

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign"
)

BUDGETS = [0, 150, 300, 500, 1000, 2000, 2500, 5000]
SEEDS = list(range(10))
FEDERATED_TAG = "federated_3vs7"
DEPLOY_AGG_METHODS_FEDERATED = ["mean", "krum", "multikrum", "median", "trmean"]
SINGLE_USER_AGG = "mean"

BRANCH_COLORS = {
    "single_user": "black",
    "federated_mean": "tab:blue",
    "federated_median": "tab:orange",
    "federated_krum": "tab:green",
    "federated_trmean": "tab:red",
    "federated_multikrum": "tab:purple",
}

BRANCH_LABELS = {
    "single_user": "Single user (mean)",
    "federated_mean": "Federated -- mean",
    "federated_median": "Federated -- median",
    "federated_krum": "Federated -- krum",
    "federated_trmean": "Federated -- trmean",
    "federated_multikrum": "Federated -- multikrum",
}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def get_final_value(npy_path):
    """Last recorded epoch value of a caccs.npy/paccs.npy trace, or NaN if missing/unreadable."""
    try:
        if not npy_path.exists() or npy_path.stat().st_size == 0:
            return np.nan
        data = np.load(npy_path, allow_pickle=True)
        if data.size == 0:
            return np.nan
        if data.ndim > 1:
            return float(data[-1][0])
        return float(data[-1])
    except Exception:
        return np.nan


def branch_dir(cell_dir, branch, agg=None, budget=None):
    if branch == "single_user":
        return cell_dir / SINGLE_USER_AGG / f"train_user_{budget}"
    return cell_dir / FEDERATED_TAG / agg / f"train_user_{budget}"


def collect_branch(model_flag, dataset, tag, branch, agg=None):
    """Mean/std of CTA and ASR over seeds, for every budget, for one branch."""
    records = []
    for budget in BUDGETS:
        cta_vals, asr_vals = [], []
        for seed in SEEDS:
            cell_dir = EXP_BASE / model_flag / dataset / tag / f"seed{seed}"
            run_dir = branch_dir(cell_dir, branch, agg=agg, budget=budget)
            cta = get_final_value(run_dir / "caccs.npy")
            asr = get_final_value(run_dir / "paccs.npy")
            if not np.isnan(cta):
                cta_vals.append(cta)
            if not np.isnan(asr):
                asr_vals.append(asr)
        records.append({
            "budget": budget,
            "n_seeds_cta": len(cta_vals),
            "n_seeds_asr": len(asr_vals),
            "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
            "cta_std": np.std(cta_vals) if cta_vals else np.nan,
            "asr_mean": np.mean(asr_vals) if asr_vals else np.nan,
            "asr_std": np.std(asr_vals) if asr_vals else np.nan,
        })
    return records


def collect_dataset(model_flag, dataset, tag):
    """{branch_key: records} for single_user + every federated aggregator, one dataset."""
    branches = {"single_user": collect_branch(model_flag, dataset, tag, "single_user")}
    for agg in DEPLOY_AGG_METHODS_FEDERATED:
        branches[f"federated_{agg}"] = collect_branch(
            model_flag, dataset, tag, "federated", agg=agg
        )
    return branches


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_metric_vs_budget(branches, dataset, model_flag, tag, metric, ylabel, out_path):
    plt.figure(figsize=(8, 6))

    any_data = False
    for branch_key, records in branches.items():
        budgets = np.array([r["budget"] for r in records])
        means = np.array([r[f"{metric}_mean"] for r in records]) * 100
        stds = np.array([r[f"{metric}_std"] for r in records]) * 100
        valid = ~np.isnan(means)
        if not valid.any():
            continue
        any_data = True
        color = BRANCH_COLORS[branch_key]
        plt.plot(
            budgets[valid], means[valid], "-o",
            color=color, label=BRANCH_LABELS[branch_key], markersize=5,
        )
        plt.fill_between(
            budgets[valid], means[valid] - stds[valid], means[valid] + stds[valid],
            color=color, alpha=0.15,
        )

    if not any_data:
        plt.close()
        return False

    plt.xlabel("Deployment budget (poisoned samples)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs budget -- {model_flag} / {dataset} / tag={tag}")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_cta_vs_asr(branches, dataset, model_flag, tag, out_path):
    plt.figure(figsize=(7.5, 6.5))

    any_data = False
    for branch_key, records in branches.items():
        asr = np.array([r["asr_mean"] for r in records]) * 100
        cta = np.array([r["cta_mean"] for r in records]) * 100
        valid = ~(np.isnan(asr) | np.isnan(cta))
        if not valid.any():
            continue
        any_data = True
        color = BRANCH_COLORS[branch_key]
        plt.plot(asr[valid], cta[valid], "-", color=color, alpha=0.8)
        plt.scatter(asr[valid], cta[valid], color=color, s=45, label=BRANCH_LABELS[branch_key])

    if not any_data:
        plt.close()
        return False

    plt.xlabel("ASR (%)")
    plt.ylabel("CTA (%)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title(f"CTA vs ASR -- {model_flag} / {dataset} / tag={tag}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="r32p")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--datasets", nargs="+", default=["cifar", "svhn"])
    parser.add_argument("--out-dir", default="out/graphs/paper_main_campaign")
    args = parser.parse_args()

    out_root = Path(args.out_dir) / f"{args.model}_{args.tag}"

    for dataset in args.datasets:
        print(f"=== {args.model} / {dataset} / tag={args.tag} ===")
        branches = collect_dataset(args.model, dataset, args.tag)

        for branch_key, records in branches.items():
            n_cta = sum(r["n_seeds_cta"] for r in records)
            n_asr = sum(r["n_seeds_asr"] for r in records)
            print(f"  {branch_key:24s} seeds found: cta={n_cta} asr={n_asr} (of {len(SEEDS)*len(BUDGETS)} cells)")

        ok_cta = plot_metric_vs_budget(
            branches, dataset, args.model, args.tag,
            metric="cta", ylabel="CTA (%)",
            out_path=out_root / dataset / "cta_vs_budget.png",
        )
        ok_asr = plot_metric_vs_budget(
            branches, dataset, args.model, args.tag,
            metric="asr", ylabel="ASR (%)",
            out_path=out_root / dataset / "asr_vs_budget.png",
        )
        ok_tradeoff = plot_cta_vs_asr(
            branches, dataset, args.model, args.tag,
            out_path=out_root / dataset / "cta_vs_asr.png",
        )

        if not (ok_cta or ok_asr or ok_tradeoff):
            print(f"  [WARNING] no data found under {EXP_BASE / args.model / dataset / args.tag} -- "
                  f"nothing plotted for {dataset}.")
        else:
            print(f"  Saved plots under {out_root / dataset}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
