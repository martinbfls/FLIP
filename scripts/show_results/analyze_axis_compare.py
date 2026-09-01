"""
Head-to-head console summary for the three federated_generate_labels_trigger_joint proof-of-
concept axis comparisons launched together by orchestrate_slurm/
orchestrate_runs_trigger_joint_axis_sweep_slurm.sh (1-poisoned/0-honest, "mean" aggregation,
loosely-constrained trigger -- delta_min_frac=0.0, epsilon=1.0, lambda_align=0.0):

  - init:                    "stripe" (baseline) vs "random"
                              (gen_configs_init_compare.py)
  - expert_retrain_interval: 0 [disabled] vs 1 [retrain every outer iteration]
                              (gen_configs_expert_retrain_compare.py)
  - detach_param_dist:       False [param_dist stays fully differentiable, the module's
                              original behavior] vs True [param_dist.detach() in mtt_term_k's
                              denominator]
                              (gen_configs_detach_param_dist_compare.py)

These are three INDEPENDENT one-axis comparisons (not a cross product -- each fixes the other
two axes at gen_configs.py's own defaults), so they're reported as three separate per-budget
tables, not merged into one. show_results.py already produces per-campaign CSVs/plots/LaTeX
tables; this script is the quick "did it help, at which budgets" console read complementing
those, in the same spirit as analyze_trigger_joint_sweep.py's ranking for the OFAT
hyperparameter sweep.

Usage: python scripts/show_results/analyze_axis_compare.py
"""
import sys
from pathlib import Path

import numpy as np

# This file lives at FLIP/scripts/show_results/ -- three levels below the repo root, which
# must be on sys.path for `modules.*` to import (see show_results.py's own note).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.federated_generate_labels_trigger_joint.gen_configs_init_compare import (
    EXP_BASE as IC_EXP_BASE,
    SEEDS as IC_SEEDS,
    BUDGETS as IC_BUDGETS,
    INITS,
    cell_name as ic_cell_name,
)
from modules.federated_generate_labels_trigger_joint.gen_configs_expert_retrain_compare import (
    EXP_BASE as ERC_EXP_BASE,
    SEEDS as ERC_SEEDS,
    BUDGETS as ERC_BUDGETS,
    EXPERT_RETRAIN_INTERVALS,
    cell_name as erc_cell_name,
)
from modules.federated_generate_labels_trigger_joint.gen_configs_detach_param_dist_compare import (
    EXP_BASE as DPD_EXP_BASE,
    SEEDS as DPD_SEEDS,
    BUDGETS as DPD_BUDGETS,
    DETACH_PARAM_DISTS,
    cell_name as dpd_cell_name,
)


def get_final_value(npy_path):
    try:
        if not npy_path.exists() or npy_path.stat().st_size == 0:
            return np.nan
        data = np.load(npy_path, allow_pickle=True)
        if data.size == 0:
            return np.nan
        return float(data[-1][0]) if data.ndim > 1 else float(data[-1])
    except Exception:
        return np.nan


def fmt(x):
    return "pending" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"


def load_axis(exp_base, cell_name_fn, values, budgets, seeds):
    """{value: {budget: (cta_mean, asr_mean)}} -- mean over seeds of each (value, budget)
    cell's final CTA/ASR."""
    results = {}
    for value in values:
        per_budget = {}
        for budget in budgets:
            ctas, asrs = [], []
            for seed in seeds:
                train_user_dir = exp_base / cell_name_fn(value, seed) / f"train_user_{budget}"
                cta = get_final_value(train_user_dir / "caccs.npy")
                asr = get_final_value(train_user_dir / "paccs.npy")
                if not np.isnan(cta):
                    ctas.append(cta)
                if not np.isnan(asr):
                    asrs.append(asr)
            per_budget[budget] = (
                float(np.mean(ctas)) if ctas else np.nan,
                float(np.mean(asrs)) if asrs else np.nan,
            )
        results[value] = per_budget
    return results


def print_axis_table(title, results, values, budgets):
    print(f"\n=== {title} ===")
    header = f"{'budget':>8} " + " ".join(f"{str(v)+' (CTA/ASR)':>22}" for v in values)
    print(header)
    for budget in budgets:
        row = [f"{budget:>8}"]
        for value in values:
            cta, asr = results[value][budget]
            row.append(f"{fmt(cta):>10}/{fmt(asr):<10}")
        print(" ".join(row))

    baseline = values[0]
    print(f"\n-- vs baseline ({baseline!r}) --")
    for value in values[1:]:
        wins, losses, ties = 0, 0, 0
        for budget in budgets:
            base_asr = results[baseline][budget][1]
            cand_asr = results[value][budget][1]
            if np.isnan(base_asr) or np.isnan(cand_asr):
                continue
            if cand_asr > base_asr:
                wins += 1
            elif cand_asr < base_asr:
                losses += 1
            else:
                ties += 1
        print(
            f"  {value!r}: ASR higher than baseline at {wins} budget(s), lower at {losses}, "
            f"tied at {ties} (of {len(budgets)} total, pending cells excluded)"
        )


if __name__ == "__main__":
    ic_results = load_axis(IC_EXP_BASE, ic_cell_name, INITS, IC_BUDGETS, IC_SEEDS)
    print_axis_table("init comparison (stripe vs random)", ic_results, INITS, IC_BUDGETS)

    erc_results = load_axis(
        ERC_EXP_BASE, erc_cell_name, EXPERT_RETRAIN_INTERVALS, ERC_BUDGETS, ERC_SEEDS,
    )
    print_axis_table(
        "expert_retrain_interval comparison (0=disabled vs 1=every iteration)",
        erc_results, EXPERT_RETRAIN_INTERVALS, ERC_BUDGETS,
    )

    dpd_results = load_axis(
        DPD_EXP_BASE, dpd_cell_name, DETACH_PARAM_DISTS, DPD_BUDGETS, DPD_SEEDS,
    )
    print_axis_table(
        "detach_param_dist comparison (False=differentiable denominator vs True=detached)",
        dpd_results, DETACH_PARAM_DISTS, DPD_BUDGETS,
    )
