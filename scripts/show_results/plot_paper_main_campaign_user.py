#!/usr/bin/env python3
"""
Clean plots + LaTeX tables + trigger visuals for the paper_main_campaign TRAIN_USER results (see
modules/federated_generate_labels_trigger_joint/gen_configs_paper_main_campaign.py and
orchestrate_student_machines/orchestrate_runs_paper_main_campaign_user.sh).

Same house style as scripts/show_results/show_results.py (compute_cta_pta_mean_var /
plot_cta_vs_pta / annotate_key_points / format_cell / build_table / save_trigger_visual_in) --
this script just points that same machinery at the paper_main_campaign's own directory layout,
which none of show_results.py's existing compute_cta_pta_mean_var_*/save_all_trigger_visuals_*
variants match:

  experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign/
    {model_flag}/{dataset}/{tag}/seed{seed}/
      gen_labels_trigger_joint/trigger/                <- the generated trigger (.pt + .png)
      mean/train_user_{budget}/                         <- single_user branch (1v0/mean)
      federated_3vs7/{agg}/train_user_{budget}/         <- federated branch (3v7/agg)

caccs.npy/paccs.npy (final-epoch value = CTA / ASR) are read and averaged over seeds exactly as
show_results.py's get_final_value / compute_cta_pta_mean_var do. Sweep axes (EXP_BASE,
SEEDS, DEPLOY_BUDGETS, DEPLOY_AGG_METHODS_FEDERATED, GEN_NUM_POISONED/HONESTS, GEN_INIT,
SOURCE_LABEL/TARGET_LABEL, cell_name) are imported straight from gen_configs_paper_main_campaign.py
itself (same convention show_results.py uses for its own main campaign) so this can't drift from
wherever that generator actually wrote its configs.

Usage:
  python scripts/show_results/plot_paper_main_campaign_user.py
  python scripts/show_results/plot_paper_main_campaign_user.py --model r32p --tag baseline \
      --datasets cifar svhn
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

# This file lives at FLIP/scripts/show_results/ -- same sys.path fix as show_results.py, needed
# for `modules.*` to import when run directly (`python scripts/show_results/....py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.base_utils.datasets import load_dataset, pick_poisoner
from modules.federated_generate_labels_trigger_joint.gen_configs_paper_main_campaign import (
    EXP_BASE,
    MODEL_FLAGS,
    DATASETS,
    SEEDS,
    DEPLOY_BUDGETS,
    DEPLOY_SINGLE_USER_AGG_METHOD,
    FEDERATED_TAG,
    DEPLOY_AGG_METHODS_FEDERATED,
    GEN_NUM_POISONED,
    GEN_NUM_HONESTS,
    SOURCE_LABEL,
    TARGET_LABEL,
    GEN_INIT,
    cell_name,
)

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

# Display name for our own attack (previously called BRoADflip in this codebase's own
# comments/docstrings; the paper's legend now calls it JOLT) -- used for every branch that is
# OUR method (single_user + each federated_* branch), as opposed to FLIP_LABEL below (the FLIP
# paper's own baseline).
JOLT_LABEL = "JOLT"

BRANCH_LABELS = {
    "single_user": f"{JOLT_LABEL} (Centralized, {DEPLOY_SINGLE_USER_AGG_METHOD})",
    "federated_mean": f"{JOLT_LABEL} (mean)",
    "federated_median": f"{JOLT_LABEL} (median)",
    "federated_krum": f"{JOLT_LABEL} (krum)",
    "federated_trmean": f"{JOLT_LABEL} (trmean)",
    "federated_multikrum": f"{JOLT_LABEL} (multikrum)",
}

# Filename suffix per branch for the per-config trade-off plots (plot_cta_vs_pta_per_branch) --
# matches the naming the user's own LaTeX subfigure grid expects ({dataset}_{suffix}.png, e.g.
# cifar_mean.png / cifar_krum.png / ... / cifar_centralized.png), one file per branch so each
# can be grouped as its own \subfigure with a caption set externally in LaTeX.
BRANCH_FILE_SUFFIX = {
    "single_user": "centralized",
    "federated_mean": "mean",
    "federated_median": "median",
    "federated_krum": "krum",
    "federated_trmean": "trmean",
    "federated_multikrum": "multikrum",
}

# FLIP baseline (the paper's own comparison attack, using its own 1xs sinusoidal trigger)
# comparison series -- same color as its JOLT counterpart branch, dotted instead of solid (see
# _branch_linestyle), so each per-branch plot pairs JOLT against FLIP directly:
#   - one per federated aggregator:
#     out_neurips/{model}/{tag_suffix}_FLIP/{dataset}/backdoor/{agg}/1xs/{seed}/{budget}/
#   - one for the centralized/single_user setting (no aggregation, 1v0):
#     out_neurips/{model}/{1vs0}/{dataset}/backdoor/{DEPLOY_SINGLE_USER_AGG_METHOD}/1xs/{seed}/{budget}/
# Convention ported from modules/base_utils/show_results.py's own compute_cta_pta_mean_var(...,
# use_flip=True/False, poisoner_flag="1xs").
FLIP_LABEL = "FLIP"
FLIP_ATTACK = "backdoor"
FLIP_POISONER_FLAG = "1xs"
for _agg in DEPLOY_AGG_METHODS_FEDERATED:
    BRANCH_COLORS[f"flip_{_agg}"] = BRANCH_COLORS[f"federated_{_agg}"]
    BRANCH_LABELS[f"flip_{_agg}"] = f"{FLIP_LABEL} ({_agg})"
    BRANCH_FILE_SUFFIX[f"flip_{_agg}"] = f"federated_{_agg}"

BRANCH_COLORS["flip_single_user"] = BRANCH_COLORS["single_user"]
BRANCH_LABELS["flip_single_user"] = f"{FLIP_LABEL} (Centralized, {FLIP_POISONER_FLAG})"
BRANCH_FILE_SUFFIX["flip_single_user"] = BRANCH_FILE_SUFFIX["single_user"]

# Column order/labels for the LaTeX table -- matches the paper's own header (MEAN, CW-MEDIAN,
# KRUM, TRMEAN, MULTIKRUM), independent of DEPLOY_AGG_METHODS_FEDERATED's own order. "centralized"
# is a synthetic series (not a real deployment aggregator) read from the single_user branch
# instead of federated_3vs7 -- see build_dataset_table's own handling of it below.
CENTRALIZED_SERIES = "centralized"
TABLE_AGG_ORDER = ["mean", "median", "krum", "trmean", "multikrum"]
TABLE_COLUMNS_WITH_CENTRALIZED = [CENTRALIZED_SERIES] + TABLE_AGG_ORDER
TABLE_AGG_DISPLAY = {
    CENTRALIZED_SERIES: "CENTRALIZED",
    "mean": "MEAN", "median": "CW-MEDIAN", "krum": "KRUM",
    "trmean": "TRMEAN", "multikrum": "MULTIKRUM",
}

# Tag -> table block name. Extend as more paper_main_campaign tags (eps_0p50_lpips_0p1,
# eps_16_255) finish training -- see gen_configs_paper_main_campaign.py's own CONFIG_TAGS /
# STEALTH_GRID docstring for what each tag means ("baseline" = the plain optimized joint
# trigger, reference config with no extra stealth regularization).
TAG_DISPLAY_NAMES = {
    "baseline": "Optimized Trigger -- \\broad{}",
    "eps_0p50_lpips_0p1": "Optimized Trigger (eps=0.5, LPIPS=0.1) -- \\broad{}",
    "eps_16_255": "Optimized Trigger (eps=16/255) -- \\broad{}",
}

DATASET_DISPLAY = {"cifar": "CIFAR-10", "svhn": "SVHN"}
MODEL_DISPLAY = {"r32p": "ResNet-32", "convnext_micro": "ConvNeXt-Micro"}


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
def _branch_run_dir(model_flag, dataset, tag, seed, branch, agg=None, exp_base=EXP_BASE):
    cell_dir = exp_base / cell_name(model_flag, dataset, tag, seed)
    if branch == "single_user":
        return cell_dir / DEPLOY_SINGLE_USER_AGG_METHOD
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
    exp_base=EXP_BASE,
):
    """Same shape/semantics as show_results.py's own compute_cta_pta_mean_var: for each budget,
    averages caccs.npy/paccs.npy's final value across `seeds`, skipping any (budget, seed) cell
    whose files don't exist yet (never crashes on a partial/in-progress campaign). `exp_base`
    defaults to the imported EXP_BASE (the local/orchestrator-relative experiments/ tree) but can
    be overridden -- e.g. to JOLT's own results mirrored onto a cluster mount under a different
    root, same subtree from {model_flag}/{dataset}/{tag}/seed{seed}/ down (see --jolt-root)."""
    records = []

    for budget in budgets:
        cta_vals, pta_vals = [], []
        for seed in seeds:
            run_dir = (
                _branch_run_dir(model_flag, dataset, tag, seed, branch, agg, exp_base=exp_base)
                / f"train_user_{budget}"
            )
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


def _flip_run_dir(flip_root, model_flag, dataset, agg, seed, budget):
    tag_suffix = FEDERATED_TAG.split("_", 1)[-1]  # "federated_3vs7" -> "3vs7"
    return (
        Path(flip_root) / model_flag / f"{tag_suffix}_FLIP" / dataset / FLIP_ATTACK
        / agg / FLIP_POISONER_FLAG / str(seed) / str(budget)
    )


def _flip_centralized_run_dir(flip_root, model_flag, dataset, seed, budget):
    """Centralized (1v0, undefended, no aggregation) counterpart of _flip_run_dir -- e.g.
    out_neurips/convnext_micro/1vs0/cifar/backdoor/mean/1xs/2/{budget}/. No "_FLIP" suffix on the
    "1vs0" segment (unlike the federated tree): the undefended single-victim setting doesn't need
    one to disambiguate from JOLT's own centralized results, since it's already keyed by the 1xs
    poisoner flag instead of "optimized"."""
    return (
        Path(flip_root) / model_flag / f"{GEN_NUM_POISONED}vs{GEN_NUM_HONESTS}" / dataset
        / FLIP_ATTACK / DEPLOY_SINGLE_USER_AGG_METHOD / FLIP_POISONER_FLAG / str(seed) / str(budget)
    )


def compute_cta_pta_mean_var_flip(
    flip_root, model_flag, dataset, agg, budgets, seeds,
    cta_file="caccs.npy", pta_file="paccs.npy",
):
    """Same shape/semantics as compute_cta_pta_mean_var above, but reads the FLIP paper
    baseline's own results tree instead of this campaign's own experiments/ tree -- see
    modules/base_utils/show_results.py's compute_cta_pta_mean_var(..., use_flip=True,
    poisoner_flag="1xs") for the path convention this mirrors."""
    records = []

    for budget in budgets:
        cta_vals, pta_vals = [], []
        for seed in seeds:
            run_dir = _flip_run_dir(flip_root, model_flag, dataset, agg, seed, budget)
            cta_path = run_dir / cta_file
            pta_path = run_dir / pta_file
            if not (cta_path.exists() and pta_path.exists()):
                continue
            cta_vals.append(get_final_value(cta_path))
            pta_vals.append(get_final_value(pta_path))

        records.append({
            "dataset": dataset,
            "branch": f"flip_{agg}",
            "agg": agg,
            "budget": budget,
            "model": model_flag,
            "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
            "cta_var": np.var(cta_vals) if cta_vals else np.nan,
            "pta_mean": np.mean(pta_vals) if pta_vals else np.nan,
            "pta_var": np.var(pta_vals) if pta_vals else np.nan,
        })

    return pd.DataFrame.from_records(records)


def compute_cta_pta_mean_var_flip_centralized(
    flip_root, model_flag, dataset, budgets, seeds,
    cta_file="caccs.npy", pta_file="paccs.npy",
):
    """Centralized counterpart of compute_cta_pta_mean_var_flip, reading _flip_centralized_run_dir
    instead (FLIP's own 1xs-trigger results in the undefended single-victim setting)."""
    records = []

    for budget in budgets:
        cta_vals, pta_vals = [], []
        for seed in seeds:
            run_dir = _flip_centralized_run_dir(flip_root, model_flag, dataset, seed, budget)
            cta_path = run_dir / cta_file
            pta_path = run_dir / pta_file
            if not (cta_path.exists() and pta_path.exists()):
                continue
            cta_vals.append(get_final_value(cta_path))
            pta_vals.append(get_final_value(pta_path))

        records.append({
            "dataset": dataset,
            "branch": "flip_single_user",
            "agg": None,
            "budget": budget,
            "model": model_flag,
            "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
            "cta_var": np.var(cta_vals) if cta_vals else np.nan,
            "pta_mean": np.mean(pta_vals) if pta_vals else np.nan,
            "pta_var": np.var(pta_vals) if pta_vals else np.nan,
        })

    return pd.DataFrame.from_records(records)


def collect_dataset(
    model_flag, dataset, tag, budgets=DEPLOY_BUDGETS, seeds=SEEDS, flip_root=None, exp_base=EXP_BASE,
):
    """{branch_key: DataFrame} for single_user + every federated aggregator, one dataset. When
    flip_root is given, also adds a flip_{agg} entry per aggregator read from that FLIP baseline
    results tree (missing files there just yield all-NaN rows, same fail-soft behavior as the
    rest of this campaign's own data -- never crashes on a partial/absent cluster mount).
    `exp_base` overrides where OUR (JOLT) own results are read from -- e.g. a cluster mount
    mirroring the same {model_flag}/{dataset}/{tag}/seed{seed}/... subtree (see --jolt-root)."""
    all_data = {
        "single_user": compute_cta_pta_mean_var(
            model_flag, dataset, tag, "single_user", budgets, seeds, exp_base=exp_base,
        ),
    }
    if flip_root:
        all_data["flip_single_user"] = compute_cta_pta_mean_var_flip_centralized(
            flip_root, model_flag, dataset, budgets, seeds,
        )
    for agg in DEPLOY_AGG_METHODS_FEDERATED:
        all_data[f"federated_{agg}"] = compute_cta_pta_mean_var(
            model_flag, dataset, tag, "federated", budgets, seeds, agg=agg, exp_base=exp_base,
        )
        if flip_root:
            all_data[f"flip_{agg}"] = compute_cta_pta_mean_var_flip(
                flip_root, model_flag, dataset, agg, budgets, seeds,
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


def _branch_linestyle(branch_key):
    """single_user is dashed, the FLIP baseline is dotted (so it reads as "the comparison
    series" next to its same-colored federated_* solid line), everything else solid."""
    if branch_key == "single_user":
        return "--"
    if branch_key.startswith("flip_"):
        return ":"
    return "-"


def plot_cta_vs_pta(all_data, dataset, model_flag, tag, save_dir=None, filename=None):
    """One figure, one line per branch (single_user + each federated aggregator). Directly
    modeled on show_results.py's plot_cta_vs_pta (error bars = sqrt(var), best-budget point
    highlighted and annotated, equal-aspect axes, dashed grid)."""
    plt.figure(figsize=(7.5, 6))
    any_data = False

    for branch_key, df in all_data.items():
        # subset= is required: single_user's DataFrame has "agg"=None for every row (that
        # branch has no aggregator), and a bare df.dropna() treats None as missing on ANY
        # column -- silently dropping every row of that branch's df regardless of whether
        # cta/pta actually have data.
        df = df.dropna(subset=["cta_mean", "cta_var", "pta_mean", "pta_var"])
        if df.empty:
            continue
        any_data = True

        df = df.sort_values("budget")

        x = df["pta_mean"].values * 100
        y = df["cta_mean"].values * 100

        xerr = np.sqrt(df["pta_var"].values) * 100
        yerr = np.sqrt(df["cta_var"].values) * 100

        color = BRANCH_COLORS.get(branch_key)
        linestyle = _branch_linestyle(branch_key)

        plt.plot(
            x, y, linestyle=linestyle, linewidth=2.2, color=color, alpha=0.85,
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


def _draw_tradeoff_series(df, color, linestyle, label=None):
    """Draws one branch's CTA-vs-ASR curve (line + error bars + scatter + best-budget highlight
    + annotation) onto the current figure. Returns False (nothing drawn) if df has no complete
    (cta, pta) cell -- shared by plot_cta_vs_pta_per_branch's single_user and per-aggregator
    (BRoADflip vs FLIP) cases below."""
    # subset= is required: single_user's DataFrame has "agg"=None for every row (that branch has
    # no aggregator), and a bare df.dropna() treats None as missing on ANY column -- silently
    # dropping every row of that branch's df regardless of whether cta/pta actually have data.
    df = df.dropna(subset=["cta_mean", "cta_var", "pta_mean", "pta_var"])
    if df.empty:
        return False

    df = df.sort_values("budget")

    x = df["pta_mean"].values * 100
    y = df["cta_mean"].values * 100
    xerr = np.sqrt(df["pta_var"].values) * 100
    yerr = np.sqrt(df["cta_var"].values) * 100

    plt.plot(x, y, linestyle=linestyle, linewidth=2.2, color=color, alpha=0.85, label=label, zorder=3)
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
    return True


def plot_cta_vs_pta_per_branch(all_data, dataset, model_flag, tag, save_dir=None):
    """One standalone figure per aggregator (single_user + each federated aggregator), instead
    of plot_cta_vs_pta's single overlaid figure -- meant to be grouped externally into a LaTeX
    subfigure grid (one \\subfigure per aggregator, caption set by the caller). When a flip_{agg}
    entry is present in all_data (the FLIP paper baseline, see compute_cta_pta_mean_var_flip), it
    is overlaid on the SAME per-aggregator figure as a dotted same-colored curve, with a small
    legend distinguishing the two so each plot doubles as a BRoADflip-vs-FLIP comparison; a lone
    single_user figure keeps no legend/title (redundant with the external caption), same as
    before. Saved as f"{dataset}_{BRANCH_FILE_SUFFIX[branch]}.png". Returns the list of paths
    actually written."""
    saved = []

    # (own_key, flip_key, own_linestyle) pairs to overlay on one figure each -- single_user vs.
    # FLIP's own centralized (1v0) result, then each federated aggregator vs. its FLIP
    # counterpart. A legend is only drawn when both halves of a pair are actually present (a lone
    # own_key keeps the original unlabeled, uncluttered single-curve look).
    pairs = [("single_user", "flip_single_user", "--")]
    pairs += [(f"federated_{agg}", f"flip_{agg}", "-") for agg in DEPLOY_AGG_METHODS_FEDERATED]

    groups = {}
    for own_key, flip_key, own_linestyle in pairs:
        series = []
        if own_key in all_data:
            has_flip = flip_key in all_data
            series.append((
                all_data[own_key], BRANCH_COLORS[own_key], own_linestyle,
                BRANCH_LABELS[own_key] if has_flip else None,
            ))
        if flip_key in all_data:
            series.append((all_data[flip_key], BRANCH_COLORS[flip_key], ":", BRANCH_LABELS[flip_key]))
        if series:
            groups[own_key] = series

    for branch_key, series in groups.items():
        plt.figure(figsize=(7.5, 6))
        any_drawn = False
        has_legend = False
        for df, color, linestyle, label in series:
            drawn = _draw_tradeoff_series(df, color, linestyle, label=label)
            any_drawn = any_drawn or drawn
            has_legend = has_legend or (drawn and label is not None)

        if not any_drawn:
            plt.close()
            print(f"[WARNING] branch={branch_key!r} has no complete (cta,pta) cell yet for "
                  f"{model_flag}/{dataset}/tag={tag} -- skipping its plot.")
            continue

        plt.xlabel("ASR (%)")
        plt.ylabel("CTA (%)")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, linestyle="--", alpha=0.25)
        if has_legend:
            plt.legend(frameon=True, fontsize=10, loc="lower left")
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            suffix = BRANCH_FILE_SUFFIX.get(branch_key, branch_key)
            path = os.path.join(save_dir, f"{dataset}_{suffix}.png")
            plt.savefig(path, dpi=300)
            print(f"[INFO] Saved plot: {path}")
            saved.append(path)

        plt.close()

    return saved


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
        linestyle = _branch_linestyle(branch_key)

        plt.plot(
            budgets, means, marker="o", linestyle=linestyle, color=color, alpha=0.85, markersize=5,
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
# TABLE -- format_cell/compute_best_second/build_table ported verbatim from show_results.py
# (modules/base_utils/show_results.py's own render_block/build_final_table predate this and use
# a slightly different signature; scripts/show_results/show_results.py's build_table below is
# the one actually producing the paper's tables, so that's the one this mirrors).
# =========================
def format_cell(cta, cta_var, asr, asr_var):
    if np.isnan(cta):
        return "XXX"
    cta_std = np.sqrt(cta_var) * 100
    asr_std = np.sqrt(asr_var) * 100
    return f"{cta * 100:.1f}$\\pm${cta_std:.1f}/{asr * 100:.1f}$\\pm${asr_std:.1f}"


def compute_best_second(block, budgets, series_labels):
    """best/second-best budget per column, ranked by ASR (mean, index 2 of the stored 4-tuple)
    -- \\textbf{}/\\underline{} targets for build_table below."""
    best, second = {}, {}

    for series in series_labels:
        values = np.array([block.get((b, series), (np.nan,) * 4)[2] for b in budgets])
        valid_idx = np.where(~np.isnan(values))[0]

        if len(valid_idx) == 0:
            best[series] = second[series] = None
            continue

        sorted_idx = valid_idx[np.argsort(values[valid_idx])]
        best[series] = budgets[sorted_idx[-1]]
        second[series] = budgets[sorted_idx[-2] if len(sorted_idx) > 1 else sorted_idx[-1]]

    return best, second


def render_block(name, block, budgets, series_labels):
    best, second = compute_best_second(block, budgets, series_labels)
    lines = [f"\\multicolumn{{{1 + len(series_labels)}}}{{c}}{{\\textbf{{{name}}}}} \\\\", "\\midrule"]

    for b in budgets:
        row = [str(b)]
        for series in series_labels:
            cell = format_cell(*block.get((b, series), (np.nan,) * 4))
            if best[series] == b:
                cell = f"\\textbf{{{cell}}}"
            elif second[series] == b:
                cell = f"\\underline{{{cell}}}"
            row.append(cell)
        lines.append(" & ".join(row) + " \\\\")

    return "\n".join(lines)


def build_table(blocks, budgets, series_labels, caption, label):
    """`blocks` is an ordered dict: block name -> block dict (as filled by the MAIN loop below,
    {(budget, series): (cta_mean, cta_var, pta_mean, pta_var)}) -- one \\multicolumn block per
    tag/attack-variant, same layout as the paper's own table (Optimized Trigger / Sinusoidal
    Trigger / FLIP). Values are reported CTA/ASR (%) with std; bold/underline mark the
    highest/second-highest ASR per column within its own block."""
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "",
        f"\\caption{{{caption}}}",
        "\\vspace{2mm}",
        "\\begin{tabular}{c " + " ".join(["c"] * len(series_labels)) + "}",
        "\\toprule",
        "Budget & " + " & ".join(TABLE_AGG_DISPLAY.get(s, str(s).upper()) for s in series_labels) + " \\\\",
    ]

    first = True
    for name, block in blocks.items():
        lines.append("\\midrule")
        lines.append(render_block(name, block, budgets, series_labels))
        first = False

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def build_dataset_table(
    model_flag, dataset, tags, budgets=DEPLOY_BUDGETS, seeds=SEEDS, include_centralized=True,
    exp_base=EXP_BASE,
):
    """One table for (model_flag, dataset): one \\multicolumn block per `tags` entry that has
    ANY data on disk (missing tags -- not yet trained -- are silently skipped, never crash).
    Columns are the federated_3vs7 robust-aggregation rules (TABLE_AGG_ORDER), plus -- when
    include_centralized -- a leading CENTRALIZED column read off the single_user branch instead
    (the undefended 1-victim/no-aggregation deployment of the SAME attack, directly comparable
    at fixed budget/seed since it's the same generated trigger). `exp_base` overrides where OUR
    (JOLT) own results are read from, same as collect_dataset (see --jolt-root)."""
    columns = TABLE_COLUMNS_WITH_CENTRALIZED if include_centralized else TABLE_AGG_ORDER

    blocks = {}
    for tag in tags:
        block = {}
        any_cell = False
        for series in columns:
            if series == CENTRALIZED_SERIES:
                df = compute_cta_pta_mean_var(
                    model_flag, dataset, tag, "single_user", budgets, seeds, exp_base=exp_base,
                )
            else:
                df = compute_cta_pta_mean_var(
                    model_flag, dataset, tag, "federated", budgets, seeds, agg=series, exp_base=exp_base,
                )
            for _, row in df.iterrows():
                if not np.isnan(row["cta_mean"]):
                    any_cell = True
                block[(row["budget"], series)] = (
                    row["cta_mean"], row["cta_var"], row["pta_mean"], row["pta_var"],
                )
        if any_cell:
            blocks[TAG_DISPLAY_NAMES.get(tag, tag)] = block
        else:
            print(f"[INFO] tag={tag!r} has no data yet for {model_flag}/{dataset} -- skipped in table.")

    if not blocks:
        return None

    dataset_name = DATASET_DISPLAY.get(dataset, dataset)
    model_name = MODEL_DISPLAY.get(model_flag, model_flag)
    centralized_clause = " and the centralized (undefended, single-victim) deployment" if include_centralized else ""
    caption = (
        f"Impact of poisoning budget on ASR and CTA across robust aggregation rules{centralized_clause} on "
        f"{dataset_name} with a {model_name} model. Results are reported as CTA/ASR (in \\%) "
        f"with standard deviation. Bold and underlined values indicate the highest and "
        f"second-highest ASR."
    )
    label = f"tab:{dataset}_{model_flag}_budget_vs_asr"
    return build_table(blocks, budgets, columns, caption, label)


# =========================
# TRIGGER VISUALS -- trigger_path_in/save_trigger_visual_in/save_all_trigger_visuals ported from
# show_results.py, pointed at paper_main_campaign's own module_dir layout.
# =========================
def trigger_path_in(module_dir, model_flag, dataset):
    """Path to the .pt trigger written by federated_generate_labels_trigger_joint's run_module.py
    -- matches gen_configs_paper_main_campaign.generate_cell()'s own trigger_path build exactly
    (GEN_INIT/GEN_NUM_POISONED/GEN_NUM_HONESTS imported straight from that module, see top of
    this file)."""
    return (
        module_dir / "trigger"
        / f"opt_trig_direct_joint_{GEN_INIT}_{model_flag}_{dataset}_{GEN_NUM_POISONED}vs{GEN_NUM_HONESTS}.pt"
    )


def save_trigger_visual_in(module_dir, model_flag, dataset, label, sample_seed=0):
    """Renders clean-vs-poisoned side by side for one generated trigger and saves the PNG next
    to the .pt file itself (same `trigger/` directory) -- identical rendering to
    show_results.py's own save_trigger_visual_in."""
    trig_path = trigger_path_in(module_dir, model_flag, dataset)
    if not trig_path.exists():
        return None

    dataset_obj = load_dataset(dataset, train=True)
    indices = [i for i, (_, y) in enumerate(dataset_obj) if y == SOURCE_LABEL]
    idx = np.random.RandomState(sample_seed).choice(indices)
    img, _ = dataset_obj[idx]

    clean_img = img
    if isinstance(clean_img, torch.Tensor):
        clean_img = transforms.ToPILImage()(clean_img)

    poisoner = pick_poisoner("optimized", dataset, TARGET_LABEL, delta=str(trig_path))
    poisoned_img, _ = poisoner.poison((img, SOURCE_LABEL))
    if isinstance(poisoned_img, torch.Tensor):
        poisoned_img = transforms.ToPILImage()(poisoned_img)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    axes[0].imshow(clean_img)
    axes[0].set_title("Clean image", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(poisoned_img)
    axes[1].set_title(f"Poisoned ({label})", fontsize=11)
    axes[1].axis("off")
    plt.tight_layout()

    out_path = trig_path.with_suffix(".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved trigger visual: {out_path}")
    return out_path


def save_all_trigger_visuals(model_flags, datasets, tags, seeds, exp_base=EXP_BASE):
    """paper_main_campaign wrapper: builds each cell's module_dir via that generator's own
    cell_name/exp_base (defaults to the imported EXP_BASE, overridable -- see --jolt-root), one
    trigger per (model_flag, dataset, tag, seed)."""
    saved, missing = [], []
    for model_flag in model_flags:
        for dataset in datasets:
            for tag in tags:
                for seed in seeds:
                    module_dir = (
                        exp_base / cell_name(model_flag, dataset, tag, seed) / "gen_labels_trigger_joint"
                    )
                    out_path = save_trigger_visual_in(
                        module_dir, model_flag, dataset, label=f"{tag}, seed{seed}",
                    )
                    if out_path is None:
                        missing.append((model_flag, dataset, tag, seed))
                    else:
                        saved.append(out_path)

    if missing:
        print(f"[INFO] {len(missing)} trigger(s) not found yet (skipped):")
        for model_flag, dataset, tag, seed in missing:
            module_dir = exp_base / cell_name(model_flag, dataset, tag, seed) / "gen_labels_trigger_joint"
            print(f"  {trigger_path_in(module_dir, model_flag, dataset)}")

    return saved


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="r32p")
    parser.add_argument("--tag", default="baseline", help="single tag to plot per-dataset curves for")
    parser.add_argument(
        "--table-tags", nargs="+", default=None,
        help="tags to include as table blocks, in order (default: just --tag; add more once "
             "trained, e.g. --table-tags baseline eps_16_255 eps_0p50_lpips_0p1)",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--out-dir", default="./plots_paper_main_campaign_user")
    parser.add_argument("--csv-dir", default="./results_csv_paper_main_campaign_user")
    parser.add_argument("--table-dir", default="./tables_paper_main_campaign_user")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--flip-root", default="out_neurips",
        help="root of the FLIP paper baseline's results tree (out_neurips/{model}/"
             "{tag_suffix}_FLIP/{dataset}/backdoor/{agg}/1xs/{seed}/{budget}/{caccs,paccs}.npy), "
             "overlaid per-aggregator as a comparison series on the per-branch trade-off plots; "
             "pass an empty string to disable",
    )
    parser.add_argument(
        "--jolt-root", default=None,
        help="override where OUR (JOLT) own results are read from, in place of the imported "
             "EXP_BASE ('experiments/federated_experiments/"
             "threat_model_direct_trigger_joint_paper_main_campaign') -- e.g. a cluster mount "
             "mirroring that same {model_flag}/{dataset}/{tag}/seed{seed}/... subtree under a "
             "different root, to avoid local<->cluster file transfers "
             "(e.g. /shared/data1/Projects/DLWP/j1067582/martin/FLIP/out_iclr). "
             "Default: use EXP_BASE unchanged.",
    )
    parser.add_argument("--skip-trigger-visuals", action="store_true")
    parser.add_argument(
        "--no-centralized-column", action="store_true",
        help="drop the CENTRALIZED (single_user) column from the table, matching the "
             "federated-aggregators-only style of the original show_results.py table",
    )
    args = parser.parse_args()

    table_tags = args.table_tags or [args.tag]
    exp_base = Path(args.jolt_root) if args.jolt_root else EXP_BASE

    out_root = Path(args.out_dir) / f"{args.model}_{args.tag}"
    csv_root = Path(args.csv_dir) / f"{args.model}_{args.tag}"
    table_root = Path(args.table_dir)
    table_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_trigger_visuals:
        print(f"\n=== [paper_main_campaign] Trigger visuals ({args.model}, tags={table_tags}) ===")
        save_all_trigger_visuals([args.model], args.datasets, table_tags, args.seeds, exp_base=exp_base)

    for dataset in args.datasets:
        print(f"\n=== [paper_main_campaign/user] {args.model} / {dataset} / tag={args.tag} ===")
        all_data = collect_dataset(
            args.model, dataset, args.tag, seeds=args.seeds, flip_root=args.flip_root or None,
            exp_base=exp_base,
        )

        csv_dir = csv_root / dataset
        csv_dir.mkdir(parents=True, exist_ok=True)
        for branch_key, df in all_data.items():
            n_cta = df["cta_mean"].notna().sum()
            n_asr = df["pta_mean"].notna().sum()
            print(f"   -> {branch_key:24s} budgets with data: cta={n_cta}/{len(DEPLOY_BUDGETS)} asr={n_asr}/{len(DEPLOY_BUDGETS)}")
            df.to_csv(csv_dir / f"{branch_key}.csv", index=False)

        save_dir = str(out_root / dataset)

        # Per-branch trade-off plots -- one file per config (cifar_mean.png, cifar_krum.png,
        # ..., cifar_centralized.png), saved under out_root (which already keys on
        # {model}_{tag} -- see above) / {dataset}_{federated_tag_suffix}, so results from a
        # DIFFERENT tag never collide/overwrite these files (each tag gets its own
        # {model}_{tag}/ subtree). Matches the img_neurips/ subfigure-grid convention (e.g.
        # img_neurips/r32p_cifar_3vs7/cifar_mean.png) one level deeper -- group them into a
        # LaTeX subfigure grid by hand from there.
        tag_suffix = FEDERATED_TAG.split("_", 1)[-1]  # "federated_3vs7" -> "3vs7"
        per_branch_dir = str(out_root / f"{dataset}_{tag_suffix}")
        saved_per_branch = plot_cta_vs_pta_per_branch(
            all_data, dataset, args.model, args.tag, save_dir=per_branch_dir,
        )
        ok_tradeoff = bool(saved_per_branch)

        ok_cta = plot_metric_vs_budget(
            all_data, dataset, args.model, args.tag,
            metric="cta", ylabel="CTA (%)", save_dir=save_dir,
        )
        ok_asr = plot_metric_vs_budget(
            all_data, dataset, args.model, args.tag,
            metric="pta", ylabel="ASR (%)", save_dir=save_dir,
        )

        if not (ok_tradeoff or ok_cta or ok_asr):
            print(f"[WARNING] no data found under {exp_base / cell_name(args.model, dataset, args.tag, '*')} -- "
                  f"nothing plotted for {dataset}.")

        table = build_dataset_table(
            args.model, dataset, table_tags, seeds=args.seeds,
            include_centralized=not args.no_centralized_column, exp_base=exp_base,
        )
        if table is None:
            print(f"[WARNING] no data found for table blocks {table_tags} -- nothing written for {dataset}.")
        else:
            table_path = table_root / f"{args.model}_{dataset}.tex"
            table_path.write_text(table)
            print(f"[INFO] Saved table: {table_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
