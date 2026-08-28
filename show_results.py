import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from modules.base_utils.datasets import load_dataset, pick_poisoner
from modules.federated_generate_labels_trigger_joint.gen_configs import (
    EXP_BASE,
    MODEL_FLAGS,
    DATASETS,
    AGG_METHODS,
    SEEDS,
    BUDGETS,
    NUM_POISONED,
    NUM_HONESTS,
    SOURCE_LABEL,
    TARGET_LABEL,
    cell_name,
)


# =========================
# Style (Paper-ready)
# =========================
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "lines.linewidth": 2.5,
    }
)


# =========================
# Colors -- keyed by agg_method, since aggregator is the axis this campaign
# sweeps (unlike the older federated_optimizing_trigger show_results.py this
# replaces, which swept trigger-generation strategies instead).
# =========================
AGG_COLORS = {
    "mean": "tab:blue",
    "median": "tab:orange",
    "krum": "tab:green",
    "trmean": "tab:red",
    "multikrum": "tab:purple",
}


# =========================
# Utils
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
def compute_cta_pta_mean_var(
    model_flag,
    dataset,
    agg_method,
    budgets,
    seeds,
    cta_file="caccs.npy",
    pta_file="paccs.npy",
):
    """Reads federated_train_user's outputs for each (budget, seed) cell of this
    threat_model_direct_trigger_joint campaign. Directory layout matches
    gen_configs.generate_cell() exactly: EXP_BASE / cell_name(...) /
    train_user_{budget}/{caccs,paccs}.npy -- cell_name() is imported straight
    from gen_configs.py so this can't drift from wherever the configs actually
    got written.
    """
    records = []

    for budget in budgets:
        cta_vals, pta_vals = [], []

        for seed in seeds:
            run_dir = (
                EXP_BASE
                / cell_name(model_flag, dataset, agg_method, seed)
                / f"train_user_{budget}"
            )

            cta_path = run_dir / cta_file
            pta_path = run_dir / pta_file

            if not (cta_path.exists() and pta_path.exists()):
                continue

            cta_vals.append(get_final_value(cta_path))
            pta_vals.append(get_final_value(pta_path))

        records.append(
            {
                "dataset": dataset,
                "aggregator": agg_method,
                "budget": budget,
                "model": model_flag,
                "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
                "cta_var": np.var(cta_vals) if cta_vals else np.nan,
                "pta_mean": np.mean(pta_vals) if pta_vals else np.nan,
                "pta_var": np.var(pta_vals) if pta_vals else np.nan,
            }
        )

    return pd.DataFrame.from_records(records)


# =========================
# TRIGGER VISUALS
# =========================
def trigger_path(model_flag, dataset, agg_method, seed):
    """Path to the .pt trigger written by federated_generate_labels_trigger_joint's
    run_module.py (torch.save(delta, ...), see its `trig_path` build). Mirrors
    gen_configs.py's JOINT_TRIGGER_TEMPLATE exactly: output_dir_trigger =
    "{module_dir}/trigger" where module_dir = cell_dir/"gen_labels_trigger_joint",
    and the filename is init="stripe" (the module's own default, unchanged by
    gen_configs.py) with run_tag=f"{num_poisoned}vs{num_honests}"."""
    return (
        EXP_BASE
        / cell_name(model_flag, dataset, agg_method, seed)
        / "gen_labels_trigger_joint"
        / "trigger"
        / f"opt_trig_direct_joint_stripe_{model_flag}_{dataset}_{NUM_POISONED}vs{NUM_HONESTS}.pt"
    )


def save_trigger_visual(model_flag, dataset, agg_method, seed, sample_seed=0):
    """Renders clean-vs-poisoned side by side for one generated trigger and saves
    the PNG next to the .pt file itself (same `trigger/` directory)."""
    trig_path = trigger_path(model_flag, dataset, agg_method, seed)
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
    axes[1].set_title(f"Poisoned ({agg_method}, seed{seed})", fontsize=11)
    axes[1].axis("off")
    plt.tight_layout()

    out_path = trig_path.with_suffix(".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved trigger visual: {out_path}")
    return out_path


def save_all_trigger_visuals(model_flags, datasets, agg_methods, seeds):
    saved, missing = [], []
    for model_flag in model_flags:
        for dataset in datasets:
            for agg_method in agg_methods:
                for seed in seeds:
                    out_path = save_trigger_visual(model_flag, dataset, agg_method, seed)
                    if out_path is None:
                        missing.append((model_flag, dataset, agg_method, seed))
                    else:
                        saved.append(out_path)
    if missing:
        print(f"[INFO] {len(missing)} trigger(s) not found yet (skipped):")
        for model_flag, dataset, agg_method, seed in missing:
            print(f"  {trigger_path(model_flag, dataset, agg_method, seed)}")
    return saved


def compute_best_second(block, budgets, aggregators):
    best = {}
    second = {}

    for agg in aggregators:
        values = []

        for b in budgets:
            val = block.get((b, agg), (np.nan,) * 4)
            values.append(val[2])  # ASR

        values = np.array(values)

        # ignore NaNs
        valid_idx = np.where(~np.isnan(values))[0]

        if len(valid_idx) == 0:
            best[agg] = None
            second[agg] = None
            continue

        sorted_idx = valid_idx[np.argsort(values[valid_idx])]

        best_idx = sorted_idx[-1]
        second_idx = sorted_idx[-2] if len(sorted_idx) > 1 else sorted_idx[-1]

        best[agg] = budgets[best_idx]
        second[agg] = budgets[second_idx]

    return best, second


# =========================
# PLOT
# =========================
def annotate_key_points(df, x, y, score, color):

    budgets = df["budget"].values

    if np.all(np.isnan(score)):
        return

    # max du score (ignore NaN)
    max_score = np.nanmax(score)

    # indices candidats (égalité tolérante aux floats)
    candidates = np.where(np.isclose(score, max_score, atol=1e-12))[0]

    # choisir le plus petit budget
    idx_best = candidates[np.argmin(budgets[candidates])]

    key_indices = [idx_best]

    # offsets adaptatifs
    x_range = max(np.nanmax(x) - np.nanmin(x), 1e-6)
    y_range = max(np.nanmax(y) - np.nanmin(y), 1e-6)

    dx = 0.03 * x_range
    dy = 0.03 * y_range

    directions = [
        (+dx, +dy),
        (+dx, -dy),
        (-dx, +dy),
        (-dx, -dy),
        (0, +1.5 * dy),
        (+1.5 * dx, 0),
    ]

    for k, i in enumerate(key_indices):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue

        dx_i, dy_i = directions[k % len(directions)]

        plt.text(
            x[i] + dx_i,
            y[i] + dy_i,
            str(int(budgets[i])),
            fontsize=11,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.8),
            zorder=10,
        )


def plot_cta_vs_pta_per_aggregator(all_data, dataset, save_dir=None):
    """One figure per dataset, one line per aggregator (the axis this
    campaign actually sweeps)."""

    plt.figure(figsize=(7.5, 6))

    for agg_method, df in all_data.items():
        df = df.dropna()
        if df.empty:
            continue

        df = df.sort_values("budget")

        x = df["pta_mean"].values * 100
        y = df["cta_mean"].values * 100

        xerr = np.sqrt(df["pta_var"].values) * 100
        yerr = np.sqrt(df["cta_var"].values) * 100

        color = AGG_COLORS.get(agg_method, None)

        plt.plot(
            x, y, linestyle="-", linewidth=2.2, color=color, alpha=0.85,
            label=agg_method, zorder=3,
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

    plt.xlabel("ASR (%)")
    plt.ylabel("CTA (%)")

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.25)

    plt.legend(frameon=True, fontsize=11, loc="lower left")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{dataset}.png")
        plt.savefig(path, dpi=300)
        print(f"[INFO] Saved plot: {path}")

    plt.close()


# =========================
# TABLE
# =========================
def format_cell(cta, cta_var, asr, asr_var):

    if np.isnan(cta):
        return "XXX"

    cta_std = np.sqrt(cta_var) * 100
    asr_std = np.sqrt(asr_var) * 100

    return f"{cta * 100:.1f}$\\pm${cta_std:.1f}/{asr * 100:.1f}$\\pm${asr_std:.1f}"


def build_table(block, budgets, aggregators, name):

    best, second = compute_best_second(block, budgets, aggregators)

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("")
    lines.append("\\begin{tabular}{c cccccc}")
    lines.append("\\toprule")
    lines.append("Budget & " + " & ".join([a.upper() for a in aggregators]) + " \\\\")
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{1 + len(aggregators)}}}{{c}}{{\\textbf{{{name}}}}} \\\\")
    lines.append("\\midrule")

    for b in budgets:
        row = [str(b)]

        for agg in aggregators:
            val = block.get((b, agg), (np.nan,) * 4)
            cell = format_cell(*val)

            if best[agg] == b:
                cell = f"\\textbf{{{cell}}}"
            elif second[agg] == b:
                cell = f"\\underline{{{cell}}}"

            row.append(cell)

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Sweep axes are read from gen_configs.py itself (not redeclared here) so
    # this script always looks in the same places the campaign actually wrote
    # to -- edit gen_configs.py's grid, not this file, to change what's read.
    CSV_DIR = "./results_csv_trigger_joint"
    TABLE_DIR = "./tables_trigger_joint"
    PLOT_DIR = "./plots_trigger_joint"

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("\n=== Trigger visuals ===")
    save_all_trigger_visuals(MODEL_FLAGS, DATASETS, AGG_METHODS, SEEDS)

    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            print(f"\n=== {model_flag} / {dataset} ===")

            all_data = {}
            block = {}

            for agg_method in AGG_METHODS:
                print(f"   -> {agg_method}")

                df = compute_cta_pta_mean_var(
                    model_flag=model_flag,
                    dataset=dataset,
                    agg_method=agg_method,
                    budgets=BUDGETS,
                    seeds=SEEDS,
                )

                df.to_csv(
                    f"{CSV_DIR}/{model_flag}_{dataset}_{agg_method}.csv",
                    index=False,
                )

                all_data[agg_method] = df

                for _, row in df.iterrows():
                    block[(row["budget"], agg_method)] = (
                        row["cta_mean"],
                        row["cta_var"],
                        row["pta_mean"],
                        row["pta_var"],
                    )

            latex_table = build_table(
                block, BUDGETS, AGG_METHODS, name=f"{model_flag}/{dataset}"
            )
            with open(f"{TABLE_DIR}/{model_flag}_{dataset}.tex", "w") as f:
                f.write(latex_table)
            print(f"[INFO] Saved LaTeX table for {model_flag}/{dataset}")

            plot_cta_vs_pta_per_aggregator(
                all_data, dataset=dataset, save_dir=f"{PLOT_DIR}/{model_flag}/"
            )
