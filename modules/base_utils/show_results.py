import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2.5,
})

AGG_COLORS = {
    "mean": "tab:blue",
    "median": "tab:orange",
    "krum": "tab:green",
    "trmean": "tab:red",
    "multikrum": "tab:purple",
}

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

def compute_cta_pta_mean_var(
    dataset,
    aggregator,
    budgets,
    runs,
    base_path=".",
    cta_file="caccs.npy",
    pta_file="paccs.npy",
    num_poisoned=1,
    num_clean=2,
    attack="backdoor",
    poisoner_flag="1xp",
    model_flag="r32p",
    use_flip=False,
):
    records = []
    if use_flip:
        base_dir = f"out/{model_flag}/{num_poisoned}vs{num_clean}_FLIP/{dataset}/{attack}/{aggregator}/{poisoner_flag}"
    else:
        base_dir = f"out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner_flag}"
    for budget in budgets:
        cta_vals, pta_vals = [], []
        for run in runs:
            run_dir = os.path.join(
                base_path,
                base_dir,
                str(run),
                str(budget),
            )
            cta_path = os.path.join(run_dir, cta_file)
            pta_path = os.path.join(run_dir, pta_file)
            if not (os.path.exists(cta_path) and os.path.exists(pta_path)):
                continue
            cta_vals.append(get_final_value(cta_path))
            pta_vals.append(get_final_value(pta_path))
        records.append({
            "dataset": dataset,
            "aggregator": aggregator,
            "budget": budget,
            "model": model_flag,
            "trigger": poisoner_flag,
            "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
            "cta_var": np.var(cta_vals) if cta_vals else np.nan,
            "pta_mean": np.mean(pta_vals) if pta_vals else np.nan,
            "pta_var": np.var(pta_vals) if pta_vals else np.nan,
        })

    return pd.DataFrame.from_records(records)

def compute_best_second(block, budgets, aggregators):
    best = {}
    second = {}

    for agg in aggregators:
        values = []

        for b in budgets:
            val = block.get((b, agg), (np.nan,)*4)
            values.append(val[2])  # ASR

        values = np.array(values)

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

def plot_cta_vs_pta_per_aggregator(df, dataset, aggregator, save_dir=None):

    df = df.dropna()
    if df.empty:
        return

    plt.figure(figsize=(7.5, 6))

    x = df["pta_mean"].values * 100
    y = df["cta_mean"].values * 100

    color = AGG_COLORS.get(aggregator, None)

    plt.plot(x, y, "-", color=color, alpha=0.8)
    plt.scatter(x, y, color=color, s=50)

    score = y - x
    best_idx = np.argmax(score)

    plt.scatter(
        x[best_idx], y[best_idx],
        s=120, edgecolor="black", linewidth=1.5, color=color
    )

    plt.xlabel("ASR (%)")
    plt.ylabel("CTA (%)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{dataset}_{aggregator}.png")
        plt.savefig(path, dpi=300)

    plt.close()

def format_cell(cta, cta_var, asr, asr_var):
    if np.isnan(cta):
        return "XXX"
    cta_std = np.sqrt(cta_var) * 100
    asr_std = np.sqrt(asr_var) * 100
    return f"{cta*100:.1f}$\\pm${cta_std:.1f}/{asr*100:.1f}$\\pm${asr_std:.1f}"

def render_block(name, block, budgets, aggregators):
    best, second = compute_best_second(block, budgets, aggregators)
    lines = []
    lines.append(f"\\multicolumn{{6}}{{c}}{{\\textbf{{{name}}}}} \\\\")
    lines.append("\\midrule")
    for b in budgets:
        row = [str(b)]
        for agg in aggregators:
            val = block.get((b, agg), (np.nan,)*4)
            cell = format_cell(*val)
            if best[agg] == b:
                cell = f"\\textbf{{{cell}}}"
            elif second[agg] == b:
                cell = f"\\underline{{{cell}}}"
            row.append(cell)
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines)

def build_final_table(tables, budgets, aggregators):

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\scriptsize")
    latex.append("\\setlength{\\tabcolsep}{3pt}")
    latex.append("")
    latex.append("\\begin{tabular}{c cccccc}")
    latex.append("\\toprule")
    latex.append("Budget & " + " & ".join([a.upper() for a in aggregators]) + " \\\\")
    latex.append("\\midrule")
    first = True
    for name, block in tables.items():
        if not first:
            latex.append("\\midrule")
        first = False
        latex.append(render_block(name, block, budgets, aggregators))

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


if __name__ == "__main__":

    DATASETS = ["cifar", "svhn"]
    AGGREGATORS = ["mean", "median", "krum", "trmean", "multikrum"]
    BUDGETS = [0, 150, 300, 500, 1000, 2000, 2500, 5000]
    RUNS = range(1, 11)

    BASE_PATH = "."
    CSV_DIR = "./results_csv"
    TABLE_DIR = "./tables"
    PLOT_DIR = "./plots"

    NUM_POISONED = 3
    NUM_CLEAN = 7
    ATTACK = "backdoor"
    MODEL_FLAG = "r32p"

    CONFIGS = [
        ("Optimized Trigger -- BRoAD\\textit{flip}", "optimized", False),
        ("Sinusoidal Trigger -- BRoAD\\textit{flip}", "1xs", False),
        ("Sinusoidal Trigger -- FLIP", "1xs", True),
    ]

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    for dataset in DATASETS:

        tables = {}

        for name, poisoner_flag, use_flip in CONFIGS:

            print(f"\n=== {name} - {dataset} ===")

            block = {}

            for aggregator in AGGREGATORS:

                print(f"   -> {aggregator}")

                df = compute_cta_pta_mean_var(
                    dataset=dataset,
                    aggregator=aggregator,
                    budgets=BUDGETS,
                    runs=RUNS,
                    base_path=BASE_PATH,
                    num_poisoned=NUM_POISONED,
                    num_clean=NUM_CLEAN,
                    attack=ATTACK,
                    poisoner_flag=poisoner_flag,
                    model_flag=MODEL_FLAG,
                    use_flip=use_flip
                )

                df.to_csv(
                    f"{CSV_DIR}/{MODEL_FLAG}_{poisoner_flag}_{dataset}_{aggregator}.csv",
                    index=False
                )

                plot_cta_vs_pta_per_aggregator(
                    df,
                    dataset=dataset,
                    aggregator=aggregator,
                    save_dir=f"{PLOT_DIR}/{MODEL_FLAG}_{poisoner_flag}_{dataset}/",
                )

                for _, row in df.iterrows():
                    block[(row["budget"], aggregator)] = (
                        row["cta_mean"],
                        row["cta_var"],
                        row["pta_mean"],
                        row["pta_var"],
                    )

            tables[name] = block

        latex_table = build_final_table(tables, BUDGETS, AGGREGATORS)

        with open(f"{TABLE_DIR}/{MODEL_FLAG}_{dataset}.tex", "w") as f:
            f.write(latex_table)
