"""
scripts/collect_policy_campaign.py -- Etape 6.3 of the "solveur QP couple" follow-up task.

Walks a policy_solver comparison campaign's output directory (the layout
gen_configs.py's generate_policy_solver_campaign / scripts/run_policy_campaign.sh produce:
EXP_BASE/<model>/<dataset>/<p>vs<h>/budget<b>/seed<s>/<solver>_<agg>/{policy_opt,
train_user_<b>}), assembles a tidy CSV (solver, seed, beta, agg, metric, value), and writes a
report.md with:
  - Etape 5's comparison table, aggregated over seeds as median [min, max] per (solver, beta, agg)
  - Etape 4's four-term decomposition (B2_span / alpha_tilde^2 / B2_QP / B2_current, all
    already logged per-batch in diagnostics.jsonl when diag_span_projection=true, the default)
  - Etape 3's discretization cost (mass/nnz requested vs. realized, for both the current policy
    and the QP reference -- also already in diagnostics.jsonl when diag_discretization=true)

Every number in report.md also appears in the CSV -- "every curve has a numeric twin" (8-12
point tables), per the task's own requirement.

Usage:
    python scripts/collect_policy_campaign.py <exp_base> [--out-csv PATH] [--out-report PATH]

<exp_base> is the campaign's root, e.g.
    experiments/federated_experiments/threat_model_expert_policy
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import median

import numpy as np


CELL_TAG_RE = re.compile(r"^(?P<solver>[a-z]+)_(?P<agg>[a-z]+)$")


def _read_last_diag_records(diagnostics_path, n=5):
    if not diagnostics_path.exists():
        return []
    records = []
    with open(diagnostics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") == "inner_solve":
                continue
            records.append(r)
    return records[-n:]


def _read_final_accuracy(npy_path):
    if not npy_path.exists():
        return None
    arr = np.load(npy_path, allow_pickle=True)
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    # paccs.npy/caccs.npy are typically (epochs,) or (epochs, k) -- take the last epoch's
    # (mean, if multiple values per epoch) value.
    last = arr[-1]
    return float(np.mean(last))


def _mean_over_records(records, key):
    vals = [r[key] for r in records if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def find_cells(exp_base):
    """
    Walks exp_base for directories matching <solver>_<agg>/policy_opt, inferring
    (model, dataset, num_poisoned vs num_honests, budget, seed, solver, agg) from the path --
    see gen_configs.py's cell_name()/generate_policy_solver_campaign for the layout this mirrors.
    """
    exp_base = Path(exp_base)
    cells = []
    for policy_opt_dir in exp_base.glob("*/*/*/budget*/seed*/*/policy_opt"):
        cell_tag_dir = policy_opt_dir.parent
        m = CELL_TAG_RE.match(cell_tag_dir.name)
        if not m:
            continue
        seed_dir = cell_tag_dir.parent
        budget_dir = seed_dir.parent
        pvsh_dir = budget_dir.parent
        dataset_dir = pvsh_dir.parent
        model_dir = dataset_dir.parent

        seed = int(seed_dir.name.replace("seed", ""))
        budget = budget_dir.name.replace("budget", "")

        config_path = policy_opt_dir / "config.toml"
        beta = None
        if config_path.exists():
            for line in config_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("beta =") or line.startswith("beta="):
                    try:
                        beta = float(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
                    break

        train_user_dir = seed_dir / cell_tag_dir.name / f"train_user_{budget}"
        cells.append({
            "model": model_dir.name, "dataset": dataset_dir.name, "pvsh": pvsh_dir.name,
            "budget": budget, "seed": seed, "solver": m.group("solver"), "agg": m.group("agg"),
            "beta": beta, "policy_opt_dir": policy_opt_dir, "train_user_dir": train_user_dir,
        })
    return cells


def collect_cell_metrics(cell):
    """Returns a dict metric_name -> value for one cell, or None values for anything missing
    (NEVER fabricated -- absence in the source data means absence in the output, per the
    "diagnostics" tasks' own repeated instruction not to invent numbers)."""
    records = _read_last_diag_records(cell["policy_opt_dir"] / "diagnostics.jsonl")
    metrics = {
        "B2_current_normalized": _mean_over_records(records, "B2_current_continuous"),
        "B2_qp_normalized": _mean_over_records(records, "B2_qp_continuous"),
        "B2_span_normalized": _mean_over_records(records, "B2_span"),
        "alpha_tilde_sq": _mean_over_records(records, "alpha_tilde_sq"),
        "budget_fraction_used": _mean_over_records(records, "current_policy_budget_fraction"),
        "cosine_Au_v": _mean_over_records(records, "current_cosine"),
        "nnz_current_requested": _mean_over_records(records, "current_nnz_requested"),
        "nnz_current_after_rounding": _mean_over_records(records, "current_nnz_after_rounding"),
        "nnz_qp_requested": _mean_over_records(records, "qp_nnz_requested"),
        "nnz_qp_after_rounding": _mean_over_records(records, "qp_nnz_after_rounding"),
        "mass_realized_fraction_current": _mean_over_records(records, "current_mass_realized_fraction"),
        "mass_realized_fraction_qp": _mean_over_records(records, "qp_mass_realized_fraction"),
    }
    metrics["ASR_final"] = _read_final_accuracy(cell["train_user_dir"] / "paccs.npy")
    metrics["clean_acc_final"] = _read_final_accuracy(cell["train_user_dir"] / "caccs.npy")
    # NOTE: "temps par pas externe" (Etape 5's per-outer-step wall time) is NOT collected here --
    # it lives in Slurm job logs (LOG_DIR from slurm_lib.sh), not in any artifact this script
    # reads. Left out rather than approximated; add a log-timestamp parser here if needed.
    return metrics


def write_csv(cells, out_path):
    rows = []
    for cell in cells:
        metrics = collect_cell_metrics(cell)
        for metric, value in metrics.items():
            rows.append({
                "solver": cell["solver"], "seed": cell["seed"], "beta": cell["beta"],
                "agg": cell["agg"], "metric": metric, "value": value,
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["solver", "seed", "beta", "agg", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _agg_median_min_max(rows, solver, beta, agg, metric):
    vals = [
        r["value"] for r in rows
        if r["solver"] == solver and r["beta"] == beta and r["agg"] == agg
        and r["metric"] == metric and r["value"] is not None
    ]
    if not vals:
        return None
    return median(vals), min(vals), max(vals)


def _fmt(triple):
    if triple is None:
        return "n/a"
    med, lo, hi = triple
    return f"{med:.4g} [{lo:.4g}, {hi:.4g}]"


def write_report(rows, cells, out_path):
    solvers = sorted({c["solver"] for c in cells})
    betas = sorted({c["beta"] for c in cells if c["beta"] is not None})
    aggs = sorted({c["agg"] for c in cells})

    lines = ["# Policy solver campaign report", ""]
    lines.append(
        "Every table below is the numeric twin of what a figure would show -- 8-12 points, "
        "readable without any plot. Values are `median [min, max]` over seeds."
    )
    lines.append("")

    # Etape 5: comparison table, per (beta, agg), solver as columns.
    lines.append("## Etape 5 -- solver comparison")
    for agg in aggs:
        for beta in betas:
            lines.append(f"\n### agg={agg}, beta_local={beta}\n")
            lines.append("| metric | " + " | ".join(solvers) + " |")
            lines.append("|---|" + "---|" * len(solvers))
            for metric, label in [
                ("B2_current_normalized", "B2 final (normalized)"),
                ("budget_fraction_used", "budget used (‖u‖_1/beta_local)"),
                ("cosine_Au_v", "cos(Au, v)"),
                ("nnz_current_requested", "directions active (requested)"),
                ("nnz_current_after_rounding", "directions active (after rounding)"),
                ("ASR_final", "ASR (final)"),
            ]:
                cells_str = [
                    _fmt(_agg_median_min_max(rows, s, beta, agg, metric)) for s in solvers
                ]
                lines.append(f"| {label} | " + " | ".join(cells_str) + " |")

    # Etape 4: four-term decomposition.
    lines.append("\n## Etape 4 -- four-term decomposition (all / ‖v‖^2)\n")
    lines.append("| solver | beta | agg | B2_span | alpha_tilde^2 | B2_QP | B2_current |")
    lines.append("|---|---|---|---|---|---|---|")
    for solver in solvers:
        for beta in betas:
            for agg in aggs:
                row = [
                    _fmt(_agg_median_min_max(rows, solver, beta, agg, m))
                    for m in ("B2_span_normalized", "alpha_tilde_sq", "B2_qp_normalized",
                              "B2_current_normalized")
                ]
                lines.append(f"| {solver} | {beta} | {agg} | " + " | ".join(row) + " |")

    # Etape 3: discretization cost.
    lines.append("\n## Etape 3 -- discretization cost (current policy vs. QP reference)\n")
    lines.append(
        "| solver | beta | agg | mass realized (current) | mass realized (QP) | "
        "nnz current (req->after) | nnz QP (req->after) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for solver in solvers:
        for beta in betas:
            for agg in aggs:
                mass_cur = _agg_median_min_max(rows, solver, beta, agg, "mass_realized_fraction_current")
                mass_qp = _agg_median_min_max(rows, solver, beta, agg, "mass_realized_fraction_qp")
                nnz_cur_req = _agg_median_min_max(rows, solver, beta, agg, "nnz_current_requested")
                nnz_cur_after = _agg_median_min_max(rows, solver, beta, agg, "nnz_current_after_rounding")
                nnz_qp_req = _agg_median_min_max(rows, solver, beta, agg, "nnz_qp_requested")
                nnz_qp_after = _agg_median_min_max(rows, solver, beta, agg, "nnz_qp_after_rounding")
                nnz_cur_str = f"{_fmt(nnz_cur_req)} -> {_fmt(nnz_cur_after)}"
                nnz_qp_str = f"{_fmt(nnz_qp_req)} -> {_fmt(nnz_qp_after)}"
                lines.append(
                    f"| {solver} | {beta} | {agg} | {_fmt(mass_cur)} | {_fmt(mass_qp)} | "
                    f"{nnz_cur_str} | {nnz_qp_str} |"
                )

    lines.append(
        "\n*(agg is a LOGGING-ONLY axis -- see generate_policy_solver_campaign's docstring; "
        "rows differing only in agg are functionally identical runs.)*"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_base")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    exp_base = Path(args.exp_base)
    out_csv = Path(args.out_csv) if args.out_csv else exp_base / "campaign_results.csv"
    out_report = Path(args.out_report) if args.out_report else exp_base / "report.md"

    cells = find_cells(exp_base)
    if not cells:
        print(f"No cells found under {exp_base} -- has the campaign been run yet?")
        sys.exit(1)

    rows = write_csv(cells, out_csv)
    write_report(rows, cells, out_report)
    print(f"Found {len(cells)} cells. Wrote {out_csv} ({len(rows)} rows) and {out_report}.")


if __name__ == "__main__":
    main()
