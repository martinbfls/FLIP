"""
Builds an index of every run tracked under experiments/, from the run_manifest.json files
ExperimentTracker writes at experiments/<experiment_name>/plots/<module_name>/run_manifest.json
(see modules/base_utils/experiment_tracker.py).

For each run, reports: experiment name, module, config path, whether it finished (metrics
file present and non-empty) or looks crashed/incomplete (manifest exists but metrics file is
missing or empty -- ExperimentTracker.finalize() always writes both together, so a mismatch
means the process died between them, or the metrics file was deleted after the fact), plot
files, and W&B run info if enabled.

Run:  python scripts/list_experiments.py [root] [--format csv|table] [--out path]
Prints a table to stdout by default; --format csv with --out writes a CSV file instead.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_runs(root: Path):
    """Yields one dict per run_manifest.json found under root."""
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            yield {
                "experiment_name": "?", "module_name": "?",
                "status": f"UNREADABLE ({e})", "manifest_path": str(manifest_path),
            }
            continue

        metrics_path = Path(manifest.get("metrics_log_path", ""))
        if metrics_path.exists() and metrics_path.stat().st_size > 0:
            status = "finished"
        else:
            status = "crashed/incomplete (manifest present, metrics missing/empty)"

        wandb_info = manifest.get("wandb", {})
        yield {
            "experiment_name": manifest.get("experiment_name", "?"),
            "module_name": manifest.get("module_name", "?"),
            "status": status,
            "config_path": manifest.get("config_path", ""),
            "metrics_log_path": str(metrics_path),
            "n_plots": len(manifest.get("plots", [])),
            "wandb_enabled": wandb_info.get("enabled", False),
            "wandb_run_url": wandb_info.get("run_url") or "",
            "manifest_path": str(manifest_path),
        }


def print_table(rows):
    if not rows:
        print("No runs found (no run_manifest.json under the given root).")
        return
    cols = ["experiment_name", "module_name", "status", "n_plots", "wandb_enabled"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print(f"\n{len(rows)} run(s) indexed.")


def write_csv(rows, out_path):
    if not rows:
        print("No runs found -- not writing an empty CSV.")
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="experiments")
    parser.add_argument("--format", choices=["table", "csv"], default="table")
    parser.add_argument("--out", default=None, help="CSV output path (required with --format csv)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"{root} does not exist -- nothing to index.")
        return 0

    rows = list(find_runs(root))

    if args.format == "csv":
        out_path = args.out or "experiments_registry.csv"
        write_csv(rows, out_path)
    else:
        print_table(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
