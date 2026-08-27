"""
Offline analysis for Etape 0bis of the "switch to the exact QP solver" task -- reads a
diagnostics.jsonl produced with diag_oneshot_gap=true and reports:

  (0bis.1) J(ubar) vs B2_current vs mean_k(B2_QP,k), raw AND ||v||^2-window-normalized, on the
           SAME window -- requires inner_solve.py's B2_current_window_raw/_v2norm,
           B2_per_checkpoint_mean_raw/_v2norm, B2_coupled_raw/_v2norm fields (added after the
           first oneshot-gap-audit run; older logs fall back to the den-based fields, with a
           caveat that B2_current_continuous there is evaluated on a SINGLE representative
           checkpoint, not the whole window).

  (0bis.2) intra-checkpoint vs inter-checkpoint pairwise cosine of the per-checkpoint QP optima
           u*_k -- requires per_checkpoint_u_star (same as above, newly added). Separates
           "minibatch/estimation noise at fixed theta_k" (intra) from "genuine trajectory
           drift" (inter).

  (0bis.3) absolute values reported alongside relative ones; batches with
           B2_per_checkpoint_mean < min_b2_qp (default 1e-4) excluded from relative-value
           summary statistics (a near-zero denominator makes the ratio explode without a
           meaningful underlying gap) -- works on ANY oneshot-gap-audit log, old or new.

Usage: python prelim/analyze_oneshot_gap_diagnostics.py path/to/diagnostics.jsonl [min_b2_qp]
"""
import json
import sys

import numpy as np


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") == "inner_solve":
                continue
            if "oneshot_gap_absolute" not in r:
                continue
            records.append(r)
    return records


def summarize_0bis1(records):
    print("=== 0bis.1 -- J(ubar) vs B2_current vs mean_k(B2_QP,k), same window ===")
    have_window_fields = all("B2_current_window_raw" in r for r in records)
    if not have_window_fields:
        print(
            "[window-normalized fields absent in this log -- rerun with the updated "
            "inner_solve.py to get B2_current_window_raw/_v2norm etc. Falling back to the "
            "existing den-based fields below: B2_current_continuous there is evaluated on a "
            "SINGLE representative checkpoint, NOT the whole window B2_per_checkpoint_mean/"
            "B2_coupled use -- not a same-window comparison, shown for reference only.]"
        )
        for r in records:
            print(
                f"  batch {r['batch_idx']:>4} | B2_current_continuous(1 ckpt)="
                f"{r['B2_current_continuous']:.6g} | B2_per_checkpoint_mean(window)="
                f"{r['B2_per_checkpoint_mean']:.6g} | B2_coupled(window)="
                f"{r['B2_coupled']:.6g}"
            )
        return

    rows = [
        (
            r["batch_idx"], r["B2_current_window_raw"], r["B2_per_checkpoint_mean_raw"],
            r["B2_coupled_raw"], r["B2_current_window_v2norm"],
            r["B2_per_checkpoint_mean_v2norm"], r["B2_coupled_v2norm"],
        )
        for r in records
    ]
    print(
        f"{'batch':>6} {'B2_current(raw)':>16} {'B2_QPmean(raw)':>16} {'B2_coupled(raw)':>16}"
        f" {'B2_current(v2n)':>16} {'B2_QPmean(v2n)':>16} {'B2_coupled(v2n)':>16}"
    )
    for batch, cur_raw, qp_raw, cpl_raw, cur_n, qp_n, cpl_n in rows:
        print(
            f"{batch:>6} {cur_raw:>16.6g} {qp_raw:>16.6g} {cpl_raw:>16.6g}"
            f" {cur_n:>16.6g} {qp_n:>16.6g} {cpl_n:>16.6g}"
        )

    def med(vals):
        return float(np.median(vals))

    print("\nmedians:")
    print(f"  B2_current   raw={med([r[1] for r in rows]):.6g}  v2norm={med([r[4] for r in rows]):.6g}")
    print(f"  B2_QP_mean   raw={med([r[2] for r in rows]):.6g}  v2norm={med([r[5] for r in rows]):.6g}")
    print(f"  B2_coupled   raw={med([r[3] for r in rows]):.6g}  v2norm={med([r[6] for r in rows]):.6g}")


def summarize_0bis2(records):
    print("\n=== 0bis.2 -- intra- vs inter-checkpoint cosine of u*_k ===")
    have_tags = any("per_checkpoint_u_star" in r for r in records)
    if not have_tags:
        print("[per_checkpoint_u_star absent in this log -- rerun with the updated inner_solve.py]")
        return

    observations = []  # (batch_idx, checkpoint_id, vector)
    for r in records:
        for ckpt_id, vec in r.get("per_checkpoint_u_star", {}).items():
            observations.append((r["batch_idx"], ckpt_id, np.asarray(vec, dtype=np.float64)))

    intra, inter = [], []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            b1, k1, v1 = observations[i]
            b2, k2, v2 = observations[j]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < 1e-12:
                continue
            cos = float(np.dot(v1, v2) / denom)
            (intra if k1 == k2 else inter).append(cos)

    def stats(name, vals):
        if not vals:
            print(f"  {name}: no pairs")
            return
        print(
            f"  {name}: n={len(vals)}, mean={np.mean(vals):.4f}, median={np.median(vals):.4f}, "
            f"min={np.min(vals):.4f}, max={np.max(vals):.4f}"
        )

    stats("intra-checkpoint (same theta_k, different batch/minibatch)", intra)
    stats("inter-checkpoint (different theta_k)", inter)

    if intra and inter:
        gap = np.mean(inter) - np.mean(intra)
        print(f"\n  mean(inter) - mean(intra) = {gap:+.4f}")
        if abs(gap) < 0.1:
            print(
                "  -> intra ~ inter: the disagreement looks dominated by MINIBATCH/ESTIMATION "
                "NOISE at fixed theta_k, not trajectory drift -- the remedy is to aggregate Q "
                "and c over several batches before solving, not to restrict the checkpoint "
                "window."
            )
        else:
            print(
                "  -> inter >> intra: genuine TRAJECTORY DRIFT dominates -- try restricting "
                "the checkpoint window (expert_config's min/max in gen_configs.py) to a "
                "narrower [mid, end] range and re-measure the one-shot gap there."
            )
    elif not intra:
        print(
            "  (no intra-checkpoint pairs found -- no checkpoint id recurred across the "
            "sampled batches in this log; cannot separate noise from drift with this data. "
            "A longer or more frequently-diagnosed run would give sample_checkpoints more "
            "chances to redraw the same checkpoint.)"
        )


def summarize_0bis3(records, min_b2_qp=1e-4):
    print(
        f"\n=== 0bis.3 -- absolute values, excluding batches with "
        f"B2_per_checkpoint_mean < {min_b2_qp:g} ==="
    )
    kept = [r for r in records if r["B2_per_checkpoint_mean"] >= min_b2_qp]
    excluded = [r for r in records if r["B2_per_checkpoint_mean"] < min_b2_qp]
    print(f"kept {len(kept)}/{len(records)} batches ({len(excluded)} excluded as near-zero-denominator)")
    if excluded:
        print("  excluded batches: " + ", ".join(str(r["batch_idx"]) for r in excluded))

    if not kept:
        print("  nothing left to summarize.")
        return

    rel = [r["oneshot_gap_relative"] for r in kept]
    absv = [r["oneshot_gap_absolute"] for r in kept]
    print(
        f"  oneshot_gap_relative: median={np.median(rel):.3f}, mean={np.mean(rel):.3f}, "
        f"min={np.min(rel):.3f}, max={np.max(rel):.3f}"
    )
    print(
        f"  oneshot_gap_absolute: median={np.median(absv):.3g}, mean={np.mean(absv):.3g}, "
        f"min={np.min(absv):.3g}, max={np.max(absv):.3g}"
    )
    print("\n  per-batch (absolute alongside relative):")
    for r in kept:
        print(
            f"    batch {r['batch_idx']:>4}: gap_abs={r['oneshot_gap_absolute']:.3g}  "
            f"gap_rel={r['oneshot_gap_relative']:.3f}  "
            f"B2_per_checkpoint_mean={r['B2_per_checkpoint_mean']:.3g}"
        )


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python prelim/analyze_oneshot_gap_diagnostics.py path/to/diagnostics.jsonl [min_b2_qp]")
        sys.exit(1)
    min_b2_qp = float(sys.argv[2]) if len(sys.argv) == 3 else 1e-4
    records = load_records(sys.argv[1])
    if not records:
        print("No oneshot-gap diagnostic records found (diag_oneshot_gap must be true).")
        sys.exit(1)
    summarize_0bis1(records)
    summarize_0bis2(records)
    summarize_0bis3(records, min_b2_qp=min_b2_qp)


if __name__ == "__main__":
    main()
