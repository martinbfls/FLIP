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

    n_coupled_beats_current = sum(1 for r in rows if r[6] < r[4])
    print(
        f"\n  B2_coupled(v2n) < B2_current(v2n) in {n_coupled_beats_current}/{len(rows)} batches "
        "-- the decisive comparison: does the coupled ubar, on the SAME window, actually beat "
        "the current co-descended policy?"
    )

    if all("coupled_converged" in r for r in records):
        n_converged = sum(1 for r in records if r["coupled_converged"])
        iters = [r["coupled_actual_iters"] for r in records]
        print(
            f"  coupled QP solver converged on {n_converged}/{len(records)} batches "
            f"(actual_iters median={np.median(iters):.0f}, max={np.max(iters)}) -- if this is "
            "well below len(records) and/or actual_iters is pinned at the max_iters budget "
            "on most batches, B2_coupled above is likely an OVERESTIMATE (the true, converged "
            "J(ubar) is <= what's reported here); it would only strengthen a 'B2_coupled beats "
            "B2_current' conclusion, never weaken it."
        )


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


def load_m_sweep_records(path):
    """Etape 1a.1: 'inner_solve' events (policy_solver='qp') carrying an 'm_sweep' field --
    see run_module.py's diag_m_sweep wiring / inner_solve.m_sweep_cosine's docstring."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") == "inner_solve" and "m_sweep" in r:
                records.append(r)
    return records


def summarize_m_sweep(records):
    print("\n=== 1a.1 -- cos(u*(m)) vs. u*(max(m)), per checkpoint, median over observations ===")
    if not records:
        print("[no m_sweep records found -- rerun with policy_solver='qp' and diag_m_sweep=true "
              "(gen_configs.py --qp-m-sweep-audit), and long enough for history to accumulate "
              "(need >= max(diag_m_sweep_values) batches touching the SAME checkpoint).]")
        return

    by_m = {}
    for r in records:
        for m_str, cos in r["m_sweep"]["cosine_by_m"].items():
            if cos is not None:
                by_m.setdefault(int(m_str), []).append(cos)

    print(f"{'m':>6} {'n_obs':>7} {'median_cos':>12} {'min':>8} {'max':>8}")
    for m in sorted(by_m):
        vals = by_m[m]
        print(f"{m:>6} {len(vals):>7} {np.median(vals):>12.4f} {np.min(vals):>8.4f} {np.max(vals):>8.4f}")

    ordered_m = sorted(by_m)
    if len(ordered_m) >= 2:
        plateau_m, prev_m = ordered_m[-1], ordered_m[-2]
        plateau_val = np.median(by_m[plateau_m])
        step_gain = plateau_val - np.median(by_m[prev_m])
        print(f"\n  largest-m median cosine = {plateau_val:.4f} (m={plateau_m}); "
              f"last step's gain over m={prev_m} was {step_gain:+.4f}.")
        if plateau_val > 0.98:
            print("  -> plateau near 1: the residual disagreement (Diagnostic 0bis.2) is "
                  "essentially all MINIBATCH/ESTIMATION NOISE, fully averaged out by this m.")
        else:
            print(f"  -> plateau strictly below 1 ({plateau_val:.4f}): this residual IS "
                  "trajectory drift (prop:oneshot-gap net of noise), not an averaging artifact.")
        print("  Use the SMALLEST m where the step-to-step gain is already small as "
              "qp_batches_per_checkpoint in production.")


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python prelim/analyze_oneshot_gap_diagnostics.py path/to/diagnostics.jsonl [min_b2_qp]")
        sys.exit(1)
    min_b2_qp = float(sys.argv[2]) if len(sys.argv) == 3 else 1e-4
    path = sys.argv[1]

    records = load_records(path)
    m_sweep_records = load_m_sweep_records(path)
    if not records and not m_sweep_records:
        print("No oneshot-gap or m-sweep diagnostic records found (diag_oneshot_gap or "
              "diag_m_sweep must be true).")
        sys.exit(1)

    if records:
        summarize_0bis1(records)
        summarize_0bis2(records)
        summarize_0bis3(records, min_b2_qp=min_b2_qp)
    else:
        print("[no diag_oneshot_gap records in this log -- skipping 0bis.1/0bis.2/0bis.3]")
    summarize_m_sweep(m_sweep_records)


if __name__ == "__main__":
    main()
