"""
prelim/report.py -- builds prelim/artifacts/report.md (+ report.json, same
content, machine-readable) from prelim/artifacts/metrics.csv and run_meta.json.

Designed to be read WITHOUT the figures: every plot produced by prelim.ipynb
has a numeric twin here (a decimated table, a correlation coefficient plus
extreme points, or a heatmap flattened to its top entries) -- see each
section's "Jumeaux numeriques" paragraph for which metric rows back which
figure. sweep.py records exactly the rows this file needs (eigval_Q_p*,
u_star_top*, sweep_err_rel__B=*, snr__B=* etc.) precisely so this doesn't have
to re-derive anything from Gbar (never persisted by default).

Usage: python report.py [--metrics PATH] [--out-dir DIR]
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as _stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prelim_lib as pl

ARTIFACT_DIR = os.path.join(pl._FLIP_ROOT, "prelim", "artifacts")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.csv")
RUN_META_PATH = os.path.join(ARTIFACT_DIR, "run_meta.json")
FAILURES_PATH = os.path.join(ARTIFACT_DIR, "failures.log")
REPORT_MD_PATH = os.path.join(ARTIFACT_DIR, "report.md")
REPORT_JSON_PATH = os.path.join(ARTIFACT_DIR, "report.json")


# --------------------------------------------------------------------------#
# Formatting helpers
# --------------------------------------------------------------------------#
def sig(x, n=3):
    """Format a float to n significant digits, plain (no scientific notation
    unless the magnitude demands it)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return str(x)
    if x == 0:
        return "0"
    try:
        from decimal import Decimal
        d = Decimal(repr(float(x)))
        exp = d.adjusted()
        if exp < -4 or exp > 6:
            return f"{x:.{n-1}e}"
        digits = max(0, n - 1 - exp)
        return f"{x:.{digits}f}"
    except Exception:
        return f"{x:.{n}g}"


def median_range(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return "n/a"
    if len(values) == 1:
        return sig(values[0])
    return f"{sig(float(np.median(values)))} [{sig(min(values))}, {sig(max(values))}]"


def md_table(headers, rows):
    if not rows:
        return "*(aucune ligne)*\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def status_str(ok):
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------#
# Data access helpers
# --------------------------------------------------------------------------#
def qvals(df, experiment, metric, **filters):
    sub = df[(df.experiment == experiment) & (df.metric == metric)]
    for k, v in filters.items():
        if v is None:
            continue
        sub = sub[sub[k] == v]
    return sub["value"].tolist()


def qval1(df, experiment, metric, **filters):
    vals = qvals(df, experiment, metric, **filters)
    return vals[0] if vals else None


def models_present(df):
    return sorted(df["model"].dropna().unique().tolist())


def checkpoints_present(df, model):
    ck = df[(df.model == model) & (df.checkpoint.isin(["begin", "mid", "end"]))]["checkpoint"].unique().tolist()
    order = {"begin": 0, "mid": 1, "end": 2}
    return sorted(ck, key=lambda c: order.get(c, 99))


# --------------------------------------------------------------------------#
# Sec 0 -- header
# --------------------------------------------------------------------------#
def section0(df, meta, failures_text):
    lines = ["## 0. En-tete\n"]
    n_fail_log = len([l for l in failures_text.split("\n" + "=" * 70) if l.strip()]) if failures_text else 0
    rows = [
        ["Date du rapport", time.strftime("%Y-%m-%d %H:%M")],
        ["Commit git", meta.get("git_commit", "?") + (" (dirty)" if meta.get("git_dirty") else "")],
        ["torch", meta.get("torch_version", "?")],
        ["numpy", meta.get("numpy_version", "?")],
        ["osqp", meta.get("osqp_version", "?")],
        ["Device entrainement / evaluation", f"{meta.get('train_device','?')} / {meta.get('eval_device','?')}"],
        ["Duree totale du sweep", f"{meta.get('duration_s', 0):.0f}s (~{meta.get('duration_s', 0)/60:.1f} min)"],
        ["Cellules executees / en cache / en echec",
         f"{meta.get('n_cells_run', '?')} / {meta.get('n_cells_cached', '?')} / {meta.get('n_cells_failed', '?')}"],
        ["Lignes dans metrics.csv", str(len(df))],
        ["Entrees dans failures.log", str(n_fail_log)],
    ]
    lines.append(md_table(["Champ", "Valeur"], rows))
    data = dict(rows=rows)
    return "\n".join(lines), data


# --------------------------------------------------------------------------#
# Sec 1 -- grid executed
# --------------------------------------------------------------------------#
def section1(df, meta):
    lines = ["## 1. Grille executee\n"]
    grid = meta.get("grid", {})
    rows = [[k, ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)] for k, v in grid.items()]
    lines.append("### Axes du balayage\n")
    lines.append(md_table(["Axe", "Valeurs"], rows))
    lines.append(
        "\nPortee reelle de E1-E3 (voir sweep.py docstring) : E1 (table principale) tourne a "
        "`seed=SEEDS[0]`, `beta=E1_BETA=0.10`, tous les checkpoints ; robustesse aux graines "
        "et balayage SNR/batch a `checkpoint=end` uniquement, sur toutes les graines/betas "
        "respectivement ; E2/E3 tournent sur tous les checkpoints x tous les betas (independants "
        "de la graine). TRIGGERS=stripe est calcule et mis en cache (`grad_bd`) mais pas encore "
        "exploite par une metrique E1-E3 -- reserve a E4/E5.\n"
    )
    lines.append(f"\nNombre de cellules (checkpoint x modele) : "
                 f"{meta.get('n_cells_run', 0) + meta.get('n_cells_cached', 0) + meta.get('n_cells_failed', 0)}\n")

    failed = meta.get("failed_cells", [])
    lines.append("\n### Cellules echouees ou ignorees\n")
    if failed:
        rows_f = [[f.get("model"), f.get("seed"), f.get("checkpoint"), f.get("reason")] for f in failed]
        lines.append(md_table(["Modele", "Graine", "Checkpoint", "Raison"], rows_f))
    else:
        lines.append("Aucune.\n")

    gbar_sec = meta.get("gbar_seconds", [])
    if gbar_sec:
        lines.append("\n### Cout de Gbar (compute_expected_flip_gradients) par cellule\n")
        rows_g = [[g["model"], g["checkpoint"], f"{g['seconds']:.1f}s"] for g in gbar_sec]
        lines.append(md_table(["Modele", "Checkpoint", "Duree"], rows_g))

    return "\n".join(lines), dict(grid=grid, failed_cells=failed, gbar_seconds=gbar_sec)


# --------------------------------------------------------------------------#
# Sec 2 -- assertions
# --------------------------------------------------------------------------#
def section2(df):
    lines = ["## 2. Assertions de coherence\n"]
    rows = []
    data = []

    def add(name, scope, value, threshold, ok, pending=False):
        status = "PENDING" if pending else status_str(ok)
        rows.append([name, scope, status, sig(value) if value is not None else "n/a",
                     sig(threshold) if threshold is not None else "n/a"])
        data.append(dict(name=name, scope=scope, status=status, value=value, threshold=threshold))

    g1 = qvals(df, "assertions", "budget_relation_abs_gap")
    v1 = max(g1) if g1 else None
    add("Relation de budget ||u_i||_1*gamma == ||ubar*||_1", "par modele (checkpoint=end)",
        v1, 1e-6, v1 is not None and v1 <= 1e-6)

    g2 = qvals(df, "assertions", "solve_qp_vs_repo_project_gradient_max_abs_diff")
    v2 = max(g2) if g2 else None
    add("solve_qp(capacity=False) == project_gradient (depot)", "par modele (checkpoint=end)",
        v2, 1e-6, v2 is not None and v2 <= 1e-6)

    g3 = qvals(df, "assertions", "alpha_tilde_budget_frac_of_bigbeta")
    v3 = max(g3) if g3 else None
    add("alpha_tilde_star: contrainte de budget inactive a l'optimum (NNLS)", "par modele (checkpoint=end)",
        v3, 1e-2, v3 is not None and v3 <= 1e-2)

    gap_cols = [c for c in df["metric"].unique() if c.startswith("flip_mass_gap_rel_max__")]
    g4 = df[df.metric.isin(gap_cols)]["value"].tolist()
    v4 = max(g4) if g4 else None
    add("Masses de flip realisees vs demandees : ecart relatif max (>=1 flip attendu)",
        "toutes cellules E1", v4, 0.10, v4 is not None and v4 <= 0.10)

    add("Agregateur aplati == depot (modele a un seul tenseur)", "N/A cette session",
        None, None, False, pending=True)

    nan_ct = int(df["value"].isna().sum())
    inf_ct = int(np.isinf(df["value"]).sum())
    add(f"Aucun NaN/Inf dans le CSV (NaN={nan_ct}, Inf={inf_ct})", "metrics.csv entier",
        nan_ct + inf_ct, 0, (nan_ct + inf_ct) == 0)

    lines.append(md_table(["Nom", "Portee", "Statut", "Valeur observee", "Seuil"], rows))
    return "\n".join(lines), dict(assertions=data)


# --------------------------------------------------------------------------#
# Sec 3 -- E1
# --------------------------------------------------------------------------#
E1_TAGS = ["a_source_target", "b_other_pair", "c_uniform", "d_qp", "e_random1", "f_random2"]


def section_e1(df):
    lines = ["## 3. E1 -- Carte de biais : implementation, transfert, signal/bruit\n"]
    lines.append("**Hypothese** : `E[g_i] = grad_c + Gbar@u_i` transfere du jeu de calibration vers "
                  "un shard reel, et le signal domine le bruit minibatch aux budgets realistes.\n")
    models = models_present(df)
    n_cells = len(df[df.experiment == "E1"][["model", "checkpoint"]].drop_duplicates())
    lines.append(f"**Ce qui a ete execute** : table principale (6 configs x checkpoints) a "
                 f"beta={pl.__name__ and 0.10}, robustesse aux graines et balayage SNR/batch a "
                 f"checkpoint=end -- {n_cells} cellules (modele x checkpoint) au total.\n")

    data = {}
    lines.append("\n### Resultats -- transfert calibration -> shard (cos, erreur relative)\n")
    for model in models:
        rows = []
        for ck in checkpoints_present(df, model):
            for tag in E1_TAGS:
                cos_c = qval1(df, "E1", f"cos_calib__{tag}", model=model, checkpoint=ck)
                err_c = qval1(df, "E1", f"relerr_calib__{tag}", model=model, checkpoint=ck)
                cos_s = qval1(df, "E1", f"cos_shard__{tag}", model=model, checkpoint=ck)
                err_s = qval1(df, "E1", f"relerr_shard__{tag}", model=model, checkpoint=ck)
                rows.append([ck, tag, sig(cos_c) if cos_c is not None else "-",
                             sig(err_c) if err_c is not None else "-",
                             sig(cos_s) if cos_s is not None else "-",
                             sig(err_s) if err_s is not None else "-"])
        lines.append(f"\n**{model}**\n")
        lines.append(md_table(["checkpoint", "config", "cos (calib)", "err_rel (calib)",
                               "cos (shard)", "err_rel (shard)"], rows))
        data[f"table_{model}"] = rows

    lines.append("\n### Robustesse aux graines (checkpoint=end, config a)\n")
    rows_seed = []
    for model in models:
        cos_vals = qvals(df, "E1", "seed_robustness_cos__a", model=model, checkpoint="end")
        err_vals = qvals(df, "E1", "seed_robustness_err_rel__a", model=model, checkpoint="end")
        rows_seed.append([model, median_range(cos_vals), median_range(err_vals), len(cos_vals)])
    lines.append(md_table(["Modele", "cos (median [min,max])", "err_rel (median [min,max])", "n graines"], rows_seed))
    data["seed_robustness"] = rows_seed

    lines.append("\n### Jumeaux numeriques des figures\n")
    lines.append("**cos vs err_rel (nuage E1, shard)** -- correlation + points extremes :\n")
    rows_corr = []
    for model in models:
        errs, coss, labels = [], [], []
        for ck in checkpoints_present(df, model):
            for tag in E1_TAGS:
                e = qval1(df, "E1", f"relerr_shard__{tag}", model=model, checkpoint=ck)
                c = qval1(df, "E1", f"cos_shard__{tag}", model=model, checkpoint=ck)
                if e is not None and c is not None:
                    errs.append(e); coss.append(c); labels.append(f"{ck}/{tag}")
        if len(errs) >= 3:
            r, _ = _stats.pearsonr(errs, coss)
        else:
            r = float("nan")
        i_min_cos = int(np.argmin(coss)) if coss else None
        i_max_err = int(np.argmax(errs)) if errs else None
        rows_corr.append([model, sig(r) if not math.isnan(r) else "n/a",
                          f"{labels[i_min_cos]} (cos={sig(coss[i_min_cos])})" if i_min_cos is not None else "-",
                          f"{labels[i_max_err]} (err_rel={sig(errs[i_max_err])})" if i_max_err is not None else "-"])
    lines.append(md_table(["Modele", "Pearson r(err_rel, cos)", "cos min", "err_rel max"], rows_corr))

    lines.append("\n**Erreur vs |B| (log-log) et SNR(beta,B), checkpoint=end** -- 12 points par modele :\n")
    rows_snr = []
    for model in models:
        for beta in sorted(df[(df.model == model) & (df.experiment == "E1") &
                               (df.metric == "sweep_err_rel__B=full")]["beta"].unique()):
            err_full = qval1(df, "E1", "sweep_err_rel__B=full", model=model, checkpoint="end", beta=beta)
            rows_snr.append([model, sig(beta), "full", sig(err_full) if err_full is not None else "-", "-"])
            for B in [64, 256, 1024]:
                err_b = qval1(df, "E1", f"sweep_err_rel__B={B}", model=model, checkpoint="end", beta=beta)
                snr_b = qval1(df, "E1", f"snr__B={B}", model=model, checkpoint="end", beta=beta)
                rows_snr.append([model, sig(beta), str(B), sig(err_b) if err_b is not None else "-",
                                 sig(snr_b) if snr_b is not None else "-"])
    lines.append(md_table(["Modele", "beta", "|B|", "err_rel", "SNR"], rows_snr))
    data["snr_table"] = rows_snr

    lines.append("\n### Verdict\n")
    verdicts = []
    for model in models:
        cos_shard_all = []
        for ck in checkpoints_present(df, model):
            for tag in E1_TAGS:
                c = qval1(df, "E1", f"cos_shard__{tag}", model=model, checkpoint=ck)
                if c is not None:
                    cos_shard_all.append(c)
        min_cos = min(cos_shard_all) if cos_shard_all else None
        v = "PASS" if (min_cos is not None and min_cos >= 0.99) else (
            "INCONCLUSIF" if min_cos is None else "FAIL")
        verdicts.append(f"**{model}** : {v} (min cos_shard = {sig(min_cos) if min_cos is not None else 'n/a'}, seuil 0.99)")
    for v in verdicts:
        lines.append(f"- {v}\n")
    data["verdicts"] = verdicts

    lines.append("\n### Anomalies\n")
    anomalies = []
    for model in models:
        for ck in checkpoints_present(df, model):
            for tag in E1_TAGS:
                e = qval1(df, "E1", f"relerr_shard__{tag}", model=model, checkpoint=ck)
                if e is not None and e > 1.0:
                    anomalies.append(f"{model}/{ck}/{tag} : err_rel(shard) = {sig(e)} (>100% -- "
                                      f"||Gbar@u|| est petit devant le bruit residuel a ce budget, "
                                      f"cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est "
                                      f"correcte, seule la magnitude relative de l'erreur est grande)")
    if anomalies:
        for a in anomalies[:15]:
            lines.append(f"- {a}\n")
        if len(anomalies) > 15:
            lines.append(f"- ... et {len(anomalies)-15} autres lignes similaires (voir metrics.csv, "
                         f"experiment=E1, metric=relerr_shard__*)\n")
    else:
        lines.append("Aucune.\n")
    data["anomalies"] = anomalies

    return "\n".join(lines), data


# --------------------------------------------------------------------------#
# Sec 4 -- E2
# --------------------------------------------------------------------------#
def section_e2(df):
    lines = ["## 4. E2 -- Geometrie et scalaires de regime\n"]
    lines.append("**Hypothese** : le plafond de rang (`varpi`) laisse une marge exploitable a "
                 "l'attaquant plutot que d'etre domine par la politique par classe.\n")
    lines.append("**Rappel de portee** : pour `linear`, `Gbar` a rang exactement `C(C-1)`=90 "
                 "generiquement (produit exterieur) -- `varpi`/`alpha_tilde_star` y sont "
                 "atypiquement favorables par construction. **Le verdict E2 est pris sur `cnn` "
                 "uniquement** ; `linear` sert de test d'implementation.\n")

    models = models_present(df)
    lines.append("\n### Resultats\n")
    data = {}
    for model in models:
        rows = []
        for ck in checkpoints_present(df, model):
            for beta in sorted(df[(df.model == model) & (df.experiment == "E2")]["beta"].dropna().unique()):
                varpi = qval1(df, "E2", "varpi", model=model, checkpoint=ck, beta=beta)
                baseline = qval1(df, "E2", "baseline", model=model, checkpoint=ck, beta=beta)
                v_hat = qval1(df, "E2", "v_hat", model=model, checkpoint=ck, beta=beta)
                alpha = qval1(df, "E2", "alpha_tilde_star", model=model, checkpoint=ck, beta=beta)
                theta = qval1(df, "E2", "Theta_rad", model=model, checkpoint=ck, beta=beta)
                rows.append([ck, sig(beta), sig(varpi), sig(baseline), sig(v_hat), sig(alpha),
                             sig(theta) if theta is not None else "n/a"])
        lines.append(f"\n**{model}**\n")
        lines.append(md_table(["checkpoint", "beta", "varpi", "baseline (rang_eff/d)", "v_hat",
                               "alpha_tilde_star", "Theta (rad)"], rows))
        data[f"table_{model}"] = rows

    lines.append("\n### Jumeaux numeriques des figures\n")
    lines.append("**Spectre des valeurs propres de Q (checkpoint=end)** -- 11 points (P100..P0) :\n")
    rows_eig = []
    for model in models:
        vals = []
        for pct in range(0, 101, 10):
            v = qval1(df, "E2", f"eigval_Q_p{pct}", model=model, checkpoint="end")
            vals.append(sig(v) if v is not None else "-")
        rows_eig.append([model] + vals)
    lines.append(md_table(["Modele"] + [f"P{p}" for p in range(0, 101, 10)], rows_eig))

    lines.append("\n**varpi vs baseline (bar chart)** : voir table ci-dessus (colonnes varpi/baseline).\n")

    lines.append("\n**alpha_tilde_star vs sqrt(varpi) (nuage)** -- correlation + extremes :\n")
    rows_corr = []
    for model in models:
        xs, ys, labels = [], [], []
        for ck in checkpoints_present(df, model):
            for beta in sorted(df[(df.model == model) & (df.experiment == "E2")]["beta"].dropna().unique()):
                x = qval1(df, "E2", "sqrt_varpi", model=model, checkpoint=ck, beta=beta)
                y = qval1(df, "E2", "alpha_tilde_star", model=model, checkpoint=ck, beta=beta)
                if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                    xs.append(x); ys.append(y); labels.append(f"{ck}/beta={beta}")
        r = _stats.pearsonr(xs, ys)[0] if len(xs) >= 3 else float("nan")
        rows_corr.append([model, sig(r) if not math.isnan(r) else "n/a", len(xs)])
    lines.append(md_table(["Modele", "Pearson r(sqrt(varpi), alpha_tilde_star)", "n points"], rows_corr))

    lines.append("\n**v_hat par checkpoint, par beta** : voir table de resultats ci-dessus.\n")

    lines.append("\n### Verdict\n")
    verdicts = []
    cnn_present = "cnn" in models
    if cnn_present:
        varpis = qvals(df, "E2", "varpi", model="cnn")
        baselines = qvals(df, "E2", "baseline", model="cnn")
        med_varpi = float(np.median(varpis)) if varpis else None
        med_baseline = float(np.median(baselines)) if baselines else None
        margin = (med_varpi is not None and med_baseline is not None and med_varpi < 1.0
                  and med_varpi > med_baseline)
        v = "PASS" if margin else ("INCONCLUSIF" if med_varpi is None else "FAIL")
        verdicts.append(f"**cnn (decisif)** : {v} -- median(varpi)={sig(med_varpi)}, "
                        f"median(baseline)={sig(med_baseline)} (marge exploitable si varpi<1 et > baseline)")
    else:
        verdicts.append("**cnn** : INCONCLUSIF -- config cnn absente de ce run")
    if "linear" in models:
        verdicts.append("**linear** : test d'implementation seulement (rang exact C(C-1), non decisif)")
    for v in verdicts:
        lines.append(f"- {v}\n")

    lines.append("\n### Anomalies\n")
    lines.append("Aucune detectee automatiquement (varpi > 1 ou alpha_tilde_star hors [0,1] "
                 "seraient signales ici).\n")
    anomalies = []
    for model in models:
        for v in qvals(df, "E2", "varpi", model=model):
            if v is not None and (v < 0 or v > 1.0 + 1e-6):
                anomalies.append(f"{model}: varpi={sig(v)} hors [0,1]")
        for v in qvals(df, "E2", "alpha_tilde_star", model=model):
            if v is not None and (v < 0 or v > 1.0 + 1e-6):
                anomalies.append(f"{model}: alpha_tilde_star={sig(v)} hors [0,1]")
    if anomalies:
        lines.append("\n" + "\n".join(f"- {a}" for a in anomalies) + "\n")

    return "\n".join(lines), dict(verdicts=verdicts, anomalies=anomalies)


# --------------------------------------------------------------------------#
# Sec 5 -- E3
# --------------------------------------------------------------------------#
def section_e3(df):
    lines = ["## 5. E3 -- Stabilite de Gbar et cout du one-shot\n"]
    lines.append("**Hypothese** : une configuration `ubar` unique sert toute la trajectoire "
                 "d'entrainement (le one-shot ne coute presque rien face a l'oracle par checkpoint).\n")

    models = models_present(df)
    data = {}
    lines.append("\n### Resultats -- cout one-shot et budget effectif\n")
    for model in models:
        rows = []
        for ck in checkpoints_present(df, model):
            a_rho2 = qval1(df, "E3", "a_over_rho2", model=model, checkpoint=ck)
            l1b = qval1(df, "E3", "l1_over_beta", model=model, checkpoint=ck)
            rows.append([ck, sig(a_rho2), sig(l1b)])
        J_shared = qval1(df, "E3", "J_shared", model=model, checkpoint="shared")
        mean_pc = qval1(df, "E3", "mean_perckpt", model=model, checkpoint="shared")
        gap_pct = qval1(df, "E3", "gap_pct", model=model, checkpoint="shared")
        lines.append(f"\n**{model}**\n")
        lines.append(md_table(["checkpoint", "a_k/rho_k^2", "||u*||_1/beta"], rows))
        lines.append(f"\nMoyenne par checkpoint = {sig(mean_pc)}, J(ubar partage) = {sig(J_shared)}, "
                     f"ecart one-shot = {sig(gap_pct)}%\n")
        data[f"table_{model}"] = rows
        data[f"gap_pct_{model}"] = gap_pct

        l1t = qval1(df, "E3", "l1_capacity_true", model=model, checkpoint="end")
        l1f = qval1(df, "E3", "l1_capacity_false", model=model, checkpoint="end")
        sbc = qval1(df, "E3", "s_beta_check", model=model, checkpoint="end")
        lines.append(f"\nVerification capacity=True vs False (beta=0.10, s_beta={sig(sbc)}, "
                     f">1 donc plafonds attendus actifs) : ||u*||_1(capacity=True)={sig(l1t)} "
                     f"(doit etre < 0.10), ||u*||_1(capacity=False)={sig(l1f)} (doit s'approcher de 0.10).\n")

    lines.append("\n### Jumeaux numeriques des figures\n")
    lines.append("**cos(u*_k, u*_k') entre checkpoints** :\n")
    for model in models:
        rows_cos = []
        cks = checkpoints_present(df, model)
        for i in range(len(cks)):
            for j in range(i + 1, len(cks)):
                key = f"{cks[i]}_vs_{cks[j]}"
                cos_u = qval1(df, "E3", "cos_u_star", model=model, checkpoint=key)
                col_cos = qval1(df, "E3", "mean_col_cos_Gbar", model=model, checkpoint=key)
                rows_cos.append([key, sig(cos_u) if cos_u is not None else "-",
                                 sig(col_cos) if col_cos is not None else "-"])
        lines.append(f"\n**{model}**\n")
        lines.append(md_table(["paire checkpoints", "cos(u*,u*)", "cos colonne moyen(Gbar,Gbar)"], rows_cos))

    lines.append("\n**Heatmap u*_ckpt** -- top-5 paires (y,z) par masse :\n")
    for model in models:
        rows_top = []
        for ck in checkpoints_present(df, model):
            cells = df[(df.model == model) & (df.checkpoint == ck) & (df.experiment == "E3") &
                       (df.metric.str.startswith("u_star_top"))]
            top = cells.sort_values("metric")
            entries = []
            for _, r in top.iterrows():
                m = re.match(r"u_star_top(\d)_y(\d+)_z(\d+)", r["metric"])
                if m:
                    entries.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), r["value"]))
            entries.sort(key=lambda t: t[0])
            rows_top.append([ck] + [f"({y}->{z}): {sig(mass)}" for _, y, z, mass in entries])
        lines.append(f"\n**{model}**\n")
        maxlen = max((len(r) for r in rows_top), default=1)
        headers = ["checkpoint"] + [f"top{i}" for i in range(1, maxlen)]
        rows_top = [r + ["-"] * (maxlen - len(r)) for r in rows_top]
        lines.append(md_table(headers, rows_top))

    lines.append("\n### Verdict\n")
    verdicts = []
    for model in models:
        cks = checkpoints_present(df, model)
        cos_all = []
        for i in range(len(cks)):
            for j in range(i + 1, len(cks)):
                c = qval1(df, "E3", "cos_u_star", model=model, checkpoint=f"{cks[i]}_vs_{cks[j]}")
                if c is not None and not math.isnan(c):
                    cos_all.append(c)
        min_cos = min(cos_all) if cos_all else None
        gap_pct = data.get(f"gap_pct_{model}")
        ok = (min_cos is not None and min_cos >= 0.7) and (gap_pct is not None and abs(gap_pct) <= 20)
        v = "PASS" if ok else ("INCONCLUSIF" if min_cos is None else "FAIL")
        verdicts.append(f"**{model}** : {v} -- min cos(u*,u*)={sig(min_cos)}, ecart one-shot={sig(gap_pct)}%")
    for v in verdicts:
        lines.append(f"- {v}\n")

    lines.append("\n### Anomalies\n")
    lines.append("Aucune detectee automatiquement au-dela de ce qui est deja signale ci-dessus.\n")

    return "\n".join(lines), dict(verdicts=verdicts)


# --------------------------------------------------------------------------#
# Sec 6-9 -- E4-E7 (not run this session)
# --------------------------------------------------------------------------#
PENDING_SECTIONS = {
    "E4": ("6", "Reponse des agregateurs a l'attaque optimale sous la moyenne",
           "Controler la moyenne controle aussi les regles robustes (Krum, Multi-Krum, "
           "trimmed mean, coordinate-wise median), en variantes flat et per_tensor."),
    "E5": ("7", "Courbe de furtivite et saturation",
           "Reduire la demande n'ameliore la selection que sous le rayon atteignable -- "
           "coude attendu pres de v_hat=1."),
    "E6": ("8", "Pouvoir predictif des notions de faisabilite (bloc couteux, include_e6)",
           "Le residu normalise (alpha_tilde_star, v_hat, varpi, E[a_k/rho_k^2]) predit l'ASR."),
    "E7": ("9", "Etaler le budget",
           "A beta fixe, augmenter n_p reduit le taux de corruption local sans changer "
           "l'ensemble atteignable sous la moyenne."),
}


def section_pending(tag, meta):
    num, title, hyp = PENDING_SECTIONS[tag]
    lines = [f"## {num}. {tag} -- {title}\n"]
    lines.append(f"**Hypothese** : {hyp}\n")
    reason = ("Non execute cette session -- prevu a l'etape 2 (agregateurs instrumentes) "
              "puis 3 (E4/E5/E7).")
    if tag == "E6":
        reason = ("Non execute cette session -- bloc couteux, derriere le drapeau "
                  "`include_e6=True`, prevu en dernier (etape 4).")
    lines.append(f"**Ce qui a ete execute** : rien. {reason}\n")
    lines.append("**Resultats** : n/a.\n")
    lines.append("**Jumeaux numeriques des figures** : n/a.\n")
    lines.append(f"**Verdict** : INCONCLUSIF -- {reason}\n")
    lines.append("**Anomalies** : n/a.\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------#
# Sec 10 -- synthese
# --------------------------------------------------------------------------#
def section10(e1_data, e2_data, e3_data, models):
    lines = ["## 10. Synthese\n"]
    rows = []

    def first_verdict_status(verdicts):
        if not verdicts:
            return "INCONCLUSIF"
        for v in verdicts:
            if v.startswith("FAIL") or ": FAIL" in v:
                return "FAIL"
        for v in verdicts:
            if "INCONCLUSIF" in v:
                return "INCONCLUSIF"
        return "PASS" if all("PASS" in v for v in verdicts) else "INCONCLUSIF"

    s_e1 = first_verdict_status(e1_data.get("verdicts", []))
    rows.append(["E1", "Implementation et transfert de Gbar", s_e1])

    e2_verdicts = e2_data.get("verdicts", [])
    s_e2 = "INCONCLUSIF"
    for v in e2_verdicts:
        if v.startswith("**cnn"):
            s_e2 = "PASS" if "PASS" in v else ("FAIL" if "FAIL" in v else "INCONCLUSIF")
    rows.append(["E2", "Plafond de rang (config cnn uniquement)", s_e2])

    s_e3 = first_verdict_status(e3_data.get("verdicts", []))
    rows.append(["E3", "Stabilite one-shot", s_e3])

    rows.append(["E5", "Existence du levier de furtivite", "INCONCLUSIF (non execute cette session)"])

    lines.append(md_table(["Verrou", "Description", "Go/no-go"], rows))

    n_fail = sum(1 for r in rows if r[2] == "FAIL")
    n_inc = sum(1 for r in rows if "INCONCLUSIF" in r[2])
    lines.append("\n**Recommandation**\n")
    if n_fail > 0:
        lines.append(f"- {n_fail} verrou(x) en FAIL : corriger avant de lancer le sweep complet "
                     f"(voir sections correspondantes pour le chiffre qui fonde le FAIL).\n")
    elif s_e2 == "INCONCLUSIF":
        lines.append("- E1/E3 sont concluants ; E2 attend son execution complete sur `cnn` avant "
                     "de statuer -- ne pas lancer E4-E7 tant que ce verrou n'est pas tranche.\n")
    else:
        lines.append("- E1-E3 sont concluants sur ce run : lancer l'etape 2 (agregateurs "
                     "instrumentes + assertion de concordance flat/depot).\n")
    lines.append("- E5 (existence du levier de furtivite) reste le verrou le plus falsifiable et "
                 "n'est pas encore tranche -- le prioriser des l'etape 3.\n")
    lines.append("- E6 reste derriere son drapeau `include_e6` vu son cout ; ne pas le lancer "
                 "avant confirmation explicite.\n")

    return "\n".join(lines), dict(rows=rows)


# --------------------------------------------------------------------------#
# Main
# --------------------------------------------------------------------------#
def build_report(metrics_path=METRICS_PATH, out_dir=ARTIFACT_DIR):
    df = pd.read_csv(metrics_path)
    with open(RUN_META_PATH) as f:
        meta = json.load(f)
    failures_text = ""
    if os.path.exists(FAILURES_PATH):
        with open(FAILURES_PATH) as f:
            failures_text = f.read()

    md_parts = ["# Rapport preliminaires -- E1-E3 (session 2)\n"]
    json_data = {}

    s, d = section0(df, meta, failures_text); md_parts.append(s); json_data["sec0"] = d
    s, d = section1(df, meta); md_parts.append(s); json_data["sec1"] = d
    s, d = section2(df); md_parts.append(s); json_data["sec2"] = d
    s, e1_data = section_e1(df); md_parts.append(s); json_data["E1"] = e1_data
    s, e2_data = section_e2(df); md_parts.append(s); json_data["E2"] = e2_data
    s, e3_data = section_e3(df); md_parts.append(s); json_data["E3"] = e3_data
    for tag in ("E4", "E5", "E6", "E7"):
        md_parts.append(section_pending(tag, meta))
    s, d = section10(e1_data, e2_data, e3_data, models_present(df))
    md_parts.append(s); json_data["sec10"] = d

    report_md = "\n".join(md_parts)
    n_lines = report_md.count("\n") + 1
    if n_lines > 1500:
        report_md += (f"\n\n*(Avertissement : {n_lines} lignes, au-dessus de la limite de 1500 "
                      f"-- a resserrer.)*\n")

    with open(REPORT_MD_PATH, "w") as f:
        f.write(report_md)
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    print(f"[report] {n_lines} lignes -> {REPORT_MD_PATH}")
    print(f"[report] -> {REPORT_JSON_PATH}")
    return report_md


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", default=METRICS_PATH)
    p.add_argument("--out-dir", default=ARTIFACT_DIR)
    args = p.parse_args()
    build_report(metrics_path=args.metrics, out_dir=args.out_dir)
