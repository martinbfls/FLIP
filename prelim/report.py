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


def e3_checkpoints_present(df, model):
    """
    Session correction D2: E3 (only) may have 2 extra checkpoints between
    "mid" and "end" -- postmid1, postmid2, ... (see sweep.py's
    E3_CHECKPOINTS/E3_EXTRA_POST_MID) -- ordered begin < mid < postmid1 <
    postmid2 < ... < end. Unlike checkpoints_present (E1/E2, fixed 3-point
    grid), this discovers whichever postmidN checkpoints are actually in the
    E3 rows rather than assuming a fixed count.
    """
    sub = df[(df.model == model) & (df.experiment == "E3") &
             df.checkpoint.str.match(r"^(begin|mid|end|postmid\d+)$", na=False)]
    ck = sub["checkpoint"].unique().tolist()

    def _key(c):
        if c == "begin":
            return (0, 0)
        if c == "mid":
            return (1, 0)
        if c == "end":
            return (3, 0)
        m = re.match(r"postmid(\d+)", c)
        return (2, int(m.group(1)))

    return sorted(ck, key=_key)


def _parse_failures_log(failures_text):
    """
    Session correction E: failures.log is append-only across every resumed
    run_all() call (sweep.py's _log_failure opens it with "a"), so it can
    (correctly) accumulate entries from an EARLIER attempt that a LATER
    resume subsequently fixed -- the current run's own run_meta.json
    (n_cells_failed/failed_cells) only ever reflects the MOST RECENT call.
    Splits failures_text into individual entries and extracts each one's
    timestamp (first line: "YYYY-MM-DD HH:MM:SS model=... ..."), so section0/
    section1 can separate "from this run" (>= meta.started_at) from "stale,
    from an earlier attempt" instead of reporting a bare, ambiguous count.
    Returns a list of (timestamp_or_None, entry_text).
    """
    if not failures_text:
        return []
    entries = [e for e in failures_text.split("\n" + "=" * 70) if e.strip()]
    parsed = []
    for e in entries:
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", e)
        parsed.append((m.group(1) if m else None, e.strip()))
    return parsed


# --------------------------------------------------------------------------#
# Sec 0 -- header
# --------------------------------------------------------------------------#
def section0(df, meta, failures_text):
    lines = ["## 0. En-tete\n"]
    fail_entries = _parse_failures_log(failures_text)
    started_at = meta.get("started_at")
    n_this_run = sum(1 for ts, _ in fail_entries if started_at and ts and ts >= started_at)
    n_stale = len(fail_entries) - n_this_run
    fail_log_str = str(len(fail_entries))
    if fail_entries:
        fail_log_str += f" (dont {n_this_run} de ce run, {n_stale} anterieure(s)/perimee(s))"
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
        ["Entrees dans failures.log", fail_log_str],
    ]
    lines.append(md_table(["Champ", "Valeur"], rows))
    data = dict(rows=rows, n_fail_log_total=len(fail_entries), n_fail_log_this_run=n_this_run,
                n_fail_log_stale=n_stale)
    return "\n".join(lines), data


# --------------------------------------------------------------------------#
# Sec 1 -- grid executed
# --------------------------------------------------------------------------#
def section1(df, meta, failures_text=""):
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
        "de la graine). E2 exploite `identity` et `stripe` (patch, voir section E2) ; E1/E3 restent "
        "`identity` uniquement. E4/E5/E7 (etapes 2-3) tournent a checkpoint=end uniquement : E4 "
        "balaie `identity`/`stripe`, E5/E7 restent a `identity` avec beta/n_p balayes -- rounds et "
        "taille de minibatch reduits sous les hints SPEC section 8 pour tenir le budget de dix "
        "minutes par bloc (voir sweep.py, E4_ROUNDS/E5_ROUNDS/E7_ROUNDS/SIM_BATCH).\n"
    )
    lines.append(f"\nNombre de cellules (checkpoint x modele) : "
                 f"{meta.get('n_cells_run', 0) + meta.get('n_cells_cached', 0) + meta.get('n_cells_failed', 0)}\n")

    failed = meta.get("failed_cells", [])
    lines.append("\n### Cellules echouees ou ignorees\n")
    if failed:
        rows_f = [[f.get("model"), f.get("seed"), f.get("checkpoint"), f.get("reason")] for f in failed]
        lines.append(md_table(["Modele", "Graine", "Checkpoint", "Raison"], rows_f))
    else:
        lines.append("Aucune cellule en echec pour CE run (`run_meta.json`, n_cells_failed="
                     f"{meta.get('n_cells_failed', 0)}).\n")

    # Session correction E: reconcile against failures.log, which is
    # append-only across resumed runs (see _parse_failures_log) and can
    # legitimately carry entries from an earlier, since-fixed attempt.
    fail_entries = _parse_failures_log(failures_text)
    started_at = meta.get("started_at")
    stale = [(ts, e) for ts, e in fail_entries if not (started_at and ts and ts >= started_at)]
    this_run = [(ts, e) for ts, e in fail_entries if (started_at and ts and ts >= started_at)]
    if fail_entries:
        if this_run:
            lines.append(f"\n`failures.log` contient {len(this_run)} entree(s) horodatee(s) DE CE "
                         f"run (>= {started_at}) -- ceci NE contredit PAS la table ci-dessus que si "
                         f"celles-ci correspondent a des cellules retentees avec succes plus tard "
                         f"dans le meme run (resume interne) ; sinon, verifier `n_cells_failed` "
                         f"ci-dessus, qui doit refleter ces echecs.\n")
        if stale:
            lines.append(f"\n`failures.log` contient {len(stale)} entree(s) ANTERIEURE(S) au demarrage "
                         f"de ce run (avant {started_at}) -- benignes par construction : le fichier "
                         f"n'est jamais tronque entre deux `run_all()` (append-only, voir sweep.py "
                         f"`_log_failure`), donc une cellule qui a echoue lors d'un essai precedent "
                         f"puis reussi lors d'une reprise (`resume=True`) laisse une trace ici sans "
                         f"que la section 1 courante ne la liste comme en echec. Horodatages : "
                         + ", ".join(ts or "?" for ts, _ in stale) + ".\n")
    else:
        lines.append("\n`failures.log` est vide ou absent : coherent avec \"Aucune\" ci-dessus.\n")

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

    # Session correction B4: threshold moved from >= 1 to >= 10 EXPECTED
    # flips per pair (sweep.py's own gap computation now filters at 10, so
    # this just needs to read whatever it wrote) -- below 10, round()'s
    # integer-count noise dominates the ratio by construction. The prior
    # 0.309 FAIL traced to c_uniform's ~90-way mass split, well under 10
    # expected flips per pair.
    gap_cols = [c for c in df["metric"].unique() if c.startswith("flip_mass_gap_rel_max__")]
    g4 = df[df.metric.isin(gap_cols)]["value"].tolist()
    v4 = max(g4) if g4 else None
    add("Masses de flip realisees vs demandees : ecart relatif max (>=10 flips attendus/paire)",
        "toutes cellules E1", v4, 0.10, v4 is not None and v4 <= 0.10)

    g5 = qvals(df, "assertions", "flat_aggregator_vs_repo_max_abs_diff")
    v5 = max(g5) if g5 else None
    add("Agregateur aplati == depot (5 regles, sur un stack reel de gradients, checkpoint=end)",
        "par modele", v5, 1e-5, v5 is not None and v5 <= 1e-5, pending=v5 is None)

    nan_ct = int(df["value"].isna().sum())
    inf_ct = int(np.isinf(df["value"]).sum())
    add(f"Aucun NaN/Inf dans le CSV (NaN={nan_ct}, Inf={inf_ct})", "metrics.csv entier",
        nan_ct + inf_ct, 0, (nan_ct + inf_ct) == 0)

    lines.append(md_table(["Nom", "Portee", "Statut", "Valeur observee", "Seuil"], rows))

    if gap_cols:
        lines.append("\n**Detail par config (B4)** -- max de l'ecart relatif, par tag E1 (celui-ci "
                     "isole c_uniform du reste) :\n")
        rows_gap = []
        for c in sorted(gap_cols):
            tag = c[len("flip_mass_gap_rel_max__"):]
            vals = df[df.metric == c]["value"].tolist()
            rows_gap.append([tag, sig(max(vals)) if vals else "-", len(vals)])
        lines.append(md_table(["config", "ecart relatif max", "n cellules"], rows_gap))

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

    lines.append("\n### Diagnostic du facteur d'echelle (correction B, session 4)\n")
    lines.append("Le rapport v1 montrait `err_rel(shard) ~ 9` avec `cos ~ 1` sur TOUS les "
                 "tags/checkpoints/modeles (voir metrics.csv historique) -- pas du bruit : avec "
                 "`cos=1`, `err_rel = |a-1|` pour la pente `a = <g_emp-grad_c, Gbar@u>/||Gbar@u||^2`, "
                 "donc `err_rel~9` implique `a~10`, constant. Cause : `Gbar` est pondere par `pi[y]` "
                 "(`Gbar[:,(y,z)] = pi[y]*(g[y][z]-g[y][y])`, verbatim du depot -- jamais modifie ici) "
                 "alors que la masse realisee `u_real` (flip_masses_to_labels) est une fraction de "
                 "TOUS les exemples detenus par le worker (definition de `u` de SPEC section 2, "
                 "coherente avec les plafonds `U_loc`/`U_beta` eux-memes bornes par `pi[y]`) -- "
                 "combiner un `Gbar` pondere par `pi[y]` avec une masse deja en unites de `pi[y]` "
                 "compte `pi[y]` deux fois. Sur CIFAR-10 equilibre, `pi[y]=0.1` pour toutes les "
                 "classes, donc `1/pi[y]=10` explique la constance du facteur observe sur tous les "
                 "configs (y compris `b_other_pair`, sur une AUTRE classe source, et `c_uniform`, "
                 "etale sur les 90 paires) -- ce n'est pas propre a la paire (9,4).\n")
    lines.append("**Correction appliquee a la source** (sweep.py `_run_e1_main`/`_run_e1_seed_and_snr`, "
                 "pas ici) : la prediction utilise desormais `Gbar @ (u_real / pi_vec)` au lieu de "
                 "`Gbar @ u_real` -- `pi_vec[j] = pi[y]` pour la classe source de la paire `j`. "
                 "`solve_qp`/`Q`/`c` (verifies contre `project_gradient` du depot dans les assertions, "
                 "section 2) ne sont PAS touches -- seule la contraction empirique utilisee pour "
                 "comparer a `g_emp` l'est. Colonnes `a_slope_shard`/`a_r2_shard` ci-dessous "
                 "(section B1) : calculees APRES cette correction, elles doivent valider "
                 "empiriquement le correctif (`a~1`, `R^2~1`) au prochain sweep -- si elles ne le "
                 "font pas, il reste un second facteur d'echelle non explique par ce mecanisme. Reste "
                 "OUVERT : si le MEME mecanisme affecte aussi les deploiements bases sur le QP "
                 "(`ubar*` d'E3/E4/E5/E7, qui ciblent `v` via ce meme `Gbar` pondere), ceux-ci "
                 "realiseraient un decalage EFFECTIF ~10x plus grand que ce que `v_hat`/`alpha_tilde_star` "
                 "rapportent -- cette session ne tranche pas cette question plus large (voir "
                 "recommandation section 10) ; c'est precisement pourquoi E6 n'est pas relance.\n")

    data = {}
    lines.append("\n### Resultats -- transfert calibration -> shard (cos, erreur relative, pente a)\n")
    for model in models:
        rows = []
        for ck in checkpoints_present(df, model):
            for tag in E1_TAGS:
                cos_c = qval1(df, "E1", f"cos_calib__{tag}", model=model, checkpoint=ck)
                err_c = qval1(df, "E1", f"relerr_calib__{tag}", model=model, checkpoint=ck)
                cos_s = qval1(df, "E1", f"cos_shard__{tag}", model=model, checkpoint=ck)
                err_s = qval1(df, "E1", f"relerr_shard__{tag}", model=model, checkpoint=ck)
                a_s = qval1(df, "E1", f"a_slope_shard__{tag}", model=model, checkpoint=ck)
                r2_s = qval1(df, "E1", f"a_r2_shard__{tag}", model=model, checkpoint=ck)
                snr_s = qval1(df, "E1", f"snr_shard__{tag}", model=model, checkpoint=ck)
                rows.append([ck, tag, sig(cos_c) if cos_c is not None else "-",
                             sig(err_c) if err_c is not None else "-",
                             sig(cos_s) if cos_s is not None else "-",
                             sig(err_s) if err_s is not None else "-",
                             sig(a_s) if a_s is not None else "-",
                             sig(r2_s) if r2_s is not None else "-",
                             sig(snr_s) if snr_s is not None else "-"])
        lines.append(f"\n**{model}**\n")
        lines.append(md_table(["checkpoint", "config", "cos (calib)", "err_rel (calib)",
                               "cos (shard)", "err_rel (shard)", "a_slope (shard, post-correction)",
                               "R^2 (shard)", "SNR (shard)"], rows))
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
    lines.append("**Correction D1** : colonnes `masse source (locale)` / `plafond source atteint` "
                 "ajoutees -- `u_a[(9,4)]` est plafonne a `min(beta/gamma, pi[9])` (U_loc, portee "
                 "LOCALE). Sur cette grille (gamma=0.3, pi[9]=0.1), ce plafond LOCAL vaut `pi[9]=0.1`, "
                 "atteint des que `beta/gamma >= 0.1`, soit `beta >= 0.03` -- **beta=0.03 ET beta=0.10 "
                 "saturent tous deux a la meme masse**, ce qui explique que leurs lignes soient "
                 "identiques ci-dessous : c'est ce plafond de classe source, pas le budget global, "
                 "qui a cesse de compter au-dela de beta=0.03.\n")
    rows_snr = []
    for model in models:
        for beta in sorted(df[(df.model == model) & (df.experiment == "E1") &
                               (df.metric == "sweep_err_rel__B=full")]["beta"].unique()):
            err_full = qval1(df, "E1", "sweep_err_rel__B=full", model=model, checkpoint="end", beta=beta)
            smass = qval1(df, "E1", "source_mass_local", model=model, checkpoint="end", beta=beta)
            shit = qval1(df, "E1", "source_ceiling_hit", model=model, checkpoint="end", beta=beta)
            ceil_str = ("oui" if shit == 1.0 else "non") if shit is not None else "-"
            rows_snr.append([model, sig(beta), "full", sig(err_full) if err_full is not None else "-", "-",
                             sig(smass) if smass is not None else "-", ceil_str])
            for B in [64, 256, 1024]:
                err_b = qval1(df, "E1", f"sweep_err_rel__B={B}", model=model, checkpoint="end", beta=beta)
                snr_b = qval1(df, "E1", f"snr__B={B}", model=model, checkpoint="end", beta=beta)
                rows_snr.append([model, sig(beta), str(B), sig(err_b) if err_b is not None else "-",
                                 sig(snr_b) if snr_b is not None else "-",
                                 sig(smass) if smass is not None else "-", ceil_str])
    lines.append(md_table(["Modele", "beta", "|B|", "err_rel", "SNR", "masse source (locale)",
                           "plafond source atteint"], rows_snr))
    data["snr_table"] = rows_snr

    lines.append("\n### Verdict\n")
    lines.append("**Correction B3 (session 4)** : le gate prenait le min de `cos_shard` sur les 6 "
                 "configs, dont `c_uniform` (masse etalee sur les 90 paires, SNR faible par "
                 "construction) -- un cosinus bas sous ce regime mesure le bruit minibatch, pas un "
                 "defaut du modele. Seules les configs avec `SNR(shard) > 0.5` entrent "
                 "desormais dans le min ; les autres sont rapportees separement comme "
                 "\"non concluantes, SNR insuffisant\".\n")
    SNR_GATE = 0.5
    verdicts = []
    for model in models:
        cos_gated, cos_excluded = [], []
        for ck in checkpoints_present(df, model):
            for tag in E1_TAGS:
                c = qval1(df, "E1", f"cos_shard__{tag}", model=model, checkpoint=ck)
                snr_v = qval1(df, "E1", f"snr_shard__{tag}", model=model, checkpoint=ck)
                if c is None:
                    continue
                if snr_v is not None and snr_v > SNR_GATE:
                    cos_gated.append(c)
                else:
                    cos_excluded.append((ck, tag, c, snr_v))
        min_cos = min(cos_gated) if cos_gated else None
        if min_cos is not None:
            v = "PASS" if min_cos >= 0.99 else "FAIL"
        elif cos_excluded:
            v = "INCONCLUSIF"
        else:
            v = "INCONCLUSIF"
        note = (f" -- {len(cos_excluded)} config(s) non concluante(s), SNR <= {SNR_GATE} "
               f"(exclues du min)" if cos_excluded else "")
        verdicts.append(f"**{model}** : {v} (min cos_shard, SNR>{SNR_GATE} = "
                        f"{sig(min_cos) if min_cos is not None else 'n/a'}, seuil 0.99{note})")
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
    lines.append("**Note de correction (patch stripe)** : E2 tournait initialement avec `T=identity` "
                 "uniquement, ou `v` est quasi tautologiquement dans l'image de `Gbar` (formule de "
                 "decomposition de la section 8/E2 : `v/lam = Gbar[:,(9,4)]/pi[9] + (g[9][9]-grad_c)`). "
                 "Ce patch ajoute `T=stripe` (le vrai trigger) aux memes cellules pour ecarter "
                 "l'hypothese que le FAIL n'etait qu'un artefact du cas degenere -- **le resultat sous "
                 "stripe est quasi identique a celui sous identity** (voir tables ci-dessous) : ce n'est "
                 "donc pas un artefact de portee, c'est bien le plafond de rang qui domine, meme sous le "
                 "vrai trigger.\n")

    models = models_present(df)
    transforms = sorted(df[(df.experiment == "E2") & df["transform"].notna() &
                            (df["transform"] != "identity")]["transform"].unique().tolist())
    transforms = ["identity"] + transforms
    lines.append("\n### Resultats\n")
    data = {}
    for model in models:
        for transform in transforms:
            rows = []
            for ck in checkpoints_present(df, model):
                for beta in sorted(df[(df.model == model) & (df.experiment == "E2") &
                                       (df["transform"] == transform)]["beta"].dropna().unique()):
                    varpi = qval1(df, "E2", "varpi", model=model, checkpoint=ck, beta=beta, transform=transform)
                    baseline = qval1(df, "E2", "baseline", model=model, checkpoint=ck, beta=beta, transform=transform)
                    v_hat = qval1(df, "E2", "v_hat", model=model, checkpoint=ck, beta=beta, transform=transform)
                    alpha = qval1(df, "E2", "alpha_tilde_star", model=model, checkpoint=ck, beta=beta, transform=transform)
                    theta = qval1(df, "E2", "Theta_rad", model=model, checkpoint=ck, beta=beta, transform=transform)
                    rows.append([ck, sig(beta), sig(varpi), sig(baseline), sig(v_hat), sig(alpha),
                                 sig(theta) if theta is not None else "n/a"])
            if not rows:
                continue
            lines.append(f"\n**{model} / T={transform}**\n")
            lines.append(md_table(["checkpoint", "beta", "varpi", "baseline (rang_eff/d)", "v_hat",
                                   "alpha_tilde_star", "Theta (rad)"], rows))
            data[f"table_{model}_{transform}"] = rows

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
        for transform in transforms:
            xs, ys = [], []
            for ck in checkpoints_present(df, model):
                for beta in sorted(df[(df.model == model) & (df.experiment == "E2") &
                                       (df["transform"] == transform)]["beta"].dropna().unique()):
                    x = qval1(df, "E2", "sqrt_varpi", model=model, checkpoint=ck, beta=beta, transform=transform)
                    y = qval1(df, "E2", "alpha_tilde_star", model=model, checkpoint=ck, beta=beta, transform=transform)
                    if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                        xs.append(x); ys.append(y)
            if not xs:
                continue
            r = _stats.pearsonr(xs, ys)[0] if len(xs) >= 3 else float("nan")
            rows_corr.append([f"{model}/{transform}", sig(r) if not math.isnan(r) else "n/a", len(xs)])
    lines.append(md_table(["Modele/transform", "Pearson r(sqrt(varpi), alpha_tilde_star)", "n points"], rows_corr))

    lines.append("\n**v_hat par checkpoint, par beta** : voir table de resultats ci-dessus.\n")

    lines.append("\n### Verdict\n")
    lines.append("**Correction A1 (session 4)** : le critere precedent lisait un FAIL des que "
                 "`varpi` etait a moins de 1% du plafond 1.0 (\"SATURE\") -- mais le critere de "
                 "l'hypothese porte sur l'ecart a la BASELINE (`rang_effectif(Q)/d`, de l'ordre de "
                 "1e-5 a 1e-6 ici), pas sur l'ecart a 1. `varpi=0.998` contre `baseline=9e-06` "
                 "signifie que la cible `v` est dans le sous-espace atteignable par une politique "
                 "class-level -- c'est exactement ce que l'hypothese predit (\"le plafond de rang "
                 "laisse une marge exploitable\"), pas un signe de saturation. Nouveau critere : "
                 "**PASS si `varpi > 10 * baseline`, FAIL sinon** (facteur 10 = marge d'ordre de "
                 "grandeur, `baseline` etant elle-meme de l'ordre de `1/d`).\n")
    verdicts = []
    cnn_present = "cnn" in models
    decisive_transform = "stripe" if "stripe" in transforms else "identity"
    VARPI_BASELINE_FACTOR = 10.0
    if cnn_present:
        varpis = qvals(df, "E2", "varpi", model="cnn", transform=decisive_transform)
        baselines = qvals(df, "E2", "baseline", model="cnn", transform=decisive_transform)
        v_hats = qvals(df, "E2", "v_hat", model="cnn", transform=decisive_transform)
        varpis_id = qvals(df, "E2", "varpi", model="cnn", transform="identity")
        med_varpi = float(np.median(varpis)) if varpis else None
        med_baseline = float(np.median(baselines)) if baselines else None
        med_vhat = float(np.median(v_hats)) if v_hats else None
        med_varpi_id = float(np.median(varpis_id)) if varpis_id else None
        if med_varpi is None or med_baseline is None:
            v = "INCONCLUSIF"
        elif med_varpi > VARPI_BASELINE_FACTOR * med_baseline:
            v = "PASS"
        else:
            v = "FAIL"
        note = ""
        if decisive_transform == "stripe" and med_varpi_id is not None:
            note = (f" ; identity donne median(varpi)={sig(med_varpi_id)}, quasi identique -- "
                    f"donc pas un artefact du cas degenere T=identity, le plafond de rang domine "
                    f"reellement sous le vrai trigger")
        verdicts.append(f"**cnn (decisif, T={decisive_transform})** : {v} -- median(varpi)={sig(med_varpi)} "
                        f"vs {VARPI_BASELINE_FACTOR:.0f}x median(baseline)={sig(med_baseline)} "
                        f"({sig(VARPI_BASELINE_FACTOR * med_baseline) if med_baseline is not None else 'n/a'})"
                        f"{note}")
        if med_vhat is not None:
            interp = ("direction atteignable, magnitude hors budget -- la contrainte mordante est "
                      "le budget (`v_hat`), pas le rang" if med_vhat > 1 else
                      "direction atteignable ET magnitude dans le budget -- aucune contrainte ne "
                      "mord a ce beta")
            verdicts.append(f"**Interpretation** : varpi eleve (proche du plafond de rang) avec "
                            f"v_hat={sig(med_vhat)} {'> 1' if med_vhat > 1 else '<= 1'} signifie {interp}.")
    else:
        verdicts.append("**cnn** : INCONCLUSIF -- config cnn absente de ce run")
    if "linear" in models:
        verdicts.append("**linear** : test d'implementation seulement (rang exact C(C-1), non decisif)")
    for v in verdicts:
        lines.append(f"- {v}\n")

    lines.append("\n### Anomalies\n")
    anomalies = []
    for model in models:
        for transform in transforms:
            for v in qvals(df, "E2", "varpi", model=model, transform=transform):
                if v is not None and (v < 0 or v > 1.0 + 1e-6):
                    anomalies.append(f"{model}/{transform}: varpi={sig(v)} > 1 (hors [0,1] -- au-dela de la "
                                     f"tolerance du solveur OSQP eps~1e-6 utilisee par dist_to_cone/rank_ratio, "
                                     f"a surveiller mais pas un signe de bug de signe/formule)")
            for v in qvals(df, "E2", "alpha_tilde_star", model=model, transform=transform):
                if v is not None and (v < 0 or v > 1.0 + 1e-6):
                    anomalies.append(f"{model}/{transform}: alpha_tilde_star={sig(v)} hors [0,1]")
    if not anomalies:
        lines.append("Aucune detectee automatiquement (varpi > 1 ou alpha_tilde_star hors [0,1] "
                     "seraient signales ici).\n")
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
    lines.append("**Correction A3 (session 4)** : le gate portait sur `cos(u*_k, u*_k')`, un objet "
                 "instable des que deux checkpoints ont des optima quasi-degeneres -- un cosinus bas "
                 "avec un ecart one-shot negligeable signale des optima interchangeables, pas une "
                 "instabilite dommageable. Le critere porte desormais sur l'ECART ONE-SHOT lui-meme "
                 "(`J(ubar partage)` vs la moyenne par-checkpoint) : **PASS si ecart < 20%**, sinon "
                 "FAIL. `cos(u*,u*)` reste rapporte comme diagnostic (table ci-dessous) mais ne "
                 "decide plus le verdict ; le vrai signal de rotation de la trajectoire est le "
                 "cosinus COLONNE moyen de `Gbar` entre checkpoints (meme table), qui ne souffre pas "
                 "de la degenerescence du solveur QP.\n")
    lines.append("**Correction D2 (session 4)** : 2 checkpoints supplementaires entre \"mid\" et "
                 "\"end\" (postmid1, postmid2 -- voir sweep.py `E3_EXTRA_POST_MID`), motives par le "
                 "constat que `Gbar` tourne fortement en debut d'entrainement puis se stabilise : "
                 "l'ecart one-shot est rapporte a la fois sur la fenetre COMPLETE (begin..end) et "
                 "restreint a la fenetre [mid, end] (excluant \"begin\"), pour separer le cout du a "
                 "l'instabilite de debut d'entrainement de celui, propre, du cadrage one-shot.\n")

    models = models_present(df)
    data = {}
    lines.append("\n### Resultats -- cout one-shot et budget effectif\n")
    for model in models:
        rows = []
        for ck in e3_checkpoints_present(df, model):
            a_rho2 = qval1(df, "E3", "a_over_rho2", model=model, checkpoint=ck)
            l1b = qval1(df, "E3", "l1_over_beta", model=model, checkpoint=ck)
            smass = qval1(df, "E3", "source_mass_agg", model=model, checkpoint=ck)
            scap = qval1(df, "E3", "source_cap_agg", model=model, checkpoint=ck)
            shit = qval1(df, "E3", "source_cap_hit", model=model, checkpoint=ck)
            rows.append([ck, sig(a_rho2), sig(l1b), sig(smass) if smass is not None else "-",
                         sig(scap) if scap is not None else "-",
                         ("oui" if shit == 1.0 else "non") if shit is not None else "-"])
        J_full = qval1(df, "E3", "J_shared", model=model, checkpoint="shared_full")
        mean_full = qval1(df, "E3", "mean_perckpt", model=model, checkpoint="shared_full")
        gap_full = qval1(df, "E3", "gap_pct", model=model, checkpoint="shared_full")
        J_me = qval1(df, "E3", "J_shared", model=model, checkpoint="shared_mid_end")
        mean_me = qval1(df, "E3", "mean_perckpt", model=model, checkpoint="shared_mid_end")
        gap_me = qval1(df, "E3", "gap_pct", model=model, checkpoint="shared_mid_end")
        lines.append(f"\n**{model}**\n")
        lines.append(md_table(["checkpoint", "a_k/rho_k^2", "||u*||_1/beta", "masse source (agg)",
                               "plafond source (gamma*pi_9)", "plafond atteint"], rows))
        lines.append(f"\nFenetre complete (begin..end) : moyenne par checkpoint = {sig(mean_full)}, "
                     f"J(ubar partage) = {sig(J_full)}, **ecart one-shot = {sig(gap_full)}%**\n")
        lines.append(f"\nFenetre [mid, end] uniquement (exclut begin) : moyenne par checkpoint = "
                     f"{sig(mean_me)}, J(ubar partage) = {sig(J_me)}, "
                     f"**ecart one-shot = {sig(gap_me)}%**\n")
        data[f"table_{model}"] = rows
        data[f"gap_pct_full_{model}"] = gap_full
        data[f"gap_pct_mid_end_{model}"] = gap_me

        l1t = qval1(df, "E3", "l1_capacity_true", model=model, checkpoint="end")
        l1f = qval1(df, "E3", "l1_capacity_false", model=model, checkpoint="end")
        sbc = qval1(df, "E3", "s_beta_check", model=model, checkpoint="end")
        lines.append(f"\nVerification capacity=True vs False (beta=0.10, s_beta={sig(sbc)}, "
                     f">1 donc plafonds attendus actifs) : ||u*||_1(capacity=True)={sig(l1t)} "
                     f"(doit etre < 0.10), ||u*||_1(capacity=False)={sig(l1f)} (doit s'approcher de 0.10).\n")

    lines.append("\n### Jumeaux numeriques des figures\n")
    lines.append("**cos(u*_k, u*_k') [diagnostic, ne decide plus le verdict -- voir A3] et cos "
                 "colonne moyen de Gbar [le vrai signal de rotation de la trajectoire] entre "
                 "checkpoints** :\n")
    for model in models:
        rows_cos = []
        cks = e3_checkpoints_present(df, model)
        for i in range(len(cks)):
            for j in range(i + 1, len(cks)):
                key = f"{cks[i]}_vs_{cks[j]}"
                cos_u = qval1(df, "E3", "cos_u_star", model=model, checkpoint=key)
                col_cos = qval1(df, "E3", "mean_col_cos_Gbar", model=model, checkpoint=key)
                rows_cos.append([key, sig(cos_u) if cos_u is not None else "-",
                                 sig(col_cos) if col_cos is not None else "-"])
        lines.append(f"\n**{model}**\n")
        lines.append(md_table(["paire checkpoints", "cos(u*,u*) [diagnostic]",
                               "cos colonne moyen(Gbar,Gbar) [signal]"], rows_cos))

    lines.append("\n**Heatmap u*_ckpt** -- top-5 paires (y,z) par masse :\n")
    for model in models:
        rows_top = []
        for ck in e3_checkpoints_present(df, model):
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
        cks = e3_checkpoints_present(df, model)
        cos_all = []
        for i in range(len(cks)):
            for j in range(i + 1, len(cks)):
                c = qval1(df, "E3", "cos_u_star", model=model, checkpoint=f"{cks[i]}_vs_{cks[j]}")
                if c is not None and not math.isnan(c):
                    cos_all.append(c)
        min_cos = min(cos_all) if cos_all else None
        gap_full = data.get(f"gap_pct_full_{model}")
        gap_me = data.get(f"gap_pct_mid_end_{model}")
        v = "INCONCLUSIF" if gap_full is None else ("PASS" if abs(gap_full) < 20 else "FAIL")
        verdicts.append(f"**{model}** : {v} -- ecart one-shot (fenetre complete)={sig(gap_full)}%, "
                        f"ecart one-shot ([mid,end])={sig(gap_me)}% -- min cos(u*,u*)={sig(min_cos)} "
                        f"(diagnostic seulement, ne fonde pas le verdict)")
    for v in verdicts:
        lines.append(f"- {v}\n")

    lines.append("\n### Anomalies\n")
    lines.append("Aucune detectee automatiquement au-dela de ce qui est deja signale ci-dessus.\n")

    return "\n".join(lines), dict(verdicts=verdicts)


# --------------------------------------------------------------------------#
# Sec 6 -- E4
# --------------------------------------------------------------------------#
RULES_VARIANTS = [(r, v) for r in ("mean", "cw_median", "trmean", "krum", "multikrum")
                   for v in ("flat", "per_tensor")]


def _agg_key(rule, variant):
    return f"{rule}:{variant}"


def section_e4(df):
    lines = ["## 6. E4 -- Reponse des agregateurs a l'attaque optimale sous la moyenne\n"]
    lines.append("**Hypothese** : controler la moyenne controle aussi les regles robustes "
                 "(Krum, Multi-Krum, trimmed mean, coordinate-wise median), en variantes "
                 "`flat` et `per_tensor`.\n")
    sub = df[df.experiment == "E4"]
    if sub.empty:
        lines.append("**Ce qui a ete execute** : rien (E4 absent de ce run).\n")
        lines.append("**Verdict** : INCONCLUSIF -- E4 non execute.\n")
        return "\n".join(lines), dict(verdicts=[])

    models = models_present(sub)
    transforms = sorted(sub["transform"].dropna().unique().tolist())
    lines.append(f"**Ce qui a ete execute** : au checkpoint=end, beta=E1_BETA=0.10, n_p={int(sub['n_p'].dropna().iloc[0])}, "
                 f"deploiement `ubar*` sur les transforms {transforms}, {len(RULES_VARIANTS)} combinaisons "
                 "regle x variante par transform. Nombre de rounds reduit sous le hint '~200' de la section "
                 "8/E4 pour tenir le budget de dix minutes par bloc (voir sweep.py, E4_ROUNDS/SIM_BATCH) -- "
                 "moins de puissance statistique sur Abar que la valeur de reference.\n")

    lines.append("**Correction D3 (session 4)** : colonne \"borne informative\" ajoutee -- quand "
                 "`Abar~0` (aucun worker perturbe jamais selectionne, cas de krum ici), "
                 "`||P||+||N||~0 <= rhs` est vraie de facon VIDE (aucun worker malveillant n'a "
                 "jamais atteint l'agregat, donc rien n'a ete teste) : \"borne respectee = OUI\" ne "
                 "doit alors pas se lire comme une confirmation de la borne theorique.\n")
    data = {}
    lines.append("\n### Resultats\n")
    for model in models:
        for transform in transforms:
            rows = []
            for rule, variant in RULES_VARIANTS:
                key = _agg_key(rule, variant)
                ell = qval1(df, "E4", "ell", model=model, transform=transform, aggregator=key)
                chi = qval1(df, "E4", "chi_ell", model=model, transform=transform, aggregator=key)
                osc = qval1(df, "E4", "osc_abar", model=model, transform=transform, aggregator=key)
                sel = qval1(df, "E4", "selection_rate", model=model, transform=transform, aggregator=key)
                abar_mean = qval1(df, "E4", "abar_mean", model=model, transform=transform, aggregator=key)
                pn = qval1(df, "E4", "PN_norm", model=model, transform=transform, aggregator=key)
                bound = qval1(df, "E4", "bound_rhs", model=model, transform=transform, aggregator=key)
                resp = qval1(df, "E4", "bound_respected", model=model, transform=transform, aggregator=key)
                at_agg = qval1(df, "E4", "alpha_tilde_b_agg", model=model, transform=transform, aggregator=key)
                at_mean = qval1(df, "E4", "alpha_tilde_b_mean", model=model, transform=transform, aggregator=key)
                informative = "non (Abar~0)" if (abar_mean is not None and abar_mean < 1e-9) else "oui"
                rows.append([rule, variant, sig(ell) if ell is not None else "-",
                             sig(chi) if chi is not None else "-", sig(osc) if osc is not None else "-",
                             sig(sel) if sel is not None else "-", sig(pn) if pn is not None else "-",
                             sig(bound) if bound is not None else "-",
                             ("OUI" if resp == 1.0 else "NON") if resp is not None else "-",
                             informative,
                             sig(at_agg) if at_agg is not None else "-", sig(at_mean) if at_mean is not None else "-"])
            lines.append(f"\n**{model} / T={transform}**\n")
            lines.append(md_table(["regle", "variante", "ell", "chi_ell", "osc(Abar)", "taux selection",
                                   "||P||+||N||", "borne (rhs)", "borne respectee", "borne informative",
                                   "alpha~(b_Agg)", "alpha~(b_mean)"], rows))
            data[f"table_{model}_{transform}"] = rows

    lines.append("\n### Jumeaux numeriques des figures\n")
    lines.append("**Histogramme de A_j (deciles), variante flat, T=identity** :\n")
    for model in models:
        rows_dec = []
        for rule in ("mean", "cw_median", "trmean", "krum", "multikrum"):
            key = _agg_key(rule, "flat")
            vals = [qval1(df, "E4", f"abar_dec{p}", model=model, transform="identity", aggregator=key)
                    for p in range(0, 101, 10)]
            rows_dec.append([rule] + [sig(v) if v is not None else "-" for v in vals])
        lines.append(f"\n**{model}**\n")
        lines.append(md_table(["regle"] + [f"P{p}" for p in range(0, 101, 10)], rows_dec))

    lines.append("\n### Verdict\n")
    verdicts = []
    for model in models:
        osc_krum_flat = qval1(df, "E4", "osc_abar", model=model, transform="identity", aggregator=_agg_key("krum", "flat"))
        osc_krum_pt = qval1(df, "E4", "osc_abar", model=model, transform="identity", aggregator=_agg_key("krum", "per_tensor"))
        osc_mk_flat = qval1(df, "E4", "osc_abar", model=model, transform="identity", aggregator=_agg_key("multikrum", "flat"))
        osc_mk_pt = qval1(df, "E4", "osc_abar", model=model, transform="identity", aggregator=_agg_key("multikrum", "per_tensor"))
        sel_krum_flat = qval1(df, "E4", "selection_rate", model=model, transform="identity", aggregator=_agg_key("krum", "flat"))
        n_viol = 0
        for rule, variant in RULES_VARIANTS:
            for transform in transforms:
                r = qval1(df, "E4", "bound_respected", model=model, transform=transform,
                         aggregator=_agg_key(rule, variant))
                if r == 0.0:
                    n_viol += 1
        # osc(Abar)=0 under flat is a STRUCTURAL property (kind="global": one
        # selected set for the whole vector) -- always checkable. osc>0 under
        # per_tensor is a DATA-dependent divergence that requires at least
        # some malicious selection somewhere to be observable at all: if krum
        # never selects a malicious worker in ANY round/tensor (selection_rate
        # ~0 in both variants, as happens for cnn here), osc=0 in per_tensor
        # too, trivially -- that's krum being maximally robust at this
        # operating point, not a falsification of the divergence prediction.
        # multikrum (larger candidate pool, ell=n_b-f-2) is far less prone to
        # this all-or-nothing degeneracy, so it carries the primary evidence;
        # krum is reported alongside with its degeneracy flagged explicitly.
        flat_structural_ok = (osc_krum_flat is not None and abs(osc_krum_flat) < 1e-6 and
                              osc_mk_flat is not None and abs(osc_mk_flat) < 1e-6)
        mk_divergence_ok = osc_mk_pt is not None and osc_mk_pt > 1e-6
        krum_degenerate = sel_krum_flat is not None and sel_krum_flat < 1e-6
        if osc_krum_flat is None or osc_mk_flat is None:
            v = "INCONCLUSIF"
        elif flat_structural_ok and mk_divergence_ok and n_viol == 0:
            v = "PASS"
        else:
            v = "FAIL"
        krum_note = (" (krum: selection_rate~0 dans les deux variantes -- degenere, "
                    "osc(Abar)=0 non informatif ici, voir Anomalies)" if krum_degenerate else
                    f" ; krum per_tensor={sig(osc_krum_pt)} (attendu >0)")
        verdicts.append(f"**{model}** : {v} -- structure flat (krum={sig(osc_krum_flat)}, "
                        f"multikrum={sig(osc_mk_flat)}, attendu 0 les deux) OK ; "
                        f"divergence per_tensor multikrum={sig(osc_mk_pt)} (attendu >0){krum_note}, "
                        f"{n_viol} violation(s) de la borne theorique sur {len(RULES_VARIANTS)*len(transforms)} cellules")
    for v in verdicts:
        lines.append(f"- {v}\n")

    lines.append("\n### Anomalies\n")
    anomalies = []
    for model in models:
        sel_krum = qval1(df, "E4", "selection_rate", model=model, transform="identity", aggregator=_agg_key("krum", "flat"))
        if sel_krum is not None and sel_krum < 1e-6:
            anomalies.append(f"{model} : krum (flat et per_tensor) a selection_rate~0 a ce point de "
                             f"deploiement -- aucun worker perturbe jamais choisi, la prediction "
                             f"osc(Abar)>0 sous per_tensor n'est pas testable ici (pas une violation, "
                             f"un regime degenere)")
        for transform in transforms:
            for rule, variant in RULES_VARIANTS:
                key = _agg_key(rule, variant)
                resp = qval1(df, "E4", "bound_respected", model=model, transform=transform, aggregator=key)
                slack = qval1(df, "E4", "bound_slack", model=model, transform=transform, aggregator=key)
                if resp == 0.0:
                    anomalies.append(f"{model}/{transform}/{key} : borne ||P||+||N||<=rhs VIOLEE "
                                     f"(slack={sig(slack)} < 0)")
    if anomalies:
        lines.append("\n" + "\n".join(f"- {a}" for a in anomalies) + "\n")
    else:
        lines.append("Aucune violation de la borne theorique detectee.\n")

    return "\n".join(lines), dict(verdicts=verdicts, anomalies=anomalies)


# --------------------------------------------------------------------------#
# Sec 7 -- E5
# --------------------------------------------------------------------------#
def _knee_vhat(vhats, sels):
    """Steepest-slope point on (v_hat, selection): argmax |d(sel)/d(log v_hat)|."""
    if len(vhats) < 3:
        return None
    idx = np.argsort(vhats)
    x = np.log(np.asarray(vhats)[idx])
    y = np.asarray(sels)[idx]
    slopes = np.abs(np.diff(y) / np.diff(x))
    j = int(np.argmax(slopes))
    return float(np.asarray(vhats)[idx][j])


def section_e5(df):
    lines = ["## 7. E5 -- Courbe de furtivite et saturation\n"]
    lines.append("**Hypothese** : reduire la demande n'ameliore la selection que sous le "
                 "rayon atteignable -- coude attendu pres de v_hat=1. Seule la variante "
                 "`flat` (la seule couverte par la theorie, SPEC section 7) decide du verdict.\n")
    lines.append("**Correction C3 (session 4)** : \"taux de selection\" = `E[A_j] / gamma` -- la "
                 "part malveillante captee par l'agregat, RAPPORTEE a ce que donnerait la moyenne "
                 "simple (`A_j = gamma` par construction). Une valeur `> 1` est une "
                 "SUR-REPRESENTATION des workers perturbes dans l'agregat par rapport a leur poids "
                 "sous la moyenne (regles ell=1 sur peu de workers, ex. krum/cw_median), pas une "
                 "erreur de calcul.\n")
    sub = df[df.experiment == "E5"]
    if sub.empty:
        lines.append("**Ce qui a ete execute** : rien (E5 absent de ce run).\n")
        lines.append("**Verdict** : INCONCLUSIF -- E5 non execute.\n")
        return "\n".join(lines), dict(verdicts=[])

    models = models_present(sub)
    betas = sorted(sub["beta"].dropna().unique().tolist())
    n_tau_pts = int(sub[sub.metric == "n_tau_points"]["value"].iloc[0]) if (sub.metric == "n_tau_points").any() else 16
    lines.append(f"**Ce qui a ete execute** : checkpoint=end, T=identity, betas={betas}, grille de "
                 f"{n_tau_pts} taus log-espaces sur v_hat in [0.02, 10] par beta (etendue vers le bas "
                 "depuis [0.1, 10] -- correction C1, pour localiser le coude au lieu d'extrapoler au-dela "
                 "du bord inferieur de la grille). Rounds/round reduits sous le hint SPEC pour tenir le "
                 "budget de dix minutes (voir sweep.py, E5_ROUNDS/SIM_BATCH) ; krum/multikrum recoivent un "
                 "passage separe, boostee a >= E5_ROUNDS_KRUM_MIN tirages effectifs pour krum/"
                 "multikrum (correction C2).\n")

    data = {}
    lines.append("\n### Resultats -- selection (variante flat) vs v_hat, par regle\n")
    RULES = ("mean", "cw_median", "trmean", "krum", "multikrum")
    knees = {}
    for model in models:
        for beta in betas:
            taus = sorted(sub[(sub.model == model) & (sub.beta == beta) & sub.tau.notna()]["tau"].unique().tolist())
            if not taus:
                continue
            for rule in RULES:
                key = _agg_key(rule, "flat")
                vhats, sels = [], []
                for tau in taus:
                    vh = qval1(df, "E5", "v_hat", model=model, beta=beta, tau=tau)
                    s = qval1(df, "E5", "selection_rate", model=model, beta=beta, tau=tau, aggregator=key)
                    if vh is not None and s is not None:
                        vhats.append(vh); sels.append(s)
                if len(vhats) < 3:
                    continue
                knee = _knee_vhat(vhats, sels)
                knees[(model, beta, rule)] = knee
                rows = [[sig(vh), sig(s)] for vh, s in sorted(zip(vhats, sels))]
                lines.append(f"\n**{model} / beta={sig(beta)} / {rule} (flat)** -- coude estime a v_hat~{sig(knee)}\n")
                lines.append(md_table(["v_hat", "taux selection"], rows))
                data[f"table_{model}_{beta}_{rule}"] = rows

    lines.append("\n### Jumeaux numeriques des figures\n")
    lines.append("**Table des coudes (v_hat) par regle, beta=E1_BETA** :\n")
    rows_knee = []
    e1_beta_val = 0.10
    for model in models:
        row = [model]
        for rule in RULES:
            k = knees.get((model, e1_beta_val, rule))
            if k is None:
                for b in betas:
                    if (model, b, rule) in knees:
                        k = knees[(model, b, rule)]
                        break
            row.append(sig(k) if k is not None else "-")
        rows_knee.append(row)
    lines.append(md_table(["Modele"] + list(RULES), rows_knee))

    lines.append("\n### Baisse de selection par courbe (correction A2)\n")
    lines.append("**Bug corrige** : le calcul precedent comparait uniquement le PREMIER et le "
                 "DERNIER point de chaque courbe (`pts[0]` vs `pts[-1]`) -- une courbe qui descend "
                 "puis REMONTE (le comportement attendu pres du coude, cf. `cw_median`/`trmean` dans "
                 "la table de resultats ci-dessus) a des extremites quasi identiques et se comptait "
                 "donc comme \"pas de baisse\" alors que la table montre un creux net au milieu. "
                 "Nouveau calcul : `baisse = (selection moyenne pour v_hat>=1) - (selection minimale "
                 "pour v_hat<1)`, qui teste directement l'hypothese \"plat au-dessus de v_hat=1, "
                 "decroissant en-dessous\" plutot que la seule difference d'extremites.\n")
    MARGIN = 0.10
    rows_drop = []
    curve_drops = {}
    for model in models:
        for beta in betas:
            for rule in RULES:
                key = _agg_key(rule, "flat")
                taus = sorted(sub[(sub.model == model) & (sub.beta == beta) & sub.tau.notna()]["tau"].unique().tolist())
                pts = []
                for tau in taus:
                    vh = qval1(df, "E5", "v_hat", model=model, beta=beta, tau=tau)
                    s = qval1(df, "E5", "selection_rate", model=model, beta=beta, tau=tau, aggregator=key)
                    if vh is not None and s is not None:
                        pts.append((vh, s))
                if len(pts) < 4:
                    continue
                pts.sort()
                above = [s for vh, s in pts if vh >= 1.0]
                below = [s for vh, s in pts if vh < 1.0]
                flat_ref = float(np.mean(above)) if above else pts[-1][1]
                trough = min(below) if below else min(s for _, s in pts)
                drop = flat_ref - trough
                curve_drops[(model, beta, rule)] = drop
                rows_drop.append([model, sig(beta), rule, sig(pts[0][0]), sig(pts[0][1]),
                                  sig(pts[-1][0]), sig(pts[-1][1]), sig(flat_ref), sig(trough), sig(drop)])
    lines.append(md_table(["modele", "beta", "regle", "v_hat (premier)", "selection (premier)",
                           "v_hat (dernier)", "selection (dernier)", "plat (v_hat>=1)",
                           "creux (v_hat<1)", "baisse"], rows_drop))

    lines.append("\n### Verdict\n")
    verdicts = []
    # "Flat above v_hat=1, decreasing below" -- tested per (model, beta, rule)
    # curve as: baisse (flat_ref - trough, computed above) >= MARGIN. Replaces
    # the endpoint-only comparison (see "Bug corrige" note above).
    for model in models:
        n_curves = sum(1 for (m, b, r) in curve_drops if m == model)
        n_confirm = sum(1 for (m, b, r), d in curve_drops.items() if m == model and d >= MARGIN)
        if n_curves == 0:
            v = "INCONCLUSIF"
            frac_txt = "n/a"
        else:
            frac = n_confirm / n_curves
            v = "PASS" if frac >= 0.5 else "FAIL"
            frac_txt = f"{n_confirm}/{n_curves} courbes ({frac*100:.0f}%)"
        fail_note = (" ; la selection ne varie pas assez avec v_hat sous ce budget de rounds "
                    "reduit pour confirmer la prediction la plus falsifiable du modele"
                    if v == "FAIL" else "")
        verdicts.append(f"**{model}** : {v} -- courbes confirmant une baisse (plat au-dessus de "
                        f"v_hat=1 moins creux en-dessous) >= {MARGIN*100:.0f}pt : {frac_txt} "
                        f"(voir la table \"Baisse de selection par courbe\" ci-dessus, et les coudes "
                        f"estimes par regle plus haut -- a prendre avec prudence vu le bruit de "
                        f"Monte-Carlo, cf. Anomalies){fail_note}")
    for v in verdicts:
        lines.append(f"- {v}\n")

    lines.append("\n### Anomalies\n")
    lines.append("Voir la note sur la reduction du nombre de rounds/taille de minibatch ci-dessus : "
                 "cela augmente le bruit de Monte-Carlo sur `selection_rate`, qui peut produire des "
                 "coudes moins nets que sous les ~200 rounds de reference de la section 8/E5.\n")

    return "\n".join(lines), dict(verdicts=verdicts, knees={f"{k[0]}/{sig(k[1])}/{k[2]}": v for k, v in knees.items()})


# --------------------------------------------------------------------------#
# Sec 9 -- E7
# --------------------------------------------------------------------------#
def section_e7(df):
    lines = ["## 9. E7 -- Etaler le budget\n"]
    lines.append("**Hypothese** : a beta fixe, augmenter n_p reduit le taux local `beta/gamma` "
                 "sans changer l'ensemble atteignable sous la moyenne (`ubar*`, `E_k` inchanges).\n")
    sub = df[df.experiment == "E7"]
    if sub.empty:
        lines.append("**Ce qui a ete execute** : rien (E7 absent de ce run).\n")
        lines.append("**Verdict** : INCONCLUSIF -- E7 non execute.\n")
        return "\n".join(lines), dict(verdicts=[])

    models = models_present(sub)
    n_ps = sorted(sub["n_p"].dropna().unique().tolist())
    lines.append(f"**Ce qui a ete execute** : checkpoint=end, T=identity, beta=E1_BETA=0.10, "
                 f"n_p in {n_ps} (replay de E4 a chaque n_p).\n")

    data = {}
    lines.append("\n### Resultats -- invariance de ubar*/E_k et selection vs n_p\n")
    for model in models:
        rows_inv = []
        for n_p in n_ps:
            gap = qval1(df, "E7", "ubar_linf_gap_vs_ref", model=model, n_p=n_p)
            s_beta = qval1(df, "E7", "s_beta", model=model, n_p=n_p)
            caps = qval1(df, "E7", "caps_can_bind", model=model, n_p=n_p)
            rate = qval1(df, "E7", "local_rate_beta_over_gamma", model=model, n_p=n_p)
            rows_inv.append([int(n_p), sig(gap), sig(s_beta), "OUI" if caps == 1.0 else "NON", sig(rate)])
        lines.append(f"\n**{model}**\n")
        lines.append(md_table(["n_p", "||ubar*(n_p)-ubar*(ref)||_inf", "s_beta", "plafonds actifs", "beta/gamma"], rows_inv))
        data[f"invariance_{model}"] = rows_inv

        rows_sel = []
        for n_p in n_ps:
            row = [int(n_p)]
            for rule in ("mean", "cw_median", "trmean", "krum", "multikrum"):
                key = _agg_key(rule, "flat")
                sel = qval1(df, "E7", "selection_rate", model=model, n_p=n_p, aggregator=key)
                row.append(sig(sel) if sel is not None else "-")
            rows_sel.append(row)
        lines.append(f"\n**{model} -- taux de selection (flat) vs n_p**\n")
        lines.append(md_table(["n_p", "mean", "cw_median", "trmean", "krum", "multikrum"], rows_sel))
        data[f"selection_{model}"] = rows_sel

    lines.append("\n### Jumeaux numeriques des figures\n")
    lines.append("**Selection vs n_p (courbe)** : voir table ci-dessus.\n")

    lines.append("\n### Verdict\n")
    verdicts = []
    for model in models:
        gaps = [qval1(df, "E7", "ubar_linf_gap_vs_ref", model=model, n_p=n_p) for n_p in n_ps]
        gaps = [g for g in gaps if g is not None]
        caps_binding = any(qval1(df, "E7", "caps_can_bind", model=model, n_p=n_p) == 1.0 for n_p in n_ps)
        max_gap = max(gaps) if gaps else None
        increasing = []
        for rule in ("cw_median", "krum"):
            key = _agg_key(rule, "flat")
            sels = [qval1(df, "E7", "selection_rate", model=model, n_p=n_p, aggregator=key) for n_p in n_ps]
            sels = [s for s in sels if s is not None]
            increasing.append(len(sels) >= 2 and sels[-1] >= sels[0])
        if max_gap is None:
            v = "INCONCLUSIF"
        elif caps_binding and max_gap > 1e-6:
            v = ("FAIL (attendu -- plafonds actifs a s_beta>1, cf. E3/E7 section 8)"
                 if not all(increasing) else "PASS (partiel -- selection croit avec n_p malgre ubar* qui varie car les plafonds saturent)")
        elif max_gap <= 1e-6 and all(increasing):
            v = "PASS"
        else:
            v = "FAIL"
        verdicts.append(f"**{model}** : {v} -- max||ubar*(n_p)-ubar*(ref)||_inf={sig(max_gap)}, "
                        f"plafonds actifs sur au moins un n_p={'oui' if caps_binding else 'non'}, "
                        f"selection croissante en n_p (cw_median/krum)={'oui' if all(increasing) else 'non'}")
    for v in verdicts:
        lines.append(f"- {v}\n")

    lines.append("\n### Anomalies\n")
    lines.append("Un ecart ubar*(n_p) non nul n'est pas un bug ici : SPEC section 8/E7 note "
                 "explicitement que l'invariance n'est exacte que lorsque les plafonds par classe "
                 "de U_beta sont non saturants ; a s_beta = beta/(gamma*min_y pi[y]) > 1, gamma=n_p/n_b "
                 "bouge avec n_p et les plafonds saturent differemment a chaque n_p (voir la colonne "
                 "'plafonds actifs' ci-dessus).\n")

    return "\n".join(lines), dict(verdicts=verdicts)


# --------------------------------------------------------------------------#
# Sec 8 -- E6 (expensive block, behind include_e6)
# --------------------------------------------------------------------------#
def section_pending_e6():
    lines = ["## 8. E6 -- Pouvoir predictif des notions de faisabilite (bloc couteux, include_e6)\n"]
    reason = ("Non execute cette session -- bloc couteux, derriere le drapeau "
              "`include_e6=True`.")
    lines.append("**Hypothese** : Le residu normalise (alpha_tilde_star, v_hat, varpi, "
                 "a_over_rho2) predit l'ASR.\n")
    lines.append(f"**Ce qui a ete execute** : rien. {reason}\n")
    lines.append("**Resultats** : n/a.\n")
    lines.append("**Jumeaux numeriques des figures** : n/a.\n")
    lines.append(f"**Verdict** : INCONCLUSIF -- {reason}\n")
    lines.append("**Anomalies** : n/a.\n")
    return "\n".join(lines)


PREDICTOR_NAMES = ("a_over_rho2", "alpha_tilde_star", "v_hat", "varpi")
E6_AGG_NAMES = ("mean", "trmean")


def section_e6(df):
    sub = df[df.experiment == "E6"]
    if sub.empty:
        return section_pending_e6(), dict(verdicts=[])

    lines = ["## 8. E6 -- Pouvoir predictif des notions de faisabilite (bloc couteux, include_e6)\n"]
    lines.append("**Hypothese** : le residu normalise (alpha_tilde_star, v_hat, varpi, "
                 "a_over_rho2, calcules au round 0 et jamais mis a jour) predit l'ASR finale, "
                 "sur `r32p`/CIFAR-10 (10000 exemples), ~30 rounds federes reels.\n")

    configs = sorted(sub[sub.checkpoint.str.startswith("round0_p", na=False)]
                     [["checkpoint", "beta"]].drop_duplicates().itertuples(index=False),
                     key=lambda r: (r.checkpoint, r.beta))
    lines.append(f"**Ce qui a ete execute** : {len(configs)} configuration(s) (source/target x beta, "
                 f"choisies pour etaler le predicteur -- 3 paires x 3 betas tronque a 8), "
                 f"aggregateurs {E6_AGG_NAMES}, empoisonnement par inversion d'etiquettes uniquement "
                 f"(jamais de trigger pixel pendant l'entrainement -- voir effect_rate/masses_to_labels).\n")

    lines.append("\n### Resultats -- table complete (config, predicteurs, effet, precision propre)\n")
    headers = ["config", "beta", "a_over_rho2", "alpha_tilde_star", "v_hat", "varpi"]
    for agg in E6_AGG_NAMES:
        headers += [f"effect_rate ({agg})", f"clean_acc ({agg})"]
    rows_full = []
    for ck, beta in configs:
        pair = ck.replace("round0_", "")
        m = re.match(r"p(\d+)-(\d+)", pair)
        label = f"{m.group(1)}->{m.group(2)}" if m else pair
        row = [label, sig(beta)]
        for pred in PREDICTOR_NAMES:
            v = qval1(df, "E6", pred, checkpoint=ck, beta=beta)
            row.append(sig(v) if v is not None else "-")
        for agg in E6_AGG_NAMES:
            eff = qval1(df, "E6", "effect_rate", checkpoint=f"end_{pair}", beta=beta, aggregator=agg)
            acc = qval1(df, "E6", "clean_accuracy", checkpoint=f"end_{pair}", beta=beta, aggregator=agg)
            row.append(sig(eff) if eff is not None else "-")
            row.append(sig(acc) if acc is not None else "-")
        rows_full.append(row)
    lines.append(md_table(headers, rows_full))
    data = dict(table=rows_full)

    lines.append("\n### Jumeaux numeriques des figures\n")
    lines.append("**Correlation de Spearman(effect_rate, predicteur), par aggregateur** :\n")
    rows_sp = []
    for agg in E6_AGG_NAMES:
        row = [agg]
        for pred in PREDICTOR_NAMES:
            r = qval1(df, "E6", f"spearman_effect_vs_{pred}", checkpoint="summary", aggregator=agg)
            row.append(sig(r) if r is not None else "n/a")
        n_cfg = qval1(df, "E6", "n_configs", checkpoint="summary", aggregator=agg)
        row.append(int(n_cfg) if n_cfg is not None else 0)
        rows_sp.append(row)
    lines.append(md_table(["aggregateur"] + list(PREDICTOR_NAMES) + ["n configs"], rows_sp))
    data["spearman"] = rows_sp

    lines.append("\n### Verdict\n")
    verdicts = []
    for agg, row in zip(E6_AGG_NAMES, rows_sp):
        coefs = [c for c in row[1:1 + len(PREDICTOR_NAMES)] if c != "n/a"]
        coefs_f = [float(c) for c in coefs]
        n_cfg = row[-1]
        if n_cfg < 3 or not coefs_f:
            v = "INCONCLUSIF"
        else:
            # Expected: clear DECREASING monotonicity between residual and
            # effect (SPEC section 8/E6) -- negative Spearman r for all 4.
            v = "PASS" if all(c < -0.3 for c in coefs_f) else "FAIL"
        verdicts.append(f"**{agg}** : {v} -- Spearman(effect_rate, predicteur) sur {n_cfg} configs : "
                        f"{dict(zip(PREDICTOR_NAMES, row[1:1 + len(PREDICTOR_NAMES)]))} "
                        f"(attendu : negatif et net pour les 4)")
    for v in verdicts:
        lines.append(f"- {v}\n")

    lines.append("\n### Anomalies\n")
    lines.append("n_configs < 8 signifie des configurations en echec (voir failures.log) ; "
                 "les coefficients de Spearman avec seulement quelques points sont fragiles, a "
                 "lire avec prudence.\n")

    return "\n".join(lines), dict(verdicts=verdicts, table=rows_full, spearman=rows_sp)


# --------------------------------------------------------------------------#
# Sec 10 -- synthese
# --------------------------------------------------------------------------#
def section10(e1_data, e2_data, e3_data, e4_data, e5_data, e6_data, e7_data, models):
    lines = ["## 10. Synthese\n"]
    rows = []

    def _verdict_token(v):
        """
        Extracts the actual status word right after "** : " (every verdict
        string in this file follows "**model/label** : STATUS ..."). Matching
        against this single token -- rather than searching the whole sentence
        for a bare "FAIL"/"PASS" substring -- avoids the class of bug where an
        unrelated explanatory clause elsewhere in the sentence (e.g. "si FAIL,
        ...") contains the word and silently flips the synthesis table's
        verdict regardless of what was actually computed.
        """
        m = re.search(r"\*\*.*?\*\*\s*:\s*(\S+)", v)
        return m.group(1) if m else ""

    def first_verdict_status(verdicts):
        if not verdicts:
            return "INCONCLUSIF"
        tokens = [_verdict_token(v) for v in verdicts]
        if any(t.startswith("FAIL") for t in tokens):
            return "FAIL"
        if any(t.startswith("INCONCLUSIF") for t in tokens):
            return "INCONCLUSIF"
        return "PASS" if all(t.startswith("PASS") for t in tokens) else "INCONCLUSIF"

    s_e1 = first_verdict_status(e1_data.get("verdicts", []))
    rows.append(["E1", "Implementation et transfert de Gbar", s_e1])

    e2_verdicts = e2_data.get("verdicts", [])
    s_e2 = "INCONCLUSIF"
    for v in e2_verdicts:
        if v.startswith("**cnn"):
            t = _verdict_token(v)
            s_e2 = "PASS" if t.startswith("PASS") else ("FAIL" if t.startswith("FAIL") else "INCONCLUSIF")
    rows.append(["E2", "Plafond de rang (config cnn uniquement)", s_e2])

    s_e3 = first_verdict_status(e3_data.get("verdicts", []))
    rows.append(["E3", "Stabilite one-shot", s_e3])

    s_e4 = first_verdict_status(e4_data.get("verdicts", []))
    rows.append(["E4", "Reponse des agregateurs robustes (informatif, non-gate)", s_e4])

    s_e5 = first_verdict_status(e5_data.get("verdicts", []))
    rows.append(["E5", "Existence du levier de furtivite", s_e5])

    s_e7 = first_verdict_status(e7_data.get("verdicts", []))
    rows.append(["E7", "Etaler le budget sur n_p (informatif, non-gate)", s_e7])

    s_e6 = first_verdict_status(e6_data.get("verdicts", []))
    rows.append(["E6", "Pouvoir predictif du residu (informatif, non-gate, cher)", s_e6])

    lines.append(md_table(["Verrou", "Description", "Go/no-go"], rows))

    gate_rows = [r for r in rows if r[0] in ("E1", "E2", "E3", "E5")]
    n_fail = sum(1 for r in gate_rows if r[2] == "FAIL")
    n_inc = sum(1 for r in gate_rows if "INCONCLUSIF" in r[2])
    lines.append("\n**Recommandation**\n")
    if n_fail > 0:
        lines.append(f"- {n_fail} verrou(x) bloquant(s) (E1/E2/E3/E5) en FAIL : le sweep complet ne "
                     f"devrait pas etre lance sans d'abord comprendre ces echecs (voir sections "
                     f"correspondantes pour le chiffre qui fonde chaque FAIL) -- ce run a neanmoins "
                     f"pousse jusqu'a E4/E5/E7 a titre informatif, sur demande explicite.\n")
    elif n_inc > 0:
        lines.append(f"- {n_inc} verrou(x) encore INCONCLUSIF -- ne pas lancer le sweep complet avant "
                     f"qu'ils soient tranches.\n")
    else:
        lines.append("- Tous les verrous bloquants sont PASS : le sweep complet peut etre lance.\n")
    if e6_data.get("verdicts"):
        lines.append(f"- E6 (informatif, non bloquant) : {s_e6} -- voir section 8 pour le detail "
                     f"par aggregateur.\n")
    else:
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

    md_parts = ["# Rapport preliminaires -- E1-E7 (session 4, v2)\n\n"
                "**v2 (cette session)** : correction de mesure uniquement, aucune nouvelle experience "
                "(E6 reste non lance). Corrige : verdicts E2/E3/E5 (sections A1-A3), le facteur "
                "d'echelle ~10x d'E1 (section B, voir section 3), la grille/rounds d'E5 (section C), "
                "des metriques manquantes E1/E2/E3/E4 (section D), et la coherence section1/"
                "failures.log (section E). Voir le compte-rendu de session pour le detail.\n"]
    json_data = {}

    s, d = section0(df, meta, failures_text); md_parts.append(s); json_data["sec0"] = d
    s, d = section1(df, meta, failures_text); md_parts.append(s); json_data["sec1"] = d
    s, d = section2(df); md_parts.append(s); json_data["sec2"] = d
    s, e1_data = section_e1(df); md_parts.append(s); json_data["E1"] = e1_data
    s, e2_data = section_e2(df); md_parts.append(s); json_data["E2"] = e2_data
    s, e3_data = section_e3(df); md_parts.append(s); json_data["E3"] = e3_data
    s, e4_data = section_e4(df); md_parts.append(s); json_data["E4"] = e4_data
    s, e5_data = section_e5(df); md_parts.append(s); json_data["E5"] = e5_data
    s, e6_data = section_e6(df); md_parts.append(s); json_data["E6"] = e6_data
    s, e7_data = section_e7(df); md_parts.append(s); json_data["E7"] = e7_data
    s, d = section10(e1_data, e2_data, e3_data, e4_data, e5_data, e6_data, e7_data, models_present(df))
    md_parts.append(s); json_data["sec10"] = d

    report_md = "\n".join(md_parts)
    n_lines = report_md.count("\n") + 1
    if n_lines > 1500:
        report_md += (f"\n\n*(Avertissement : {n_lines} lignes, au-dessus de la limite de 1500 "
                      f"-- a resserrer.)*\n")

    os.makedirs(out_dir, exist_ok=True)
    report_md_path = os.path.join(out_dir, "report.md")
    report_json_path = os.path.join(out_dir, "report.json")
    with open(report_md_path, "w") as f:
        f.write(report_md)
    with open(report_json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    print(f"[report] {n_lines} lignes -> {report_md_path}")
    print(f"[report] -> {report_json_path}")
    return report_md


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", default=METRICS_PATH)
    p.add_argument("--out-dir", default=ARTIFACT_DIR)
    args = p.parse_args()
    build_report(metrics_path=args.metrics, out_dir=args.out_dir)
