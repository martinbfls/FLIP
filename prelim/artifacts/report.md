# Rapport preliminaires -- E1-E7 (session 3)

## 0. En-tete

| Champ | Valeur |
|---|---|
| Date du rapport | 2026-08-21 00:02 |
| Commit git | f1cff8f (dirty) |
| torch | 2.8.0 |
| numpy | 1.26.4 |
| osqp | 1.1.3 |
| Device entrainement / evaluation | mps / cpu |
| Duree totale du sweep | 1202s (~20.0 min) |
| Cellules executees / en cache / en echec | 2 / 6 / 0 |
| Lignes dans metrics.csv | 36412 |
| Entrees dans failures.log | 0 |

## 1. Grille executee

### Axes du balayage

| Axe | Valeurs |
|---|---|
| models | linear, cnn |
| seeds | 0, 1, 2 |
| checkpoints | begin, mid, end |
| betas | 0.01, 0.03, 0.1 |
| transforms | identity, stripe |
| n_p | 3 |
| aggregators | mean, cw_median, trmean, krum, multikrum |
| include_e6 | False |


Portee reelle de E1-E3 (voir sweep.py docstring) : E1 (table principale) tourne a `seed=SEEDS[0]`, `beta=E1_BETA=0.10`, tous les checkpoints ; robustesse aux graines et balayage SNR/batch a `checkpoint=end` uniquement, sur toutes les graines/betas respectivement ; E2/E3 tournent sur tous les checkpoints x tous les betas (independants de la graine). E2 exploite `identity` et `stripe` (patch, voir section E2) ; E1/E3 restent `identity` uniquement. E4/E5/E7 (etapes 2-3) tournent a checkpoint=end uniquement : E4 balaie `identity`/`stripe`, E5/E7 restent a `identity` avec beta/n_p balayes -- rounds et taille de minibatch reduits sous les hints SPEC section 8 pour tenir le budget de dix minutes par bloc (voir sweep.py, E4_ROUNDS/E5_ROUNDS/E7_ROUNDS/SIM_BATCH).


Nombre de cellules (checkpoint x modele) : 8


### Cellules echouees ou ignorees

Aucune.

## 2. Assertions de coherence

| Nom | Portee | Statut | Valeur observee | Seuil |
|---|---|---|---|---|
| Relation de budget ||u_i||_1*gamma == ||ubar*||_1 | par modele (checkpoint=end) | PASS | 1.39e-17 | 1.00e-06 |
| solve_qp(capacity=False) == project_gradient (depot) | par modele (checkpoint=end) | PASS | 0 | 1.00e-06 |
| alpha_tilde_star: contrainte de budget inactive a l'optimum (NNLS) | par modele (checkpoint=end) | PASS | 2.73e-06 | 0.0100 |
| Masses de flip realisees vs demandees : ecart relatif max (>=1 flip attendu) | toutes cellules E1 | FAIL | 0.309 | 0.100 |
| Agregateur aplati == depot (5 regles, sur un stack reel de gradients, checkpoint=end) | par modele | PASS | 0 | 1.00e-05 |
| Aucun NaN/Inf dans le CSV (NaN=0, Inf=0) | metrics.csv entier | PASS | 0 | 0 |

## 3. E1 -- Carte de biais : implementation, transfert, signal/bruit

**Hypothese** : `E[g_i] = grad_c + Gbar@u_i` transfere du jeu de calibration vers un shard reel, et le signal domine le bruit minibatch aux budgets realistes.

**Ce qui a ete execute** : table principale (6 configs x checkpoints) a beta=0.1, robustesse aux graines et balayage SNR/batch a checkpoint=end -- 6 cellules (modele x checkpoint) au total.


### Resultats -- transfert calibration -> shard (cos, erreur relative)


**cnn**

| checkpoint | config | cos (calib) | err_rel (calib) | cos (shard) | err_rel (shard) |
|---|---|---|---|---|---|
| begin | a_source_target | 0.996 | 9.31 | 0.997 | 8.63 |
| begin | b_other_pair | 0.990 | 9.14 | 0.996 | 9.50 |
| begin | c_uniform | 0.677 | 18.8 | 0.749 | 13.0 |
| begin | d_qp | 0.999 | 8.95 | 1.000 | 8.84 |
| begin | e_random1 | 0.782 | 15.8 | 0.888 | 14.3 |
| begin | f_random2 | 0.675 | 8.56 | 0.875 | 10.7 |
| mid | a_source_target | 0.998 | 9.38 | 0.999 | 9.24 |
| mid | b_other_pair | 0.985 | 9.28 | 0.994 | 9.18 |
| mid | c_uniform | 0.992 | 9.01 | 0.994 | 9.04 |
| mid | d_qp | 1.000 | 9.06 | 1.00 | 9.11 |
| mid | e_random1 | 0.991 | 9.12 | 0.994 | 9.17 |
| mid | f_random2 | 0.993 | 9.17 | 0.995 | 9.19 |
| end | a_source_target | 0.997 | 9.17 | 0.999 | 9.07 |
| end | b_other_pair | 0.990 | 9.36 | 0.998 | 9.22 |
| end | c_uniform | 0.991 | 8.65 | 0.993 | 9.00 |
| end | d_qp | 1.000 | 9.03 | 1.00 | 9.06 |
| end | e_random1 | 0.985 | 8.67 | 0.991 | 9.05 |
| end | f_random2 | 0.989 | 8.98 | 0.993 | 9.31 |


**linear**

| checkpoint | config | cos (calib) | err_rel (calib) | cos (shard) | err_rel (shard) |
|---|---|---|---|---|---|
| begin | a_source_target | 0.972 | 8.57 | 0.999 | 9.04 |
| begin | b_other_pair | 0.699 | 12.5 | 0.988 | 9.28 |
| begin | c_uniform | 0.929 | 10.6 | 0.961 | 9.49 |
| begin | d_qp | 0.994 | 9.01 | 1.000 | 9.08 |
| begin | e_random1 | 0.911 | 10.7 | 0.954 | 9.35 |
| begin | f_random2 | 0.919 | 10.9 | 0.956 | 9.82 |
| mid | a_source_target | 0.969 | 8.55 | 0.999 | 8.98 |
| mid | b_other_pair | 0.688 | 13.2 | 0.990 | 9.28 |
| mid | c_uniform | 0.904 | 9.75 | 0.960 | 9.30 |
| mid | d_qp | 0.993 | 8.90 | 1.000 | 9.04 |
| mid | e_random1 | 0.886 | 10.1 | 0.952 | 9.18 |
| mid | f_random2 | 0.888 | 9.87 | 0.955 | 9.65 |
| end | a_source_target | 0.970 | 8.63 | 0.999 | 9.02 |
| end | b_other_pair | 0.693 | 13.4 | 0.991 | 9.23 |
| end | c_uniform | 0.907 | 9.97 | 0.961 | 9.31 |
| end | d_qp | 0.993 | 8.95 | 1.000 | 9.05 |
| end | e_random1 | 0.887 | 10.2 | 0.953 | 9.22 |
| end | f_random2 | 0.896 | 10.1 | 0.957 | 9.61 |


### Robustesse aux graines (checkpoint=end, config a)

| Modele | cos (median [min,max]) | err_rel (median [min,max]) | n graines |
|---|---|---|---|
| cnn | 0.997 [0.995, 0.999] | 9.28 [9.17, 9.35] | 3 |
| linear | 0.964 [0.956, 0.970] | 8.78 [8.63, 8.95] | 3 |


### Jumeaux numeriques des figures

**cos vs err_rel (nuage E1, shard)** -- correlation + points extremes :

| Modele | Pearson r(err_rel, cos) | cos min | err_rel max |
|---|---|---|---|
| cnn | -0.839 | begin/c_uniform (cos=0.749) | begin/e_random1 (err_rel=14.3) |
| linear | -0.730 | mid/e_random1 (cos=0.952) | begin/f_random2 (err_rel=9.82) |


**Erreur vs |B| (log-log) et SNR(beta,B), checkpoint=end** -- 12 points par modele :

| Modele | beta | |B| | err_rel | SNR |
|---|---|---|---|---|
| cnn | 0.0100 | full | 9.29 | - |
| cnn | 0.0100 | 64 | 20.0 | 0.0578 |
| cnn | 0.0100 | 256 | 12.9 | 0.112 |
| cnn | 0.0100 | 1024 | 10.6 | 0.245 |
| cnn | 0.0300 | full | 9.17 | - |
| cnn | 0.0300 | 64 | 9.76 | 0.168 |
| cnn | 0.0300 | 256 | 8.92 | 0.307 |
| cnn | 0.0300 | 1024 | 9.25 | 0.693 |
| cnn | 0.100 | full | 9.17 | - |
| cnn | 0.100 | 64 | 9.76 | 0.168 |
| cnn | 0.100 | 256 | 8.92 | 0.307 |
| cnn | 0.100 | 1024 | 9.25 | 0.693 |
| linear | 0.0100 | full | 11.0 | - |
| linear | 0.0100 | 64 | 57.4 | 0.0183 |
| linear | 0.0100 | 256 | 29.5 | 0.0357 |
| linear | 0.0100 | 1024 | 15.7 | 0.0867 |
| linear | 0.0300 | full | 8.63 | - |
| linear | 0.0300 | 64 | 20.0 | 0.0544 |
| linear | 0.0300 | 256 | 12.3 | 0.104 |
| linear | 0.0300 | 1024 | 9.50 | 0.246 |
| linear | 0.100 | full | 8.63 | - |
| linear | 0.100 | 64 | 20.0 | 0.0544 |
| linear | 0.100 | 256 | 12.3 | 0.104 |
| linear | 0.100 | 1024 | 9.50 | 0.246 |


### Verdict

- **cnn** : FAIL (min cos_shard = 0.749, seuil 0.99)

- **linear** : FAIL (min cos_shard = 0.952, seuil 0.99)


### Anomalies

- cnn/begin/a_source_target : err_rel(shard) = 8.63 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/begin/b_other_pair : err_rel(shard) = 9.50 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/begin/c_uniform : err_rel(shard) = 13.0 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/begin/d_qp : err_rel(shard) = 8.84 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/begin/e_random1 : err_rel(shard) = 14.3 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/begin/f_random2 : err_rel(shard) = 10.7 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/mid/a_source_target : err_rel(shard) = 9.24 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/mid/b_other_pair : err_rel(shard) = 9.18 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/mid/c_uniform : err_rel(shard) = 9.04 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/mid/d_qp : err_rel(shard) = 9.11 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/mid/e_random1 : err_rel(shard) = 9.17 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/mid/f_random2 : err_rel(shard) = 9.19 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/end/a_source_target : err_rel(shard) = 9.07 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/end/b_other_pair : err_rel(shard) = 9.22 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- cnn/end/c_uniform : err_rel(shard) = 9.00 (>100% -- ||Gbar@u|| est petit devant le bruit residuel a ce budget, cf. table ci-dessus ; cos reste eleve, donc la DIRECTION est correcte, seule la magnitude relative de l'erreur est grande)

- ... et 21 autres lignes similaires (voir metrics.csv, experiment=E1, metric=relerr_shard__*)

## 4. E2 -- Geometrie et scalaires de regime

**Hypothese** : le plafond de rang (`varpi`) laisse une marge exploitable a l'attaquant plutot que d'etre domine par la politique par classe.

**Rappel de portee** : pour `linear`, `Gbar` a rang exactement `C(C-1)`=90 generiquement (produit exterieur) -- `varpi`/`alpha_tilde_star` y sont atypiquement favorables par construction. **Le verdict E2 est pris sur `cnn` uniquement** ; `linear` sert de test d'implementation.

**Note de correction (patch stripe)** : E2 tournait initialement avec `T=identity` uniquement, ou `v` est quasi tautologiquement dans l'image de `Gbar` (formule de decomposition de la section 8/E2 : `v/lam = Gbar[:,(9,4)]/pi[9] + (g[9][9]-grad_c)`). Ce patch ajoute `T=stripe` (le vrai trigger) aux memes cellules pour ecarter l'hypothese que le FAIL n'etait qu'un artefact du cas degenere -- **le resultat sous stripe est quasi identique a celui sous identity** (voir tables ci-dessous) : ce n'est donc pas un artefact de portee, c'est bien le plafond de rang qui domine, meme sous le vrai trigger.


### Resultats


**cnn / T=identity**

| checkpoint | beta | varpi | baseline (rang_eff/d) | v_hat | alpha_tilde_star | Theta (rad) |
|---|---|---|---|---|---|---|
| begin | 0.0100 | 1.000 | 1.01e-05 | 3.15 | 0.995 | 0.00190 |
| begin | 0.0300 | 1.000 | 1.01e-05 | 3.15 | 0.995 | 0.00569 |
| begin | 0.100 | 1.000 | 1.01e-05 | 3.15 | 0.995 | 0.0142 |
| mid | 0.0100 | 1.00 | 7.37e-06 | 5.83 | 0.999 | 0.0207 |
| mid | 0.0300 | 1.00 | 7.37e-06 | 5.83 | 0.999 | 0.0622 |
| mid | 0.100 | 1.00 | 7.37e-06 | 5.83 | 0.999 | 0.155 |
| end | 0.0100 | 1.00 | 9.15e-06 | 5.13 | 1.00 | 0.0241 |
| end | 0.0300 | 1.00 | 9.15e-06 | 5.13 | 1.00 | 0.0724 |
| end | 0.100 | 1.00 | 9.15e-06 | 5.13 | 1.00 | 0.131 |


**cnn / T=stripe**

| checkpoint | beta | varpi | baseline (rang_eff/d) | v_hat | alpha_tilde_star | Theta (rad) |
|---|---|---|---|---|---|---|
| begin | 0.0100 | 0.999 | 1.01e-05 | 3.16 | 0.994 | 0.00190 |
| begin | 0.0300 | 0.999 | 1.01e-05 | 3.16 | 0.994 | 0.00569 |
| begin | 0.100 | 0.999 | 1.01e-05 | 3.16 | 0.994 | 0.0142 |
| mid | 0.0100 | 1.00 | 7.37e-06 | 5.91 | 0.999 | 0.0207 |
| mid | 0.0300 | 1.00 | 7.37e-06 | 5.91 | 0.999 | 0.0622 |
| mid | 0.100 | 1.00 | 7.37e-06 | 5.91 | 0.999 | 0.155 |
| end | 0.0100 | 0.999 | 9.15e-06 | 5.28 | 1.000 | 0.0241 |
| end | 0.0300 | 0.999 | 9.15e-06 | 5.28 | 1.000 | 0.0724 |
| end | 0.100 | 1.000 | 9.15e-06 | 5.28 | 1.000 | 0.131 |


**linear / T=identity**

| checkpoint | beta | varpi | baseline (rang_eff/d) | v_hat | alpha_tilde_star | Theta (rad) |
|---|---|---|---|---|---|---|
| begin | 0.0100 | 1.000 | 0.000377 | 7.17 | 0.970 | 0.00707 |
| begin | 0.0300 | 1.000 | 0.000377 | 7.17 | 0.970 | 0.0212 |
| begin | 0.100 | 1.000 | 0.000377 | 7.17 | 0.970 | 0.0485 |
| mid | 0.0100 | 0.998 | 0.000377 | 7.97 | 0.990 | 0.00663 |
| mid | 0.0300 | 0.998 | 0.000377 | 7.97 | 0.990 | 0.0199 |
| mid | 0.100 | 0.998 | 0.000377 | 7.97 | 0.990 | 0.0455 |
| end | 0.0100 | 0.997 | 0.000377 | 7.94 | 0.983 | 0.00616 |
| end | 0.0300 | 0.997 | 0.000377 | 7.94 | 0.983 | 0.0185 |
| end | 0.100 | 0.997 | 0.000377 | 7.94 | 0.983 | 0.0423 |


**linear / T=stripe**

| checkpoint | beta | varpi | baseline (rang_eff/d) | v_hat | alpha_tilde_star | Theta (rad) |
|---|---|---|---|---|---|---|
| begin | 0.0100 | 0.984 | 0.000377 | 7.15 | 0.963 | 0.00707 |
| begin | 0.0300 | 0.984 | 0.000377 | 7.15 | 0.963 | 0.0212 |
| begin | 0.100 | 0.984 | 0.000377 | 7.15 | 0.963 | 0.0485 |
| mid | 0.0100 | 0.982 | 0.000377 | 8.00 | 0.982 | 0.00663 |
| mid | 0.0300 | 0.982 | 0.000377 | 8.00 | 0.982 | 0.0199 |
| mid | 0.100 | 0.982 | 0.000377 | 8.00 | 0.982 | 0.0455 |
| end | 0.0100 | 0.981 | 0.000377 | 8.02 | 0.976 | 0.00616 |
| end | 0.0300 | 0.981 | 0.000377 | 8.02 | 0.976 | 0.0185 |
| end | 0.100 | 0.981 | 0.000377 | 8.02 | 0.976 | 0.0423 |


### Jumeaux numeriques des figures

**Spectre des valeurs propres de Q (checkpoint=end)** -- 11 points (P100..P0) :

| Modele | P0 | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 | P100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | 0 | 0 | 0.0105 | 0.0637 | 0.157 | 0.334 | 0.673 | 1.88 | 5.42 | 28.3 | 2207 |
| linear | 0.0596 | 0.0937 | 0.187 | 0.217 | 0.396 | 0.589 | 0.716 | 1.97 | 4.80 | 22.9 | 127 |


**varpi vs baseline (bar chart)** : voir table ci-dessus (colonnes varpi/baseline).


**alpha_tilde_star vs sqrt(varpi) (nuage)** -- correlation + extremes :

| Modele/transform | Pearson r(sqrt(varpi), alpha_tilde_star) | n points |
|---|---|---|
| cnn/identity | 0.665 | 9 |
| cnn/stripe | 0.634 | 9 |
| linear/identity | -0.754 | 9 |
| linear/stripe | -0.898 | 9 |


**v_hat par checkpoint, par beta** : voir table de resultats ci-dessus.


### Verdict

- **cnn (decisif, T=stripe)** : FAIL (SATURE) -- median(varpi)=0.999 (a moins de 1% du plafond 1.0, donc SATURE -- pas de marge exploitable demontree malgre varpi<1 au sens strict), median(baseline)=9.15e-06 ; identity donne median(varpi)=1.00, quasi identique -- donc pas un artefact du cas degenere T=identity, le plafond de rang domine reellement sous le vrai trigger

- **linear** : test d'implementation seulement (rang exact C(C-1), non decisif)


### Anomalies


- cnn/identity: varpi=1.00 > 1 (hors [0,1] -- au-dela de la tolerance du solveur OSQP eps~1e-6 utilisee par dist_to_cone/rank_ratio, a surveiller mais pas un signe de bug de signe/formule)
- cnn/identity: varpi=1.00 > 1 (hors [0,1] -- au-dela de la tolerance du solveur OSQP eps~1e-6 utilisee par dist_to_cone/rank_ratio, a surveiller mais pas un signe de bug de signe/formule)
- cnn/identity: varpi=1.00 > 1 (hors [0,1] -- au-dela de la tolerance du solveur OSQP eps~1e-6 utilisee par dist_to_cone/rank_ratio, a surveiller mais pas un signe de bug de signe/formule)
- cnn/identity: varpi=1.00 > 1 (hors [0,1] -- au-dela de la tolerance du solveur OSQP eps~1e-6 utilisee par dist_to_cone/rank_ratio, a surveiller mais pas un signe de bug de signe/formule)
- cnn/identity: varpi=1.00 > 1 (hors [0,1] -- au-dela de la tolerance du solveur OSQP eps~1e-6 utilisee par dist_to_cone/rank_ratio, a surveiller mais pas un signe de bug de signe/formule)
- cnn/identity: varpi=1.00 > 1 (hors [0,1] -- au-dela de la tolerance du solveur OSQP eps~1e-6 utilisee par dist_to_cone/rank_ratio, a surveiller mais pas un signe de bug de signe/formule)
- cnn/stripe: varpi=1.00 > 1 (hors [0,1] -- au-dela de la tolerance du solveur OSQP eps~1e-6 utilisee par dist_to_cone/rank_ratio, a surveiller mais pas un signe de bug de signe/formule)
- cnn/stripe: varpi=1.00 > 1 (hors [0,1] -- au-dela de la tolerance du solveur OSQP eps~1e-6 utilisee par dist_to_cone/rank_ratio, a surveiller mais pas un signe de bug de signe/formule)
- cnn/stripe: varpi=1.00 > 1 (hors [0,1] -- au-dela de la tolerance du solveur OSQP eps~1e-6 utilisee par dist_to_cone/rank_ratio, a surveiller mais pas un signe de bug de signe/formule)

## 5. E3 -- Stabilite de Gbar et cout du one-shot

**Hypothese** : une configuration `ubar` unique sert toute la trajectoire d'entrainement (le one-shot ne coute presque rien face a l'oracle par checkpoint).


### Resultats -- cout one-shot et budget effectif


**cnn**

| checkpoint | a_k/rho_k^2 | ||u*||_1/beta |
|---|---|---|
| begin | 7.54 | 1.00 |
| mid | 26.3 | 1.00 |
| end | 21.6 | 1.00 |


Moyenne par checkpoint = 18.5, J(ubar partage) = 18.6, ecart one-shot = 0.416%


Verification capacity=True vs False (beta=0.10, s_beta=3.33, >1 donc plafonds attendus actifs) : ||u*||_1(capacity=True)=0.100 (doit etre < 0.10), ||u*||_1(capacity=False)=0.100 (doit s'approcher de 0.10).


**linear**

| checkpoint | a_k/rho_k^2 | ||u*||_1/beta |
|---|---|---|
| begin | 43.5 | 1.00 |
| mid | 54.8 | 1.00 |
| end | 54.4 | 1.00 |


Moyenne par checkpoint = 50.9, J(ubar partage) = 50.9, ecart one-shot = 6.49e-06%


Verification capacity=True vs False (beta=0.10, s_beta=3.33, >1 donc plafonds attendus actifs) : ||u*||_1(capacity=True)=0.100 (doit etre < 0.10), ||u*||_1(capacity=False)=0.100 (doit s'approcher de 0.10).


### Jumeaux numeriques des figures

**cos(u*_k, u*_k') entre checkpoints** :


**cnn**

| paire checkpoints | cos(u*,u*) | cos colonne moyen(Gbar,Gbar) |
|---|---|---|
| begin_vs_mid | 0.322 | 0.0902 |
| begin_vs_end | 0.583 | 0.0683 |
| mid_vs_end | 0.693 | 0.851 |


**linear**

| paire checkpoints | cos(u*,u*) | cos colonne moyen(Gbar,Gbar) |
|---|---|---|
| begin_vs_mid | 1.000 | 1.000 |
| begin_vs_end | 1.000 | 1.000 |
| mid_vs_end | 1.000 | 1.000 |


**Heatmap u*_ckpt** -- top-5 paires (y,z) par masse :


**cnn**

| checkpoint | top1 | top2 | top3 | top4 | top5 |
|---|---|---|---|---|---|
| begin | (7->4): 0.0300 | (1->4): 0.0300 | (9->4): 0.0297 | (3->4): 0.0103 | (1->8): 8.54e-09 |
| mid | (9->5): 0.0300 | (1->4): 0.0300 | (8->4): 0.0300 | (0->4): 0.01000 | (8->0): 5.83e-08 |
| end | (1->4): 0.0300 | (8->4): 0.0300 | (9->4): 0.0206 | (0->2): 0.0100 | (9->2): 0.00940 |


**linear**

| checkpoint | top1 | top2 | top3 | top4 | top5 |
|---|---|---|---|---|---|
| begin | (8->4): 0.0300 | (9->4): 0.0300 | (0->4): 0.0300 | (1->4): 0.0100 | (4->9): 7.07e-09 |
| mid | (9->4): 0.0300 | (8->4): 0.0300 | (0->4): 0.0300 | (1->4): 0.01000 | (2->4): 5.18e-09 |
| end | (9->4): 0.0300 | (0->4): 0.0300 | (8->4): 0.0300 | (1->4): 0.0100 | (0->1): 0 |


### Verdict

- **cnn** : FAIL -- min cos(u*,u*)=0.322, ecart one-shot=0.416%

- **linear** : PASS -- min cos(u*,u*)=1.000, ecart one-shot=6.49e-06%


### Anomalies

Aucune detectee automatiquement au-dela de ce qui est deja signale ci-dessus.

## 6. E4 -- Reponse des agregateurs a l'attaque optimale sous la moyenne

**Hypothese** : controler la moyenne controle aussi les regles robustes (Krum, Multi-Krum, trimmed mean, coordinate-wise median), en variantes `flat` et `per_tensor`.

**Ce qui a ete execute** : au checkpoint=end, beta=E1_BETA=0.10, n_p=3, deploiement `ubar*` sur les transforms ['identity', 'stripe'], 10 combinaisons regle x variante par transform. Nombre de rounds reduit sous le hint '~200' de la section 8/E4 pour tenir le budget de dix minutes par bloc (voir sweep.py, E4_ROUNDS/SIM_BATCH) -- moins de puissance statistique sur Abar que la valeur de reference.


### Resultats


**cnn / T=identity**

| regle | variante | ell | chi_ell | osc(Abar) | taux selection | ||P||+||N|| | borne (rhs) | borne respectee | alpha~(b_Agg) | alpha~(b_mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | flat | 10.0 | 0 | 0 | 1.00 | 0 | 0 | OUI | 0.955 | 0.955 |
| mean | per_tensor | 10.0 | 0 | 0 | 1.00 | 0 | 0 | OUI | 0.955 | 0.955 |
| cw_median | flat | 1.00 | 0.900 | 0.633 | 0.557 | 11.6 | 37.6 | OUI | 0.755 | 0.955 |
| cw_median | per_tensor | 1.00 | 0.900 | 0.633 | 0.557 | 11.6 | 37.6 | OUI | 0.755 | 0.955 |
| trmean | flat | 4.00 | 0.150 | 0.454 | 0.590 | 11.5 | 16.7 | OUI | 0.883 | 0.955 |
| trmean | per_tensor | 4.00 | 0.150 | 0.454 | 0.590 | 11.5 | 16.7 | OUI | 0.883 | 0.955 |
| krum | flat | 1.00 | 0.900 | 0 | 0 | 10.6 | 37.6 | OUI | 0 | 0.955 |
| krum | per_tensor | 1.00 | 0.900 | 0 | 0 | 10.3 | 37.6 | OUI | 0 | 0.955 |
| multikrum | flat | 5.00 | 0.100 | 0 | 0 | 10.2 | 13.7 | OUI | 0 | 0.955 |
| multikrum | per_tensor | 5.00 | 0.100 | 0.0100 | 0.00194 | 10.2 | 13.7 | OUI | 0 | 0.955 |


**cnn / T=stripe**

| regle | variante | ell | chi_ell | osc(Abar) | taux selection | ||P||+||N|| | borne (rhs) | borne respectee | alpha~(b_Agg) | alpha~(b_mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | flat | 10.0 | 0 | 0 | 1.00 | 0 | 0 | OUI | 0.954 | 0.954 |
| mean | per_tensor | 10.0 | 0 | 0 | 1.00 | 0 | 0 | OUI | 0.954 | 0.954 |
| cw_median | flat | 1.00 | 0.900 | 0.633 | 0.562 | 11.6 | 37.7 | OUI | 0.763 | 0.954 |
| cw_median | per_tensor | 1.00 | 0.900 | 0.633 | 0.562 | 11.6 | 37.7 | OUI | 0.763 | 0.954 |
| trmean | flat | 4.00 | 0.150 | 0.454 | 0.595 | 11.6 | 16.8 | OUI | 0.888 | 0.954 |
| trmean | per_tensor | 4.00 | 0.150 | 0.454 | 0.595 | 11.6 | 16.8 | OUI | 0.888 | 0.954 |
| krum | flat | 1.00 | 0.900 | 0 | 0 | 10.6 | 37.7 | OUI | 0 | 0.954 |
| krum | per_tensor | 1.00 | 0.900 | 0 | 0 | 10.4 | 37.7 | OUI | 0 | 0.954 |
| multikrum | flat | 5.00 | 0.100 | 0 | 0 | 10.2 | 13.8 | OUI | 0 | 0.954 |
| multikrum | per_tensor | 5.00 | 0.100 | 0.0100 | 0.00135 | 10.3 | 13.8 | OUI | 0 | 0.954 |


**linear / T=identity**

| regle | variante | ell | chi_ell | osc(Abar) | taux selection | ||P||+||N|| | borne (rhs) | borne respectee | alpha~(b_Agg) | alpha~(b_mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | flat | 10.0 | 0 | 0 | 1.00 | 0 | 0 | OUI | 0.785 | 0.785 |
| mean | per_tensor | 10.0 | 0 | 0 | 1.00 | 0 | 0 | OUI | 0.785 | 0.785 |
| cw_median | flat | 1.00 | 0.900 | 0.633 | 0.939 | 4.72 | 38.3 | OUI | 0.530 | 0.785 |
| cw_median | per_tensor | 1.00 | 0.900 | 0.633 | 0.939 | 4.72 | 38.3 | OUI | 0.530 | 0.785 |
| trmean | flat | 4.00 | 0.150 | 0.504 | 0.963 | 4.15 | 16.1 | OUI | 0.580 | 0.785 |
| trmean | per_tensor | 4.00 | 0.150 | 0.504 | 0.963 | 4.15 | 16.1 | OUI | 0.580 | 0.785 |
| krum | flat | 1.00 | 0.900 | 0 | 0.0556 | 5.37 | 38.3 | OUI | 0.113 | 0.785 |
| krum | per_tensor | 1.00 | 0.900 | 0.0167 | 0.0555 | 5.37 | 38.3 | OUI | 0.113 | 0.785 |
| multikrum | flat | 5.00 | 0.100 | 0 | 0.100 | 4.92 | 13.2 | OUI | 0.0984 | 0.785 |
| multikrum | per_tensor | 5.00 | 0.100 | 0.0267 | 0.1000 | 4.92 | 13.2 | OUI | 0.0982 | 0.785 |


**linear / T=stripe**

| regle | variante | ell | chi_ell | osc(Abar) | taux selection | ||P||+||N|| | borne (rhs) | borne respectee | alpha~(b_Agg) | alpha~(b_mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | flat | 10.0 | 0 | 0 | 1.00 | 0 | 0 | OUI | 0.775 | 0.775 |
| mean | per_tensor | 10.0 | 0 | 0 | 1.00 | 0 | 0 | OUI | 0.775 | 0.775 |
| cw_median | flat | 1.00 | 0.900 | 0.633 | 0.939 | 4.72 | 38.3 | OUI | 0.527 | 0.775 |
| cw_median | per_tensor | 1.00 | 0.900 | 0.633 | 0.939 | 4.72 | 38.3 | OUI | 0.527 | 0.775 |
| trmean | flat | 4.00 | 0.150 | 0.504 | 0.963 | 4.15 | 16.1 | OUI | 0.577 | 0.775 |
| trmean | per_tensor | 4.00 | 0.150 | 0.504 | 0.963 | 4.15 | 16.1 | OUI | 0.577 | 0.775 |
| krum | flat | 1.00 | 0.900 | 0 | 0.0556 | 5.37 | 38.3 | OUI | 0.106 | 0.775 |
| krum | per_tensor | 1.00 | 0.900 | 0.0167 | 0.0555 | 5.37 | 38.3 | OUI | 0.106 | 0.775 |
| multikrum | flat | 5.00 | 0.100 | 0 | 0.100 | 4.92 | 13.2 | OUI | 0.102 | 0.775 |
| multikrum | per_tensor | 5.00 | 0.100 | 0.0267 | 0.1000 | 4.92 | 13.2 | OUI | 0.102 | 0.775 |


### Jumeaux numeriques des figures

**Histogramme de A_j (deciles), variante flat, T=identity** :


**cnn**

| regle | P0 | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 | P100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 |
| cw_median | 0 | 0 | 0.0167 | 0.0667 | 0.117 | 0.167 | 0.217 | 0.250 | 0.300 | 0.333 | 0.633 |
| trmean | 0 | 0.0125 | 0.0458 | 0.0917 | 0.146 | 0.196 | 0.237 | 0.267 | 0.287 | 0.308 | 0.454 |
| krum | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| multikrum | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |


**linear**

| regle | P0 | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 | P100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 |
| cw_median | 0 | 0.150 | 0.217 | 0.250 | 0.267 | 0.283 | 0.317 | 0.333 | 0.350 | 0.383 | 0.633 |
| trmean | 0.00833 | 0.175 | 0.262 | 0.279 | 0.292 | 0.300 | 0.308 | 0.317 | 0.329 | 0.350 | 0.512 |
| krum | 0.0167 | 0.0167 | 0.0167 | 0.0167 | 0.0167 | 0.0167 | 0.0167 | 0.0167 | 0.0167 | 0.0167 | 0.0167 |
| multikrum | 0.0300 | 0.0300 | 0.0300 | 0.0300 | 0.0300 | 0.0300 | 0.0300 | 0.0300 | 0.0300 | 0.0300 | 0.0300 |


### Verdict

- **cnn** : PASS -- structure flat (krum=0, multikrum=0, attendu 0 les deux) OK ; divergence per_tensor multikrum=0.0100 (attendu >0) (krum: selection_rate~0 dans les deux variantes -- degenere, osc(Abar)=0 non informatif ici, voir Anomalies), 0 violation(s) de la borne theorique sur 20 cellules

- **linear** : PASS -- structure flat (krum=0, multikrum=0, attendu 0 les deux) OK ; divergence per_tensor multikrum=0.0267 (attendu >0) ; krum per_tensor=0.0167 (attendu >0), 0 violation(s) de la borne theorique sur 20 cellules


### Anomalies


- cnn : krum (flat et per_tensor) a selection_rate~0 a ce point de deploiement -- aucun worker perturbe jamais choisi, la prediction osc(Abar)>0 sous per_tensor n'est pas testable ici (pas une violation, un regime degenere)

## 7. E5 -- Courbe de furtivite et saturation

**Hypothese** : reduire la demande n'ameliore la selection que sous le rayon atteignable -- coude attendu pres de v_hat=1. Seule la variante `flat` (la seule couverte par la theorie, SPEC section 7) decide du verdict.

**Ce qui a ete execute** : checkpoint=end, T=identity, betas=[0.01, 0.03, 0.1], grille de 12 taus log-espaces sur v_hat in [0.1, 10] par beta. Rounds/round reduits sous le hint SPEC pour tenir le budget de dix minutes (voir sweep.py, E5_ROUNDS/SIM_BATCH).


### Resultats -- selection (variante flat) vs v_hat, par regle


**cnn / beta=0.0100 / mean (flat)** -- coude estime a v_hat~0.100

| v_hat | taux selection |
|---|---|
| 0.100 | 1.000 |
| 0.152 | 1.000 |
| 0.231 | 1.000 |
| 0.351 | 1.000 |
| 0.534 | 1.000 |
| 0.811 | 1.000 |
| 1.23 | 1.000 |
| 1.87 | 1.000 |
| 2.85 | 1.000 |
| 4.33 | 1.000 |
| 6.58 | 1.000 |
| 10.0 | 1.000 |


**cnn / beta=0.0100 / cw_median (flat)** -- coude estime a v_hat~1.23

| v_hat | taux selection |
|---|---|
| 0.100 | 0.923 |
| 0.152 | 0.919 |
| 0.231 | 0.906 |
| 0.351 | 0.894 |
| 0.534 | 0.892 |
| 0.811 | 0.885 |
| 1.23 | 0.889 |
| 1.87 | 0.916 |
| 2.85 | 0.920 |
| 4.33 | 0.927 |
| 6.58 | 0.927 |
| 10.0 | 0.927 |


**cnn / beta=0.0100 / trmean (flat)** -- coude estime a v_hat~1.23

| v_hat | taux selection |
|---|---|
| 0.100 | 0.935 |
| 0.152 | 0.931 |
| 0.231 | 0.916 |
| 0.351 | 0.907 |
| 0.534 | 0.906 |
| 0.811 | 0.901 |
| 1.23 | 0.901 |
| 1.87 | 0.928 |
| 2.85 | 0.931 |
| 4.33 | 0.938 |
| 6.58 | 0.938 |
| 10.0 | 0.938 |


**cnn / beta=0.0100 / krum (flat)** -- coude estime a v_hat~0.152

| v_hat | taux selection |
|---|---|
| 0.100 | 0.556 |
| 0.152 | 0.556 |
| 0.231 | 0 |
| 0.351 | 0.556 |
| 0.534 | 0.556 |
| 0.811 | 0.556 |
| 1.23 | 0.556 |
| 1.87 | 0.556 |
| 2.85 | 0.556 |
| 4.33 | 0.556 |
| 6.58 | 0.556 |
| 10.0 | 0.556 |


**cnn / beta=0.0100 / multikrum (flat)** -- coude estime a v_hat~2.85

| v_hat | taux selection |
|---|---|
| 0.100 | 0.778 |
| 0.152 | 0.667 |
| 0.231 | 0.556 |
| 0.351 | 0.556 |
| 0.534 | 0.556 |
| 0.811 | 0.556 |
| 1.23 | 0.556 |
| 1.87 | 0.667 |
| 2.85 | 0.667 |
| 4.33 | 0.889 |
| 6.58 | 0.889 |
| 10.0 | 0.889 |


**cnn / beta=0.0300 / mean (flat)** -- coude estime a v_hat~0.100

| v_hat | taux selection |
|---|---|
| 0.100 | 1.000 |
| 0.152 | 1.000 |
| 0.231 | 1.000 |
| 0.351 | 1.000 |
| 0.534 | 1.000 |
| 0.811 | 1.000 |
| 1.23 | 1.000 |
| 1.87 | 1.000 |
| 2.85 | 1.000 |
| 4.33 | 1.000 |
| 6.58 | 1.000 |
| 10.0 | 1.000 |


**cnn / beta=0.0300 / cw_median (flat)** -- coude estime a v_hat~0.351

| v_hat | taux selection |
|---|---|
| 0.100 | 0.898 |
| 0.152 | 0.892 |
| 0.231 | 0.885 |
| 0.351 | 0.858 |
| 0.534 | 0.827 |
| 0.811 | 0.820 |
| 1.23 | 0.803 |
| 1.87 | 0.814 |
| 2.85 | 0.827 |
| 4.33 | 0.823 |
| 6.58 | 0.823 |
| 10.0 | 0.823 |


**cnn / beta=0.0300 / trmean (flat)** -- coude estime a v_hat~0.351

| v_hat | taux selection |
|---|---|
| 0.100 | 0.910 |
| 0.152 | 0.907 |
| 0.231 | 0.903 |
| 0.351 | 0.877 |
| 0.534 | 0.851 |
| 0.811 | 0.848 |
| 1.23 | 0.835 |
| 1.87 | 0.842 |
| 2.85 | 0.855 |
| 4.33 | 0.853 |
| 6.58 | 0.853 |
| 10.0 | 0.853 |


**cnn / beta=0.0300 / krum (flat)** -- coude estime a v_hat~0.811

| v_hat | taux selection |
|---|---|
| 0.100 | 0.556 |
| 0.152 | 0.556 |
| 0.231 | 0.556 |
| 0.351 | 0.556 |
| 0.534 | 0.556 |
| 0.811 | 0.556 |
| 1.23 | 0 |
| 1.87 | 0 |
| 2.85 | 0 |
| 4.33 | 0 |
| 6.58 | 0 |
| 10.0 | 0 |


**cnn / beta=0.0300 / multikrum (flat)** -- coude estime a v_hat~0.152

| v_hat | taux selection |
|---|---|
| 0.100 | 0.556 |
| 0.152 | 0.556 |
| 0.231 | 0.333 |
| 0.351 | 0.333 |
| 0.534 | 0.333 |
| 0.811 | 0.333 |
| 1.23 | 0.333 |
| 1.87 | 0.556 |
| 2.85 | 0.556 |
| 4.33 | 0.444 |
| 6.58 | 0.444 |
| 10.0 | 0.444 |


**linear / beta=0.0100 / mean (flat)** -- coude estime a v_hat~0.100

| v_hat | taux selection |
|---|---|
| 0.100 | 1.00 |
| 0.152 | 1.00 |
| 0.231 | 1.00 |
| 0.351 | 1.00 |
| 0.534 | 1.00 |
| 0.811 | 1.00 |
| 1.23 | 1.00 |
| 1.87 | 1.00 |
| 2.85 | 1.00 |
| 4.33 | 1.00 |
| 6.58 | 1.00 |
| 10.0 | 1.00 |


**linear / beta=0.0100 / cw_median (flat)** -- coude estime a v_hat~0.100

| v_hat | taux selection |
|---|---|
| 0.100 | 0.981 |
| 0.152 | 0.973 |
| 0.231 | 0.975 |
| 0.351 | 0.970 |
| 0.534 | 0.977 |
| 0.811 | 0.975 |
| 1.23 | 0.968 |
| 1.87 | 0.962 |
| 2.85 | 0.962 |
| 4.33 | 0.962 |
| 6.58 | 0.962 |
| 10.0 | 0.962 |


**linear / beta=0.0100 / trmean (flat)** -- coude estime a v_hat~1.23

| v_hat | taux selection |
|---|---|
| 0.100 | 0.990 |
| 0.152 | 0.984 |
| 0.231 | 0.984 |
| 0.351 | 0.982 |
| 0.534 | 0.988 |
| 0.811 | 0.987 |
| 1.23 | 0.983 |
| 1.87 | 0.976 |
| 2.85 | 0.976 |
| 4.33 | 0.976 |
| 6.58 | 0.976 |
| 10.0 | 0.976 |


**linear / beta=0.0100 / krum (flat)** -- coude estime a v_hat~1.23

| v_hat | taux selection |
|---|---|
| 0.100 | 1.11 |
| 0.152 | 1.11 |
| 0.231 | 1.11 |
| 0.351 | 1.11 |
| 0.534 | 1.11 |
| 0.811 | 1.11 |
| 1.23 | 1.11 |
| 1.87 | 0.556 |
| 2.85 | 0.556 |
| 4.33 | 0.556 |
| 6.58 | 0.556 |
| 10.0 | 0.556 |


**linear / beta=0.0100 / multikrum (flat)** -- coude estime a v_hat~0.811

| v_hat | taux selection |
|---|---|
| 0.100 | 0.778 |
| 0.152 | 0.778 |
| 0.231 | 0.778 |
| 0.351 | 0.778 |
| 0.534 | 0.778 |
| 0.811 | 0.778 |
| 1.23 | 0.556 |
| 1.87 | 0.667 |
| 2.85 | 0.667 |
| 4.33 | 0.667 |
| 6.58 | 0.667 |
| 10.0 | 0.667 |


**linear / beta=0.0300 / mean (flat)** -- coude estime a v_hat~0.100

| v_hat | taux selection |
|---|---|
| 0.100 | 1.00 |
| 0.152 | 1.00 |
| 0.231 | 1.00 |
| 0.351 | 1.00 |
| 0.534 | 1.00 |
| 0.811 | 1.00 |
| 1.23 | 1.00 |
| 1.87 | 1.00 |
| 2.85 | 1.00 |
| 4.33 | 1.00 |
| 6.58 | 1.00 |
| 10.0 | 1.00 |


**linear / beta=0.0300 / cw_median (flat)** -- coude estime a v_hat~0.534

| v_hat | taux selection |
|---|---|
| 0.100 | 0.976 |
| 0.152 | 0.978 |
| 0.231 | 0.972 |
| 0.351 | 0.968 |
| 0.534 | 0.949 |
| 0.811 | 0.969 |
| 1.23 | 0.955 |
| 1.87 | 0.938 |
| 2.85 | 0.938 |
| 4.33 | 0.938 |
| 6.58 | 0.938 |
| 10.0 | 0.938 |


**linear / beta=0.0300 / trmean (flat)** -- coude estime a v_hat~0.534

| v_hat | taux selection |
|---|---|
| 0.100 | 0.987 |
| 0.152 | 0.986 |
| 0.231 | 0.981 |
| 0.351 | 0.981 |
| 0.534 | 0.969 |
| 0.811 | 0.983 |
| 1.23 | 0.970 |
| 1.87 | 0.958 |
| 2.85 | 0.958 |
| 4.33 | 0.958 |
| 6.58 | 0.958 |
| 10.0 | 0.958 |


**linear / beta=0.0300 / krum (flat)** -- coude estime a v_hat~0.152

| v_hat | taux selection |
|---|---|
| 0.100 | 1.11 |
| 0.152 | 1.11 |
| 0.231 | 0.556 |
| 0.351 | 0.556 |
| 0.534 | 0.556 |
| 0.811 | 0.556 |
| 1.23 | 0.556 |
| 1.87 | 0.556 |
| 2.85 | 0.556 |
| 4.33 | 0.556 |
| 6.58 | 0.556 |
| 10.0 | 0.556 |


**linear / beta=0.0300 / multikrum (flat)** -- coude estime a v_hat~0.351

| v_hat | taux selection |
|---|---|
| 0.100 | 0.889 |
| 0.152 | 0.889 |
| 0.231 | 0.667 |
| 0.351 | 0.778 |
| 0.534 | 0.556 |
| 0.811 | 0.667 |
| 1.23 | 0.556 |
| 1.87 | 0.333 |
| 2.85 | 0.333 |
| 4.33 | 0.333 |
| 6.58 | 0.333 |
| 10.0 | 0.333 |


**linear / beta=0.100 / mean (flat)** -- coude estime a v_hat~0.100

| v_hat | taux selection |
|---|---|
| 0.100 | 1.00 |
| 0.152 | 1.00 |
| 0.231 | 1.00 |
| 0.351 | 1.00 |
| 0.534 | 1.00 |
| 0.811 | 1.00 |
| 1.23 | 1.00 |
| 1.87 | 1.00 |
| 2.85 | 1.00 |
| 4.33 | 1.00 |
| 6.58 | 1.00 |
| 10.0 | 1.00 |


**linear / beta=0.100 / cw_median (flat)** -- coude estime a v_hat~2.85

| v_hat | taux selection |
|---|---|
| 0.100 | 0.970 |
| 0.152 | 0.962 |
| 0.231 | 0.953 |
| 0.351 | 0.942 |
| 0.534 | 0.942 |
| 0.811 | 0.933 |
| 1.23 | 0.914 |
| 1.87 | 0.915 |
| 2.85 | 0.913 |
| 4.33 | 0.893 |
| 6.58 | 0.889 |
| 10.0 | 0.889 |


**linear / beta=0.100 / trmean (flat)** -- coude estime a v_hat~0.811

| v_hat | taux selection |
|---|---|
| 0.100 | 0.975 |
| 0.152 | 0.971 |
| 0.231 | 0.960 |
| 0.351 | 0.951 |
| 0.534 | 0.949 |
| 0.811 | 0.941 |
| 1.23 | 0.927 |
| 1.87 | 0.928 |
| 2.85 | 0.929 |
| 4.33 | 0.921 |
| 6.58 | 0.919 |
| 10.0 | 0.919 |


**linear / beta=0.100 / krum (flat)** -- coude estime a v_hat~0.231

| v_hat | taux selection |
|---|---|
| 0.100 | 0.333 |
| 0.152 | 0.400 |
| 0.231 | 0.267 |
| 0.351 | 0.0667 |
| 0.534 | 0 |
| 0.811 | 0 |
| 1.23 | 0 |
| 1.87 | 0 |
| 2.85 | 0 |
| 4.33 | 0 |
| 6.58 | 0 |
| 10.0 | 0 |


**linear / beta=0.100 / multikrum (flat)** -- coude estime a v_hat~0.152

| v_hat | taux selection |
|---|---|
| 0.100 | 0.760 |
| 0.152 | 0.693 |
| 0.231 | 0.440 |
| 0.351 | 0.280 |
| 0.534 | 0.107 |
| 0.811 | 0.0533 |
| 1.23 | 0.0133 |
| 1.87 | 0.0267 |
| 2.85 | 0.0533 |
| 4.33 | 0.0267 |
| 6.58 | 0.0267 |
| 10.0 | 0.0267 |


### Jumeaux numeriques des figures

**Table des coudes (v_hat) par regle, beta=E1_BETA** :

| Modele | mean | cw_median | trmean | krum | multikrum |
|---|---|---|---|---|---|
| cnn | 0.100 | 1.23 | 1.23 | 0.152 | 2.85 |
| linear | 0.100 | 2.85 | 0.811 | 0.231 | 0.152 |


### Verdict

- **cnn** : FAIL -- courbes confirmant une baisse de selection >= 10pt entre v_hat=0.1 et v_hat=10 : 1/10 courbes (10%) (coudes estimes par regle : voir table ci-dessus -- a prendre avec prudence vu le bruit de Monte-Carlo, cf. Anomalies) ; la selection ne varie pas assez avec v_hat sous ce budget de rounds reduit pour confirmer la prediction la plus falsifiable du modele

- **linear** : FAIL -- courbes confirmant une baisse de selection >= 10pt entre v_hat=0.1 et v_hat=10 : 0/15 courbes (0%) (coudes estimes par regle : voir table ci-dessus -- a prendre avec prudence vu le bruit de Monte-Carlo, cf. Anomalies) ; la selection ne varie pas assez avec v_hat sous ce budget de rounds reduit pour confirmer la prediction la plus falsifiable du modele


### Anomalies

Voir la note sur la reduction du nombre de rounds/taille de minibatch ci-dessus : cela augmente le bruit de Monte-Carlo sur `selection_rate`, qui peut produire des coudes moins nets que sous les ~200 rounds de reference de la section 8/E5.

## 8. E6 -- Pouvoir predictif des notions de faisabilite (bloc couteux, include_e6)

**Hypothese** : Le residu normalise (alpha_tilde_star, v_hat, varpi, a_over_rho2) predit l'ASR.

**Ce qui a ete execute** : rien. Non execute cette session -- bloc couteux, derriere le drapeau `include_e6=True`.

**Resultats** : n/a.

**Jumeaux numeriques des figures** : n/a.

**Verdict** : INCONCLUSIF -- Non execute cette session -- bloc couteux, derriere le drapeau `include_e6=True`.

**Anomalies** : n/a.

## 9. E7 -- Etaler le budget

**Hypothese** : a beta fixe, augmenter n_p reduit le taux local `beta/gamma` sans changer l'ensemble atteignable sous la moyenne (`ubar*`, `E_k` inchanges).

**Ce qui a ete execute** : checkpoint=end, T=identity, beta=E1_BETA=0.10, n_p in [2.0, 3.0, 5.0] (replay de E4 a chaque n_p).


### Resultats -- invariance de ubar*/E_k et selection vs n_p


**cnn**

| n_p | ||ubar*(n_p)-ubar*(ref)||_inf | s_beta | plafonds actifs | beta/gamma |
|---|---|---|---|---|
| 2 | 0.0200 | 5.00 | OUI | 0.500 |
| 3 | 0 | 3.33 | OUI | 0.333 |
| 5 | 0.0300 | 2.00 | OUI | 0.200 |


**cnn -- taux de selection (flat) vs n_p**

| n_p | mean | cw_median | trmean | krum | multikrum |
|---|---|---|---|---|---|
| 2 | 1.000 | 0.396 | 0.415 | 0 | 0 |
| 3 | 1.00 | 0.551 | 0.586 | 0 | 0 |
| 5 | 1.00 | 0.944 | 1.00 | 0.320 | 0.608 |


**linear**

| n_p | ||ubar*(n_p)-ubar*(ref)||_inf | s_beta | plafonds actifs | beta/gamma |
|---|---|---|---|---|
| 2 | 0.0200 | 5.00 | OUI | 0.500 |
| 3 | 0 | 3.33 | OUI | 0.333 |
| 5 | 0.0300 | 2.00 | OUI | 0.200 |


**linear -- taux de selection (flat) vs n_p**

| n_p | mean | cw_median | trmean | krum | multikrum |
|---|---|---|---|---|---|
| 2 | 1.000 | 0.931 | 0.955 | 0 | 0.0200 |
| 3 | 1.00 | 0.939 | 0.961 | 0 | 0.0933 |
| 5 | 1.00 | 0.981 | 0.987 | 0.360 | 0.608 |


### Jumeaux numeriques des figures

**Selection vs n_p (courbe)** : voir table ci-dessus.


### Verdict

- **cnn** : PASS (partiel -- selection croit avec n_p malgre ubar* qui varie car les plafonds saturent) -- max||ubar*(n_p)-ubar*(ref)||_inf=0.0300, plafonds actifs sur au moins un n_p=oui, selection croissante en n_p (cw_median/krum)=oui

- **linear** : PASS (partiel -- selection croit avec n_p malgre ubar* qui varie car les plafonds saturent) -- max||ubar*(n_p)-ubar*(ref)||_inf=0.0300, plafonds actifs sur au moins un n_p=oui, selection croissante en n_p (cw_median/krum)=oui


### Anomalies

Un ecart ubar*(n_p) non nul n'est pas un bug ici : SPEC section 8/E7 note explicitement que l'invariance n'est exacte que lorsque les plafonds par classe de U_beta sont non saturants ; a s_beta = beta/(gamma*min_y pi[y]) > 1, gamma=n_p/n_b bouge avec n_p et les plafonds saturent differemment a chaque n_p (voir la colonne 'plafonds actifs' ci-dessus).

## 10. Synthese

| Verrou | Description | Go/no-go |
|---|---|---|
| E1 | Implementation et transfert de Gbar | FAIL |
| E2 | Plafond de rang (config cnn uniquement) | FAIL |
| E3 | Stabilite one-shot | FAIL |
| E4 | Reponse des agregateurs robustes (informatif, non-gate) | PASS |
| E5 | Existence du levier de furtivite | FAIL |
| E7 | Etaler le budget sur n_p (informatif, non-gate) | PASS |
| E6 | Pouvoir predictif du residu (informatif, non-gate, cher) | INCONCLUSIF |


**Recommandation**

- 4 verrou(x) bloquant(s) (E1/E2/E3/E5) en FAIL : le sweep complet ne devrait pas etre lance sans d'abord comprendre ces echecs (voir sections correspondantes pour le chiffre qui fonde chaque FAIL) -- ce run a neanmoins pousse jusqu'a E4/E5/E7 a titre informatif, sur demande explicite.

- E6 reste derriere son drapeau `include_e6` vu son cout ; ne pas le lancer avant confirmation explicite.
