# Audit — `federated_optimizing_trigger_policy` et `federated_policy_to_flips`

Référence normative : `docs/theory/threat_model.tex` (déposé le 2026-08-25). Audit en lecture
seule — **aucune correction de code fonctionnel appliquée dans cette session** ; l'annotation
(Étape 3) ne touche que docstrings, commentaires et noms de variables locales.

Voir `docs/policy_module_mapping.md` pour la table théorie ↔ code complète (Étape 1).

---

## Écarts classés par gravité

### 1. Erreur numérique silencieuse

**`lambda_poison` (donc `lambda` du run() docstring) résout à `beta` local, pas au `beta` global exigé par `lambda=beta`.**
`federated_optimizing_trigger_policy/run_module.py:477-478` (appel à `resolve_beta_and_lambda_poison`,
`federated_optimizing_trigger/utils.py:397-398` pour la résolution elle-même). Voir §2.6
ci-dessous pour l'analyse complète. Conséquence mesurable : `theta_bar_k` (la trajectoire de
référence) est réentraîné à un taux de poisoning `1/gamma` fois plus élevé que ce que la
théorie prescrit, et `v_k`'s `lambda_effective` (le taux réellement appliqué par batch dans
`_compute_step_policy`) est biaisé dans la même proportion. `gamma <= 1` toujours, donc ce biais
est **systématiquement à la hausse**, jamais à la baisse — pas un bruit aléatoire. Ce n'est pas
une divergence de nommage : c'est la MÊME variable `lambda_poison` qui alimente à la fois le
réentraînement réel (`get_poison_dataset`, `run_module.py:560`) et le calcul de `v_k` dans
l'objectif (`_compute_step_policy`, `run_module.py:121`) — l'écart se propage donc partout où
`v_k`/`theta_bar_k` sont utilisés.

### 2. Incohérence de portée entre modules

**Aucune trouvée.** C'est le résultat le plus rassurant de cet audit : la portée LOCALE de
`beta` (voir §2.2) est maintenue de façon cohérente aux quatre points vérifiés
(`project_policy_budget`, le calcul de `rho_k`, `resolve_beta_and_lambda_poison`,
`materialize_policy_flips`) — voir la table de l'Étape 1. Le point 1 ci-dessus n'est PAS une
incohérence de portée entre modules : c'est une incohérence entre la portée choisie pour `beta`
(locale) et la portée qu'exige la théorie pour `lambda` (globale) — un écart au document de
référence, pas entre les deux modules du dépôt.

### 3. Écarts de méthode documentés mais admissibles

- **Solveur simultané (b) au lieu du solveur exact (a)** (`rem:solver`) — voir §2.5. Documenté
  explicitement dans le code (`optimize_trigger_policy_step`'s docstring,
  `run_module.py:276-283`) avec les deux conséquences exactes que la théorie exige de signaler
  (surestimation de `B2`, non-applicabilité de Danskin). Diagnostic (`B2_qp`) présent et actif
  par défaut (`diag_every=50`).
- **`B2_qp` résout sur un polytope légèrement plus large que `U_loc`** (pas de plafond par
  classe côté `project_gradient`, fonction partagée) — documenté dans le code
  (`run_module.py:102`), sous-estime donc légèrement l'écart que `B2_qp` est censé quantifier.
  Direction opposée au biais du point précédent — les deux se documentent mais ne s'annulent
  pas exactement.

### 4. Simples défauts d'annotation (comblés en Étape 3)

- Aucune ancre théorique (`def:`, `prop:`, `eq:`, `rem:`) nulle part dans les deux modules avant
  cette session — vérifié par recherche exhaustive.
- Aucun docstring de tête de module (les deux `run_module.py` documentent leurs fonctions
  individuellement mais pas le fichier dans son ensemble avant `run()`).
- `varsigma_k` n'existe pas comme variable nommée séparément (fondu dans le calcul de `rho_k`).
- `source_label`/`target_label` écrits dans le `.npz` de policy mais jamais relus par
  `federated_policy_to_flips` — probablement anodin (métadonnées de provenance, la
  matérialisation des flips ne dépend pas de quelle paire est le couple source/cible du
  backdoor), mais non documenté comme tel.
- La portée de `beta` (locale) n'est pas explicitement étiquetée comme telle dans le `.npz`
  lui-même — implicite, cohérente par construction (les deux modules partagent la même
  fonction de résolution), mais un lecteur du seul fichier `.npz` ne peut pas le déduire sans
  lire le code.

---

## Étape 2 — Sept vérifications ciblées

### 2.1 — Convention des colonnes et facteur `gamma`

**Conforme.** `federated_optimizing_trigger_policy/run_module.py:168-171` :
```python
scale = torch.tensor([gamma / pi[y] for (y, c) in pairs_k], ...)
G_obj = G_k * scale
```
`G_k[:,(y,c)] = pi_y*(g_{y,c}-g_{y,y})` (convention de la fonction partagée,
`federated_optimizing_trigger/utils.py:186`) devient `G_obj[:,(y,c)] = gamma*(g_{y,c}-g_{y,y})
= gamma*Ḡ_{y,c}` — exactement la lecture locale de `eq:P` (`gamma*Ḡ(theta_bar_k)*u^i`). Les
deux facteurs (`1/pi_y` et `gamma`) sont bien présents, appliqués **une seule fois** (au
remplissage du cache `flip_grad_cache`, pas par batch), et documentés séparément comme deux
corrections distinctes dans le docstring de `_compute_step_policy`
(`run_module.py:79-94`, points (i) et (ii)).

### 2.2 — Portée du budget

**`beta` (code) est LOCAL partout, de façon cohérente aux quatre points demandés.**

1. `project_policy_budget` (`federated_optimizing_trigger_policy/utils.py:47-62`) : `‖u‖_1 <=
   beta` documenté explicitement comme "the fraction of that worker's whole shard" — LOCAL.
2. `rho_k` (`run_module.py:174`) : `beta_local * gamma * varsigma_k`. Puisque
   `beta_local = beta_theory/gamma` (portée locale), le produit vaut exactement
   `beta_theory*varsigma_k` — LOCAL en entrée, mais le résultat final correspond bien au
   `rho_k` GLOBAL de la théorie (`eq:rho`) grâce au `gamma` déjà présent dans `G_obj`.
3. `resolve_beta_and_lambda_poison` (`federated_optimizing_trigger/utils.py:352-405`, partagé) :
   docstring "beta -- the fraction of the attacker's OWN shard" — LOCAL, et sa formule
   `flip_budget = round(beta*num_poisoned*n_train/n_w)` suppose cette portée (vérifié
   algébriquement : `beta_local*n_p*n/n_w = (beta_theory/gamma)*n_p*n/n_b =
   beta_theory*n = N_flip`, cohérent avec `def:budget`).
4. `materialize_policy_flips` (`federated_policy_to_flips/utils.py:4-73`) : `n_yc =
   round(u_yc*gamma*n_train)` — exactement `rem:units`'s formule locale
   (`gamma*n*u^i_{y,z}`).

**Aucune rupture de portée entre ces quatre points.** (Le point 1 du classement par gravité,
ci-dessus, est un écart à la théorie, pas une rupture de portée ENTRE ces points.)

### 2.3 — Plafonds par classe

**Conforme.** `project_policy_budget`'s ensemble admissible est exactement `U_loc` de
`eq:Uloc` : `sum_c u_{y,c} <= pi[y]` (PAS `gamma*pi[y]`), portée locale cohérente avec 2.2.
Le docstring anticipe explicitement et écarte la mauvaise alternative
(`federated_optimizing_trigger_policy/utils.py:58`, "NOT gamma * pi[y]").

### 2.4 — Comptes de flips

**Conforme.** `materialize_policy_flips` (portée locale, `n_yc = round(u_yc*gamma*n_train)`)
correspond exactement à `rem:units`. Accord vérifié algébriquement avec
`resolve_beta_and_lambda_poison`'s `flip_budget` (voir 2.2, point 3, et le test de cohérence
permanent `federated_policy_to_flips/run_module.py:123-147`, qui compare le compte réalisé au
compte attendu à une tolérance près). Troncature par capacité **signalée, pas silencieuse** :
`materialize_policy_flips` imprime un `WARNING` explicite (`federated_policy_to_flips/utils.py:57-62`)
quand une paire `(y,c)` demande plus de flips que d'exemples de classe `y` disponibles.

### 2.5 — Jointure de l'optimisation

**Solveur (b) — descente simultanée — implémenté, conforme à `rem:solver`, documenté comme
tel.** Trace :
- `u` (paramètre `torch.nn.Parameter`-like via `requires_grad_(True)`, `utils.py:12`) reçoit
  bien un gradient réel de `B2_k`/`step_loss` : `Gu = G_obj @ u.to(...)` (`run_module.py:177`)
  ne détache jamais `u`.
- `delta` reçoit également un gradient réel de la MÊME quantité, via `v = mu_p - g_c`
  (`run_module.py:156`), `mu_p` provenant d'un `compute_batch_gradients(..., create_graph=True)`
  sur `x_poisoned` qui incorpore `delta` (`raw_to_trigger_preprocess`).
- Les deux optimiseurs (`optimizer_delta`, `optimizer_policy`) sont stepés séparément
  (`run_module.py:341-342`) après un seul `.backward()` commun sur `step_loss` (ou `L_tot`
  selon `checkpoint_backward`) — confirme la descente SIMULTANÉE, pas un solve QP interne
  détaché (aucun appel à `project_gradient` en dehors du bloc diagnostic `if run_diag:`).
- `project_gradient` (fonction partagée du solveur exact (a)) N'EST utilisée QUE dans le
  diagnostic (`run_module.py:193-199`, gardé par `run_diag`), jamais pour produire le `u`
  réellement optimisé — confirme qu'il ne s'agit PAS du solveur (a).

Les deux conséquences requises par `rem:solver` sont documentées dans le code
(`optimize_trigger_policy_step`'s docstring, `run_module.py:276-283`) : `B2` surestime
`E_k[a_k/rho_k^2]`, et Danskin ne s'applique pas. Le diagnostic (`B2_qp`, comparaison à la
valeur QP exacte sur le même `v_k`) est présent et **actif par défaut** (`diag_every=50`,
`run_module.py:265,743`) — satisfait explicitement l'exigence "activé par défaut" de
`rem:solver`.

Réserve mineure (déjà notée en §3 des écarts classés par gravité) : `B2_qp` résout sur un
polytope légèrement plus large que `U_loc` (pas de plafond par classe côté `project_gradient`),
documenté dans le code mais à garder en tête en lisant le diagnostic.

### 2.6 — `lambda = beta`

**Écart.** `lambda_poison` se résout bien mécaniquement "à `beta`"
(`federated_optimizing_trigger/utils.py:397-398`, `if lambda_poison == "beta": lambda_poison =
beta`) — littéralement vrai. Mais `beta` ici est le `beta` du CODE, dont ce même module établit
explicitement (voir 2.2, et `run_module.py:473-474` : "beta remains the LOCAL rate") qu'il
s'agit du taux LOCAL, `beta_theory/gamma` — **pas** le `beta` GLOBAL que `eq:P`'s contrainte
`lambda=beta` désigne (`def:budget`, "the primitive constraint" sur TOUTES les étiquettes
d'entraînement). Aucune multiplication par `gamma` n'intervient nulle part entre la résolution
de `lambda_poison` et son usage (vérifié par recherche exhaustive de `gamma` dans tout le
fichier, aucune occurrence près de `lambda_poison`).

Le taux **réellement passé à l'entraînement** (`get_poison_dataset`'s `lambda_target`,
`run_module.py:560`) est donc `beta_local = beta_theory/gamma`, `1/gamma` fois plus élevé que
la théorie ne le prescrit (`gamma <= 1` toujours ⟹ biais systématique à la hausse). C'est la
MÊME valeur `lambda_poison` qui détermine `lambda_effective` dans `_compute_step_policy`
(`run_module.py:121,128`, `target_count = round(lambda_poison*n_b)`), donc l'écart affecte à la
fois le réentraînement de l'expert de référence ET le calcul de `v_k` dans l'objectif — pas
seulement un effet de journalisation.

Taux réellement réalisé quand des exemples sont AJOUTÉS plutôt que réétiquetés :
`get_poison_dataset` (fonction partagée, ADD-based, `include_clean=True` par défaut) construit
`n_add` tel que `n_add/(n_base+n_add) == lambda_target` PAR CONSTRUCTION (résolution algébrique
inverse, `federated_optimizing_trigger/utils.py:479`) — donc, hors dépassement de capacité, le
taux réalisé égale bien `lambda_target` (= `lambda_poison`, avec l'écart de portée ci-dessus).
En cas de dépassement (`n_add > n_s`), le taux effectif réellement obtenu ET `beta_max` sont
explicitement journalisés (`federated_optimizing_trigger/utils.py:484-491`) — cette partie de
l'exigence 2.6 est satisfaite indépendamment de l'écart de portée.

### 2.7 — Cohérence inter-modules

**Conforme pour tout ce qui est effectivement consommé.** `num_honests`, `num_poisoned`,
`gamma`, `n_train` sont écrits dans le `.npz` (`run_module.py:858-870`) et vérifiés à la
lecture par `federated_policy_to_flips/run_module.py:58-100` (erreur bloquante sur
`n_train`, erreur bloquante sur `num_honests`/`num_poisoned`/`gamma` avec avertissement de
repli pour les anciens `.npz`). L'ordre des paires `(y,z)` est lu depuis le `.npz`
(`pairs_y`/`pairs_c`), jamais recalculé indépendamment côté aval — élimine tout risque de
divergence d'ordonnancement.

Deux nuances mineures (catégorie "défaut d'annotation", pas un écart fonctionnel) :
- `source_label`/`target_label` écrits mais jamais relus en aval — probablement sans
  conséquence (la matérialisation des flips est agnostique à quelle paire est le couple
  source/cible du backdoor), mais non confirmé comme intentionnel dans le code.
- La portée de `beta` (locale) n'est pas un champ explicite du `.npz` — implicite via le
  partage de `resolve_beta_and_lambda_poison`, cohérente en pratique, mais pas auto-documentée
  pour un lecteur du seul artefact.

---

## Fichiers annotés (Étape 3)

- `modules/federated_optimizing_trigger_policy/utils.py` — docstring de tête ajouté ; ancres
  théoriques sur `project_policy_budget`.
- `modules/federated_optimizing_trigger_policy/run_module.py` — docstring de tête ajouté ;
  ancres théoriques sur `_compute_step_policy` (G_obj, rho_k, B2, L_bd, diagnostic QP),
  `optimize_trigger_policy_step` (solveur simultané), `optimize_trigger_policy`
  (beta/lambda_poison, avec correction de l'affirmation "lambda=beta is enforced exactly" qui
  contredisait la propre portée locale documentée du module — voir §2.6 ci-dessus), `run()`.
- `modules/federated_policy_to_flips/utils.py` — ancre théorique sur `materialize_policy_flips`.
- `modules/federated_policy_to_flips/run_module.py` — docstring de tête enrichi ; ancre
  théorique sur le cross-check `num_honests`/`num_poisoned`/`gamma`/`n_train`.
