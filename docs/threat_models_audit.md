# Audit comparatif des trois modules de threat model

Date : 2026-08-23, corrections appliquées le 2026-08-24 (voir marques **[2026-08-24]**
ci-dessous — le corps du texte n'a pas été réécrit rétroactivement pour ne pas perdre la trace
de ce que l'audit avait trouvé avant correction). Portée : `modules/federated_generate_labels_trigger`,
`modules/federated_generate_labels_trigger_joint`, `modules/federated_optimizing_trigger_policy`,
et les modules dont ils dépendent directement (`federated_optimizing_trigger`,
`federated_generate_labels`, `federated_policy_to_flips`, `train_expert`, `base_utils.datasets`).

**Méthode** : chaque affirmation ci-dessous vient d'une lecture directe du code, avec référence
`fichier:ligne`. Aucune reformulation de docstring n'est prise pour argent comptant sans avoir
vérifié le code qu'elle décrit. Les écarts trouvés entre docstring et code sont dans **§0**.

---

## §0 — Écarts docstring / code trouvés pendant l'audit

| # | Fichier | Docstring dit | Code fait | Statut |
|---|---|---|---|---|
| 1 | `schemas/train_expert.toml:23` | clé `schedule_kwargs` | `modules/train_expert/run_module.py:49` lit `args.get("scheduler_kwargs", {})` | **[2026-08-24] Corrigé** : schéma aligné sur le code (`scheduler_kwargs`), PAS l'inverse — changer le code aurait changé le comportement des checkpoints déjà produits. Le schéma porte maintenant une note explicite : les experts entraînés avant cette correction ont utilisé le scheduler par défaut quel que soit `schedule_kwargs`/`scheduler_kwargs` dans leur config (`schemas/train_expert.toml:23`). |
| 2 | `schemas/federated_generate_labels_trigger_joint.toml` | Section `[OPTIONAL]` (lignes 50-69) | `modules/federated_generate_labels_trigger_joint/run_module.py:203-207` lit `trigger_constraint`, `align_kappa`, `lambda_align`, `lambda_mag`, `delta_min_frac` | **[2026-08-24] Corrigé** : les 5 clés ajoutées au schéma (`schemas/federated_generate_labels_trigger_joint.toml:72-76`), plus `checkpoint_sampling` (correction F, voir §4). Vérifié de bout en bout via `run_experiment.py` (run court, config temporaire, aucune erreur "Malformed config", point-4 check toujours à `0.000e+00`). |
| 3 | `modules/federated_generate_labels_trigger/run_module.py:1-11` (avant la correction E1, pour mémoire) | annonçait une optimisation conjointe de `(labels_syn, delta)` | `x_t_adv[is_poisoned_dev] = x_trig.detach()` (ligne 278) coupe `delta` de `param_loss` | **Déjà corrigé** (E1, session précédente) — mentionné ici pour traçabilité, la docstring actuelle (lignes 1-11, 51-68) est maintenant exacte, vérifié en §3/§5. |

Aucun autre écart trouvé : les docstrings de `federated_generate_labels_trigger_joint` et
`federated_optimizing_trigger_policy` correspondent au code relu ligne à ligne (§2/§3
ci-dessous en sont la preuve de travail).

---

## §1 — Tableau comparatif

| Axe | `federated_generate_labels_trigger` | `federated_generate_labels_trigger_joint` | `federated_optimizing_trigger_policy` |
|---|---|---|---|
| **Décision côté labels** | `labels_syn` : logits continus par exemple de `mtt_dataset.distill`, shape `(N_distill, n_classes)` (`extract_labels`, run_module.py:153-155) | Identique (run_module.py:240-242) | `u` : politique continue par paire de classes ordonnée `(y,c)`, shape `(P,)`, `P = |{(y,c): y≠c, y a des échantillons}|` (`init_policy`, policy/utils.py:4-13) |
| **Décision côté trigger** | `delta`, shape = `mu.shape` (image brute `C×H×W`) ; contrainte `‖δ‖_∞≤ε` par `clamp_` (run_module.py:370) | Identique + option `trigger_constraint` : `"penalty"` (clamp, défaut) ou `"projection"` (cône + plancher de norme, joint/utils.py:171-187) | `delta`, même shape ; `‖δ‖_∞≤ε` par `clamp_` (policy/run_module.py:389) |
| **Terme d'appariement** | `param_loss/param_dist`, `MSELoss(reduction="sum")` (`total_mse_distance`, base_utils/util.py:176) entre `student_update` et `expert` **réel post-step** (run_module.py:338-350), `×gamma` (schéma "stealth", pas l'agrégation — voir ligne gamma) | Identique en forme, mais contre `expert_next` **différentiable** (via `sgd_step` sur `grads_e`, joint/run_module.py:479-511) au lieu de `expert_params` réel | `B2_k = ‖G_obj@u − v‖²/ρ_k²` (défaut, `normalization="rho"`) ou `/(‖v‖²+ε)` (`"v"`, legacy) — policy/utils.py:221-227 |
| **Termes additionnels, poids par défaut** | `L_bd` isolé (`lambda_bd=1.0`) ; `L_pen` (stealth, `lambda_penalty=0.0`) ; `‖δ‖` (`lambda_delta=0.0`) ; `L_tv` (`lambda_tv=0.0`) — run_module.py:126-130,357-361 | Mêmes 4 + `L_align` (`lambda_align=1.0`), `L_mag` (`lambda_mag=1.0`), floor `align_kappa=0.6` — joint/run_module.py:180-207,513-534 | `L_bd` **non isolé** (dans le même `step_loss`, policy/utils.py:246) ; `L_pen`(0.0), `‖δ‖`(0.0), `L_tv`(0.0) — policy/run_module.py:753-757 |
| **Où le budget entre** | Nulle part en boucle — appliqué en aval par `federated_select_flips` (`budgets`, docstring run_module.py:81-83). Aucun paramètre `beta`/`budget` dans ce module. | Identique | **Structurel** : `beta` fixe `ρ_k` (dénominateur de `B2`, policy/utils.py:218) ET la projection dure `project_policy_budget(u,beta,pairs,pi)` (policy/utils.py:390) — les deux à chaque batch. |
| **Portée de `beta`** | N/A (pas de `beta`) | N/A | **Locale** : fraction du shard d'UN worker corrompu (`project_policy_budget` docstring, policy/utils.py:53-62 ; `resolve_beta_and_lambda_poison`, federated_optimizing_trigger/utils.py:352-405). |
| **Usage de `gamma_stealth`** | **[2026-08-24]** `args.get("gamma_stealth", ...)` — poids scalaire "stealthy/backdoor" multipliant `grand_loss` (run_module.py:112-132,374). Renommé depuis `gamma` (toujours accepté, déprécié avec `DeprecationWarning`, run_module.py:117-130) précisément pour lever la collision décrite ci-dessous. **Ce n'est PAS** `num_poisoned/(num_poisoned+num_honests)` — aucune notion d'agrégation fédérée dans ce module. | Identique (même renommage — joint/run_module.py:168-188,532) | `gamma = num_poisoned/(num_poisoned+num_honests)` (policy/run_module.py:524) — entre dans `G_obj` (échelle `gamma/pi_y`) et dans `materialize_policy_flips` en aval. Nom inchangé ici (c'est le sens original, correct pour ce module) — la collision qui existait avec les deux autres modules est résolue par leur renommage, pas par un changement ici. |
| **Usage de `pi_y`** | Aucun | Aucun | Deux usages : (i) rescaling colonnes `G_obj = (gamma/pi_y)·G_k` (policy/utils.py:209-219, corrections A+B3b) ; (ii) plafond par classe `sum_c u_{y,c}≤pi_y` dans `project_policy_budget` (policy/utils.py:51, correction B3a). Convention colonnes de `G_k` (le shared, non modifié) : `G_k[:,(y,c)] = pi_y·(g_{y,c}-g_{y,y})` (federated_optimizing_trigger/utils.py:128-160). |
| **Agrégateur** | `agg_method` utilisé dans la boucle (`agg(...)`, run_module.py:320-325) ; avertissement si `≠"mean"` (run_module.py:113-120). | Identique (joint/run_module.py:169-176,458-463) | **Aucun paramètre** `agg_method` — l'objectif `(P^mean)` suppose la moyenne par construction (`v = mu_p - g_c`, une quantité par-batch, pas une agrégation explicite sur workers). Absent du schéma. |
| **Échantillonnage checkpoints experts** | **[2026-08-24]** `checkpoint_sampling` (défaut `"uniform"`, son comportement d'origine) : `"uniform"` → `extract_experts` (`federated_generate_labels.utils.py:31-49`, tirage uniforme aléatoire) ; `"biased"` → `extract_experts_biased`, dupliquée dans `federated_generate_labels_trigger/utils.py` (correction F). | **[2026-08-24]** `checkpoint_sampling` (défaut `"biased"`, son comportement d'origine) : `"biased"` → `extract_experts_biased` (joint/utils.py:58-83, tirage biaisé exponentiel `exp(-alpha_ckpt·k)`) ; `"uniform"` → `extract_experts` (importée de `federated_generate_labels.utils`, correction F). Les deux modules ont maintenant le MÊME paramètre, valeurs possibles identiques — mettre les deux à la même valeur retire ce facteur d'une comparaison. | **Deux niveaux distincts** : `extract_experts` (`federated_optimizing_trigger.utils.py:408-415`) énumère **déterministe, exhaustif** tous les `(expert,epoch,s)` du rang — puis `sample_checkpoints` (federated_optimizing_trigger/utils.py:46-55) sous-échantillonne `num_chckpt` parmi eux, biais exponentiel, garantit le dernier inclus (policy/run_module.py:329-331). Toujours pas de `checkpoint_sampling` ici — cette architecture à deux niveaux (énumération puis sous-échantillonnage d'un pool retenu en mémoire, `expert_models`) est structurellement différente du tirage direct des deux modules `direct` et ne s'y prête pas telle quelle. |
| **Entraînement de l'expert, taux d'empoisonnement** | **Pas entraîné par ce module** — charge des checkpoints pré-existants (`input_pths`/`opt_pths`), produits ailleurs (`train_expert`, `budget` optionnel, ADD non replace — base_utils/datasets.py:710-716). | Identique | **Réentraîné à chaque step** par ce module même : `mini_train` sur `get_poison_dataset(..., lambda_target=lambda_poison, lambda_overflow)` (policy/run_module.py:597-606), taux = `lambda_poison` = `beta` par défaut (`lambda=beta`). |
| **Artefacts produits** | `labels.npy`, `true.npy`, `losses.npy`, trigger `.pt` (run_module.py:384-397) | Identiques + `metrics_log_path` JSON (joint/run_module.py:671-674), trigger `.pt` nommé `opt_trig_direct_joint_...` | Trigger `.pt`, politique `.npz` (`u,pairs,beta,n_train,...`, policy/run_module.py:885-907) + checkpoints intermédiaires du `model` réentraîné (`checkpoint_callback`, side-effect à chaque step) |
| **Artefacts consommés** | `input_pths`/`opt_pths` (checkpoints experts pré-existants) | Identiques | `expert_path`/`expert_config` (checkpoints experts pré-existants, via la 3ᵉ `extract_experts`) |
| **Module suivant** | `federated_select_flips` (budgets appliqués là) | Non documenté explicitement, mais même format de sortie → `federated_select_flips` | `federated_policy_to_flips` |

---

## §2 — L'objectif réellement optimisé, par module

### `federated_generate_labels_trigger` (indirect)

```
grand_loss = gamma_stealth · [ Σ_p MSE_sum(student_update_p, expert_p) / Σ_p MSE_sum(init_p, expert_p) + λ·‖softmax(labels_syn)-labels_init‖_1 ]
           + λ_penalty·relu(cos(δ, μ_target-μ_source) - κ) + λ_delta·‖δ‖ + λ_tv·L_tv
```
(run_module.py:337-361 ; `expert_p` = paramètre réel du modèle expert **après** `optimizer_expert.step()`, ligne 316). Aucun facteur `pi_y` ni `gamma` d'agrégation n'est appliqué où que ce soit — ce module ne construit pas de matrice de shifts du tout. Séparément, hors de `grand_loss` :
```
delta.grad += λ_bd · ∇_δ CE(f_θ(T_δ(x)), y_target)   (torch.autograd.grad isolé, run_module.py:295-303)
```

### `federated_generate_labels_trigger_joint` (couplage réel)

Même formule que ci-dessus, mais `expert_p` remplacé par `expert_next_p = expert_start_p − lr·grads_e_p` (joint/run_module.py:502-504), avec `grads_e = ∇_{θ_e} CE(f_{θ_e}(x^{adv}(δ)), y)` calculé via `torch.autograd.grad(..., create_graph=True)` (joint/run_module.py:400-402) — **différentiable en `δ`**. Plus, ajouté au même `grand_loss` (pas isolé, contrairement à `L_bd`) sous `trigger_constraint=="penalty"` (défaut) :
```
+ λ_align·relu(align_kappa − cos(δ, μ_target)) + λ_mag·relu(delta_min − ‖δ‖_2)
```
(joint/run_module.py:523-534). Toujours aucun `pi_y` ni `gamma` d'agrégation.

### `federated_optimizing_trigger_policy` (P^mean)

```
min_{‖δ‖_∞≤ε, u∈U_loc}  E_k[ ‖G_obj_k @ u − v_k(δ)‖² / ρ_k² ]  +  λ_bd · E_k[ CE(f_{θ_k}(T_δ(x)), y_target) ]
                        + λ_penalty·L_pen + λ_delta·‖δ‖ + λ_tv·L_tv
```
avec, construit **une fois par checkpoint** et mis en cache (policy/utils.py:204-219) :
```
G_obj_k = G_k · diag(scale),   scale[(y,c)] = gamma / pi_y
```
c'est-à-dire : la matrice de shifts brute `G_k` (colonnes `pi_y·(g_{y,c}-g_{y,y})`, convention du fichier partagé, federated_optimizing_trigger/utils.py:143-146) est rééchelonnée par **`gamma/pi_y`, colonne par colonne** — le seul des trois modules qui applique un facteur multiplicatif à la matrice de shifts. `ρ_k = beta·max_col_norm(G_obj_k)` (policy/utils.py:218). `U_loc = {u≥0, sum(u)≤beta, sum_c u_{y,c}≤pi_y ∀y}` (`project_policy_budget`, policy/utils.py:47-119).

---

## §3 — Trace du flux de gradient

Vérifié par lecture **et** exécution (`prelim/audit_gradient_flow.py`, §5) — voir la table de
sortie en §5, reproduite ici avec les références de code correspondantes.

| Module | Variable | Termes qui lui envoient du gradient | Mécanisme (référence) |
|---|---|---|---|
| `federated_generate_labels_trigger` | `delta` | **`L_bd` isolé uniquement** | `torch.autograd.grad(λ_bd·L_bd_cid, [delta], retain_graph=False)` (run_module.py:295-298), accumulé manuellement dans `delta.grad` (lignes 299-303). `x_t_adv[mask]=x_trig.detach()` (ligne 278) coupe `param_loss` de `delta` avant que `loss_e`/`loss_s` ne le voient. |
| — | `labels_syn` | `param_loss` (via `loss_s`) | `loss_s = clf_loss(student_model(x_d), softmax(y_d))` avec `y_d=labels_syn[idx]` (run_module.py:308), `grads_s = torch.autograd.grad(loss_s, student_params, create_graph=True)` (ligne 309-311) → `student_update` → `param_loss`. Non ré-exécuté dans le script §5 (motif structurellement identique au chemin `delta` du module joint, non dupliqué — voir note dans le script). |
| — | `expert_params` (réel) | `loss_e.backward()`, agrégé sur TOUS les clients | **[2026-08-25, corrigé]** `expert_grad_buf` collecte le `.grad` de chaque client (honnête et empoisonné), puis `agg(expert_params, expert_grad_buf, agg_method, f=num_poisoned)` (symétrique à l'agrégation déjà appliquée côté student) est appelée AVANT `optimizer_expert.step()`. Auparavant : `expert_model.zero_grad()` avant CHAQUE client écrasait la contribution des précédents — seul le DERNIER client (toujours un client empoisonné) survivait jusqu'à `optimizer_expert.step()`, une mise à jour mono-client, non fédérée. Voir Annexe 2. |
| `federated_generate_labels_trigger_joint` | `delta` | **`param_loss` (via l'agrégat fédéré) ET `L_bd` isolé** | `param_loss` : chaque client empoisonné ajoute son `grads_e` (`create_graph=True`) NON détaché à `expert_grad_buf` ; `agg_expert_grads = agg(expert_params, expert_grad_buf, agg_method, f=num_poisoned)` (symétrique à `agg_student_grads`) → `expert_next_param` (`sgd_step`) → `param_loss` → `grand_loss.backward()`. `L_bd` : identique au module indirect, `retain_graph=True` (nécessaire — `x_trig` partagé avec `x_t_adv`). Les deux s'accumulent (`.grad` jamais remis à zéro entre les deux, une seule fois par batch). |
| — | `expert_params` | AUCUN backward direct, AUCUNE mise à jour réelle | **[2026-08-25, corrigé]** `expert_params[i].grad` n'est plus JAMAIS assigné explicitement dans ce module : `agg(expert_params, ...)` le positionne comme effet de bord, mais rien ne le lit plus ensuite — `optimizer_expert.step()` et le `.load_state_dict()` qui l'alimentait ont été supprimés (`expert_model` est de toute façon rechargé depuis le disque au batch suivant). Voir Annexe 2 pour l'analyse complète et pourquoi ceci rend le "point-4 check" (comparaison à un vrai pas d'optimiseur) obsolète — remplacé par `prelim/tests/test_sgd_step.py`. |
| `federated_optimizing_trigger_policy` | `u` | `B2_k` (seul terme où `u` apparaît) | `Gu = G_obj @ u` (policy/utils.py:221), `B2_k` en dépend directement ; `u.grad` peuplé par `step_loss.backward()` (ligne 247) ou `L_tot.backward()` (ligne 383) selon `checkpoint_backward`. |
| — | `delta` | `B2_k` (via `v(δ)`) et `L_bd_k` (même `step_loss`, PAS isolé) | `v = mu_p - g_c` où `mu_p` vient de `compute_batch_gradients(..., create_graph=True)` sur `x_poisoned` (contient `T_δ(x)`, policy/utils.py:186-198) ; `L_bd_k = CE(logits_p[mask], y_poison[mask])`, `logits_p` de la MÊME forward pass que `mu_p` (ligne 195-198) — donc `delta` reçoit du gradient combiné dans le MÊME `step_loss.backward()`, contrairement aux deux modules `direct`. |

**Point non trivial découvert en construisant §5, absent de toute docstring existante (historique
— voir mise à jour 2026-08-25 ci-dessous)** : dans le module joint, `expert_params[i]` est une
feuille qui participe AUSSI au graphe `create_graph=True` de `grads_e` (via `loss_e`). Si
`.grad` est déjà positionné avant `grand_loss.backward()`, PyTorch's `AccumulateGrad`
l'additionne **en place** — y compris un `.detach().clone()`, dont le stockage propre se
retrouve donc modifié. Vérifié empiriquement (`prelim/tests/test_joint_accumgrad_hazard.py`,
3/3 PASS). **[2026-08-25]** Ce risque ne s'applique plus à aucun chemin réellement exécuté dans
ce module : `expert_params[i].grad` n'est plus jamais assigné explicitement (voir Annexe 2) —
le test est conservé comme épinglage général du comportement PyTorch (son propre en-tête le
dit maintenant explicitement), pas comme garde-fou d'un chemin de code encore actif.

---

## §4 — État des corrections

| Correction | Module(s) | Statut | Référence | Note |
|---|---|---|---|---|
| **A** — `1/pi_y` sur les colonnes de shifts | `federated_optimizing_trigger_policy` | **Fait** | policy/utils.py:209-215 (`scale = gamma/pi[y]`) | Combiné avec B3b dans la même ligne — voir §2. |
| **B3a** — plafonds par classe `sum_c u_{y,c}≤pi_y` (portée locale, sans `gamma`) | `federated_optimizing_trigger_policy` | **Fait** | policy/utils.py:47-119 (`project_policy_budget`) | Docstring de la fonction affirme explicitement "NOT gamma*pi[y]" (ligne 58). |
| **B3b** — facteur `gamma` niveau agrégat | `federated_optimizing_trigger_policy` | **Fait** | policy/utils.py:212-219 ; `gamma` calculé run_module.py:524 | |
| **B3c** — `round(u·gamma·n_train)` | `federated_policy_to_flips` | **Fait** | policy_to_flips/utils.py:49 | |
| **C** — normalisation `rho_k` | `federated_optimizing_trigger_policy` | **Fait** | policy/utils.py:218,224-227 | Les deux normalisations (`rho`,`v`) toujours calculées et loguées (`B2_rho`,`B2_v`), policy/utils.py:265-266. |
| **D** — diagnostic QP intérieur | `federated_optimizing_trigger_policy` | **Fait** | policy/utils.py:236-243 (`diag_every`, `B2_qp`) | QP sans plafonds par classe (budget global seul) — limite documentée dans le code (policy/utils.py:143-145). |
| **E1** — docstring `federated_generate_labels_trigger` corrigée | `federated_generate_labels_trigger` | **Fait** | run_module.py:1-11,51-68 | Vérifié exact par §3/§5. |
| **E2** — couplage réel | `federated_generate_labels_trigger_joint` | **Fait** | joint/run_module.py:61-115,347-405,476-511 | Vérifié empiriquement (point-4 check à 0.000e+00) et par §5. |
| **F** — avertissement `agg_method≠"mean"` | `federated_generate_labels_trigger`, `federated_generate_labels_trigger_joint` | **Fait** | run_module.py:113-120 ; joint/run_module.py:169-176 | |
| **F** — paramètre `checkpoint_sampling` dans les deux modules | `federated_generate_labels_trigger`, `federated_generate_labels_trigger_joint` | **[2026-08-24] Fait** | run_module.py (les deux), défauts = comportement d'origine de chaque module (`"uniform"`/`"biased"`) | Voir §1. Pas de régression : défauts inchangés. `extract_experts_biased` dupliquée (pas importée cross-module) dans `federated_generate_labels_trigger/utils.py`, pour garder les deux modules indépendamment autonomes. |
| **G** — commentaire rechargement checkpoint / BatchNorm | `federated_generate_labels_trigger` | **Fait, en forme de non-finding** | — | Investigué (session précédente) : `expert_model.load_state_dict(checkpoint)` recharge l'état complet (poids + buffers BN) à CHAQUE batch (run_module.py:215), et une passe train-mode sur `x_trig` normalise avec les stats du batch courant (pas les running stats) — aucun effet observable, vérifié numériquement. Pas de commentaire persistant ajouté au code ; seule trace = le rapport de la session précédente. |
| **H3** — `n_train` incohérent = erreur bloquante | `federated_policy_to_flips` | **Fait** | policy_to_flips/run_module.py:58-70 (`raise ValueError`) | |
| **[2026-08-24 nouveau]** — `num_honests`/`num_poisoned`/`gamma` écrits dans le `.npz`, vérification croisée à `federated_policy_to_flips` | `federated_optimizing_trigger_policy`, `federated_policy_to_flips` | **Fait** | policy/run_module.py:898-911 (écriture) ; policy_to_flips/run_module.py (lecture + `raise ValueError` en cas de désaccord) | `.npz` produits avant cette correction : avertissement (pas d'erreur), vérification passée. Testé synthétiquement (3 cas : ancien format, concordant, discordant) — les trois comportements confirmés. |
| **anti-effondrement** — `L_align` plancher | `federated_generate_labels_trigger_joint` | **Fait** | joint/utils.py:104-112 ; joint/run_module.py:530 | **[2026-08-24]** Accessible via TOML depuis la correction du schéma (§0.2). |
| **anti-effondrement** — plancher magnitude `‖δ‖_2` | idem | **Fait** | joint/utils.py:115-122 | **[2026-08-24]** idem ; MAIS voir §9 — `delta_min_frac=0.5` calculé sur `‖δ_init‖` **avant** clamp epsilon, potentiellement inatteignable une fois `δ` contraint (constaté pendant la vérification P0 et pendant le balayage §annexe : `mag_active_rate=1.00` en continu). Non corrigé — signalé. |
| **anti-effondrement** — `lambda_delta=0` par défaut | idem | **Fait** | joint/run_module.py:184-190 | Identique au défaut des deux autres modules, mais commenté ici comme load-bearing. |
| **anti-effondrement** — variante projection (cône) | idem | **Fait** | joint/utils.py:125-187 | Pas de test unitaire synthétique dédié écrit (la session précédente a été interrompue avant — voir historique) ; correction non revalidée par un test dans cette session (hors périmètre : lecture seule). |
| **anti-effondrement** — instrumentation (`cos`, normes, `expert_asr`, termes séparés) | idem | **Fait** | joint/run_module.py:603-662 | `align_active_rate`/`mag_active_rate` (fenêtre glissante 50, ligne 306-308,621-628) — analogue du `hinge_rate` de la table policy. |
| **anti-effondrement** — régime stable identifié pour `expert_asr` | idem | **Non fait** | — | Voir Annexe 2 (run de référence Étape 5, 2026-08-25) : `expert_asr` ne collapse plus à un ZÉRO exact et invariant comme sous l'ancien balayage (epsilon=1.0), mais reste faible (moyenne 5.4%, sous le hasard ~10%) sur un run court (68 batches) — aucun régime stable identifié, mais la signature d'échec a changé qualitativement. |
| **[2026-08-25 nouveau]** — agrégation fédérée du gradient expert (mono-client → `agg()`) | `federated_generate_labels_trigger`, `federated_generate_labels_trigger_joint` | **Fait** | run_module.py (les deux, voir §3) | Baseline `federated_generate_labels` NON touché (arbitrage utilisateur explicite) — bug identique présent, avec une tentative de correction identique déjà écrite puis mise en commentaire dans son code (utils.py:171-176 de ce module-là). Voir Annexe 2. |
| **[2026-08-25 nouveau]** — suppression du "point-4 check" / pas d'optimiseur réel côté expert | `federated_generate_labels_trigger_joint` | **Fait** | run_module.py (voir §3) | Remplacé par `prelim/tests/test_sgd_step.py` (test synthétique dédié, 13/13 PASS) — voir Annexe 2. Documente aussi un bug dormant, non corrigé (fichier partagé, hors périmètre) : `sgd_step` ne persiste pas son `momentum_buffer` entre appels successifs sur le même `opt_state` — inoffensif ici (chaque batch recharge un `opt_state` frais et n'appelle `sgd_step` qu'une fois par paramètre). |
| **[2026-08-25 nouveau]** — instrumentation H1 corrigée (`delta_init` post-clamp, ASR sur ensemble fixe) | `federated_generate_labels_trigger_joint` | **Fait** | run_module.py (voir §3, Annexe 2) | `delta_init` capturé après le premier clamp (au lieu de l'init brute pré-clamp) ; `expert_asr`/`expert_asr_frozen` mesurés sur 256 exemples fixes de la classe source au lieu du sous-ensemble empoisonné (petit, bruité) de chaque batch. Vérifié : `cos_delta_to_init==1.0`, `delta_drift_l2==0.0` exactement au batch de capture (assertion en code + vérifié sur run réel). |

---

## §5 — Vérification exécutable

Script : `prelim/audit_gradient_flow.py`. Tenseurs synthétiques uniquement (MLP jouet
`Linear(6,8)→ReLU→Linear(8,3)`, `C=3`, `N=5`), aucun dataset, aucun entraînement, < 1s.

**Limite déclarée** : les trois modules ne sont pas isolables tels quels sans leur pipeline
complet (`get_matching_datasets`, `MTTDataset`, checkpoints experts réels sur disque). Le
script reproduit fidèlement le **motif d'autograd caractéristique** de chaque module
(`.detach()` ou non, `torch.autograd.grad(create_graph=True)` isolé ou non, ordre
`backward()`/assignation `.grad`) sur des tenseurs jouets, avec référence ligne-à-ligne vers le
code réel pour chaque étape reproduite — voir les commentaires en tête de chaque section du
script. `labels_syn` (indirect) n'est pas re-dérivé séparément : son chemin réel
(`loss_s=clf_loss(student_model(x_d), softmax(y_d))`, `create_graph=True`) est structurellement
identique au chemin `delta` déjà vérifié pour le module joint — dupliquer le test n'ajoutait
pas d'information nouvelle. Idem pour `federated_optimizing_trigger_policy` : `_compute_step_
policy` n'a pas de dépendance dataset propre (`G_obj`,`u`,`v` sont déjà des tenseurs `(D,P)`/
`(P,)`/`(D,)` une fois les checkpoints chargés) — la formule exacte (policy/utils.py:221-227)
est reproduite directement, ce n'est pas un motif de remplacement mais l'objectif lui-même.

**Piège rencontré en écrivant ce script, pour mémoire** : la première version initialisait
`delta=0`. Avec `delta=0`, l'entrée triggée == l'entrée propre, et `expert`/`student` partant
des mêmes poids, `expert_next` et `student_update` coïncidaient exactement en ce point précis
— annulant artificiellement `d(param_loss)/d(delta)` (facteur `(A-B)=0` dans la dérivée du MSE)
alors que la dépendance existe bel et bien ailleurs (vérifié séparément). Corrigé en
initialisant `delta` à une valeur non nulle. Un artefact du script de test, pas du code audité
— documenté ici pour que quiconque réexécute ou étend ce script ne retombe pas dedans.

**[2026-08-24] Corrigé** : la version précédente utilisait `"no"` à la fois pour "testé, gradient
nul" et pour "non testé" (le cas `labels_syn`) — une table qui affiche `"no"` pour un cas non
testé se lit comme un résultat négatif alors que c'est une absence de preuve. Trois valeurs
désormais : `YES` / `no` / `not_tested`, portées par un paramètre explicite de `record(...)`
plutôt qu'inférées d'un tenseur potentiellement `None` pour deux raisons différentes.

Sortie (recopiée de l'exécution réelle) :

```
module                                   variable     term                       status
federated_generate_labels_trigger        delta        param_loss (grand_loss)    no
federated_generate_labels_trigger        delta        L_bd (isolated)            YES
federated_generate_labels_trigger        labels_syn   param_loss (via loss_s)    not_tested
federated_generate_labels_trigger_joint  delta        param_loss (grand_loss)    YES
federated_generate_labels_trigger_joint  delta        L_bd (isolated)            YES
federated_generate_labels_trigger_joint  delta        TOTAL delta.grad (both terms) YES
federated_optimizing_trigger_policy      u            B2 (via G_obj@u)           YES
federated_optimizing_trigger_policy      delta        B2 (via v(delta))          YES
```

Confirme exactement §3 : l'indirect ne couple `delta` que via `L_bd` ; le joint le couple par
les deux voies ; la politique n'a pas de variable "labels" séparée, `u` en tient lieu.
`labels_syn` reste explicitement `not_tested`, pas `no`.

---

## §6 — Substitut CPU à la mesure mémoire

Script : `prelim/audit_memory_proxy.py`. Exécute le motif exact du pas expert du module joint
(modèle **r32p réel**, checkpoint réel déjà sur disque `model_1_10.pth`, batch 32×3×32×32
aléatoire — pas de téléchargement, les poids suffisent), une fois avec `create_graph=True`
(E2, tel que livré), une fois sans (`.backward()` classique, motif du module indirect) —
**chacun dans son propre sous-processus** : la RSS d'un seul processus ne peut que croître
(haut-de-crue), donc mesurer les deux variantes séquentiellement dans le même processus
aurait rapporté le max des deux, pas chacune séparément.

**[2026-08-24] Corrigé** : la première version comparait des RSS **totales**, dominées par une
constante de plusieurs centaines de Mo (interpréteur, torch, poids du modèle) sans rapport avec
`create_graph`. Le script mesure maintenant une **baseline** (juste après chargement du modèle
et construction du batch, AVANT tout forward) et un **pic** (après backward), et rapporte
l'incrément `pic − baseline` pour chaque variante — c'est cet incrément, pas le total, qui
isole le coût réel de `create_graph`. Le comptage de nœuds du graphe (qui donnait "5" pour un
forward complet de r32p, manifestement faux) a été abandonné plutôt que corrigé — la RSS seule
suffit et un chiffre faux ne devait pas rester publié.

Résultat (Darwin arm64, exécution réelle) :

| Métrique | `create_graph=True` | `create_graph=False` |
|---|---|---|
| Baseline (modèle+batch chargés, avant forward) | 342.55 Mo | 339.12 Mo |
| Pic (après backward) | 725.52 Mo | 486.78 Mo |
| **Incrément (pic − baseline)** | **382.97 Mo** | **147.66 Mo** |

Ratio sur les incréments (la grandeur qui compte) : **2.59×**. Ratio sur les totaux (celui
publié dans la version précédente de cet audit) : 1.49× — la prédiction qu'un ratio calculé sur
les totaux sous-estimait fortement le coût réel de `create_graph` est confirmée : la moitié
environ de l'écart observé sur les totaux était noyée dans la constante partagée.

**Avertissement explicite, tel que demandé** : ceci reste un **proxy CPU**, pas la mesure GPU
demandée (`torch.cuda.max_memory_allocated`, toujours indisponible sur cette machine). La RSS
processus est un haut-de-crue global soumis à la fragmentation de l'allocateur mémoire du
système (macOS/glibc), pas à l'allocateur "caching" propre à CUDA que la question visait
réellement — l'ordre de grandeur (**~2.6×**, pas 1.5×) doit maintenant peser sur la décision
r18 différemment de ce que rapportait la version précédente : un facteur ~2.6× sur un modèle
~24× plus gros (r18 ≈ 11M paramètres vs r32p ≈ 0.46M) reste un ordre de grandeur informatif
mais insuffisant à lui seul pour garantir l'absence d'OOM sur r18 sans la mesure GPU réelle
(voir §9).

---

## §7 — Comparabilité

**[2026-08-24] Reformulation.** La version précédente de cette section cherchait une
comparabilité "à un seul facteur" entre `federated_optimizing_trigger_policy` et les deux
modules `direct`. Ce n'est pas l'objectif : la paramétrisation (`u` par paire de classes vs
`labels_syn` par exemple) **est** la différence entre les deux formulations théoriques — la
neutraliser reviendrait à effacer la contribution elle-même, pas à contrôler un facteur de
nuisance. Ce qui doit être aligné, ce sont les facteurs de nuisance indépendants de la
formulation : l'échantillonnage des checkpoints, la provenance et le taux d'empoisonnement des
experts, le budget, l'agrégateur. La table ci-dessous est organisée par facteur de nuisance,
pas par paire de modules.

| Facteur de nuisance | `federated_generate_labels_trigger` | `federated_generate_labels_trigger_joint` | `federated_optimizing_trigger_policy` | Alignable aujourd'hui ? |
|---|---|---|---|---|
| **Échantillonnage des checkpoints** | `checkpoint_sampling` (`"uniform"` défaut) | `checkpoint_sampling` (`"biased"` défaut) | `sample_checkpoints` sur un pool énuméré par `extract_experts` (déterministe, exhaustif) — pas de paramètre `checkpoint_sampling` | **Partiellement [2026-08-24]** : les deux modules `direct` s'alignent maintenant l'un sur l'autre en fixant `checkpoint_sampling` à la même valeur (§1, §4). `federated_optimizing_trigger_policy` reste sur une architecture à deux niveaux (énumération puis sous-échantillonnage d'un pool retenu en mémoire) structurellement différente — l'aligner demanderait de changer CETTE architecture, pas juste un paramètre. |
| **Provenance des checkpoints experts** | Pré-existants, chargés via `input_pths`/`opt_pths` (produits par `train_expert`, ailleurs) | Identique | Deux populations distinctes dans le MÊME module : `expert_models` (pré-existants, `expert_path`/`expert_config`, comme les deux modules `direct`) ET `model` (réentraîné par ce module même, checkpoints sauvegardés en side-effect) | **Oui, pour la partie `expert_path`** : pointer les trois modules vers des checkpoints produits par le MÊME `train_expert` (même `budget`, voir ligne suivante) rend cette source comparable. La seconde population de `federated_optimizing_trigger_policy` (`model` réentraîné) n'a pas d'équivalent côté `direct` — ce n'est pas un facteur à aligner, c'est une différence de conception à documenter. |
| **Taux d'empoisonnement des experts** | Fixé par le `budget` de `train_expert` au moment où les checkpoints ont été produits (pas revisité par ce module) | Identique | `model` réentraîné à `lambda_poison=beta` par ce module même, à CHAQUE step | **Non pour `model`** (structurel : policy réentraîne, les deux `direct` ne réentraînent jamais) ; **oui pour `expert_path`** si le `train_expert` en amont utilise un `budget` cohérent avec le `beta`/`flip_budget` de policy (aucune vérification croisée automatique de ceci — à faire à la main). |
| **Budget** | Aucun — appliqué en aval par `federated_select_flips` | Identique | Structurel : `beta` fixe `ρ_k` ET la projection dure `project_policy_budget` (§1) | **Non alignable au sens strict** : les deux `direct` n'ont pas de notion de budget en boucle du tout ; le comparer au budget structurel de policy reviendrait, encore une fois, à comparer les formulations plutôt qu'à neutraliser un facteur commun. Seul le budget AVAL (`federated_select_flips`, appliqué aux deux modules `direct`) est un réglage véritablement partagé — s'assurer qu'il correspond au même ordre de grandeur que le `beta` de policy est la seule forme d'alignement possible ici. |
| **Agrégateur** | `agg_method`, avertissement si `≠"mean"` | Identique | Aucun paramètre — `(P^mean)` suppose la moyenne par construction | **Oui, trivialement** : fixer `agg_method="mean"` dans les deux `direct` (déjà le défaut des deux) rend ce facteur non-croisé avec policy. |

**Ce qui reste une différence de formulation, pas un facteur de nuisance à aligner** (pour
mémoire, afin de ne pas les re-signaler comme des "manques") : la paramétrisation de la
décision (`u` vs `labels_syn`), le couplage direct/indirect (E1/E2), et la présence d'un second
modèle réentraîné dans `federated_optimizing_trigger_policy`. Les deux modules `direct` entre
eux, en revanche, ne diffèrent QUE par E1/E2 et (jusqu'à cette correction) l'échantillonnage
des checkpoints — une fois ce dernier aligné, une comparaison indirect-vs-joint isole
correctement l'effet du couplage.

---

## §8 — Ce qui manque pour exécuter chaque chaîne de bout en bout

### `federated_generate_labels_trigger` (indirect)

Séquence : `train_expert` (produit les checkpoints `.pth`/`_opt.pth`) → `federated_generate_labels_trigger` → `federated_select_flips` → `federated_train_user`.

Clés TOML obligatoires (`schemas/federated_generate_labels_trigger.toml`, section non-OPTIONAL) :
`input_pths`, `opt_pths`, `output_dir`, `output_dir_trigger`, `expert_model`, `dataset`,
`source_label`, `target_label`, `epsilon`, `lr_delta`, `lambda_bd`.

Artefacts intermédiaires : checkpoints `train_expert` (chemins `input_pths`/`opt_pths` doivent
pointer dessus — format à 3 `{}`, ex. `.../r32p_1xs/{}/model_{}_{}.pth`, vérifié fonctionnel
cette session sur un run de fumée réel) ; `labels.npy`/`true.npy` en sortie, consommés par
`federated_select_flips` (`input_label_glob`/`true_labels`).

**Ce qui n'existe pas encore** : rien d'identifié côté code — chaîne exécutable telle quelle
(vérifié : un run de fumée complet a réussi cette session, avant l'audit).

### `federated_generate_labels_trigger_joint`

Même séquence, même schéma de clés obligatoires (identique nom pour nom au module indirect,
`schemas/federated_generate_labels_trigger_joint.toml`).

**Ce qui manque, concret et bloquant** : `trigger_constraint`, `align_kappa`, `lambda_align`,
`lambda_mag`, `delta_min_frac` ne sont pas dans le schéma (§0.2) — `run_experiment.py` rejette
toute config qui les fixe. Utilisable uniquement en appelant `run()` directement (comme les
scripts de fumée de la session précédente), pas via la chaîne normale
`python run_experiment.py <experiment_name>`.

### `federated_optimizing_trigger_policy`

Séquence : (`train_expert` optionnel, pour `expert_budget`) → `federated_optimizing_trigger_policy`
(réentraîne lui-même un `model` à chaque step ET consomme des checkpoints `expert_path` déjà
existants) → `federated_policy_to_flips` → `federated_train_user`.

Clés TOML obligatoires (`schemas/federated_optimizing_trigger_policy.toml`) : `dataset`,
`model`, `source_label`, `target_label`, `optim_kwargs`, `scheduler_kwargs`, `output_dir`,
`output_dir_trigger`, `device`, `lambda_bd`, `lambda_penalty`, `lambda_delta`, `lambda_tv`,
`kappa`, `epsilon`, `lr_delta`, `lr_policy`, `n_steps`, `epochs`, `checkpoint_backward`, `beta`,
`num_honests`, `num_poisoned`, `lambda_poison`, `lambda_overflow`, `alpha_ckpt`, `num_chckpt`,
`expert_path`.

Artefacts intermédiaires : checkpoints `expert_path`/`expert_config` (pré-existants, PAS
produits par ce module) ; checkpoints du `model` réentraîné (side-effect, `output_dir`) ;
sortie `.npz` (`u,pairs,beta,n_train,num_honests,num_poisoned,gamma,...`) consommée par
`federated_policy_to_flips` (`policy_path`) qui exige `num_honests`/`num_poisoned`
**identiques** à ceux utilisés ici. **[2026-08-24] Corrigé** : `num_honests`/`num_poisoned`/
`gamma` sont maintenant écrits dans le `.npz` (policy/run_module.py:898-911), et
`federated_policy_to_flips` lève une `ValueError` si sa propre config diverge de ce que la
politique a réellement utilisé (au lieu de recalculer `gamma` en silence de son côté et de ne
jamais comparer). Les `.npz` produits avant cette correction n'ont pas ces champs — la
vérification est alors sautée avec un avertissement, pas une erreur.

**Ce qui manque** : rien d'identifié côté clés/schéma (vérifié complet, §0). Vérifié
fonctionnel cette session (run de fumée `beta=0.05`, checkpoint réel).

---

## §9 — Points ouverts

- **Mesure mémoire GPU réelle** (`torch.cuda.max_memory_allocated`, `create_graph` avec/sans,
  module joint, r32p puis r18) — toujours bloquée par l'absence de CUDA sur cette machine. Le
  proxy CPU (§6), corrigé pour mesurer des incréments plutôt que des totaux, donne maintenant
  **~2.6×** (pas 1.5×) — ordre de grandeur revu à la hausse, toujours pas un substitut à la
  mesure réelle. Débloqué par : accès à une machine avec GPU.
- **`delta_min_frac=0.5` potentiellement inatteignable** [2026-08-24, nouveau] : `delta_min`
  est calculé comme une fraction de `‖δ_init‖_2` **avant** que `δ` ne soit jamais contraint par
  `clamp_(-epsilon,epsilon)` (joint/run_module.py, calcul juste après `init_delta`, `strength=
  6.0`). Sous `epsilon=1.0`, la norme L2 maximale atteignable après clamp est
  `epsilon·sqrt(numel) ≈ 55.4` (image `3×32×32`) ; `delta_min` observé dans la vérification
  P0 et le balayage §annexe valait `115.7` — **supérieur au maximum atteignable, donc
  structurellement inatteignable quel que soit l'entraînement**. `mag_active_rate=1.00` en
  continu sur tous les runs observés le confirme. Non corrigé (hors périmètre des corrections
  demandées cette session) mais très probablement pertinent pour expliquer pourquoi
  `expert_asr` ne se stabilise pas — voir le rapport du balayage en annexe. Débloqué par :
  recalculer `delta_min` comme fraction de `epsilon·sqrt(numel)` (le maximum atteignable) plutôt
  que de `‖δ_init‖_2` (une quantité pré-clamp sans rapport avec ce qui est réellement
  accessible), ou choisir `delta_min_frac` en connaissance de cette borne.
- **Convention de `prelim/`** — mise de côté explicitement cette session (demande utilisateur).
  `varsigma`/`rho`/`v_hat`/`a_k` y restent, à la connaissance de cet audit, non revérifiés
  depuis la correction A (`1/pi_y`) appliquée dans `modules/` — `prelim/prelim_lib.py` a été
  lu lors d'une session antérieure et semblait DÉJÀ correct sur ce point (scope local/agrégat
  explicite, `solve_qp(scope=...)`), mais ceci n'a pas été revérifié pendant CETTE session
  (portée : `modules/` uniquement). Débloqué par : relire `prelim/prelim_lib.py` avec la même
  rigueur fichier:ligne que cet audit, une fois la session dédiée à `prelim/` reprise.
- **§7, paire indirect-vs-policy** — la question de savoir si "`L_bd` isolé" (indirect) et
  "`L_bd` dans le même `step_loss`" (policy) doivent compter comme la même construction
  expérimentale est un choix de design non tranché par le code ; indéterminé sans décision
  humaine explicite.
- **Voir l'annexe (balayage anti-effondrement, 2026-08-24) pour le point "régime stable pour
  `expert_asr`"** — traité par le balayage P3, résultat rapporté séparément ci-dessous plutôt
  que réécrit ici.

---

## Annexe — Balayage anti-effondrement (2026-08-24, P3) — **[2026-08-25] INVALIDÉE, voir Annexe 2**

> **Toutes les mesures ci-dessous (`expert_asr`, `param_loss`, etc.) ont été produites AVANT la
> correction de l'agrégation fédérée du gradient expert (Annexe 2, 2026-08-25) : le module joint
> avançait alors son "expert" sur le gradient d'un SEUL client (le dernier traité), pas un
> agrégat fédéré. Ne plus citer ce balayage comme référence pour le comportement d'`expert_asr`
> — conservé ci-dessous uniquement comme trace historique (le diagnostic `delta_min`
> inatteignable, lui, reste valide et n'est pas remis en cause par la correction d'agrégation :
> c'est un bug indépendant, toujours non corrigé, voir §9).**

Script : `prelim/run_anticollapse_sweep.py`. Run court (1 graine, `train_pct=0.02`, `iterations=1`,
`num_honests=num_poisoned=1`, mêmes checkpoints `r32p_1xs` réutilisés partout cette session),
`epsilon=1.0`. Grille : `lambda_align ∈ {0.1, 1, 10}` × `align_kappa ∈ {0.3, 0.6}` sous
`trigger_constraint="penalty"` (6 cellules), puis `trigger_constraint="projection"` à
`align_kappa` fixé à la valeur gagnante de la grille (`0.3`, meilleure `expert_asr_median` —
égalité en fait, voir plus bas).

### Table configuration × résultat

| Config | `expert_asr` final | `expert_asr` médiane | Terme d'appariement final | `cos(δ,μ_target)` final | `‖δ‖_2` final |
|---|---|---|---|---|---|
| penalty, λ_align=0.1, κ=0.3 | 0.0000 | 0.0000 | 0.6215 | −0.0162 | 54.43 |
| penalty, λ_align=0.1, κ=0.6 | 0.0000 | 0.0000 | 0.2912 | −0.0197 | 54.43 |
| penalty, λ_align=1.0, κ=0.3 | 0.0000 | 0.0000 | 0.3535 | −0.0150 | 54.44 |
| penalty, λ_align=1.0, κ=0.6 | 0.0000 | 0.0000 | 0.4911 | −0.0057 | 54.43 |
| penalty, λ_align=10.0, κ=0.3 | 0.0000 | 0.0000 | 0.2589 | **0.0281** | 54.38 |
| penalty, λ_align=10.0, κ=0.6 | 0.0000 | 0.0000 | 0.2756 | 0.0280 | 54.39 |
| **projection, κ=0.3** | 0.0000 | 0.0000 | 1.4233 | **0.3000** (exact) | **115.72** |

**Résultat net : `expert_asr` collapse à exactement 0 (finale ET médiane) dans les 7
configurations, sans exception.** Aucun régime stable n'a été trouvé dans cette grille.

### Courbes par step (5 points échantillonnés sur 68 batches), 3 configurations

**`penalty, λ_align=1.0, κ=0.6`** (config "milieu de grille") :

| step | `expert_asr` | terme d'appariement | `L_bd` | `cos_target` | `‖δ‖_2` | `L_mag` |
|---|---|---|---|---|---|---|
| 0 | 1.000 | 0.587 | 1.229 | −0.00086 | 52.53 | 63.19 |
| 17 | 0.000 | 0.674 | 13.131 | −0.00102 | 52.90 | 62.83 |
| 34 | 0.000 | 0.507 | 0.000 | −0.00220 | 53.50 | 62.22 |
| 51 | 0.000 | 0.523 | 12.939 | −0.00431 | 53.98 | 61.74 |
| 67 | 0.000 | 0.491 | 12.792 | −0.00569 | 54.43 | 61.29 |

**`penalty, λ_align=10.0, κ=0.3`** (pression anti-effondrement la plus forte de la grille) :

| step | `expert_asr` | terme d'appariement | `L_bd` | `cos_target` | `‖δ‖_2` | `L_mag` |
|---|---|---|---|---|---|---|
| 0 | 0.667 | 0.886 | 1.489 | −0.00115 | 52.53 | 63.19 |
| 17 | 0.000 | 0.780 | 13.045 | 0.00607 | 52.87 | 62.86 |
| 34 | 0.000 | 0.348 | 13.138 | 0.01382 | 53.41 | 62.32 |
| 51 | 0.000 | 0.680 | 12.839 | 0.02088 | 53.90 | 61.82 |
| 67 | 0.000 | 0.259 | 0.000 | 0.02809 | 54.38 | 61.34 |

**`projection, κ=0.3`** :

| step | `expert_asr` | terme d'appariement | `L_bd` | `cos_target` | `‖δ‖_2` | `‖δ‖_∞` |
|---|---|---|---|---|---|---|
| 0 | 0.500 | 0.466 | 1.453 | 0.30000 | 115.72 | **2.587** |
| 17 | 0.000 | 0.744 | 13.442 | 0.30000 | 115.72 | 2.587 |
| 34 | 0.000 | 0.696 | 13.442 | 0.30000 | 115.72 | 2.587 |
| 51 | 0.000 | 0.807 | 13.442 | 0.30000 | 115.72 | 2.587 |
| 67 | 0.000 | 1.423 | 0.000 | 0.30000 | 115.72 | 2.587 |

### Diagnostic — pourquoi aucun régime n'est stable

Le point ouvert de §9 (`delta_min` potentiellement inatteignable) n'est pas hypothétique : il
explique directement ce qui précède.

- **`penalty`** : `L_mag` reste à ~61-63 sur toute la grille, quel que soit `lambda_align` —
  `mag_active_rate=1.00` en continu (vu dans les logs bruts). `‖δ‖_2` plafonne à ~54.4, jamais
  proche de `delta_min=115.7`, parce que `‖δ‖_∞≤epsilon=1.0` borne `‖δ‖_2` à
  `epsilon·sqrt(3·32·32)≈55.4` — **`delta_min` est mathématiquement inatteignable sous ce
  epsilon**. `L_mag` (avec `lambda_mag=1.0` par défaut) domine alors `grand_loss` (`g_loss`
  observé jusqu'à ~63, quasi entièrement `L_mag`), quel que soit `lambda_align`, ce qui explique
  pourquoi balayer `lambda_align`/`align_kappa` seuls n'a aucun effet observable sur `expert_asr`
  dans cette grille — le vrai signal est noyé.
- **`projection`** : pire — `‖δ‖_∞` final vaut **2.587**, VIOLANT `epsilon=1.0`. La boucle de
  projections alternées (`project_trigger_constraints`, joint/utils.py:171-187) se termine sur
  l'étape "plancher de magnitude" (dernière du triplet clamp/cône/magnitude) ; quand
  `delta_min` est hors de portée du cône contraint par `epsilon`, cette dernière étape pousse
  systématiquement `δ` HORS de la boule `epsilon`, et rien ne re-clamp après. `cos_target` reste
  figé EXACTEMENT à `0.30000` (= `align_kappa`) du premier au dernier batch — signe que la
  projection écrase entièrement le pas Adam à chaque itération plutôt que de laisser `δ`
  évoluer. C'est un vrai défaut d'implémentation (l'intersection des trois contraintes est vide
  sous ces paramètres, et la fonction n'a aucun moyen de le signaler — elle retourne
  silencieusement un point qui viole la contrainte `epsilon`, la plus fondamentale des trois
  pour la furtivité du trigger). Non corrigé cette session (hors périmètre), mais c'est un bug,
  pas juste une limite de méthode.

**Conclusion du balayage : ni `lambda_align`/`align_kappa` (mode `penalty`) ni le mode
`projection` ne permettent d'évaluer l'hypothèse anti-effondrement telle que testée, parce que
`delta_min_frac=0.5` produit un plancher de magnitude hors d'atteinte sous `epsilon=1.0`. Avant
tout nouveau balayage, corriger la base de calcul de `delta_min` (§9) — le rapporter à
`epsilon·sqrt(numel)`, le maximum réellement atteignable, plutôt qu'à `‖δ_init‖_2` (une valeur
pré-clamp sans rapport avec ce que l'entraînement peut produire) — sans quoi tout autre
balayage de `lambda_align`/`align_kappa`/`trigger_constraint` répétera exactement ce résultat
nul.**

---

## Annexe 2 — Agrégation fédérée différentiable de l'expert (2026-08-25)

### Contexte et hypothèse motivante

Hypothèse de l'utilisateur, formulée après le refus initial de H1 (voir historique) : le
`student` avance sur un gradient AGRÉGÉ (`agg(student_params, student_grad_buf, ...)`) tandis
que l'`expert` avançait sur le gradient d'un SEUL client (le dernier traité,
`expert_model.zero_grad()` avant chaque client écrasant les précédents). Avec
`num_honests=num_poisoned=1` (config utilisée dans tout le balayage P3 et le test H1), ceci
rend la perte d'appariement MTT structurellement quasi triviale — indépendamment de tout
régularisateur — et était jugée plus probable que `delta_min` ou l'épuisement d'epsilon comme
cause principale de l'effondrement d'`expert_asr`.

### Étape 0 — Vérification (lecture seule, avant tout code)

Confirmé dans les DEUX fichiers :

- **`federated_generate_labels_trigger`** : `expert_grad_buf` peuplé par tous les clients
  (honnêtes ET empoisonnés) mais jamais lu — seul `optimizer_expert.step()` consomme `.grad`,
  écrasé à chaque client par `expert_model.zero_grad()`.
- **`federated_generate_labels`** (baseline publié) : MÊME schéma, avec une preuve plus directe
  — le code contient littéralement un appel `agg_expert_grads = agg(expert_params,
  expert_grad_buf, agg_method, f=num_poisoned)` **laissé en commentaire** (utils.py:171-176),
  suivi immédiatement de `optimizer_expert.step()` sur le `.grad` brut du dernier client.
  L'agrégation avait donc été tentée puis abandonnée par les auteurs d'origine — ce n'est pas
  un oubli local au module trigger, c'est hérité du baseline lui-même.

**Arbitrage utilisateur** : corriger `federated_generate_labels_trigger` ET
`federated_generate_labels_trigger_joint` ; NE PAS toucher au baseline
`federated_generate_labels` (portée volontairement restreinte, cohérent avec la contrainte de
compatibilité des trois modules — voir §2).

### Étape 1 — Test unitaire `sgd_step`

`prelim/tests/test_sgd_step.py` (13/13 PASS) remplace l'ancien "point-4 check" (une assertion
numérique unique, au premier batch d'un run réel) par un test synthétique dédié :
- pas unique depuis un `opt_state` frais ET depuis un `momentum_buffer` pré-chargé non nul,
  comparé exactement (`max_gap<1e-6`) à `torch.optim.SGD`, sur 4 configurations (SGD simple,
  momentum seul, momentum+weight_decay+nesterov, momentum+weight_decay+dampening) ;
- **découverte annexe (bug dormant, NON corrigé — fichier partagé
  `modules/federated_generate_labels/utils.py`, hors périmètre)** : `sgd_step` ne persiste
  jamais son `momentum_buffer` calculé dans `opt_state` (`buf = buf.mul(...).add(...)` est une
  opération hors-place qui ne réécrit jamais `opt_state['momentum_buffer']`) — un second appel
  réutilisant le même `opt_state` diverge de `torch.optim.SGD` dès le second pas (gap mesuré :
  jusqu'à 0.30 sur un cas synthétique). **Inoffensif dans ce dépôt** : les trois modules
  `run_module.py` rechargent un `opt_state` FRAIS depuis le disque à chaque batch et
  n'appellent `sgd_step` qu'une seule fois par paramètre — jamais deux fois sur le même
  `opt_state`. Signalé, non corrigé (comportement figé pour ne pas affecter les runs déjà
  produits).

### Étape 2 — Refactor (agrégation fédérée de l'expert)

Dans les deux modules corrigés : `expert_grad_buf` collecte la contribution de CHAQUE client
(honnête : `.grad` réel détaché depuis `loss.backward()` ; empoisonné, module joint uniquement :
`grads_e` différentiable, `create_graph=True`, NON détaché). Après la boucle clients,
`agg(expert_params, expert_grad_buf, agg_method, f=num_poisoned)` agrège — symétrique à
l'agrégation déjà appliquée côté student.

Conséquence dans le module joint : `optimizer_expert.step()` (le "vrai" pas d'optimiseur,
existant seulement pour fournir la comparaison du point-4 check) devient inutile et est
supprimé, de même que le `.load_state_dict()` qui l'alimentait — `expert_model` est de toute
façon rechargé intégralement depuis le disque au batch SUIVANT, donc rien ne dépendait de ce
pas réel au sein du même batch. Ceci rend également obsolète (sans le rendre incorrect) le
mécanisme d'évitement du risque `AccumulateGrad` documenté en §3 : `expert_params[i].grad`
n'est simplement plus jamais assigné explicitement.

Vérifications (Étape 4) :
- **A/B/C, synthétique** (`prelim/verify_expert_aggregation.py`, 8/8 PASS) : l'agrégat dépend de
  TOUS les clients (pas seulement le dernier), est invariant à l'ordre de traitement des
  clients, le gradient empoisonné atteint `delta.grad` à travers l'agrégat pour `mean`/
  `median`/`trmean`, et l'agrégat `mean` est exact en VALEUR et en GRADIENT (différentiabilité
  vérifiée, pas seulement la valeur avant) par rapport à une moyenne manuelle des contributions
  par client.
- **D, intégration réelle** (`prelim/verify_step4d_integration.py`, checkpoints `r32p_1xs`
  réels, 15/15 PASS) : `optimizer_expert.step()` espionné et confirmé appelé **0 fois** sur un
  run réel, pour `agg_method ∈ {mean, median, trmean}`, `num_honests=num_poisoned=2` — preuve
  comportementale, pas seulement statique (grep du code source), qu'aucun pas réel ni aucune
  dépendance à `expert_params[i].grad` ne subsiste. Confirme aussi Étape 3 (voir ci-dessous) :
  `cos_delta_to_init==1.0` et `delta_drift_l2==0.0` exactement au batch 0, pour les trois
  `agg_method`.

### Étape 3 — Correction de l'instrumentation H1

Deux corrections, motivées par le fait que le run H1 original (session précédente,
`epsilon=1.0`) donnait `cos_delta_to_init=0.931` au step 0 (attendu : exactement 1.0) et
`delta_drift_l2` dépassant la norme L2 maximale atteignable sous cet epsilon — signe que
`delta_init` était capturé AVANT tout clamp, un point de référence non faisable :

1. `delta_init` est maintenant capturé À L'INTÉRIEUR de la boucle d'entraînement, la première
   fois que l'exécution atteint le clamp_/projection de `delta` (pas avant la boucle, depuis
   l'init brute `strength=6.0`).
2. `expert_asr`/`expert_asr_frozen` sont maintenant mesurés sur un ensemble FIXE de 256 exemples
   de la classe source (`asr_eval_raw`, construit une seule fois avant la boucle), au lieu
   d'être accumulés sur le sous-ensemble empoisonné (petit, bruité) de chaque mini-batch.

### Étape 5 — Run de référence (epsilon=0.031, régularisateurs désactivés)

Un seul run, `num_honests=num_poisoned=1` (la configuration motivante), `epsilon=0.031` (8/255,
réaliste — au lieu de 1.0), `agg_method="mean"`, `lambda_align=lambda_mag=0` (désactivés, non
recalibrés avant ce run, comme demandé). 68 batches (`train_pct=0.02`, run court de
vérification, pas une campagne).

| Métrique | Moyenne | Écart-type | Min | Max |
|---|---|---|---|---|
| `expert_asr` (ensemble fixe) | 0.0542 | 0.0200 | 0.0156 | 0.1953 |
| `expert_asr_frozen` (δ_init figé) | 0.0144 | 0.0221 | 0.0117 | 0.1953 |
| `matching_term` (param_loss/param_dist) | 0.2705 | 0.0835 | 0.0885 | 0.5026 |
| `L_bd_mean` | 2.5324 | 1.4272 | 0.0000 | 5.5923 |
| `‖δ‖_2` | 1.5153 | 0.0857 | 1.2463 | 1.6694 |
| `‖·‖` contribution MTT à `delta.grad` | 0.0097 | 0.0072 | 0.0000 | 0.0360 |

**Constats :**

- `expert_asr` ne s'effondre plus à un ZÉRO exact et invariant (contrairement à l'ancien
  balayage §Annexe, `epsilon=1.0` : `0.0000` sur les 7 configurations, sans exception) — il
  fluctue faiblement (1.6%–19.5%, moyenne 5.4%), sous le niveau du hasard (~10% pour 10 classes)
  après le step 0. La signature d'échec a changé qualitativement : bruit faible plutôt
  qu'effondrement déterministe. Ceci suggère que l'ancien collapse à exactement 0 était
  probablement amplifié — sinon causé — par la saturation `epsilon=1.0` (préoccupation H2) et/ou
  le plancher `delta_min` inatteignable (§9), PLUS que par le bug d'agrégation mono-client seul
  ; le bug d'agrégation reste néanmoins une vraie correction structurelle (symétrie avec le
  student), indépendamment de son effet mesuré ici.
- La contribution MTT à `delta.grad` est non nulle sur 57/68 batches (moyenne 0.0097) —
  confirme EN CONDITIONS RÉELLES (pas seulement synthétique, voir vérification B) que le
  gradient agrégé atteint bien `delta`, mais elle reste petite comparée à `L_bd_mean` — le
  couplage réel (E2) existe mais est faible dans ce régime.
- **Verdict H1 (re-testé implicitement, conditions non confondues cette fois — agrégation
  corrigée, epsilon réaliste) : REFUSÉ, plus nettement encore.** `expert_asr_frozen` (moyenne
  1.4%) reste PLUS BAS qu'`expert_asr` courant (moyenne 5.4%) après le step 0 — l'inverse de ce
  que H1 prédisait (trigger figé restant efficace pendant que le courant s'effondre). Le
  backdoor des checkpoints au trigger figé initial est lui-même faible ; l'entraînement du
  trigger (via le chemin MTT+L_bd maintenant correctement agrégé) tend plutôt à AMÉLIORER
  légèrement son efficacité contre le checkpoint courant, pas à s'en éloigner.
- **Confirmation en conditions réelles d'un item déjà signalé (H3, session précédente, non
  corrigé)** : 11/68 batches (16%) ont `L_bd_mean==0.0` — aucune ligne empoisonnée tirée ce
  batch (`is_poisoned.any()==False` pour tous les clients empoisonnés), pas une vraie perte
  nulle. Une moyenne naïve sur ces batches (ou de ces batches) reste biaisée vers le bas. Non
  corrigé cette session (hors périmètre de la demande d'agrégation) — toujours un point ouvert.

### Ce qui n'a PAS été fait (hors périmètre explicite de cette demande)

- Baseline `federated_generate_labels` : bug identique non corrigé (arbitrage utilisateur).
- `delta_min` (rebasage sur `epsilon·sqrt(numel)`), `project_trigger_constraints` (ordre de la
  boucle + détection d'infaisabilité), H3 (biais des batches sans ligne empoisonnée) : toujours
  non corrigés (§9) — carried over depuis le message H1/H2/H3, non redemandés explicitement
  dans cette demande d'agrégation, à clarifier si une suite est souhaitée.
- Aucune campagne relancée (conforme à l'instruction explicite) — Étape 5 est UN SEUL run de
  vérification, pas un balayage.

---

## Rapport final — écarts les plus importants, classés par impact

**[2026-08-24]** Les corrections P0-P2 demandées ont toutes été appliquées et vérifiées (schéma
débloqué et testé de bout en bout via `run_experiment.py`, `gamma_stealth` renommé avec
rétrocompatibilité, `checkpoint_sampling` ajouté aux deux modules `direct`, `.npz` de politique
vérifié croisé, mesure mémoire recalculée en incréments, table de flux de gradient corrigée,
`schedule_kwargs` aligné). La liste ci-dessous reflète l'état **après** ces corrections — les
items résolus ont été retirés plutôt que laissés comme un historique périmé (voir §0/§4 pour la
trace des corrections elles-mêmes).

**[2026-08-25]** L'agrégation fédérée du gradient expert a été corrigée dans
`federated_generate_labels_trigger` et `federated_generate_labels_trigger_joint` (mono-client →
`agg()`, symétrique au student ; baseline volontairement non touché). Voir Annexe 2 pour
l'analyse complète, le test unitaire `sgd_step` (13/13), les vérifications A-D (23/23 au total,
synthétique + intégration réelle), et le run de référence à `epsilon=0.031` : `expert_asr` ne
s'effondre plus à un zéro exact et invariant, mais reste faible (moyenne 5.4%) sur un run court
— aucun régime stable identifié. H1 est re-testé (conditions non confondues cette fois) et reste
REFUSÉ, plus nettement. L'ancien balayage anti-effondrement (annexe P3 ci-dessous) est invalidé
comme référence pour `expert_asr` par ce changement ; le diagnostic `delta_min` (item 1
ci-dessous), lui, reste valide et indépendant.

1. **Le plancher de magnitude `delta_min` est très probablement inatteignable** (annexe P3,
   §9) — découvert en exécutant le balayage anti-effondrement, maintenant débloqué : `expert_asr`
   collapse à exactement 0 dans les 7 configurations testées (`lambda_align × align_kappa`, plus
   `trigger_constraint="projection"`), sans exception. `delta_min = 0.5·‖δ_init‖_2` (calculé
   AVANT toute contrainte `epsilon`) vaut 115.7 sous la config testée, alors que
   `‖δ‖_2` ne peut pas dépasser `epsilon·sqrt(numel)≈55.4` une fois `δ` contraint — le plancher
   ne peut mathématiquement jamais être atteint. Conséquence concrète et plus grave que prévu :
   sous `trigger_constraint="projection"`, la boucle de projections alternées retourne un `δ`
   qui **viole `epsilon`** (`‖δ‖_∞=2.587` observé contre `epsilon=1.0` demandé) — un vrai bug
   d'implémentation quand l'intersection des contraintes est vide, pas seulement une limite de
   méthode. Aucun des deux modes (`penalty`/`projection`) ne peut donc être évalué correctement
   tant que la base de calcul de `delta_min` n'est pas corrigée. C'est l'écart qui bloque le plus
   directement la suite du travail (identifier un régime stable pour `expert_asr`).
2. **La comparabilité entre `federated_optimizing_trigger_policy` et les deux modules `direct`
   reste, et restera, à plusieurs facteurs** (§7, reformulé) — pas parce que des réglages
   manquent, mais parce que la paramétrisation (`u` vs `labels_syn`) et l'architecture (`policy`
   réentraîne son propre modèle, les deux `direct` non) sont la contribution elle-même, pas un
   facteur de nuisance. Les facteurs de nuisance véritables (échantillonnage des checkpoints,
   agrégateur) sont maintenant alignables ; les facteurs structurels ne le seront jamais et ne
   devraient pas être présentés comme un manque à corriger.
3. **`prelim/` reste hors périmètre et non revérifié** (§9) — `varsigma`/`rho`/`v_hat`/`a_k`
   n'ont pas été relus avec la rigueur de cet audit depuis la correction A appliquée dans
   `modules/`. Aucune preuve d'erreur, mais aucune vérification récente non plus.
