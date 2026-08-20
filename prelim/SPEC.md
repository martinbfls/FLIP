# `prelim/` — Specification

Empirical validation of a theoretical model of label-noise perturbations in federated
learning. The model predicts the bias induced on the aggregated gradient; this suite checks
whether that prediction holds, and how it interacts with robust aggregation rules. Purpose is
defence evaluation: the quantities below are what a defender would need to know to size the
tolerance parameters of an aggregation rule.

This file is the single source of truth for the task. Sessions should read it rather than have
it restated.

---

## 1. Scope and constraints

- All new code lives in `prelim/`, at the root of `FLIP/`. Imports into the repo go through
  `sys.path.insert` pointing at `FLIP/`, as `FLIP/scripts/check_trigger_objective.py` already does.
- `modules/` is **read only** for this work. Reuse what exists; never refactor it.
- `experiments/` holds pipeline configurations and is unrelated to this work. Do not touch it.
- **Never write under `out/checkpoints/`.** All artefacts go to `prelim/artifacts/`.
- No unrequested robustness layers: no retry logic, no elaborate CLI, no logging framework.
  Simple scripts, close in style to the existing code.
- Every block except E6 must run in under ten minutes.

---

## 2. Notation

`C` classes; `pi[y]` is the clean-data frequency of class `y`. `n_b` workers, `n_p` of them
perturbed, `gamma = n_p / n_b`, `f` the tolerance parameter of the aggregation rule.

At fixed parameters `theta`:

- `g[y][z]` = mean over true-class-`y` examples of `grad_theta loss(f_theta(x), z)`.
- `grad_c = sum_y pi[y] * g[y][y]` (clean gradient).
- **Shift matrix** `Gbar`, shape `d x C(C-1)`, columns indexed by `(y, z)` with `z != y`:
  `Gbar[:, (y,z)] = pi[y] * (g[y][z] - g[y][y])`.
- **Perturbation masses** `u >= 0`, indexed by `(y, z)`, `z != y`: `u[y,z]` is the proportion of
  held examples that are truly class `y` and carry label `z` after perturbation.
- **Central prediction**: the expected message of worker `i` is `grad_c + Gbar @ u_i`.
- Admissible sets:
  - per worker: `U_loc = { u >= 0 : sum_z u[y,z] <= pi[y] for all y, ||u||_1 <= beta / gamma }`
  - aggregate: `U_beta = { u >= 0 : ||u||_1 <= beta, sum_z u[y,z] <= gamma * pi[y] for all y }`
  - homogeneous deployment: `u_i = ubar / gamma` on the `n_p` perturbed workers, `0` elsewhere.
- Reachable set under the mean: `E_k = Gbar @ U_beta`.
- Target deviation: `v = grad_p - grad_c` with `grad_p = (1-lam)*grad_c + lam*grad_bd` and
  `grad_bd = grad_theta E_{x ~ source class}[ loss(f_theta(T(x)), y_target) ]`. `T` is a fixed
  input transform, never optimised in this suite.
- `varsigma = max_{y,z} ||Gbar[:, (y,z)]||`, `rho = beta * varsigma`, `v_hat = ||v|| / rho`.
- `beta = N_flip / n`: a global fraction, not a count. On CIFAR-10 (`n = 50000`), 5000 altered
  labels give `beta = 0.10`, and the local rate is `beta / gamma`.

**Budget-scope hazard.** The repo solver takes `sum(w) <= beta` with no scope attached. Any
solver wrapper here must take a `scope` argument valued `"aggregate"` or `"local"` and derive the
bound itself. Assert `||u_i||_1 * gamma == ||ubar||_1` numerically at every solve.

---

## 3. Repo assets to reuse

From `modules/federated_optimizing_trigger/utils.py`:

- `compute_class_frequencies(dataset_flag, n_classes)` → `pi`
- `compute_expected_flip_gradients(...)` → `(G, Q = G^T G, pairs)`; column convention is
  `pi[y] * (g[y][z] - g[y][y])`, matching §2. `pairs` is the ordered column index.
- `project_gradient(Q, c, beta, pairs, ridge=1e-6)` → OSQP solve of
  `min 0.5 w^T Q w - c^T w  s.t.  w >= 0, sum(w) <= beta`. **Global budget only**; no per-class
  caps exist anywhere in the repo.
- `compute_v_polytope_distance(v, G, Q, ...)` → squared distance without materialising `G @ w*`.

From `modules/base_utils/datasets.py`: `shard_dataset_indices(...)` for IID sharding,
`StripePoisoner` for the non-identity `T`.

From `modules/base_utils/aggregator/`: the aggregation rules. **They operate tensor by parameter
tensor, not on the flattened gradient.** This matters for E4/E5 and must be carried through, not
smoothed over.

Datasets available: CIFAR-10 and CIFAR-100 only. Models, smallest first: `r32p` (~0.46M params),
`convnext_micro`, `r18` (~11M), then pretrained ones. No purely linear model exists.

Source/target pair: `9 -> 4`, as in the `opt_trigger` configurations.

---

## 4. Configurations

Two model settings:

- **`linear`** — `nn.Linear(3*32*32, 10)` on flattened CIFAR-10, `d ≈ 30730`, CPU. Reference
  setting: `Gbar` is estimated precisely and everything runs in seconds.
- **`cnn`** — `r32p` on CIFAR-10 subsampled to 10000 training examples.

**The `linear` setting validates the implementation only.** For a linear softmax model,
`grad_W loss(x,z) = (p(x) - e_z) x^T`, so up to the bias term
`Gbar[:, (y,z)] = pi[y] * (e_y - e_z) (x_bar_y)^T`: the image of `Gbar` is a highly structured
subspace and `v` inherits the same structure, making the rank ratio and the cone alignment
atypically favourable. **The E2 verdict is taken on the `cnn` setting only.**

Defaults: `n_b = 10`, `n_p = 3`, `f = 3`, `beta in {0.01, 0.03, 0.10}`, `lam = beta`, three seeds.
These differ from the toy `config.toml` in the repo (1/1) and the difference is intentional: 10/3
is the grid of the real campaign, and `n_p <= f < n_b / 2` holds.

One short clean training run per `(model, seed)` yields **three checkpoints** (early, mid, late)
under `prelim/artifacts/ckpt/`. That is the only training in the suite outside E6.

Sweep grid:

```
MODELS      = ["linear", "cnn"]
SEEDS       = [0, 1, 2]
CHECKPOINTS = ["early", "mid", "late"]
BETAS       = [0.01, 0.03, 0.10]
TRANSFORMS  = ["identity", "stripe"]
N_P         = [3]            # E7 extends to [2, 3, 5]
AGGREGATORS = ["mean", "cw_median", "trmean", "krum", "multikrum"]
```

---

## 5. Deliverables

### `prelim/prelim_lib.py`

Pure, testable functions, no global state:

- `make_config(...)` → dataclass holding the axes above.
- `shard_indices(dataset, n_b, seed)` — reuse `shard_dataset_indices`.
- `class_conditional_shifts(model, calib_loader, C, device)` → `(Gbar, grad_c, pi)`, `float32` on
  CPU, shape `(d, C*(C-1))`, plus an exported `col_index(y, z)` mapping backed by `pairs`.
  Cost: `C*(C-1)` backward passes.
- `masses_to_labels(shard_targets, u, rng)` → hard relabelling realising `u` as closely as integer
  counts allow; returns the **realised** masses as well as the labels.
- `worker_gradient(model, indices, targets, batch_size, device)` → flattened mean gradient.
- `solve_qp(Q, c, beta, pi, gamma, scope, capacity: bool)` — `capacity=False` delegates to
  `project_gradient` unchanged; `capacity=True` adds the per-class rows
  `sum_z u[y,z] <= gamma * pi[y]` in a new OSQP constraint block written here, following the style
  of the existing global-budget builder. Never modify the original module.
- `dist_to_cone(Q, c, v_norm_sq)` → `min_{u >= 0} ||Gbar u - v||^2` with no budget, and
  `alpha_tilde_star = sqrt(1 - dist^2 / ||v||^2)`. This is `project_gradient` with a very large
  `beta`; assert the budget constraint is inactive at the optimum and report it if not.
- `support_function(Gbar, p, beta, pi, gamma)` → greedy water-filling: set
  `psi[y] = max(0, max_{z != y} <p, Gbar[:, (y,z)]>)`, sort classes by `psi` descending, pour the
  budget into classes in that order up to the caps `gamma * pi[y]`.
- `reachable_radius(Gbar, beta, pi, gamma, n_restarts=20)` → **bracket**, not exact value:
  upper bound `beta * varsigma`; lower bound `min(beta, gamma * pi[y*]) * varsigma` where `y*`
  indexes the largest-norm column; best value found by projected gradient ascent from random
  starts. Maximising a convex function over the polytope is not tractable exactly — say so in a
  comment and label the three numbers distinctly.
- `rank_ratio(Q, c, v_norm_sq)` → `varpi = c^T pinv(Q) c / ||v||^2`.
- Instrumented aggregation rules (see §7).

### `prelim/sweep.py`

`run_all(grid, include_e6=False, resume=True)`, writing one tidy row per measurement:

```
config_id, model, seed, checkpoint, beta, transform, aggregator, n_p, tau,
experiment, metric, value
```

- **Cache and resume.** One cell = one `config_id` (stable hash of its parameters). If its rows
  are already in the CSV and `resume=True`, skip. A failing cell is logged with its traceback to
  `prelim/artifacts/failures.log` and the sweep **continues**.
- **`Gbar` cost drives the loop order.** `Gbar` is `d x 90` — about 167 MB in float32 for `r32p`.
  The outer loop is `(model, seed, checkpoint, transform)`; derive everything that depends on it —
  all `beta`, all `tau`, all aggregators — before releasing it. Persist only reduced quantities
  (`Q`, `c`, column norms, `||grad_c||`, and the `d`-dimensional vectors genuinely reused later
  such as `b* = Gbar @ ubar*`). Optional `--cache-gbar` writes float16 to
  `prelim/artifacts/cache/`.
- **Order.** Training-free blocks (E1–E5, E7) first; E6 last and only under its flag, printing a
  runtime estimate and asking for confirmation first.
- **Reproducibility.** Seed torch, numpy and the relabelling RNG; record effective seeds in the CSV.

### `prelim/report.py`

Builds `prelim/artifacts/report.md` and a machine-readable `report.json` of the same content from
the CSV. Format spec in §6.

### `prelim/prelim.ipynb`

Thin driver: one configuration cell, a call to `run_all`, a call to the report generator, then
figure display. All logic lives in the modules.

---

## 6. Report format

**The report is read without the figures.** Every figure must have a numeric twin: a curve
becomes a table of 8–12 points, a heatmap becomes a table, a scatter becomes a correlation
coefficient plus the extreme points. Anything existing only as an image is lost.

- **§0 Header.** Date, git short hash (plus `dirty`), versions of torch / numpy / osqp, device,
  total runtime, cells run / cached / failed.
- **§1 Grid.** Table of axes and values, cell count, and the list of failed or skipped cells
  **with reasons**. No silent failures.
- **§2 Consistency assertions.** Table `name | scope | PASS/FAIL | observed | threshold`, at
  minimum: the budget-scope relation; `solve_qp(capacity=False)` agreeing with `project_gradient`
  to 1e-6; budget inactive in the cone solve; max relative gap between requested and realised
  masses; the flat reference aggregator agreeing with the repo one on a single-tensor model; no
  NaN or inf in the CSV.
- **§3–§9, one per experiment.** Same layout each time: *Hypothesis* (one line); *What ran*;
  *Results* as tables, aggregated over seeds as `median [min, max]`, three to four significant
  digits, never row-by-row dumps; *Numeric twins of the figures*; *Verdict* — `PASS` / `FAIL` /
  `INCONCLUSIVE` with the figure backing it; *Anomalies*, including any violated theoretical bound.
- **§10 Summary.** Go / no-go table for the four gates — E1, E2 (`cnn` only), E3, E5 — then a
  three-line recommendation: launch the full sweep, or fix what first.

Constraints: at most 1500 lines, markdown tables, no code blocks, no absolute paths. Figures are
still produced and saved; the report simply does not depend on them.

---

## 7. Instrumented aggregation rules

One function per rule, operating on an `(n_b, d)` stack of flattened gradients, returning
`(aggregate, selection)`, where `selection` exposes the weights `omega[i, j] = 1[i in S_j] / ell`
**without materialising** an `(n_b, d)` matrix:

- coordinate-wise rules (`cw_median`, `trmean`): an index array of shape `(ell, d)`;
- `krum` / `multikrum`: a single index set valid for all coordinates.

Derived: `A_j = |S_j ∩ M| / ell`, `w[i,j] = omega[i,j] - 1/n_b`,
`chi_ell = (n_b - ell) / (ell * n_b)`.

Two variants everywhere, reported side by side:

- **`flat`** — the rule applied to the flattened gradient; the only variant the theory covers.
- **`per_tensor`** — the rule applied tensor by tensor, **importing the repo implementation**
  rather than reimplementing it.

The distinction is not cosmetic. For `krum` and `multikrum` the theory predicts a selected set
independent of the coordinate, hence `osc(Abar) = 0` **exactly**. The per-tensor variant has a set
that is constant *within* a tensor but varies *between* tensors, hence `osc(Abar) > 0`.

---

## 8. Experiments

### E1 — Bias map: implementation and transfer *(gate)*

*Hypothesis.* `E[g_i] = grad_c + Gbar @ u`. Note this identity is exact by construction for any
model, so E1 does not test a modelling approximation. It tests three things, and the verdict must
be worded accordingly: **(i)** the implementation — signs, `pi[y]`, column ordering; **(ii)** the
transfer of a `Gbar` estimated on a calibration set to a particular shard, which is the only real
statistical content, since shards have their own class proportions and their own conditional
gradients; **(iii)** the signal-to-noise ratio.

*Protocol.* At each checkpoint: compute `Gbar`, `grad_c`, `pi` on a fixed calibration set (~512
examples per class); pick six `u` in `U_loc` — concentrated on `(9,4)`, concentrated on another
pair, uniform over all pairs, the E3 solution, and two random; for each, build the actual
perturbed shard, compute its empirical mean gradient over the whole shard, and compare against
`grad_c + Gbar @ u_realised` (realised masses, not requested); repeat with batch sizes
`{64, 256, 1024, full shard}`.

*Metrics.* Relative error `||g_emp - pred|| / ||Gbar @ u||`; `cos(g_emp - grad_c, Gbar @ u)`; the
same comparison with a `Gbar` recomputed **on the shard itself**, which separates estimation error
from implementation error; and
`SNR = ||Gbar @ u|| / (sigma_i / sqrt(|B|))`, with `sigma_i` the minibatch gradient standard
deviation estimated over several minibatches.

*Expected.* Cosine above 0.99 with the shard-recomputed `Gbar`; slope near `-1/2` on the
error-vs-batch-size log-log plot. If the cosine is well below 0.99 there, the bias map is wrong
and everything downstream is void — stop and report. If `SNR << 1` at realistic budgets, the
perturbation is buried in minibatch noise and the whole selection story is moot: that outcome is
as important as the cosine itself.

### E2 — Geometry and regime scalars *(gate; verdict on `cnn` only)*

*Hypothesis.* The rank ceiling leaves exploitable room.

*Protocol.* At each checkpoint and each `beta`, tabulate: `varsigma`, `rho`, the radius bracket,
`||v||`, `v_hat`, `varpi` and its baseline `effective_rank(Q) / d`, `alpha_tilde_star` and
`sqrt(varpi)`, `||grad_c||`, `Theta = arcsin(radius / ||grad_c||)` (NaN if the ratio exceeds 1),
`angle(grad_p, grad_c)`, and the saturation index `s_beta = beta / (gamma * min_y pi[y])`.

Also report the split of `v` into a reachable and an unreachable part: with `T = identity`,
`v / lam = Gbar[:, (9,4)] / pi[9] + (g[9][9] - grad_c)`. The first term is exactly reachable by
relabelling, the second is not, and their relative norms say where the residual comes from.

*Expected.* `varpi` well above its baseline; `v_hat > 1`; `Theta` small but nonzero. If `varpi`
sits at the baseline on the `cnn` setting, the class-level policy is too coarse and a per-example
policy is needed — which must be known before any large sweep.

### E3 — Stability of `Gbar` and the cost of a single fixed configuration *(gate, very cheap)*

*Hypothesis.* One configuration serves the whole trajectory.

*Protocol.* Solve `min_{u in U_beta} ||Gbar_k u - v_k||^2` per checkpoint (`u_k*`), then the
coupled problem with a shared `ubar`: `Q = sum_k mu_k Gbar_k^T Gbar_k`,
`c = sum_k mu_k Gbar_k^T v_k`, `mu_k = 1 / (rho_k^2 * K)`. Compare.

*Metrics.* `cos(u_k*, u_k'*)` over checkpoint pairs; column-wise cosine between `Gbar_k` and
`Gbar_k'`; the gap `J(shared ubar) - mean_k(a_k / rho_k^2)`; the support of `u_k*` and the fraction
of budget actually spent, `||u*||_1 / beta`.

*Expected.* Stable support, concentrated on pairs involving the source class; gap under 20%.
A support that rotates across checkpoints means the single-configuration framing is ill-posed —
the most important thing to learn early. At `beta = 0.10` with `gamma = 0.3` and balanced classes,
`s_beta > 1`, so the per-class caps bind and `||u*||_1` should be **strictly below** `beta` for a
single-source objective; compare `capacity=True` against `capacity=False` on exactly this point.

### E4 — Response of robust rules to the mean-optimal configuration

*Hypothesis.* Controlling the mean also controls the robust rules.

*Protocol.* At a fixed checkpoint, deploy `u_i = ubar* / gamma` on the `n_p` perturbed workers,
simulate ~200 aggregation rounds with real minibatch gradients from all 10 workers, for each rule
and each variant.

*Metrics.* Distribution of realised `A_j` and `Abar_j = E[A_j]`; `osc(Abar) / Abar_min`; `||P||`,
`||N||`, `||nu_k||`; `||b_Agg - b_mean|| / rho`; and deviation-level alignment
`alpha_tilde(b_Agg)` against `alpha_tilde(b_mean)`. Compare `||P|| + ||N||` against the theoretical
bound `Lambda * ||Gbar u_i|| + sqrt(chi_ell * (n_h * sigma_c^2 + n_p * sigma_a^2))`.

*Numeric twins.* A `rule x variant` table of the above; the `A_j` histogram summarised by deciles.

*Expected.* `osc(Abar) = 0` exactly for `krum`/`multikrum` under `flat`, nonzero under
`per_tensor`; the theoretical bound respected everywhere — a violation is either a bug or a false
assumption, and belongs in Anomalies.

### E5 — Selection response and saturation *(gate)*

*Hypothesis.* Reducing the demanded deviation improves selection **only** below the reachable
radius.

*Protocol.* No transform optimisation. Sweep `tau` over a 12-point log grid such that
`v_hat = tau * ||v|| / rho` spans `[0.1, 10]`; at each `tau`, re-solve the QP against target
`tau * v`, deploy, measure selection. Sweep `beta` as well.

*Metrics.* Selection rate `E[A_j] / gamma` — the captured share relative to what the mean would
give — against `v_hat`; against `||P_k(v)|| / (gamma * sigma_c)`; and coordinate-wise against
`|P_k(v)_j| / s_j`.

*Numeric twins.* Per rule, the 12-point `(v_hat, selection)` table plus an estimate of the knee
location, by simple segmented regression or by the steepest-slope point.

*Expected.* **Flat above `v_hat = 1`, decreasing below**, with a knee near 1. Expected ordering of
thresholds: krum more permissive than cw-median by a factor `gamma`, trimmed mean in between.
This is the most falsifiable prediction in the model: if the curve is flat everywhere, say so
explicitly in the verdict.

### E6 — Predictive power of the feasibility measures *(the only expensive block; `include_e6=True`)*

*Hypothesis.* The normalised residual predicts end-task effect.

*Protocol.* Eight configurations chosen to **spread the predictor**, not to cover the grid: three
source/target pairs by three budgets, truncated to eight, on `r32p` / CIFAR-10 at 10000 examples,
~30 federated rounds, aggregators `mean` and `trmean`. The predictor is computed at round 0 and
never updated.

*Metrics.* Final effect rate and clean accuracy; Spearman correlation between the effect rate and
each of `E[a_k / rho_k^2]`, `alpha_tilde_star`, `v_hat`, `varpi`.

*Numeric twins.* The full eight-row table `(config, predictor, effect rate, clean accuracy)` — short
enough to appear in full — plus the Spearman coefficients.

*Expected.* Clear decreasing monotonicity between residual and effect. This is what decides
whether the alignment-based formulation is worth implementing at all.

### E7 — Spreading the perturbation across workers

*Hypothesis.* At fixed `beta`, raising `n_p` lowers the local rate `beta / gamma` and hence the
outlyingness of each message, **without changing the reachable set under the mean**.

*Protocol.* Replay E4 with `n_p in {2, 3, 5}` at fixed `beta`. Assert explicitly that `ubar*` and
`E_k` are unchanged — an assertion, not a measurement.

*Expected.* Selection increasing in `n_p`, and invisible to the mean-level model.

---

## 9. Working order

1. `report.py` and `sweep.py`; re-run E1–E3 across the whole grid; deliver a first `report.md`
   and **stop** for format validation before writing E4–E7.
2. Instrumented aggregation rules plus the `flat` / repo agreement assertion.
3. E4, E5, E7.
4. E6 last, behind its flag.