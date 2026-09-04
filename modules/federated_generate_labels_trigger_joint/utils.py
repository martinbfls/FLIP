import random

import numpy as np
import torch
import torch.nn.functional as F

from modules.base_utils.datasets import MTTDataset
from modules.federated_generate_labels.utils import DEFAULT_EXPERT_CONFIG


def build_expert_pool(expert_starts, expert_opt_starts, pool_size):
    """P3 (checkpoint-pool stability fix): preloads `pool_size` distinct (params, optimizer-
    state) checkpoint pairs into RAM (float32, CPU) once, to be drawn from uniformly at random
    per outer training step -- instead of a single checkpoint indexed sequentially by the step,
    which conditioned every step's gradient on just one point of the expert trajectory. Kept in
    float32 (never half) -- this module's create_graph=True backward differentiates through
    expert_start/state_dict-derived quantities.

    `expert_starts`/`expert_opt_starts` (parallel path lists, as returned by extract_experts /
    extract_experts_biased) may contain duplicate (params_path, opt_path) pairs -- e.g. from
    independent random draws colliding -- so the pool is sampled from the DISTINCT pairs only.
    If `pool_size` exceeds the number of distinct pairs available, uses all of them instead
    (warns).

    Returns (pool, pool_size): `pool` is a list of (params_state_dict, opt_state_dict) tuples,
    both CPU/float32 tensors; `pool_size` is the (possibly clamped) actual pool size.
    """
    all_pairs = list(dict.fromkeys(zip(expert_starts, expert_opt_starts)))
    if pool_size > len(all_pairs):
        print(
            f"WARNING: pool_size={pool_size} > {len(all_pairs)} distinct available expert "
            f"checkpoints for this run -- using all {len(all_pairs)} instead."
        )
        pool_size = len(all_pairs)
    pool_paths = random.sample(all_pairs, pool_size)
    pool = [
        (
            {k: v.float().cpu() for k, v in torch.load(p_path, map_location="cpu").items()},
            torch.load(o_path, map_location="cpu"),
        )
        for p_path, o_path in pool_paths
    ]
    return pool, pool_size


class TriggerMTTDataset(MTTDataset):
    '''
    Thin wrapper around `MTTDataset` (modules/base_utils/datasets.py) that additionally
    returns a boolean `is_poisoned` flag per example: True iff this draw's "train" branch
    came from the appended poison segment (original i >= len(self.distill)), i.e. iff it is
    one of the genuinely-triggered-and-relabeled examples, as opposed to a plain pass-through
    clean example that happens to share the same ConcatDataset.

    This is needed because MTTDataset's returned `idx` (the 5th tuple element) alone cannot
    distinguish the two cases: for poisoned draws, idx is reassigned to
    `poison_inds[i % len(distill)]`, but a *non*-reassigned i (from the clean segment) can
    coincidentally also be a member of poison_inds (that source-class image, drawn
    unpoisoned) -- checking `idx in poison_inds` from outside would give false positives.

    Constructed by re-wrapping an existing MTTDataset's constituent objects (train, distill,
    poison_inds, transform, n_classes) -- e.g. the one returned by
    `modules.base_utils.datasets.get_matching_datasets` -- rather than duplicating its
    dataset-construction logic.
    '''

    def __getitem__(self, i: int):
        is_poisoned = i >= len(self.distill)
        train_x, train_oh, distill_x, distill_oh, idx = super().__getitem__(i)
        return train_x, train_oh, distill_x, distill_oh, idx, is_poisoned

    @classmethod
    def from_mtt_dataset(cls, mtt_dataset):
        return cls(
            mtt_dataset.train, mtt_dataset.distill, mtt_dataset.poison_inds,
            mtt_dataset.transform, mtt_dataset.n_classes,
        )


def _sample_trajectory_index(n_traj, alpha_ckpt):
    '''
    Single draw from the SAME exponential-bias distribution
    `federated_optimizing_trigger.utils.sample_checkpoints` uses (probability of index k
    proportional to exp(-alpha_ckpt*k), k=0..n_traj-1 -- biased toward EARLY checkpoints for
    the repo's convention alpha_ckpt>0), reimplemented here (not called) because
    sample_checkpoints's "always include the last checkpoint" augmentation is specific to its
    own use (picking a representative SUBSET alongside a guaranteed final one, once per outer
    training step) and doesn't apply to a single per-(iteration,trajectory-length) draw.
    '''
    ks = torch.arange(0, n_traj, dtype=torch.float)
    probs = torch.exp(-alpha_ckpt * ks)
    probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1).item())


def extract_experts_biased(expert_config, expert_path, iterations, alpha_ckpt, expert_opt_path=None):
    '''
    Like `federated_generate_labels.utils.extract_experts`, but draws the "how far into this
    expert's own training run" trajectory index via the SAME exponentially-biased distribution
    federated_optimizing_trigger_policy's `sample_checkpoints` uses (controlled by the SAME
    `alpha_ckpt` parameter), instead of extract_experts's `np.random.randint(min, max)`
    (uniform). This is what makes the joint module's checkpoint sampling comparable to the
    policy module's -- see this module's run() docstring, "Comparability across the three
    arms" (federated_generate_labels_trigger, unlike this module, still uses the uniform
    `extract_experts` unchanged -- see its own corrected docstring for why that divergence
    now needs to be accounted for whenever a direct/joint comparison is drawn against it).

    Args, returns: identical to extract_experts.
    '''
    config = {**DEFAULT_EXPERT_CONFIG, **expert_config}
    n_traj = config['max'] - config['min']
    expert_starts, expert_opt_starts = [], []

    for _ in range(iterations):
        for s in config['trajectories']:
            expert = np.random.randint(config['experts'])
            trajectory = config['min'] + _sample_trajectory_index(n_traj, alpha_ckpt) + 1
            expert_starts.append(expert_path.format(expert, trajectory, str(s)))
            if expert_path:
                expert_opt_starts.append(expert_opt_path.format(expert, trajectory, str(s)))
    return expert_starts, expert_opt_starts


# --------------------------------------------------------------------------- #
# Anti-collapse regularizers (see run_module.py's run() docstring for the
# failure mode these guard against: expert_asr -> 0 while the matching term
# keeps improving, because a clean expert perfectly satisfies the alignment
# objective). Distinct from federated_optimizing_trigger.utils's
# trigger_penalty_hinge -- that is a STEALTH ceiling on
# cos(delta, mu_target - mu_source); this is a FLOOR on cos(delta, mu_target)
# alone, plus a floor on ||delta||_2 without which the directional floor is
# vacuously satisfiable via delta -> 0 (see run_module.py's docstring).
# --------------------------------------------------------------------------- #

def cosine_to(delta, mu, eps=1e-8):
    '''cos(delta, mu), flattened, mu treated as a fixed (detached) target direction.'''
    return F.cosine_similarity(
        delta.reshape(1, -1), mu.reshape(1, -1).detach(), eps=eps,
    ).squeeze(0)


def directional_floor_penalty(delta, mu_target, align_kappa, eps=1e-8):
    '''
    L_align = relu(align_kappa - cos(delta, mu_target)) -- a FLOOR: zero once
    cos(delta, mu_target) >= align_kappa, active (and penalizing) below it. Do
    NOT confuse with trigger_penalty_hinge's relu(cos - kappa), which is a
    CEILING on a DIFFERENT vector pair (mu_target - mu_source).
    '''
    cos = cosine_to(delta, mu_target, eps=eps)
    return F.relu(align_kappa - cos), cos


def magnitude_floor_penalty(delta, delta_min):
    '''
    L_mag = relu(delta_min - ||delta||_2) -- prevents the directional floor
    above from being satisfied vacuously by delta -> 0 (cos is scale-invariant,
    so a vanishingly small delta can still have cos(delta, mu_target) == 1).
    '''
    norm = delta.norm()
    return F.relu(delta_min - norm), norm


def _project_onto_cone(delta, mu_target, align_kappa, eps=1e-8):
    '''
    Euclidean projection of delta onto the convex circular cone
        K = { x : cos(x, mu_target) >= align_kappa },  half-angle alpha = arccos(align_kappa)
    Closed form via the 2D reduction onto the (axis, perpendicular) plane spanned by delta and
    u = mu_target/||mu_target||: decompose delta = h*u + perp (h = <delta,u>, perp orthogonal
    to u, r = ||perp||). If already inside the cone (cos(phi) = h/||delta|| >= align_kappa),
    return delta unchanged. Otherwise the nearest point is on the cone's boundary ray at angle
    alpha from the axis: t = h*cos(alpha) + r*sin(alpha) is the (signed) distance along that
    ray; if t <= 0 the nearest point is the origin (delta's angle from the axis exceeds
    pi/2 + alpha), otherwise the projection is t*(cos(alpha)*u + sin(alpha)*perp_unit).
    '''
    u = mu_target / (mu_target.norm() + eps)
    h = (delta * u).sum()
    perp = delta - h * u
    r = perp.norm()
    delta_norm = delta.norm()
    if delta_norm < eps:
        return delta
    cos_phi = h / delta_norm
    if cos_phi.item() >= align_kappa:
        return delta

    alpha = torch.acos(
        torch.clamp(torch.tensor(align_kappa, device=delta.device, dtype=delta.dtype), -1.0, 1.0)
    )
    cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
    t = h * cos_a + r * sin_a
    if t.item() <= 0:
        return torch.zeros_like(delta)
    perp_unit = perp / (r + eps)
    return (t * cos_a) * u + (t * sin_a) * perp_unit


def _project_onto_magnitude_floor(delta, delta_min, mu_target, eps=1e-8):
    '''Pushes delta radially out to ||delta||_2 == delta_min if it fell short; a direction is
    needed only in the (measure-zero) delta==0 case, for which mu_target's own direction is
    used (already the cone's axis, so this keeps the point trivially inside the cone too).'''
    n = delta.norm()
    if n.item() >= delta_min:
        return delta
    if n.item() < eps:
        return delta_min * mu_target / (mu_target.norm() + eps)
    return delta * (delta_min / n)


def grad_mismatch_penalty(clean_grads, poison_grads, eps=1e-8):
    '''
    ||grad(L_c) - grad(L_p)(delta)||^2 / ||grad(L_c)||^2, flattened & summed across every
    parameter tensor in `clean_grads`/`poison_grads` (same param order, one tensor per
    parameter -- e.g. the per-parameter mean-over-clean-examples / mean-over-poisoned-examples
    gradients at the current checkpoint theta_k). `clean_grads` is treated as a constant
    (detached upstream); `poison_grads` may carry a live dependency on delta, in which case this
    penalty is differentiable w.r.t. delta. `eps` guards the denominator when grad(L_c) is
    (near) zero.
    '''
    diff_sq = sum((gc - gp).pow(2).sum() for gc, gp in zip(clean_grads, poison_grads))
    clean_sq = sum(gc.pow(2).sum() for gc in clean_grads)
    return diff_sq / (clean_sq + eps)


def grad_cosine_penalty(clean_grads, poison_grads, eps=1e-8):
    '''
    1 - cos(grad(L_c), grad(L_p)(delta)), flattened & concatenated across every parameter
    tensor in `clean_grads`/`poison_grads` -- same convention/inputs as `grad_mismatch_penalty`
    (see there for what the two grad lists mean), an alternative to its relative-squared-error
    ratio. Bounded in [0, 2]: 0 iff the two gradients point in exactly the same direction
    (scale-invariant -- unlike grad_mismatch_penalty, magnitude differences alone don't
    penalize), 1 iff orthogonal, 2 iff exactly opposite. `eps` guards the denominator when
    either gradient is (near) zero.
    '''
    dot = sum((gc * gp).sum() for gc, gp in zip(clean_grads, poison_grads))
    clean_norm = torch.sqrt(sum(gc.pow(2).sum() for gc in clean_grads))
    poison_norm = torch.sqrt(sum(gp.pow(2).sum() for gp in poison_grads))
    cos = dot / (clean_norm * poison_norm + eps)
    return 1 - cos


def margin_floor_penalty(logits_trig, target_label, margin_min):
    '''
    Plancher sur la marge de classification vers la cible sur les lignes declenchees :
    relu(margin_min - (logit_target - meilleur_autre_logit)). Remplace
    directional_floor_penalty comme anti-collapse (P3, cf. D2 dans le diagnostic
    threat_models_audit) : porte sur l'EFFICACITE du backdoor (une marge de decision
    positive et large) et non sur la ressemblance de delta a une image moyenne, qui
    poussait delta dans la direction opposee au prior gagnant (cf. D0).
    Retourne (L_margin, margin_mean) -- meme convention que directional_floor_penalty
    (penalty, metrique brute), pour l'instrumentation.
    '''
    top2 = logits_trig.topk(2, dim=1)
    tgt = logits_trig[:, target_label]
    runner = torch.where(top2.indices[:, 0] == target_label,
                          top2.values[:, 1], top2.values[:, 0])
    margin = tgt - runner
    return F.relu(margin_min - margin).mean(), margin.mean()


def poison_consistency(poison_grad_chunks, eps=1e-8):
    '''
    1 - cos moyen entre chaque chunk de gradient empoisonne (un par client poisonne
    contribuant ce batch, meme convention que poison_grad_mean dans run_module.py) et
    leur moyenne. Les agregateurs robustes (trimmed-mean, mediane, Krum/Multi-Krum)
    attenuent la VARIANCE inter-contributions, pas la magnitude (D0, propriete 2) --
    ce diagnostic/terme restaure une pression directe sur cette variance. Note : avec
    num_poisoned<=1 (config generation "1vs0" de la campagne principale) il n'y a
    qu'un seul chunk contribuant -> valeur triviale 0.0 (cos(g,g)=1) par construction ;
    redevient informatif des que num_poisoned>=2 (cf. protocole experimental, etape 4
    / grille finale).
    '''
    n_chunks = len(poison_grad_chunks[0])
    G = torch.stack([
        torch.cat([poison_grad_chunks[i][j].reshape(-1) for i in range(len(poison_grad_chunks))])
        for j in range(n_chunks)
    ])
    g_bar = G.mean(0)
    return 1.0 - F.cosine_similarity(G, g_bar.unsqueeze(0).expand_as(G), dim=1, eps=eps).mean()


def coordinate_budget_penalty(g_poison, honest_grads_flat, z, eps=1e-8):
    '''
    Ce qu'une defense coordinate-wise laisse passer est un decalage borne par l'ecart-type
    honnete, coordonnee par coordonnee. On l'impose directement plutot que de l'esperer
    (P5). `g_poison` : gradient AGREGE (agg_expert_grads, aplati/concatene, DIFFERENTIABLE
    w.r.t. delta) -- ce que l'agregateur reel calculerait. `honest_grads_flat` : [n_honest, d],
    DETACHE, les contributions honnetes brutes de ce meme batch/checkpoint. Seul terme du
    lot qui cible la robustesse ET reste differentiable partout (contrairement a Krum,
    argmin constant par morceaux, ou a la mediane, dont le gradient est nul des que la
    coordonnee selectionnee est honnete) -- cf. "Ce qu'il ne faut pas faire" du diagnostic :
    ne PAS reintroduire un mecanisme conscient de l'agregateur reel dans ce chemin.
    '''
    mu_h, sd_h = honest_grads_flat.mean(0), honest_grads_flat.std(0) + eps
    return F.relu((g_poison - mu_h).abs() - z * sd_h).pow(2).mean()


def lpips_penalty(delta, x_raw_clean, lpips_model):
    '''
    Perceptual (LPIPS) distance between the clean raw image(s) and their triggered
    counterpart, both in the RAW [0,1] pixel space (the same space `raw_to_trigger_preprocess`
    starts from, NOT the per-dataset normalized space it eventually produces for the
    classifier) -- rescaled to [-1, 1] here, LPIPS's own expected input range. `delta` is not
    detached, so this term is differentiable w.r.t. the trigger being optimized, exactly like
    L_bd in run_module.py (see the L_bd block this term is meant to sit next to). `lpips_model`
    is a pre-instantiated, frozen lpips.LPIPS(net=...), already moved to x_raw_clean's device.
    '''
    if x_raw_clean.dim() == 3:
        x_trig_raw = (x_raw_clean + delta).clamp(0, 1)
    else:
        x_trig_raw = (x_raw_clean + delta.unsqueeze(0)).clamp(0, 1)
    x_clean_scaled = x_raw_clean * 2 - 1
    x_trig_scaled = x_trig_raw * 2 - 1
    return lpips_model(x_trig_scaled, x_clean_scaled).mean()


def project_trigger_constraints(delta, mu_target, epsilon, align_kappa, delta_min, n_iters=8, eps=1e-8):
    '''
    Alternating-projection heuristic (see run_module.py's run() docstring,
    "trigger_constraint='projection'") onto
        { ||delta||_inf <= epsilon }  inter  K_align_kappa(mu_target)  inter  { ||delta||_2 >= delta_min }
    The third set is NOT convex (exterior of a ball), so this is not a provably-exact joint
    projection -- but each individual step is exact for its own set, and normalizing scale-up
    (the magnitude-floor step) never changes cos(delta, mu_target) (positive scaling preserves
    angle), so only the Linf clamp and the cone projection can re-violate each other; a handful
    of alternations settles this in practice. Returns delta unchanged if it is already feasible
    for all three sets (each individual projection is itself a no-op in that case).
    '''
    for _ in range(n_iters):
        delta = delta.clamp(-epsilon, epsilon)
        delta = _project_onto_cone(delta, mu_target, align_kappa, eps=eps)
        delta = _project_onto_magnitude_floor(delta, delta_min, mu_target, eps=eps)
    return delta
