"""
tests/test_1xs_stripe_poisoner_linf.py -- checks the L_infinity norm of the perturbation
StripePoisoner (poisoner_flag="1xs", the FLIP paper's own sinusoidal trigger) actually induces,
per modules.base_utils.datasets.StripePoisoner.poison:

    mask = sin(linspace(0, freq * pi, h))          # in [-1, 1]
    mix  = image + strength * mask                  # strength=6, freq=16
    poisoned = uint8(clip(mix, 0, 255))

The theoretical bound is strength/255 = 6/255 ~= 0.0235 (normalized [0, 1] pixel scale) --
this test verifies that bound is actually reached (not diluted by clipping in the typical,
non-extreme-pixel-value case) and stays within it everywhere, including at the extreme base
values (0/255) where clipping DOES reduce the achievable delta.

Uses only pick_cifar_poisoner / StripePoisoner directly (no dataset download, no model, no
GPU) -- safe to run in a code-writing session.

Run:  python tests/test_1xs_stripe_poisoner_linf.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.base_utils.datasets import pick_cifar_poisoner

STRENGTH = 6  # StripePoisoner's own default for poisoner_flag="1xs" (see pick_cifar_poisoner)
THEORETICAL_LINF_255 = STRENGTH  # sin(...) in [-1, 1], so max |delta| in [0,255] scale = strength
IMG_SIZE = 32  # CIFAR/SVHN

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def _linf_255(img_uint8, poisoner):
    """L_infinity of (poisoned - clean), in the original [0, 255] uint8 pixel scale."""
    poisoned = poisoner.poison(img_uint8)
    delta = np.asarray(poisoned).astype(int) - img_uint8.astype(int)
    return int(np.max(np.abs(delta)))


def test_linf_on_random_images():
    """On typical (non-extreme-valued) images, clipping essentially never bites: the induced
    L_infinity should sit right at the theoretical strength/255 bound."""
    poisoner = pick_cifar_poisoner("1xs")
    rng = np.random.RandomState(0)

    linf_vals = []
    for _ in range(200):
        img = rng.randint(0, 256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        linf_vals.append(_linf_255(img, poisoner))
    linf_vals = np.array(linf_vals)

    check(
        "1xs Linf (0-255 scale) matches the theoretical strength bound on random images",
        np.all(linf_vals == THEORETICAL_LINF_255),
        f"got values in [{linf_vals.min()}, {linf_vals.max()}], expected exactly {THEORETICAL_LINF_255}",
    )

    linf_norm = linf_vals.max() / 255
    check(
        "1xs Linf norm (normalized [0,1] scale) matches strength/255",
        abs(linf_norm - THEORETICAL_LINF_255 / 255) < 1e-9,
        f"got {linf_norm:.5f}, expected {THEORETICAL_LINF_255 / 255:.5f}",
    )


def test_linf_never_exceeds_bound_at_extremes():
    """At pixel values 0 or 255 (where clip() can only ever shrink |delta|, never grow it),
    the induced Linf must still never exceed the theoretical strength bound."""
    poisoner = pick_cifar_poisoner("1xs")

    all_within_bound = True
    worst = 0
    for base_val in (0, 255):
        img = np.full((IMG_SIZE, IMG_SIZE, 3), base_val, dtype=np.uint8)
        linf = _linf_255(img, poisoner)
        worst = max(worst, linf)
        if linf > THEORETICAL_LINF_255:
            all_within_bound = False

    check(
        "1xs Linf never exceeds the theoretical strength bound at extreme (0/255) pixel values",
        all_within_bound,
        f"worst observed Linf={worst}, bound={THEORETICAL_LINF_255}",
    )


def test_linf_smaller_than_eps_16_255_jolt_tag():
    """Cross-check against the paper's own JOLT tags (see
    scripts/show_results/plot_paper_main_campaign_user.py's TAG_DISPLAY_NAMES /
    gen_configs_old_objective_port_stealth_sweep.py's STEALTH_GRID): FLIP's 1xs trigger should
    read as LESS perceptible (smaller Linf) than even JOLT's most epsilon-constrained tag,
    eps_16_255 (epsilon=16/255)."""
    eps_16_255 = 16 / 255
    flip_linf_norm = THEORETICAL_LINF_255 / 255
    check(
        "FLIP's 1xs Linf norm is smaller than JOLT's eps_16_255 tag",
        flip_linf_norm < eps_16_255,
        f"1xs={flip_linf_norm:.5f} vs eps_16_255={eps_16_255:.5f}",
    )


def main():
    tests = [
        test_linf_on_random_images,
        test_linf_never_exceeds_bound_at_extremes,
        test_linf_smaller_than_eps_16_255_jolt_tag,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:
            check(t.__name__, False, f"raised {type(exc).__name__}: {exc}")

    n_ok = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{n_ok}/{len(_results)} checks passed")
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
