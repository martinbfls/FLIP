#!/usr/bin/env python3

import re
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================================
# Configuration
# ============================================================================

LOG_PATTERN = (
    "/shared/data1/Projects/DLWP/j1067582/martin/FLIP/"
    "logs_slurm/gen_r32p_1vs0_cifar_backdoor_mean_1xs_{run_id}.log"
)

OUTPUT_DIR = Path("out/graphs")

RUN_IDS = range(1, 11)


# ============================================================================
# Extraction
# ============================================================================

def extract_g_loss(log_path: Path):
    """
    Extract (iteration, g_loss) pairs from a tqdm-like log.

    If multiple values are logged for the same iteration, keep the last one.
    """

    pattern = re.compile(
        r"(\d+)/(\d+).*?g_loss=([0-9.eE+-]+)"
    )

    iterations = []
    losses = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)

            if match is None:
                continue

            iteration = int(match.group(1))
            g_loss = float(match.group(3))

            # Several tqdm updates can exist for the same iteration.
            # Keep the last observed g_loss.
            if iterations and iteration == iterations[-1]:
                losses[-1] = g_loss
            else:
                iterations.append(iteration)
                losses.append(g_loss)

    return iterations, losses


# ============================================================================
# Plotting
# ============================================================================

def plot_g_loss(iterations, losses, run_id, output_path: Path):
    plt.figure(figsize=(10, 6))

    plt.plot(iterations, losses, linewidth=1)

    plt.xlabel("Optimization step")
    plt.ylabel("g_loss")
    plt.title(f"Evolution of g_loss — run_id={run_id}")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for run_id in RUN_IDS:
        log_path = Path(
            LOG_PATTERN.format(run_id=run_id)
        )

        if not log_path.exists():
            print(f"[WARNING] File not found: {log_path}")
            continue

        print(f"Processing run_id={run_id}: {log_path}")

        iterations, losses = extract_g_loss(log_path)

        if not losses:
            print(
                f"[WARNING] No g_loss values found "
                f"for run_id={run_id}"
            )
            continue

        output_path = (
            OUTPUT_DIR
            / f"g_loss_run_{run_id}.png"
        )

        plot_g_loss(
            iterations=iterations,
            losses=losses,
            run_id=run_id,
            output_path=output_path,
        )

        print(
            f"  Found {len(losses)} values "
            f"from step {iterations[0]} "
            f"to {iterations[-1]}"
        )
        print(f"  Saved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()