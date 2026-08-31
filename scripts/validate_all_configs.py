"""
Walks experiments/ and schema-validates every config.toml found, using the same check
run_experiment.py performs before running a module (modules/base_utils/config_validation.py).

Useful to catch config/schema drift across an entire campaign tree at once -- e.g. after
editing a schema's [OPTIONAL] table, or after a gen_configs*.py run -- instead of one config
at a time when each module actually executes.

Run:  python scripts/validate_all_configs.py [root]
Exits non-zero if any config fails validation.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.base_utils.config_validation import ConfigValidationError, validate_config_file


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments")
    if not root.exists():
        print(f"[validate_all_configs] {root} does not exist -- nothing to validate.")
        return 0

    config_paths = sorted(root.rglob("config.toml"))
    if not config_paths:
        print(f"[validate_all_configs] no config.toml found under {root}.")
        return 0

    n_ok, failures = 0, []
    for path in config_paths:
        try:
            validate_config_file(path)
            n_ok += 1
        except Exception as e:
            failures.append((path, e))

    for path, e in failures:
        print(f"[FAIL] {path}: {e}")

    print(f"\n{n_ok}/{len(config_paths)} configs valid under {root}.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
