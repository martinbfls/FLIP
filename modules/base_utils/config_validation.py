"""
Shared schema validation for experiment configs.

Every module's config table is validated against `schemas/<module_name>.toml`: every schema
key must be present in the config unless it is listed under that schema's `[OPTIONAL]` table,
and every config key must exist in the schema (as a required or optional key). This is the
exact check `run_experiment.py` performs before running a module -- previously duplicated,
near-identically, in `run_experiment.py` itself and in three separate `gen_configs.py`
generators (federated_generate_labels_trigger, federated_generate_labels_trigger_joint,
federated_optimizing_trigger_policy), which had already started drifting apart. This module is
the single source of truth for that check; both run_experiment.py and every gen_configs*.py
generator should import it instead of re-implementing it.
"""

from pathlib import Path

import toml

SCHEMAS_DIR = Path("schemas")


class ConfigValidationError(AssertionError):
    """Raised when a config's keys don't match its schema. Subclasses AssertionError so
    existing `except AssertionError` / bare `assert` callers keep working unchanged."""


def validate_module_config(module_name: str, module_config: dict, schemas_dir: Path = SCHEMAS_DIR):
    """Validates one module's config table against schemas/<module_name>.toml. Raises
    ConfigValidationError on any mismatch; returns nothing on success."""
    schema_path = schemas_dir / f"{module_name}.toml"
    if not schema_path.exists():
        raise ConfigValidationError(
            f"Malformed module! Schema {schema_path} does not exist."
        )

    schema = toml.load(schema_path)
    optionals = list(schema.get("OPTIONAL", {}).keys())

    schema_keys = set(schema[module_name].keys())
    config_keys = set(module_config.keys())

    missing = [k for k in schema_keys - config_keys if k not in optionals]
    extra = [k for k in config_keys - schema_keys if k not in optionals]

    if missing:
        raise ConfigValidationError(
            f"Malformed config ([{module_name}]): missing required keys {missing}"
        )
    if extra:
        raise ConfigValidationError(
            f"Malformed config ([{module_name}]): unknown keys {extra} "
            "(not in schema, not optional)"
        )


def validate_config_file(path: Path, schemas_dir: Path = SCHEMAS_DIR):
    """Validates every module table in a config.toml file. Raises ConfigValidationError
    (with the config path in the message) on any mismatch."""
    exp_toml = toml.load(path)
    for module_name, module_config in exp_toml.items():
        try:
            validate_module_config(module_name, module_config, schemas_dir=schemas_dir)
        except ConfigValidationError as e:
            raise ConfigValidationError(f"{e} (file: {path})") from e


def write_config(path: Path, content: str):
    """Writes a generated config.toml, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[OK] Config written to {path}")
