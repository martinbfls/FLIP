"""
Run experiment based on config.
"""

import sys
import os

import toml
from collections import OrderedDict

sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'modules')
    ))

from modules.base_utils import util
from modules.base_utils.config_validation import ConfigValidationError, validate_module_config


experiment_name = sys.argv[1]
print(f"Running experiment: {experiment_name}")
args = util.extract_toml(experiment_name)
resolves_to = {}

for module_name, module_config in args.items():
    relative_path = "schemas/" + module_name + ".toml"
    full_path = util.generate_full_path(relative_path)

    # Check if path exists
    if not os.path.exists(full_path):
        print(f"Malformed module! Module {module_name} does not exist!")
        exit()

    schema = toml.load(full_path, _dict=OrderedDict)

    # Check if config is well formed (schema key presence, both directions -- see
    # modules/base_utils/config_validation.py, shared with every gen_configs*.py generator).
    try:
        validate_module_config(module_name, module_config)
    except ConfigValidationError as e:
        print(f"Malformed config: {e}")
        exit()

    # Check if module has distinct module name
    if 'INTERNAL' in schema:
        resolves_to[module_name] = schema['INTERNAL']['module_name']

    # Check for slurm and import and run module
    module_file = resolves_to.get(module_name, module_name)

    if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        slurm_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        __import__(f"{module_file}", fromlist=["run_module"]).run_module.run(
            experiment_name, module_name, slurm_id=slurm_id)
    else:
        __import__(f"{module_file}", fromlist=["run_module"]).run_module.run(
            experiment_name, module_name)
