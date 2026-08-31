.PHONY: help setup lock lint test validate-configs gen-configs smoke-test registry

help:
	@echo "Targets:"
	@echo "  setup             Install runtime dependencies (requirements.txt)"
	@echo "  lock              Regenerate requirements.lock.txt / requirements-aarch64.lock.txt"
	@echo "  lint              Run ruff (see pyproject.toml)"
	@echo "  test              Run tests/test_experiment_tracker.py"
	@echo "  gen-configs       Regenerate every campaign's configs (base_utils + per-module generators)"
	@echo "  validate-configs  Schema-validate every config.toml already under experiments/"
	@echo "  smoke-test        Run a tiny end-to-end pipeline on synthetic data"
	@echo "  registry          Build an index of experiments/ runs (see scripts/list_experiments.py)"

setup:
	pip install -r requirements.txt

lock:
	uv pip compile requirements.txt -o requirements.lock.txt --python-version 3.10
	uv pip compile requirements-aarch64.txt -o requirements-aarch64.lock.txt --python-version 3.10

lint:
	uv tool run ruff check .

test:
	python3 tests/test_experiment_tracker.py

gen-configs:
	python3 -m modules.base_utils.gen_configs
	python3 -m modules.federated_generate_labels_trigger.gen_configs
	python3 -m modules.federated_generate_labels_trigger_joint.gen_configs
	python3 -m modules.federated_optimizing_trigger_policy.gen_configs

validate-configs:
	python3 scripts/validate_all_configs.py

smoke-test:
	python3 scripts/smoke_test_pipeline.py

registry:
	python3 scripts/list_experiments.py
