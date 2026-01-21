.PHONY: check type-check schema-check env-check install-dev validate help

help:
	@echo "Available targets:"
	@echo "  make check         - Run all integrity checks"
	@echo "  make type-check    - Run mypy type checking"
	@echo "  make schema-check  - Run Alembic schema sync check"
	@echo "  make env-check     - Run environment variable audit"
	@echo "  make validate      - Run happy path validation script"
	@echo "  make install-dev   - Install development dependencies"

check: type-check schema-check env-check
	@echo ""
	@echo "✅ All integrity checks passed!"

type-check:
	@echo "Running mypy type checks..."
	cd backend && python -m mypy . --config-file mypy.ini

schema-check:
	@echo "Running Alembic schema sync check..."
	cd backend && alembic check

env-check:
	@echo "Running environment variable audit..."
	python3 scripts/check_env_vars.py

install-dev:
	@echo "Installing development dependencies..."
	pip install -r backend/requirements.txt

validate:
	@echo "Running happy path validation..."
	python -u scripts/validate_happy_path.py --url http://127.0.0.1:5000
