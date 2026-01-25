# Contributing to MeshPackageLean

Thank you for contributing to MeshPackageLean! This document outlines the development workflow and pre-flight checks that must pass before your Pull Request can be merged.

## Pre-Flight Checks

Before submitting a Pull Request, ensure all integrity checks pass locally:

```bash
make check
```

This runs three critical checks:

1. **Type Safety** (`make type-check`): Runs mypy on the backend code to catch type errors
2. **Schema Sync** (`make schema-check`): Verifies SQLAlchemy models match Alembic migrations
3. **Env Audit** (`make env-check`): Ensures all environment variables are documented in `.env.example`

### Running Checks Individually

#### Type Checking
```bash
cd backend
mypy . --config-file mypy.ini
```

**Note**: Currently, some files have `ignore_errors = True` in `mypy.ini` to allow gradual migration. These files should be fixed incrementally by:
1. Removing the `ignore_errors = True` entry for the file
2. Fixing the type errors
3. Ensuring the check still passes

#### Schema Check
```bash
cd backend
alembic check
```

This command verifies that your SQLAlchemy models in `models.py` are synchronized with Alembic migration scripts. If you've modified models, you must create a migration:

```bash
cd backend
alembic revision --autogenerate -m "description of changes"
```

Review the generated migration file in `backend/alembic/versions/` before committing.

#### Environment Variable Audit
```bash
python3 scripts/check_env_vars.py
```

This script scans all Python files in `backend/` for `os.getenv()` and `os.environ.get()` calls, then verifies each referenced variable exists in `backend/.env.example`.

## Database Migrations

When modifying `backend/models.py`:

1. **Generate a new migration:**
   ```bash
   cd backend
   alembic revision --autogenerate -m "description of changes"
   ```

2. **Review the generated migration file** in `backend/alembic/versions/`

3. **Test the migration:**
   ```bash
   alembic upgrade head
   alembic check
   ```

4. **Commit both** `models.py` and the migration file together

### Migration Best Practices

- Always review auto-generated migrations before committing
- Test migrations on a copy of production data if possible
- Use descriptive migration messages
- Never edit existing migration files (create a new one instead)

## Environment Variables

All environment variables used in code **must** be documented in `backend/.env.example`.

The `make env-check` command will fail if:
- Code uses `os.getenv('VAR')` but `VAR` is missing from `.env.example`
- Code uses `os.environ.get('VAR')` but `VAR` is missing from `.env.example`

### Adding a New Environment Variable

1. Add the variable to `backend/.env.example` with a comment explaining its purpose
2. Use the variable in your code with a sensible default if appropriate
3. Run `make env-check` to verify it's detected

## Pull Request Process

1. **Create a feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the project's coding standards

3. **Run integrity checks locally:**
   ```bash
   make check
   ```

4. **Commit your changes** with descriptive commit messages
   ```bash
   git commit -m "feat/add-new-feature"
   ```

5. **Push and create a Pull Request**

6. **Ensure all GitHub Actions checks pass** - PRs will be automatically blocked from merging if any check fails

## GitHub Actions

The Pre-Flight Integrity Checks workflow runs automatically on every Pull Request targeting `main`. It will:

- ✅ Run mypy type checking
- ✅ Verify Alembic migrations are in sync
- ✅ Audit environment variable documentation

**Your PR will be blocked from merging if any check fails.**

## Development Setup

### Prerequisites

- Python 3.11+
- PostgreSQL (for production) or SQLite (for local development)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   make install-dev
   ```

3. Copy `.env.example` to `.env` and configure:
   ```bash
   cp backend/.env.example backend/.env
   # Edit backend/.env with your local settings
   ```

4. Initialize the database:
   ```bash
   cd backend
   alembic upgrade head
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for public functions and classes
- Keep functions focused and small

## Questions?

If you have questions about the contribution process, please open an issue or contact the maintainers.

---

**Remember**: Always run `make check` before pushing your changes!

