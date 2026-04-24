# Justfile recipes for development

# default to showing the list of commands
default:
	@just --list

dev-shell:
	pixi shell --frozen -e dev

cpu-dev-shell:
	pixi shell --frozen -e dev-cpu

lint:
	pixi run -e dev-cpu ruff check src/ tests/

format:
	pixi run -e dev-cpu ruff format src/ tests/

format-check:
	pixi run -e dev-cpu ruff format --check src/ tests/

typecheck:
	pixi run -e dev-cpu pyrefly check

test:
	pixi run -e dev-cpu pytest -xvs

# Run all checks that should pass before committing.
pre-commit: lint format-check typecheck
