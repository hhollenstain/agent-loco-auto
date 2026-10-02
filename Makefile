.PHONY: sync test lint doctor run watch up-mac up-nvidia pull-model

WORKSPACE ?= examples/demo-project
GOAL ?= Make the test suite pass.

sync:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff format src tests
	uv run ruff check src tests

doctor:
	uv run loco doctor

run:
	uv run loco run --workspace $(WORKSPACE) --goal "$(GOAL)"

watch:
	uv run loco watch --workspace $(WORKSPACE)

up-mac:
	docker compose -f docker-compose.yml -f docker-compose.mac.yml up --build

up-nvidia:
	docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up --build

pull-model:
	ollama pull $(or $(MODEL),qwen2.5-coder:14b)
