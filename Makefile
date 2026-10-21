REV := d765a61998a0dc962a9c06394ab546b5a5ee0aa5
HF  := https://huggingface.co/datasets/jurifindit/JuriFindIT/resolve/$(REV)

data/raw/corpus.parquet:
	mkdir -p data/raw
	curl -fsSL -o data/raw/corpus.parquet $(HF)/corpus/corpus.parquet
	curl -fsSL -o data/raw/train.parquet $(HF)/questions/train.parquet
	curl -fsSL -o data/raw/validation.parquet $(HF)/questions/validation.parquet

download: data/raw/corpus.parquet

build-seed: download
	uv run vigilex build-seed --release data/raw --out data/seed

.PHONY: download build-seed

# Bare `make` would otherwise run the dataset download (first target above).
.DEFAULT_GOAL := test
export VIGILEX_TEST_SERVER_URL ?= postgresql://vigilex:vigilex@localhost:5432/postgres
test-unit test-integration: export VIGILEX_LLM := fake

up:
	docker compose up -d --wait db

migrate:
	uv run vigilex migrate

seed: migrate
	uv run vigilex load-seed

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

test: test-unit test-integration

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

fmt:
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: up migrate seed test-unit test-integration test lint fmt
