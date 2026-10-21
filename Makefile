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
