CREATE TABLE eval_runs (
    id                bigserial PRIMARY KEY,
    created_at        timestamptz NOT NULL DEFAULT now(),
    git_sha           text,
    config_name       text NOT NULL,
    config_hash       text NOT NULL,
    retriever_config  jsonb NOT NULL,
    split             text NOT NULL,
    n_questions       int NOT NULL,
    metrics           jsonb NOT NULL,
    per_query         jsonb NOT NULL,
    judge             boolean NOT NULL DEFAULT false
);
CREATE INDEX eval_runs_config_hash_idx ON eval_runs (config_hash, created_at DESC);
