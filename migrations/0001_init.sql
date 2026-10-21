CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE articles (
    id          bigint PRIMARY KEY,
    source_act  text NOT NULL,
    path        text NOT NULL,
    text        text NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('italian', text)) STORED
);
CREATE INDEX articles_tsv_idx ON articles USING gin (tsv);

CREATE TABLE article_chunks (
    article_id  bigint NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    ord         int NOT NULL,
    text        text NOT NULL,
    embedding   vector(1024) NOT NULL,
    PRIMARY KEY (article_id, ord)
);
CREATE INDEX article_chunks_embedding_idx ON article_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE article_refs (
    src_id  bigint NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    dst_id  bigint NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    PRIMARY KEY (src_id, dst_id)
);
CREATE INDEX article_refs_dst_idx ON article_refs (dst_id);

CREATE TABLE questions (
    id      text PRIMARY KEY,
    split   text NOT NULL,
    text    text NOT NULL,
    domain  text NOT NULL,
    topics  text NOT NULL
);

CREATE TABLE qrels (
    question_id  text NOT NULL REFERENCES questions (id) ON DELETE CASCADE,
    article_id   bigint NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    PRIMARY KEY (question_id, article_id)
);

CREATE TABLE embedding_cache (
    key        text PRIMARY KEY,
    embedding  vector(1024) NOT NULL
);
