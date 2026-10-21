CREATE TABLE session_quota (
    session_id  text NOT NULL,
    day         date NOT NULL,
    count       int NOT NULL,
    PRIMARY KEY (session_id, day)
);
