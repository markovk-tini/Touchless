-- Touchless telemetry: D1 schema. Run once with `wrangler d1 execute`
-- (see README) to bootstrap the events table + indexes used by the
-- Worker for inserts and the dashboard for queries.

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    install_id  TEXT    NOT NULL,
    event       TEXT    NOT NULL,
    properties  TEXT    NOT NULL DEFAULT '{}',  -- JSON blob
    timestamp   TEXT    NOT NULL,               -- ISO-8601 from client
    received_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS events_install_idx     ON events(install_id);
CREATE INDEX IF NOT EXISTS events_event_idx       ON events(event);
CREATE INDEX IF NOT EXISTS events_received_at_idx ON events(received_at);
CREATE INDEX IF NOT EXISTS events_event_received  ON events(event, received_at);
