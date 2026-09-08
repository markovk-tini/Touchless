-- Iris backend D1 schema.

CREATE TABLE IF NOT EXISTS users (
  license_key   TEXT PRIMARY KEY,         -- the key the app stores + sends
  tier          TEXT NOT NULL DEFAULT 'free',  -- free | plus | premium | byok
  status        TEXT NOT NULL DEFAULT 'active',-- active | canceled | suspended
  stripe_customer_id TEXT,
  email         TEXT,
  period_start  INTEGER NOT NULL DEFAULT 0, -- unix secs; quota window start
  created_at    INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

-- One row per realtime session mint; `seconds` is filled in by /v1/usage.
CREATE TABLE IF NOT EXISTS usage_log (
  session_id   TEXT PRIMARY KEY,
  license_key  TEXT NOT NULL,
  started_at   INTEGER NOT NULL,
  seconds      INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (license_key) REFERENCES users(license_key)
);

CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_log(license_key, started_at);

-- A demo free-tier key so you can test before Stripe is wired.
INSERT OR IGNORE INTO users (license_key, tier, status, period_start)
VALUES ('demo-free-key', 'free', 'active', strftime('%s','now'));
