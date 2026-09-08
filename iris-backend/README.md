# Iris backend — token broker + usage metering

The server that makes Iris **distributable with a free tier + paid increases**.
It never ships your OpenAI key to users; instead it brokers short-lived
ephemeral Realtime tokens after checking the user's quota.

## Why this exists

The Touchless app's realtime client connects **directly** to OpenAI with an
`Authorization: Bearer <token>`. You cannot embed your own `OPENAI_API_KEY`
in the distributed binary (it would be extracted and abused). So:

```
 Touchless app                Iris backend (this)              OpenAI
 ------------                 -------------------              ------
 1. POST /v1/session   ---->  auth license key
    (license key)            check quota in D1
                             POST /v1/realtime/client_secrets ---->  mint ek_...
                        <----  { value: "ek_...", session_id, limits }   <----
 2. open WebSocket  -------------------------------------------------->  (uses ek_)
    wss://api.openai.com/v1/realtime   Authorization: Bearer ek_...
 3. POST /v1/usage     ---->  debit seconds used (on session end)
    (session_id, sec)
```

- Your `OPENAI_API_KEY` lives **only** as a Worker secret.
- Quota is enforced **at mint time** (and refined by reported usage).
- Audio is **not** relayed through the server (the client talks to OpenAI
  directly with the ephemeral token), so infra cost is tiny.

## Tiers (edit in `TIER_LIMITS`)

| tier    | realtime seconds / period | custom gestures | notes                    |
|---------|---------------------------|-----------------|--------------------------|
| free    | small (e.g. 600 = 10 min) | 3               | trial                    |
| plus    | e.g. 3600 (60 min)        | 25              | paid                     |
| premium | e.g. 9000 (150 min)       | unlimited       | paid                     |
| byok    | unlimited (own key)       | unlimited       | bypasses minting entirely|

`byok` users put their own OpenAI key in the app and never hit `/v1/session`.

## Endpoints

- `POST /v1/session` — auth (license key) → quota check → mint ephemeral token.
  - 200 `{ value, session_id, model, tier, used_seconds, limit_seconds }`
  - 402 `{ error:"quota_exceeded", tier, used_seconds, limit_seconds, upgrade_url }`
  - 401 `{ error:"invalid_license" }`
- `POST /v1/usage` — `{ session_id, seconds }` debits the ledger (best-effort,
  called by the app on session end).
- `POST /stripe/webhook` — Stripe subscription events → set user tier. (Stub.)

## Data (D1) — see `schema.sql`

- `users(license_key, tier, status, period_start, ...)`
- `usage_log(session_id, license_key, started_at, seconds)`

Usage this period = `SUM(usage_log.seconds)` for the user since `period_start`.

## Deploy (Cloudflare Workers)

```bash
cd iris-backend
npm i -g wrangler            # if needed
wrangler d1 create iris            # then put the database_id in wrangler.toml
wrangler d1 execute iris --file=schema.sql --remote
wrangler secret put OPENAI_API_KEY # your real key
wrangler deploy
```

Then point the app at it (see "Client wiring" below).

## Client wiring (Touchless side — not yet done)

`live_api` currently reads `OPENAI_API_KEY` from the environment. For the
dispersed build, the realtime client should instead:
1. `POST {IRIS_BACKEND_URL}/v1/session` with the stored license key.
2. Use the returned `value` (ek_...) as the WebSocket Bearer.
3. On session end, `POST /v1/usage` with the elapsed seconds.

Add `iris_backend_url` + `iris_license_key` to `LiveApiConfig`; when both are
set, prefer the broker over the raw key. `byok` users keep the raw-key path.

## Status

Scaffold. `POST /v1/session` (mint + quota) and `POST /v1/usage` are
implemented; Stripe webhook + license provisioning are stubs to fill in.
