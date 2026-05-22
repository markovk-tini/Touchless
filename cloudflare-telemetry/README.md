# Touchless Telemetry — Cloudflare Worker + D1

This directory holds the backend that receives anonymous usage events from the Touchless app. Stack:

- **Cloudflare Worker** (TypeScript) — accepts batched event POSTs and serves a small dashboard.
- **Cloudflare D1** (serverless SQLite) — stores the events.

## What gets stored

One row per event:

| column      | type | notes                                   |
| ----------- | ---- | --------------------------------------- |
| install_id  | TEXT | random UUID4 from the user's app config |
| event       | TEXT | event name, e.g. `action_fired`         |
| properties  | TEXT | JSON blob with event-specific fields    |
| timestamp   | TEXT | client-supplied ISO-8601                |
| received_at | TEXT | server-side ISO-8601 (default UTC now)  |

No personal data: no user names, no emails, no IPs, no gesture landmarks, no voice transcripts, no file paths.

---

## One-time setup (≈10 minutes)

You need:

- [Node.js 18+](https://nodejs.org) installed locally.
- A Cloudflare account (the same one you already use for `r2:hgr-downloads`).
- A terminal in **this directory** (`cloudflare-telemetry/`).

### 1. Install Wrangler + log in

```bash
npm install
npx wrangler login
```

`wrangler login` opens your browser and asks you to authorize Wrangler against your Cloudflare account.

### 2. Create the D1 database

```bash
npx wrangler d1 create touchless-events
```

Wrangler prints a block that looks like:

```
✅ Successfully created DB 'touchless-events'

[[d1_databases]]
binding = "DB"
database_name = "touchless-events"
database_id = "abc12345-6789-..."
```

**Copy the `database_id`.** Open `wrangler.toml` and replace the `CHANGEME-paste-the-d1-database-id-here` value with it.

### 3. Apply the schema (creates the `events` table + indexes)

Local copy (for `wrangler dev`):

```bash
npx wrangler d1 execute touchless-events --file=./schema.sql
```

Remote copy (the production database):

```bash
npx wrangler d1 execute touchless-events --file=./schema.sql --remote
```

### 4. Set the shared-secret token

This is the value the Touchless app will send as `api_key` and the dashboard will require as `?token=`. Pick a long random string and keep it private — anyone with this token can write events or view the dashboard.

```bash
npx wrangler secret put SHARED_SECRET
```

It prompts you to paste the value. Use something like a UUID:

```
# Paste this when wrangler asks; KEEP IT SECRET.
0c9e8c5f9b7e4a8d8b2d7e5d3f1a2b3c
```

### 5. Deploy the Worker

```bash
npx wrangler deploy
```

Wrangler prints the deployed URL, e.g.:

```
✨ Successfully deployed
   https://touchless-telemetry.<your-subdomain>.workers.dev
```

**Copy that URL.** That's your telemetry endpoint.

### 6. Wire the URL + secret into Touchless

Two ways to do it. Pick **A** (env-var, no source change) for development; **B** (hardcoded) for builds you ship to users.

**Option A — environment variable** (easy, doesn't survive a fresh build):

Set these before running the Touchless app:

```
TOUCHLESS_TELEMETRY_HOST=https://touchless-telemetry.<your-subdomain>.workers.dev
TOUCHLESS_TELEMETRY_API_KEY=0c9e8c5f9b7e4a8d8b2d7e5d3f1a2b3c
```

On Windows (PowerShell):

```powershell
$env:TOUCHLESS_TELEMETRY_HOST = "https://touchless-telemetry.<your-subdomain>.workers.dev"
$env:TOUCHLESS_TELEMETRY_API_KEY = "0c9e8c5f9b7e4a8d8b2d7e5d3f1a2b3c"
python run_app.py
```

**Option B — bake into the build** (so end users send telemetry):

Edit `src/hgr/telemetry/config.py` and set:

```python
POSTHOG_API_KEY = "0c9e8c5f9b7e4a8d8b2d7e5d3f1a2b3c"
POSTHOG_HOST = "https://touchless-telemetry.<your-subdomain>.workers.dev"
```

Yes, the constants are still named `POSTHOG_*` — they're just the endpoint key/host pair, name kept for code stability. Could rename later.

### 7. View the dashboard

Open in a browser:

```
https://touchless-telemetry.<your-subdomain>.workers.dev/?token=YOUR_SHARED_SECRET
```

Bookmark that URL. Refresh to see new events. Empty stats are normal until your first run of Touchless with the env vars / config set.

---

## Verifying it works

1. Run Touchless with the env vars set.
2. Click around (open Settings, click Start, fire a gesture).
3. Wait ~30 seconds (the client batches every 30s).
4. Refresh the dashboard — `unique_installs` should be `1`, `total_events` should be `>0`.

You can also tail the events directly with:

```bash
npx wrangler d1 execute touchless-events --remote --command "SELECT event, COUNT(*) FROM events GROUP BY event ORDER BY 2 DESC"
```

---

## Common queries

Daily active installs in the last 14 days:

```sql
SELECT DATE(received_at) AS day, COUNT(DISTINCT install_id) AS users
FROM events
WHERE received_at >= datetime('now', '-14 days')
GROUP BY day
ORDER BY day DESC;
```

Top fired actions in the last 7 days:

```sql
SELECT json_extract(properties, '$.action_id') AS action_id, COUNT(*) AS n
FROM events
WHERE event = 'action_fired'
  AND received_at >= datetime('now', '-7 days')
GROUP BY action_id
ORDER BY n DESC
LIMIT 20;
```

Tutorial step-completion funnel:

```sql
SELECT json_extract(properties, '$.step_key') AS step,
       COUNT(DISTINCT install_id) AS users
FROM events
WHERE event = 'tutorial_step_entered'
GROUP BY step
ORDER BY users DESC;
```

Run any of these via:

```bash
npx wrangler d1 execute touchless-events --remote --command "<the SQL>"
```

---

## Cost

Free tier covers:

- 100,000 Worker requests / day
- 5 GB D1 storage
- 5 million D1 row reads / day
- 100,000 D1 row writes / day

For app-analytics scale you'll be inside the free tier indefinitely. Heavy tracking from a couple of hundred installs is still ~5,000 events/day.

---

## Updating the Worker later

Edit `src/index.ts`, then:

```bash
npx wrangler deploy
```

That's it. No restart, no migration. Schema changes go through `schema.sql` + `wrangler d1 execute --remote`.

---

## Disabling telemetry

Set `TOUCHLESS_TELEMETRY_API_KEY` to an empty string (or unset it) and rebuild Touchless. The client becomes a no-op end to end — no requests sent, no UUID exposed.
