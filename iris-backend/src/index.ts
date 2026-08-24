/**
 * Iris backend — ephemeral Realtime token broker + usage metering.
 *
 * The Touchless app calls POST /v1/session with its license key; we check the
 * user's tier + quota in D1, mint a short-lived OpenAI Realtime token (ek_...)
 * with our secret key, and return it. The app then connects directly to
 * OpenAI's WebSocket using that ephemeral token. Audio never touches us.
 *
 * See README.md for the full flow and deploy steps.
 */

export interface Env {
  DB: D1Database;
  OPENAI_API_KEY: string;
  REALTIME_MODEL: string;
  UPGRADE_URL: string;
  QUOTA_PERIOD_SECONDS: string;
  STRIPE_WEBHOOK_SECRET?: string;
}

// Realtime seconds allowed per quota window, by tier. byok = own key (never
// reaches this broker). Tune freely — these are the product's free/paid knobs.
const TIER_LIMITS: Record<string, number> = {
  free: 600, // 10 min
  plus: 3600, // 60 min
  premium: 9000, // 150 min
  byok: Number.POSITIVE_INFINITY,
};

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function now(): number {
  return Math.floor(Date.now() / 1000);
}

function licenseFromAuth(req: Request): string | null {
  const h = req.headers.get("authorization") || "";
  const m = h.match(/^Bearer\s+(.+)$/i);
  return m ? m[1].trim() : null;
}

interface UserRow {
  license_key: string;
  tier: string;
  status: string;
  period_start: number;
}

async function getUser(env: Env, license: string): Promise<UserRow | null> {
  return env.DB.prepare(
    "SELECT license_key, tier, status, period_start FROM users WHERE license_key = ?"
  )
    .bind(license)
    .first<UserRow>();
}

async function usedSeconds(env: Env, license: string, since: number): Promise<number> {
  const row = await env.DB.prepare(
    "SELECT COALESCE(SUM(seconds),0) AS s FROM usage_log WHERE license_key = ? AND started_at >= ?"
  )
    .bind(license, since)
    .first<{ s: number }>();
  return row ? Number(row.s) : 0;
}

async function mintEphemeralToken(env: Env): Promise<{ value: string; expires_at?: number }> {
  const resp = await fetch("https://api.openai.com/v1/realtime/client_secrets", {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.OPENAI_API_KEY}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({ session: { type: "realtime", model: env.REALTIME_MODEL } }),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`openai client_secrets ${resp.status}: ${text.slice(0, 300)}`);
  }
  const data: any = await resp.json();
  // Response shape: { value: "ek_...", expires_at?, session: {...} }
  const value: string = data.value || data.client_secret?.value;
  if (!value) throw new Error("no ephemeral token in OpenAI response");
  return { value, expires_at: data.expires_at };
}

async function handleSession(req: Request, env: Env): Promise<Response> {
  const license = licenseFromAuth(req);
  if (!license) return json({ error: "missing_license" }, 401);

  const user = await getUser(env, license);
  if (!user || user.status !== "active") return json({ error: "invalid_license" }, 401);

  const tier = TIER_LIMITS[user.tier] !== undefined ? user.tier : "free";
  const limit = TIER_LIMITS[tier];

  // Roll the quota window forward if it has elapsed.
  const periodLen = parseInt(env.QUOTA_PERIOD_SECONDS || "2592000", 10);
  let periodStart = user.period_start || 0;
  if (now() - periodStart > periodLen) {
    periodStart = now();
    await env.DB.prepare("UPDATE users SET period_start = ? WHERE license_key = ?")
      .bind(periodStart, license)
      .run();
  }

  const used = await usedSeconds(env, license, periodStart);
  if (used >= limit) {
    return json(
      {
        error: "quota_exceeded",
        tier,
        used_seconds: used,
        limit_seconds: limit === Infinity ? null : limit,
        upgrade_url: env.UPGRADE_URL,
      },
      402
    );
  }

  let token: { value: string; expires_at?: number };
  try {
    token = await mintEphemeralToken(env);
  } catch (e: any) {
    return json({ error: "mint_failed", detail: String(e?.message || e) }, 502);
  }

  // Record the session so /v1/usage can debit it later.
  const sessionId = crypto.randomUUID();
  await env.DB.prepare(
    "INSERT INTO usage_log (session_id, license_key, started_at, seconds) VALUES (?,?,?,0)"
  )
    .bind(sessionId, license, now())
    .run();

  return json({
    value: token.value,
    expires_at: token.expires_at,
    session_id: sessionId,
    model: env.REALTIME_MODEL,
    tier,
    used_seconds: used,
    limit_seconds: limit === Infinity ? null : limit,
  });
}

async function handleUsage(req: Request, env: Env): Promise<Response> {
  const license = licenseFromAuth(req);
  if (!license) return json({ error: "missing_license" }, 401);
  let body: any;
  try {
    body = await req.json();
  } catch {
    return json({ error: "bad_json" }, 400);
  }
  const sessionId = String(body.session_id || "");
  const seconds = Math.max(0, Math.min(86400, Math.floor(Number(body.seconds) || 0)));
  if (!sessionId) return json({ error: "missing_session_id" }, 400);
  await env.DB.prepare(
    "UPDATE usage_log SET seconds = ? WHERE session_id = ? AND license_key = ?"
  )
    .bind(seconds, sessionId, license)
    .run();
  return json({ ok: true });
}

async function handleStripeWebhook(_req: Request, _env: Env): Promise<Response> {
  // TODO: verify signature with STRIPE_WEBHOOK_SECRET, then on
  // checkout.session.completed / customer.subscription.updated|deleted:
  //   - upsert users(license_key, tier, status, stripe_customer_id, email)
  //   - generate + email a license key on first purchase
  return json({ ok: true, note: "stripe webhook stub" });
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (req.method === "POST" && url.pathname === "/v1/session") return handleSession(req, env);
    if (req.method === "POST" && url.pathname === "/v1/usage") return handleUsage(req, env);
    if (req.method === "POST" && url.pathname === "/stripe/webhook")
      return handleStripeWebhook(req, env);
    if (url.pathname === "/health") return json({ ok: true });
    return json({ error: "not_found" }, 404);
  },
};
