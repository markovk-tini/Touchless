/**
 * Touchless telemetry Worker.
 *
 * Endpoints:
 *   POST /batch/         — receive a batch of events from the client.
 *                          Body shape matches what the Python
 *                          TelemetryClient already sends (which is
 *                          PostHog-compatible), so the client code
 *                          doesn't change when swapping backends.
 *
 *   GET  /api/stats      — JSON aggregates for the dashboard. Auth:
 *                          ?token=<SHARED_SECRET> query param.
 *
 *   GET  /               — minimal HTML dashboard. Same auth as
 *                          /api/stats; surface the token via the
 *                          ?token= query param when bookmarking.
 *
 * Auth: a single SHARED_SECRET stored as a Worker secret. The
 * Python client sends it as the `api_key` field on each batch.
 * Set it once with:
 *   wrangler secret put SHARED_SECRET
 */

export interface Env {
    DB: D1Database;
    SHARED_SECRET: string;
}

interface Event {
    event?: string;
    distinct_id?: string;
    timestamp?: string;
    properties?: Record<string, unknown>;
}

interface BatchPayload {
    api_key?: string;
    batch?: Event[];
}

const JSON_HEADERS = { "Content-Type": "application/json; charset=utf-8" };
const HTML_HEADERS = { "Content-Type": "text/html; charset=utf-8" };

export default {
    async fetch(request: Request, env: Env): Promise<Response> {
        const url = new URL(request.url);

        if (request.method === "OPTIONS") {
            // Permissive CORS so the dashboard can call /api/stats
            // from any origin if you ever embed it elsewhere.
            return new Response(null, {
                headers: {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, X-Token",
                },
            });
        }

        if (request.method === "POST" && url.pathname === "/batch/") {
            return handleBatch(request, env);
        }
        if (request.method === "POST" && url.pathname === "/api/consolidate") {
            return handleConsolidate(request, env);
        }
        if (request.method === "GET" && url.pathname === "/api/sessions") {
            return handleUserSessions(request, env);
        }
        if (request.method === "GET" && url.pathname === "/api/stats") {
            return handleStats(request, env);
        }
        if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/dashboard")) {
            return new Response(DASHBOARD_HTML, { headers: HTML_HEADERS });
        }
        if (request.method === "GET" && url.pathname === "/health") {
            return new Response("ok", { headers: { "Content-Type": "text/plain" } });
        }
        return new Response("Not Found", { status: 404 });
    },
};

async function handleBatch(request: Request, env: Env): Promise<Response> {
    let body: BatchPayload;
    try {
        body = (await request.json()) as BatchPayload;
    } catch {
        return jsonResponse({ ok: false, error: "invalid json" }, 400);
    }

    if (!body || body.api_key !== env.SHARED_SECRET) {
        return jsonResponse({ ok: false, error: "unauthorized" }, 401);
    }

    const batch = Array.isArray(body.batch) ? body.batch : [];
    if (batch.length === 0) {
        return jsonResponse({ ok: true, inserted: 0 });
    }

    // Cap per-request batch size as a defensive measure — the
    // Python client uses 50 anyway. 500 is a generous ceiling.
    const truncated = batch.slice(0, 500);

    const stmt = env.DB.prepare(
        "INSERT INTO events (install_id, event, properties, timestamp) VALUES (?1, ?2, ?3, ?4)"
    );

    const statements = truncated.map((evt) => {
        const installId = String(evt.distinct_id ?? "").slice(0, 64);
        const eventName = String(evt.event ?? "").slice(0, 128);
        const props = evt.properties && typeof evt.properties === "object"
            ? JSON.stringify(evt.properties)
            : "{}";
        const ts = String(evt.timestamp ?? new Date().toISOString()).slice(0, 64);
        return stmt.bind(installId, eventName, props, ts);
    });

    try {
        await env.DB.batch(statements);
    } catch (err) {
        return jsonResponse({ ok: false, error: "db error" }, 500);
    }

    return jsonResponse({ ok: true, inserted: truncated.length });
}

interface ConsolidatePayload {
    api_key?: string;
    target_install_id?: string;
    // Optional explicit list of source IDs to merge into the target.
    merge_ids?: string[];
    // If true and merge_ids is empty/missing, merges EVERY install_id
    // in the table that isn't already the target. Use with care —
    // this is the "I'm the only user, fold all my ghosts" path.
    merge_all_others?: boolean;
}

// Re-tag historic events under a single install_id. Two modes:
//   (1) merge_ids supplied → rewrite only those rows
//   (2) merge_all_others=true → rewrite every row whose install_id
//       differs from target_install_id
// Auth: same SHARED_SECRET the /batch/ endpoint uses. The Python client
// can drive this directly after a migration; the dashboard exposes a
// button for one-shot manual consolidation.
async function handleConsolidate(request: Request, env: Env): Promise<Response> {
    let body: ConsolidatePayload;
    try {
        body = (await request.json()) as ConsolidatePayload;
    } catch {
        return jsonResponse({ ok: false, error: "invalid json" }, 400);
    }
    if (!body || body.api_key !== env.SHARED_SECRET) {
        return jsonResponse({ ok: false, error: "unauthorized" }, 401);
    }
    const target = String(body.target_install_id ?? "").slice(0, 64).trim();
    if (!target) {
        return jsonResponse({ ok: false, error: "missing target_install_id" }, 400);
    }

    let mergeIds: string[] = Array.isArray(body.merge_ids)
        ? body.merge_ids.map((s) => String(s).slice(0, 64).trim()).filter((s) => s && s !== target)
        : [];

    if (mergeIds.length === 0) {
        if (!body.merge_all_others) {
            return jsonResponse(
                { ok: false, error: "supply merge_ids or set merge_all_others=true" },
                400,
            );
        }
        // Pull every install_id != target straight from the DB. Cap
        // at 5000 IDs as a sanity bound — far above any realistic ghost
        // count and keeps the bound-parameter list inside D1's limits.
        const ghosts = await env.DB.prepare(
            "SELECT DISTINCT install_id FROM events WHERE install_id != ?1 LIMIT 5000",
        )
            .bind(target)
            .all<{ install_id: string }>();
        mergeIds = (ghosts.results ?? []).map((r) => r.install_id).filter(Boolean);
    }

    if (mergeIds.length === 0) {
        return jsonResponse({ ok: true, rows_updated: 0, ids_merged: [] });
    }

    // D1 caps bound parameters per statement. Process in chunks of 100
    // — well under any platform cap and keeps each UPDATE small.
    const CHUNK = 100;
    let totalUpdated = 0;
    for (let i = 0; i < mergeIds.length; i += CHUNK) {
        const slice = mergeIds.slice(i, i + CHUNK);
        const placeholders = slice.map((_, idx) => `?${idx + 2}`).join(", ");
        const sql = `UPDATE events SET install_id = ?1 WHERE install_id IN (${placeholders})`;
        try {
            const res = await env.DB.prepare(sql).bind(target, ...slice).run();
            totalUpdated += (res.meta?.changes ?? 0) as number;
        } catch (err) {
            return jsonResponse({ ok: false, error: "db error", updated_so_far: totalUpdated }, 500);
        }
    }

    return jsonResponse({
        ok: true,
        rows_updated: totalUpdated,
        ids_merged: mergeIds,
        target_install_id: target,
    });
}

// Per-install session breakdown — one row per app_session_started, paired
// with the matching app_session_ended (if any), plus action / gesture
// counts that fired between the start and the NEXT session start (or
// "now" when there's no follow-up start). Drives the per-user expander
// in the dashboard's Users tab.
async function handleUserSessions(request: Request, env: Env): Promise<Response> {
    if (!authorizedRead(request, env)) {
        return jsonResponse({ ok: false, error: "unauthorized" }, 401);
    }
    const url = new URL(request.url);
    const installId = String(url.searchParams.get("install_id") || "").slice(0, 64).trim();
    if (!installId) {
        return jsonResponse({ ok: false, error: "missing install_id" }, 400);
    }
    const rows = await env.DB.prepare(
        `WITH starts AS (
            SELECT
                received_at AS started_at,
                json_extract(properties, '$.app_version') AS app_version,
                LEAD(received_at) OVER (ORDER BY received_at) AS next_start
            FROM events
            WHERE install_id = ?1 AND event = 'app_session_started'
        )
        SELECT
            s.started_at,
            s.app_version,
            (SELECT received_at FROM events e
             WHERE e.install_id = ?1 AND e.event = 'app_session_ended'
               AND e.received_at >= s.started_at
               AND (s.next_start IS NULL OR e.received_at < s.next_start)
             ORDER BY e.received_at ASC LIMIT 1) AS ended_at,
            (SELECT CAST(json_extract(properties, '$.session_seconds') AS REAL)
             FROM events e
             WHERE e.install_id = ?1 AND e.event = 'app_session_ended'
               AND e.received_at >= s.started_at
               AND (s.next_start IS NULL OR e.received_at < s.next_start)
             ORDER BY e.received_at ASC LIMIT 1) AS duration_seconds,
            (SELECT COUNT(*) FROM events e
             WHERE e.install_id = ?1 AND e.event = 'action_fired'
               AND e.received_at >= s.started_at
               AND (s.next_start IS NULL OR e.received_at < s.next_start)) AS actions,
            (SELECT COUNT(*) FROM events e
             WHERE e.install_id = ?1 AND e.event = 'gesture_detected'
               AND e.received_at >= s.started_at
               AND (s.next_start IS NULL OR e.received_at < s.next_start)) AS gestures,
            (SELECT COUNT(*) FROM events e
             WHERE e.install_id = ?1 AND e.event = 'error_caught'
               AND e.received_at >= s.started_at
               AND (s.next_start IS NULL OR e.received_at < s.next_start)) AS errors,
            (SELECT COALESCE(SUM(CAST(json_extract(e.properties, '$.engine_seconds') AS REAL)), 0)
             FROM events e
             WHERE e.install_id = ?1 AND e.event = 'engine_stopped'
               AND e.received_at >= s.started_at
               AND (s.next_start IS NULL OR e.received_at < s.next_start)) AS engine_seconds,
            (SELECT COUNT(*) FROM events e
             WHERE e.install_id = ?1 AND e.event = 'engine_started'
               AND e.received_at >= s.started_at
               AND (s.next_start IS NULL OR e.received_at < s.next_start)) AS engine_runs
        FROM starts s
        ORDER BY s.started_at DESC
        LIMIT 500`
    ).bind(installId).all<{
        started_at: string;
        app_version: string | null;
        ended_at: string | null;
        duration_seconds: number | null;
        actions: number;
        gestures: number;
        errors: number;
        engine_seconds: number;
        engine_runs: number;
    }>();

    // Compute the "untracked" bucket — events the per-session query
    // can't see because they're orphaned (no preceding session_started).
    // This is the bug 1 footprint: clients on builds where init-time
    // app_session_started got dropped (because user hadn't opted in
    // yet) will have gestures/actions/durations that DON'T appear in
    // any session, but DO appear in the row totals. Surfacing this
    // bucket explains the difference between sum(sessions) and total.
    //
    // "Untracked" = events with received_at strictly EARLIER than the
    // oldest app_session_started for this install_id. Everything after
    // the oldest start is already accounted for by the session-pairing
    // CTE above (which uses LEAD to span every interval).
    const orphan = await env.DB.prepare(
        `WITH first_start AS (
            SELECT MIN(received_at) AS t FROM events
            WHERE install_id = ?1 AND event = 'app_session_started'
        )
        SELECT
            (SELECT COUNT(*) FROM events e, first_start fs
             WHERE e.install_id = ?1 AND e.event = 'gesture_detected'
               AND (fs.t IS NULL OR e.received_at < fs.t)) AS gestures,
            (SELECT COUNT(*) FROM events e, first_start fs
             WHERE e.install_id = ?1 AND e.event = 'action_fired'
               AND (fs.t IS NULL OR e.received_at < fs.t)) AS actions,
            (SELECT COUNT(*) FROM events e, first_start fs
             WHERE e.install_id = ?1 AND e.event = 'error_caught'
               AND (fs.t IS NULL OR e.received_at < fs.t)) AS errors,
            (SELECT COALESCE(SUM(CAST(json_extract(e.properties, '$.session_seconds') AS REAL)), 0)
             FROM events e, first_start fs
             WHERE e.install_id = ?1 AND e.event = 'app_session_ended'
               AND (fs.t IS NULL OR e.received_at < fs.t)) AS duration_seconds`
    ).bind(installId).first<{
        gestures: number;
        actions: number;
        errors: number;
        duration_seconds: number;
    }>();
    const orphanRow = orphan ?? { gestures: 0, actions: 0, errors: 0, duration_seconds: 0 };
    const hasOrphan = (orphanRow.gestures + orphanRow.actions + orphanRow.errors + orphanRow.duration_seconds) > 0;

    return jsonResponse({
        ok: true,
        install_id: installId,
        sessions: rows.results ?? [],
        orphan: hasOrphan ? orphanRow : null,
    });
}

// Translate a `?range=` query value into the SQLite time clause
// suffix and a human label. "all" returns "" so the WHERE drops out.
function resolveRange(raw: string | null): { clause: string; label: string; key: string } {
    const key = (raw || "7d").toLowerCase();
    if (key === "24h") return { clause: "AND received_at >= datetime('now', '-1 day')",   label: "24 hours", key };
    if (key === "30d") return { clause: "AND received_at >= datetime('now', '-30 days')", label: "30 days",  key };
    if (key === "all") return { clause: "",                                                label: "all time", key: "all" };
    return            { clause: "AND received_at >= datetime('now', '-7 days')",  label: "7 days",   key: "7d" };
}

// Convert the `?tz=` query value (browser-supplied minutes east of
// UTC, e.g. -420 for PDT) into a SQLite DATE() modifier string. The
// modifier shifts the UTC `received_at` into the user's local time
// BEFORE the DATE() / GROUP BY happens, so per-day buckets align to
// what the user thinks of as "today" rather than to UTC midnight.
//
// Returns "+0 minutes" when no tz is provided / tz is invalid, which
// is a no-op modifier — the dashboard then behaves like the previous
// UTC-only grouping. Clamped to ±14 hours to bound the value.
function tzModifier(raw: string | null): string {
    const n = parseInt(String(raw ?? ""), 10);
    if (!Number.isFinite(n)) return "+0 minutes";
    const minutes = Math.max(-14 * 60, Math.min(14 * 60, n));
    const sign = minutes >= 0 ? "+" : "-";
    return `${sign}${Math.abs(minutes)} minutes`;
}

async function handleStats(request: Request, env: Env): Promise<Response> {
    if (!authorizedRead(request, env)) {
        return jsonResponse({ ok: false, error: "unauthorized" }, 401);
    }

    const url = new URL(request.url);
    const range = resolveRange(url.searchParams.get("range"));
    const tzMod = tzModifier(url.searchParams.get("tz"));

    // ---- OVERVIEW -------------------------------------------------
    // Both COUNT(DISTINCT install_id) and the user-table rendering
    // skip ghost install_ids — those with zero sessions / actions /
    // gestures / accumulated duration. Without this filter, the
    // "Unique installs (lifetime)" stat would still include the
    // leftover empty rows even though they're hidden from the
    // Users table.
    const GHOST_FILTER_SUBQUERY = `
        SELECT install_id FROM events
        GROUP BY install_id
        HAVING SUM(CASE WHEN event = 'app_session_started' THEN 1 ELSE 0 END) > 0
            OR SUM(CASE WHEN event = 'action_fired'        THEN 1 ELSE 0 END) > 0
            OR SUM(CASE WHEN event = 'gesture_detected'    THEN 1 ELSE 0 END) > 0
            OR SUM(CASE WHEN event = 'app_session_ended'
                        THEN CAST(json_extract(properties, '$.session_seconds') AS REAL)
                        ELSE 0 END) > 0
            OR SUM(CASE WHEN event = 'engine_stopped'
                        THEN CAST(json_extract(properties, '$.engine_seconds') AS REAL)
                        ELSE 0 END) > 0
    `;
    const totals = await env.DB.prepare(
        `SELECT COUNT(DISTINCT install_id) AS unique_installs,
                COUNT(*) AS total_events
         FROM events
         WHERE install_id IN (${GHOST_FILTER_SUBQUERY})`
    ).first<{ unique_installs: number; total_events: number }>();

    // 30-day DAU trend (independent of the selected range — the
    // trend view should always cover a meaningful slice). Date
    // bucketing applies the browser's tz offset so "today" matches
    // the user's local calendar instead of UTC.
    const dauRows = await env.DB.prepare(
        `SELECT DATE(received_at, '${tzMod}') AS day,
                COUNT(DISTINCT install_id) AS users,
                COUNT(*) AS events
         FROM events
         WHERE received_at >= datetime('now', '-30 days')
         GROUP BY day
         ORDER BY day DESC`
    ).all<{ day: string; users: number; events: number }>();

    const topEvents = await env.DB.prepare(
        `SELECT event, COUNT(*) AS n
         FROM events
         WHERE 1=1 ${range.clause}
         GROUP BY event
         ORDER BY n DESC`
    ).all<{ event: string; n: number }>();

    // ---- ACTIVATION FUNNEL ---------------------------------------
    // Per-step distinct-install counts, lifetime. "Launched app"
    // is the entry to the funnel; subsequent steps are strict
    // subsets (a user who fired an action must have started the
    // engine, etc.). Counted via DISTINCT install_id so heavy users
    // don't skew the percentages.
    const funnelRow = await env.DB.prepare(
        `SELECT
            COUNT(DISTINCT install_id) AS total_installs,
            COUNT(DISTINCT CASE WHEN event = 'app_session_started' THEN install_id END) AS launched,
            COUNT(DISTINCT CASE WHEN event = 'engine_started'      THEN install_id END) AS engine,
            COUNT(DISTINCT CASE WHEN event = 'gesture_detected'    THEN install_id END) AS gestured,
            COUNT(DISTINCT CASE WHEN event = 'action_fired'        THEN install_id END) AS fired
         FROM events`
    ).first<{ total_installs: number; launched: number; engine: number; gestured: number; fired: number }>();

    // ---- RETENTION (per-day cohort, lifetime) --------------------
    // For each install, the cohort is the day of its first event.
    // For each cohort × offset, count how many came back. We only
    // need 1, 7, and 30-day offsets for the headline numbers.
    const retentionRows = await env.DB.prepare(
        `WITH first_seen AS (
            SELECT install_id, MIN(DATE(received_at, '${tzMod}')) AS cohort
            FROM events GROUP BY install_id
         ),
         active_days AS (
            SELECT DISTINCT install_id, DATE(received_at, '${tzMod}') AS day FROM events
         )
         SELECT fs.cohort AS cohort_day,
                COUNT(DISTINCT fs.install_id) AS cohort_size,
                COUNT(DISTINCT CASE WHEN ad.day = DATE(fs.cohort, '+1 day')   THEN fs.install_id END) AS d1,
                COUNT(DISTINCT CASE WHEN ad.day = DATE(fs.cohort, '+7 days')  THEN fs.install_id END) AS d7,
                COUNT(DISTINCT CASE WHEN ad.day = DATE(fs.cohort, '+30 days') THEN fs.install_id END) AS d30
         FROM first_seen fs
         LEFT JOIN active_days ad ON fs.install_id = ad.install_id
         GROUP BY fs.cohort
         ORDER BY fs.cohort DESC
         LIMIT 30`
    ).all<{ cohort_day: string; cohort_size: number; d1: number; d7: number; d30: number }>();

    // ---- FEATURE ADOPTION ----------------------------------------
    // Per-feature distinct-install count: "how many people have ever
    // used voice / drawing / mouse / spotify / volume / youtube /
    // chrome." Anchored on action_fired so we count actual use, not
    // just gesture detections.
    const featureRow = await env.DB.prepare(
        `WITH ax AS (
            SELECT install_id, json_extract(properties, '$.action_id') AS action_id
            FROM events WHERE event = 'action_fired' ${range.clause}
         )
         SELECT
            (SELECT COUNT(DISTINCT install_id) FROM events ${range.clause ? `WHERE 1=1 ${range.clause}` : ""}) AS installs,
            COUNT(DISTINCT CASE WHEN action_id IN ('voice_command_listen','voice_cancel','dictation_toggle','dictation_start','dictation_stop') THEN install_id END) AS voice,
            COUNT(DISTINCT CASE WHEN action_id LIKE 'drawing%' THEN install_id END) AS drawing,
            COUNT(DISTINCT CASE WHEN action_id LIKE 'mouse_%' THEN install_id END) AS mouse,
            COUNT(DISTINCT CASE WHEN action_id IN ('open_spotify','play_pause') OR action_id LIKE 'spotify_%' THEN install_id END) AS spotify,
            COUNT(DISTINCT CASE WHEN action_id LIKE 'volume_%' OR action_id = 'system_mute_toggle' THEN install_id END) AS volume,
            COUNT(DISTINCT CASE WHEN action_id LIKE 'youtube_%' THEN install_id END) AS youtube,
            COUNT(DISTINCT CASE WHEN action_id LIKE 'chrome_%' THEN install_id END) AS chrome
         FROM ax`
    ).first<{ installs: number; voice: number; drawing: number; mouse: number; spotify: number; volume: number; youtube: number; chrome: number }>();

    // ---- SESSIONS ------------------------------------------------
    // Distribution of sessions per install in the selected range.
    const sessionsRow = await env.DB.prepare(
        `WITH per_install AS (
            SELECT install_id,
                   SUM(CASE WHEN event = 'app_session_started' THEN 1 ELSE 0 END) AS sessions,
                   SUM(CASE WHEN event = 'action_fired' THEN 1 ELSE 0 END)        AS actions
            FROM events WHERE 1=1 ${range.clause}
            GROUP BY install_id
         )
         SELECT
            COUNT(*) AS active_installs,
            SUM(sessions) AS total_sessions,
            SUM(actions)  AS total_actions,
            ROUND(AVG(NULLIF(sessions, 0)), 2) AS avg_sessions,
            ROUND(AVG(CASE WHEN sessions > 0 THEN actions * 1.0 / sessions END), 2) AS avg_actions_per_session
         FROM per_install`
    ).first<{ active_installs: number; total_sessions: number; total_actions: number; avg_sessions: number; avg_actions_per_session: number }>();

    // Session duration stats. `session_seconds` is set by MainWindow
    // on app close; sessions that crash (no clean shutdown) won't
    // emit app_session_ended at all so they're naturally excluded.
    const sessionDuration = await env.DB.prepare(
        `SELECT
            COUNT(*) AS sessions_with_duration,
            ROUND(SUM(CAST(json_extract(properties, '$.session_seconds') AS REAL))) AS total_seconds,
            ROUND(AVG(CAST(json_extract(properties, '$.session_seconds') AS REAL))) AS avg_seconds,
            ROUND(MAX(CAST(json_extract(properties, '$.session_seconds') AS REAL))) AS max_seconds
         FROM events
         WHERE event = 'app_session_ended' ${range.clause}
           AND json_extract(properties, '$.session_seconds') IS NOT NULL`
    ).first<{ sessions_with_duration: number; total_seconds: number; avg_seconds: number; max_seconds: number }>();

    // Per-day session-time series for the Sessions tab chart.
    // Date bucketing applies the browser's tz offset so days line
    // up with the user's local calendar.
    const sessionDurationDaily = await env.DB.prepare(
        `SELECT DATE(received_at, '${tzMod}') AS day,
                ROUND(SUM(CAST(json_extract(properties, '$.session_seconds') AS REAL))) AS seconds,
                COUNT(*) AS sessions
         FROM events
         WHERE event = 'app_session_ended'
           AND received_at >= datetime('now', '-30 days')
           AND json_extract(properties, '$.session_seconds') IS NOT NULL
         GROUP BY day
         ORDER BY day ASC`
    ).all<{ day: string; seconds: number; sessions: number }>();

    const sessionBuckets = await env.DB.prepare(
        `WITH per_install AS (
            SELECT install_id, SUM(CASE WHEN event = 'app_session_started' THEN 1 ELSE 0 END) AS sessions
            FROM events WHERE 1=1 ${range.clause}
            GROUP BY install_id
            HAVING sessions > 0
         )
         SELECT
            CASE
              WHEN sessions = 1 THEN '1 (one-time)'
              WHEN sessions <= 3 THEN '2–3'
              WHEN sessions <= 10 THEN '4–10'
              WHEN sessions <= 30 THEN '11–30'
              ELSE '31+'
            END AS bucket,
            COUNT(*) AS installs,
            MIN(sessions) AS sort_key
         FROM per_install
         GROUP BY bucket
         ORDER BY sort_key`
    ).all<{ bucket: string; installs: number; sort_key: number }>();

    // ---- DETAILS -------------------------------------------------
    // Granular breakdowns the Features tab summarises into groups.

    // Every action_id, with absolute count + distinct-install reach.
    const actionsDetail = await env.DB.prepare(
        `SELECT json_extract(properties, '$.action_id') AS action_id,
                COUNT(*)                     AS n,
                COUNT(DISTINCT install_id)   AS reach
         FROM events
         WHERE event = 'action_fired' ${range.clause}
         GROUP BY action_id
         ORDER BY n DESC`
    ).all<{ action_id: string; n: number; reach: number }>();

    // Held-pose gestures detected, by gesture name + handedness.
    // Reclassifies motion-coupled stable labels (volume_pose, pinch,
    // wheel_pose, swipe_*) into the Motion bucket so the split
    // matches the user's mental model from the control guide. Older
    // rows in D1 carry the raw recognizer kind, so we apply the
    // override here at read time.
    const staticGestures = await env.DB.prepare(
        `WITH g AS (
            SELECT json_extract(properties, '$.gesture')     AS gesture,
                   json_extract(properties, '$.handedness')  AS handedness,
                   json_extract(properties, '$.kind')        AS raw_kind,
                   install_id
            FROM events
            WHERE event = 'gesture_detected' ${range.clause}
         )
         SELECT gesture, handedness, COUNT(*) AS n, COUNT(DISTINCT install_id) AS reach
         FROM g
         WHERE raw_kind = 'static'
           AND gesture NOT IN ('volume_pose', 'pinch', 'wheel_pose')
           AND gesture NOT LIKE 'swipe_%'
         GROUP BY gesture, handedness
         ORDER BY n DESC`
    ).all<{ gesture: string; handedness: string; n: number; reach: number }>();

    // Motion gestures: anything kind="dynamic" plus the motion-
    // coupled stable poses we just excluded above.
    const dynamicGestures = await env.DB.prepare(
        `WITH g AS (
            SELECT json_extract(properties, '$.gesture')     AS gesture,
                   json_extract(properties, '$.handedness')  AS handedness,
                   json_extract(properties, '$.kind')        AS raw_kind,
                   install_id
            FROM events
            WHERE event = 'gesture_detected' ${range.clause}
         )
         SELECT gesture, handedness, COUNT(*) AS n, COUNT(DISTINCT install_id) AS reach
         FROM g
         WHERE raw_kind = 'dynamic'
            OR gesture IN ('volume_pose', 'pinch', 'wheel_pose')
            OR gesture LIKE 'swipe_%'
         GROUP BY gesture, handedness
         ORDER BY n DESC`
    ).all<{ gesture: string; handedness: string; n: number; reach: number }>();

    // ---- USERS ---------------------------------------------------
    // Per-install aggregates, lifetime — first/last seen, session
    // count, total time, action count. The dashboard sorts by
    // last_seen desc by default so the most-recently-active installs
    // are at the top.
    const usersList = await env.DB.prepare(
        `SELECT install_id,
                MIN(received_at) AS first_seen,
                MAX(received_at) AS last_seen,
                SUM(CASE WHEN event = 'app_session_started' THEN 1 ELSE 0 END) AS sessions,
                SUM(CASE WHEN event = 'action_fired'        THEN 1 ELSE 0 END) AS actions,
                SUM(CASE WHEN event = 'gesture_detected'    THEN 1 ELSE 0 END) AS gestures,
                SUM(CASE WHEN event = 'app_session_ended'
                         THEN CAST(json_extract(properties, '$.session_seconds') AS REAL)
                         ELSE 0 END) AS total_seconds,
                SUM(CASE WHEN event = 'engine_stopped'
                         THEN CAST(json_extract(properties, '$.engine_seconds') AS REAL)
                         ELSE 0 END) AS engine_seconds,
                -- High-water mark of custom gestures across sessions.
                -- Client sends custom_gesture_count on every
                -- app_session_started; MAX() over all sessions gives
                -- "most they ever had", which is more useful than the
                -- latest (avoids dropping the count if a user happens
                -- to launch with 0 after deleting some).
                COALESCE(MAX(CASE WHEN event = 'app_session_started'
                                  THEN CAST(json_extract(properties, '$.custom_gesture_count') AS INTEGER)
                                  ELSE NULL END), 0) AS custom_gestures
         FROM events
         GROUP BY install_id
         HAVING
              -- Hide ghost rows from the dashboard: an install_id
              -- with ZERO sessions, ZERO actions, ZERO gestures, and
              -- ZERO accumulated session/engine seconds is a leftover
              -- from a pre-v2 build where some non-counted events
              -- (engine_started, error_caught, etc.) flowed but
              -- app_session_started got dropped by the bug-1 timing
              -- race. They no longer accrue post-v2 + post-bug-1-fix.
              -- HAVING runs after GROUP BY so D1 still aggregates
              -- every install_id; we just exclude the empty ones
              -- from the response.
              sessions > 0
              OR actions > 0
              OR gestures > 0
              OR total_seconds > 0
              OR engine_seconds > 0
         ORDER BY last_seen DESC
         LIMIT 500`
    ).all<{ install_id: string; first_seen: string; last_seen: string; sessions: number; actions: number; gestures: number; total_seconds: number; engine_seconds: number; custom_gestures: number }>();

    // Per-install "current state" lookup for the green/orange/red
    // activity dot. Walks the most recent state-edge event per
    // install_id (app_session_started / app_session_ended /
    // engine_started / engine_stopped) so the dashboard can decide:
    //   GREEN  = engine_started was the last state edge (engine ON)
    //   ORANGE = app_session_started or engine_stopped (app open, engine off)
    //   RED    = app_session_ended (app closed) or no recent activity
    // Combines with last_seen recency on the frontend; events older
    // than ~3 min collapse to RED regardless of the latched state.
    const stateRows = await env.DB.prepare(
        `WITH state_events AS (
            SELECT install_id, event, received_at,
                   ROW_NUMBER() OVER (PARTITION BY install_id ORDER BY received_at DESC) AS rn
            FROM events
            WHERE event IN (
                'app_session_started','app_session_ended',
                'engine_started','engine_stopped'
            )
        )
        SELECT install_id, event AS last_state_event, received_at AS last_state_at
        FROM state_events WHERE rn = 1`
    ).all<{ install_id: string; last_state_event: string; last_state_at: string }>();
    const stateByInstall: Record<string, { event: string; at: string }> = {};
    for (const r of stateRows.results ?? []) {
        stateByInstall[r.install_id] = { event: r.last_state_event, at: r.last_state_at };
    }
    // Splice the state info into each user row so the frontend can
    // render the dot without a second fetch.
    const usersListAugmented = (usersList.results ?? []).map((u) => {
        const s = stateByInstall[u.install_id];
        return {
            ...u,
            last_state_event: s?.event ?? null,
            last_state_at: s?.at ?? null,
        };
    });

    // Active-user counts at standard windows. Lifetime totals so they
    // don't shift with the range pill — these are headline numbers.
    const activeRow = await env.DB.prepare(
        `SELECT
            COUNT(DISTINCT CASE WHEN received_at >= datetime('now', '-1 day')   THEN install_id END) AS active_24h,
            COUNT(DISTINCT CASE WHEN received_at >= datetime('now', '-7 days')  THEN install_id END) AS active_7d,
            COUNT(DISTINCT CASE WHEN received_at >= datetime('now', '-30 days') THEN install_id END) AS active_30d,
            COUNT(DISTINCT install_id) AS total_users
         FROM events`
    ).first<{ active_24h: number; active_7d: number; active_30d: number; total_users: number }>();

    // ---- RECENT SESSIONS (one row per closed session) -----------
    const recentSessions = await env.DB.prepare(
        `SELECT install_id,
                received_at AS ended_at,
                CAST(json_extract(properties, '$.session_seconds') AS REAL) AS duration
         FROM events
         WHERE event = 'app_session_ended' ${range.clause}
           AND json_extract(properties, '$.session_seconds') IS NOT NULL
         ORDER BY received_at DESC
         LIMIT 200`
    ).all<{ install_id: string; ended_at: string; duration: number }>();

    // Voice command targets — the parsed intent's target app/action,
    // not the raw transcript.
    const voiceTargets = await env.DB.prepare(
        `SELECT json_extract(properties, '$.target')   AS target,
                json_extract(properties, '$.success')  AS success,
                COUNT(*)                     AS n,
                COUNT(DISTINCT install_id)   AS reach
         FROM events
         WHERE event = 'voice_command_executed' ${range.clause}
         GROUP BY target, success
         ORDER BY n DESC`
    ).all<{ target: string; success: number; n: number; reach: number }>();

    // ---- ERRORS --------------------------------------------------
    const errorRows = await env.DB.prepare(
        `SELECT json_extract(properties, '$.component') AS component,
                json_extract(properties, '$.exc_type')  AS exc_type,
                COUNT(*) AS n
         FROM events
         WHERE event = 'error_caught' ${range.clause}
         GROUP BY component, exc_type
         ORDER BY n DESC
         LIMIT 50`
    ).all<{ component: string; exc_type: string; n: number }>();

    const errorTotal = await env.DB.prepare(
        `SELECT COUNT(*) AS total,
                COUNT(DISTINCT install_id) AS affected_installs
         FROM events WHERE event = 'error_caught' ${range.clause}`
    ).first<{ total: number; affected_installs: number }>();

    return jsonResponse({
        range_key: range.key,
        range_label: range.label,
        // Overview
        unique_installs: totals?.unique_installs ?? 0,
        total_events: totals?.total_events ?? 0,
        daily_active: dauRows.results ?? [],
        top_events: topEvents.results ?? [],
        // Funnel
        funnel: funnelRow ?? { total_installs: 0, launched: 0, engine: 0, gestured: 0, fired: 0 },
        // Retention
        retention: retentionRows.results ?? [],
        // Features
        features: featureRow ?? { installs: 0, voice: 0, drawing: 0, mouse: 0, spotify: 0, volume: 0, youtube: 0, chrome: 0 },
        // Sessions
        sessions: sessionsRow ?? { active_installs: 0, total_sessions: 0, total_actions: 0, avg_sessions: 0, avg_actions_per_session: 0 },
        session_buckets: sessionBuckets.results ?? [],
        session_duration: sessionDuration ?? { sessions_with_duration: 0, total_seconds: 0, avg_seconds: 0, max_seconds: 0 },
        session_duration_daily: sessionDurationDaily.results ?? [],
        recent_sessions: recentSessions.results ?? [],
        // Users
        users: usersListAugmented,
        active_24h: activeRow?.active_24h ?? 0,
        active_7d: activeRow?.active_7d ?? 0,
        active_30d: activeRow?.active_30d ?? 0,
        total_users: activeRow?.total_users ?? 0,
        // Details
        actions_detail: actionsDetail.results ?? [],
        static_gestures: staticGestures.results ?? [],
        dynamic_gestures: dynamicGestures.results ?? [],
        voice_targets: voiceTargets.results ?? [],
        // Errors
        errors: errorRows.results ?? [],
        errors_total: errorTotal?.total ?? 0,
        errors_affected_installs: errorTotal?.affected_installs ?? 0,
    });
}

function authorizedRead(request: Request, env: Env): boolean {
    const url = new URL(request.url);
    const token =
        url.searchParams.get("token") ??
        request.headers.get("X-Token") ??
        request.headers.get("x-token");
    return token === env.SHARED_SECRET;
}

function jsonResponse(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body), {
        status,
        headers: { ...JSON_HEADERS, "Access-Control-Allow-Origin": "*" },
    });
}

const DASHBOARD_HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Touchless Telemetry</title>
<style>
  :root {
    --bg: #0b1426; --card: #0f1d33; --border: #1c3057;
    --text: #e5f6ff; --muted: #8aa3bf;
    --accent: #1de9b6; --accent2: #4ea3ff; --warn: #ffb454;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    max-width: 1280px; margin-left: auto; margin-right: auto;
  }
  h1 { margin: 0 0 6px; font-size: 22px; letter-spacing: 0.5px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 18px; }
  .grid {
    display: grid; gap: 14px;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    margin-bottom: 18px;
  }
  .stat {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 16px 18px;
  }
  .stat .lbl { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .stat .val { font-size: 32px; font-weight: 700; margin-top: 6px; color: var(--accent); }
  .row { display: grid; gap: 14px; grid-template-columns: 1fr 1fr; }
  @media (max-width: 800px) { .row { grid-template-columns: 1fr; } }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 16px 18px; min-height: 200px;
  }
  .card h2 { margin: 0 0 6px; font-size: 14px; color: var(--text); letter-spacing: 0.3px; }
  .card .blurb { color: var(--muted); font-size: 12px; margin-bottom: 14px; line-height: 1.5; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 12px; }
  tr:last-child td { border-bottom: none; }
  td.num { text-align: right; color: var(--accent); font-variant-numeric: tabular-nums; }
  td.muted { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
  code.install {
    color: var(--muted); font-size: 11px;
    background: #0a162b; border: 1px solid var(--border);
    padding: 2px 6px; border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    margin-left: 8px;
  }
  span.user-num {
    color: var(--text); font-weight: 700; font-size: 13px;
    letter-spacing: 0.2px;
  }
  .exp-caret {
    display: inline-block; width: 12px; color: var(--muted);
    font-size: 10px; user-select: none; transition: transform 0.15s;
  }
  tr.user-row:hover td { background: rgba(29,233,182,0.04); }
  tr.user-sessions-row td { background: #0a162b; }
  .err { background: #4a1a1a; color: #ffaaaa; padding: 10px 14px; border-radius: 10px; }
  .bar { height: 6px; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 3px; }
  .bar-cell { width: 60%; padding: 4px 0; }
  .bar-cell .bar { width: var(--w, 0%); }
  .stack { display: flex; flex-direction: column; gap: 4px; }
  .day-row { display: grid; grid-template-columns: 110px 1fr 60px; gap: 10px; align-items: center; padding: 4px 0; }
  .day-row .day { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
  .day-row .num { text-align: right; }
  .toggle {
    margin-top: 10px; background: transparent; border: 1px solid var(--border);
    color: var(--muted); padding: 6px 10px; border-radius: 8px;
    font-size: 12px; cursor: pointer; letter-spacing: 0.4px;
  }
  .toggle:hover { color: var(--text); border-color: var(--accent2); }
  tr.hidden { display: none; }
  .total { color: var(--muted); font-size: 11px; margin-top: 8px; letter-spacing: 0.3px; }

  /* Header bar with tabs + range pills side-by-side */
  .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 18px; }
  .pills {
    display: flex; gap: 4px; flex-wrap: wrap;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 4px;
  }
  .pill {
    background: transparent; border: none; color: var(--muted);
    padding: 7px 14px; border-radius: 8px; cursor: pointer;
    font-size: 13px; letter-spacing: 0.3px; font-weight: 600;
  }
  .pill:hover { color: var(--text); }
  .pill.active { background: var(--border); color: var(--accent); }
  .tabs .pill.active { color: var(--text); background: linear-gradient(180deg, #1d345f, #18294a); }
  /* Refresh button — distinct accent fill so it reads as the
     primary "reload everything" action, not a passive filter. */
  .pill.refresh {
    background: var(--accent); color: #001b13;
    font-weight: 800; padding: 7px 16px;
  }
  .pill.refresh:hover { background: #2ee7c1; color: #001b13; }
  .pill.refresh:active { transform: translateY(1px); }
  .pill.refresh.spinning { opacity: 0.65; cursor: progress; }
  .autorefresh-lbl {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 0 10px; color: var(--muted); font-size: 12px;
    cursor: pointer; user-select: none;
  }
  .autorefresh-lbl input[type=checkbox] {
    accent-color: var(--accent); width: 14px; height: 14px;
  }
  .autorefresh-lbl:hover { color: var(--text); }

  .tab-page { display: none; }
  .tab-page.active { display: block; }

  /* Funnel-specific styles */
  .funnel { display: flex; flex-direction: column; gap: 10px; }
  .funnel-step {
    display: grid; grid-template-columns: 1fr auto auto; gap: 12px; align-items: center;
    background: #0a162b; border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 14px;
  }
  .funnel-step .name { font-weight: 600; font-size: 14px; }
  .funnel-step .count { color: var(--accent); font-variant-numeric: tabular-nums; font-weight: 700; }
  .funnel-step .pct { color: var(--muted); font-size: 12px; min-width: 60px; text-align: right; }
  .funnel-bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-top: 8px; }
  .funnel-bar > div { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 3px; }
  .arrow {
    color: var(--muted); text-align: center; font-size: 12px;
    margin: -4px 0;
  }
  .arrow.bad { color: var(--warn); }

  /* Feature adoption rows */
  .feature-row {
    display: grid; grid-template-columns: 110px 1fr 70px 60px;
    gap: 10px; align-items: center; padding: 6px 0;
    border-bottom: 1px solid var(--border);
  }
  .feature-row:last-child { border-bottom: none; }
  .feature-row .name { font-weight: 600; }
  .feature-row .pct { color: var(--accent); text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
  .feature-row .count { color: var(--muted); text-align: right; font-size: 12px; font-variant-numeric: tabular-nums; }

  /* Retention table */
  .retention th, .retention td { text-align: right; }
  .retention th:first-child, .retention td:first-child { text-align: left; }
  .retention td.pct { color: var(--accent); font-variant-numeric: tabular-nums; }
  .retention td.weak { color: var(--warn); }
</style>
</head>
<body>
  <h1>Touchless Telemetry</h1>
  <div class="sub" id="sub">Loading…</div>
  <div id="error" class="err" style="display:none"></div>

  <div class="toolbar">
    <div class="pills tabs" id="tab-pills" role="tablist">
      <button class="pill active" data-tab="overview">Overview</button>
      <button class="pill" data-tab="activation">Activation</button>
      <button class="pill" data-tab="features">Features</button>
      <button class="pill" data-tab="details">Details</button>
      <button class="pill" data-tab="users">Users</button>
      <button class="pill" data-tab="sessions">Sessions</button>
      <button class="pill" data-tab="errors">Errors</button>
    </div>
    <div class="pills" id="range-pills">
      <button class="pill" data-range="24h">24h</button>
      <button class="pill active" data-range="7d">7d</button>
      <button class="pill" data-range="30d">30d</button>
      <button class="pill" data-range="all">All time</button>
    </div>
    <div class="pills" id="refresh-controls">
      <button class="pill refresh" id="refresh-btn" title="Reload current range">
        <span style="font-size:14px;line-height:1">&#x21bb;</span>&nbsp;Refresh
      </button>
      <label class="autorefresh-lbl" title="Refresh every 60 seconds">
        <input type="checkbox" id="autorefresh-cb" />
        <span>Auto&nbsp;(60s)</span>
      </label>
    </div>
  </div>

  <!-- ========== OVERVIEW ========== -->
  <section class="tab-page active" id="page-overview">
    <div class="grid">
      <div class="stat"><div class="lbl">Unique installs (lifetime)</div><div class="val" id="installs">—</div></div>
      <div class="stat"><div class="lbl">Total events (lifetime)</div><div class="val" id="events">—</div></div>
      <div class="stat"><div class="lbl">Active today</div><div class="val" id="today">—</div></div>
      <div class="stat"><div class="lbl">Active in last 7 days</div><div class="val" id="seven">—</div></div>
    </div>
    <div class="row">
      <div class="card">
        <h2>Daily active installs</h2>
        <p class="blurb">How many distinct installs sent any event each day. The trendline tells you whether the app is gaining, holding, or losing daily users.</p>
        <div id="dau" class="stack"></div>
      </div>
      <div class="card">
        <h2>All events <span id="events-range-lbl" class="sub" style="font-size:11px"></span></h2>
        <p class="blurb">Every event the app emitted in this time range, by name. A flat list of what's happening — useful as a sanity check.</p>
        <table id="top-events-tbl"><thead><tr><th>event</th><th>count</th></tr></thead><tbody></tbody></table>
        <div class="total" id="top-events-total"></div>
        <button class="toggle" id="top-events-toggle" style="display:none"></button>
      </div>
    </div>
  </section>

  <!-- ========== ACTIVATION ========== -->
  <section class="tab-page" id="page-activation">
    <div class="row">
      <div class="card" style="grid-column: 1 / -1">
        <h2>Activation funnel (lifetime)</h2>
        <p class="blurb">Of all installs that ever phoned home, how many made it through each step. Big drop-offs between two steps mean people are getting stuck somewhere — that's the spot to investigate.</p>
        <div class="funnel" id="funnel"></div>
      </div>
    </div>
    <div class="row" style="margin-top:14px">
      <div class="card" style="grid-column: 1 / -1">
        <h2>Retention (per-cohort, last 30 days)</h2>
        <p class="blurb">Each row is a group of installs that first appeared on that day. <strong>D1</strong> = how many came back the next day. <strong>D7</strong> = the day a week later. <strong>D30</strong> = a month later. Higher percentages = stickier product.</p>
        <table class="retention" id="retention-tbl">
          <thead><tr><th>First-seen day</th><th>Cohort size</th><th>D1</th><th>D7</th><th>D30</th></tr></thead>
          <tbody></tbody>
        </table>
        <div class="total" id="retention-total"></div>
      </div>
    </div>
  </section>

  <!-- ========== FEATURES ========== -->
  <section class="tab-page" id="page-features">
    <div class="row">
      <div class="card" style="grid-column: 1 / -1">
        <h2>Feature adoption <span id="features-range-lbl" class="sub" style="font-size:11px"></span></h2>
        <p class="blurb">For each subsystem (voice, drawing, mouse, etc.), what percent of active installs have ever used it. Low numbers point at features people never discover or never trust.</p>
        <div id="features-list"></div>
      </div>
    </div>
  </section>

  <!-- ========== DETAILS ========== -->
  <section class="tab-page" id="page-details">
    <div class="row">
      <div class="card" style="grid-column: 1 / -1">
        <h2>Every action_id <span id="details-range-actions" class="sub" style="font-size:11px"></span></h2>
        <p class="blurb">The exact action that fired — no grouping. <strong>Count</strong> = total fires across all installs. <strong>Reach</strong> = how many distinct installs ever fired it. High count + low reach = one user pounding on it; low count + high reach = lots of people trying once.</p>
        <table id="actions-detail-tbl"><thead><tr><th>action_id</th><th>count</th><th>reach</th></tr></thead><tbody></tbody></table>
        <div class="total" id="actions-detail-total"></div>
        <button class="toggle" id="actions-detail-toggle" style="display:none"></button>
      </div>
    </div>
    <div class="row" style="margin-top:14px">
      <div class="card">
        <h2>Held poses detected <span id="details-range-static" class="sub" style="font-size:11px"></span></h2>
        <p class="blurb">Static hand shapes the user holds in place — fist, two, three, four, peace, OK, thumbs up/down, etc. Doesn't include motion-coupled poses; those are on the right.</p>
        <table id="static-gestures-tbl"><thead><tr><th>pose</th><th>hand</th><th>count</th><th>reach</th></tr></thead><tbody></tbody></table>
        <div class="total" id="static-gestures-total"></div>
        <button class="toggle" id="static-gestures-toggle" style="display:none"></button>
      </div>
      <div class="card">
        <h2>Motion gestures detected <span id="details-range-dynamic" class="sub" style="font-size:11px"></span></h2>
        <p class="blurb">Anything where the user is moving — swipes, volume pose (motion-driven), pinch (drawing transform: move + scale min↔max), wheel pose (wheel orbit), and any other motion-coupled poses.</p>
        <table id="dynamic-gestures-tbl"><thead><tr><th>gesture</th><th>hand</th><th>count</th><th>reach</th></tr></thead><tbody></tbody></table>
        <div class="total" id="dynamic-gestures-total"></div>
        <button class="toggle" id="dynamic-gestures-toggle" style="display:none"></button>
      </div>
    </div>
    <div class="row" style="margin-top:14px">
      <div class="card" style="grid-column: 1 / -1">
        <h2>Voice commands <span id="details-range-voice" class="sub" style="font-size:11px"></span></h2>
        <p class="blurb">The parsed <em>intent target</em> of each successful voice command (e.g. <code>spotify</code>, <code>youtube</code>, <code>chrome</code>, <code>dictation</code>). Includes a success flag — high failures means transcripts aren't matching commands.</p>
        <table id="voice-targets-tbl"><thead><tr><th>target</th><th>success</th><th>count</th><th>reach</th></tr></thead><tbody></tbody></table>
        <div class="total" id="voice-targets-total"></div>
        <button class="toggle" id="voice-targets-toggle" style="display:none"></button>
      </div>
    </div>
  </section>

  <!-- ========== USERS ========== -->
  <section class="tab-page" id="page-users">
    <div class="grid">
      <div class="stat"><div class="lbl">Total users (lifetime)</div><div class="val" id="users-total">—</div></div>
      <div class="stat"><div class="lbl">Active in last 24h</div><div class="val" id="users-24h">—</div></div>
      <div class="stat"><div class="lbl">Active in last 7 days</div><div class="val" id="users-7d">—</div></div>
      <div class="stat"><div class="lbl">Active in last 30 days</div><div class="val" id="users-30d">—</div></div>
    </div>
    <div class="row">
      <div class="card" style="grid-column: 1 / -1">
        <h2>All users</h2>
        <p class="blurb">One row per anonymous install. <strong>Sessions</strong> = times the app was opened. <strong>Time</strong> = total clean-shutdown time across all sessions. <strong>Actions</strong> / <strong>Gestures</strong> = lifetime totals. Sorted by most-recently-active first.</p>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap">
          <input id="install-search" type="search" placeholder="Search install id…"
                 style="flex:0 0 280px;background:#0a162b;color:var(--text);
                        border:1px solid var(--border);border-radius:8px;
                        padding:7px 12px;font-size:13px;outline:none;" />
          <span class="muted" style="font-size:12px">Type any part of an install UUID to filter.</span>
          <button id="consolidate-btn"
                  style="margin-left:auto;background:#1DE9B6;color:#0a162b;
                         border:none;border-radius:8px;padding:7px 14px;
                         font-weight:700;font-size:13px;cursor:pointer">
            Consolidate installs…
          </button>
        </div>
        <!-- Consolidate modal: folds N install_ids into 1 by rewriting
             rows in D1. Only safe to use when YOU know all the source IDs
             belong to the same actual person (e.g. your own ghost rows
             from settings.json wipes before the MachineGuid-derived
             install_id landed). -->
        <div id="consolidate-modal" style="display:none;position:fixed;inset:0;
                  background:rgba(5,15,35,0.78);z-index:1000;align-items:center;
                  justify-content:center;padding:20px">
          <div style="background:#0e1c38;border:1px solid var(--border);
                      border-radius:12px;max-width:560px;width:100%;
                      padding:20px;color:var(--text)">
            <h2 style="margin-top:0;color:#1DE9B6">Fold install ids into one</h2>
            <p class="blurb" style="margin-top:0">
              Pick the install id to <strong>keep</strong>. Every other
              install id in the database will have its events re-tagged
              to that one. This is irreversible — use it only when you
              know the source rows all belong to the same actual person.
            </p>
            <label style="display:block;font-size:12px;color:var(--muted);
                          margin-bottom:6px">Keep install id</label>
            <select id="consolidate-target"
                    style="width:100%;background:#0a162b;color:var(--text);
                           border:1px solid var(--border);border-radius:8px;
                           padding:8px 10px;font-size:13px;
                           font-family:ui-monospace,monospace"></select>
            <p class="muted" id="consolidate-summary" style="font-size:12px;
                          margin:14px 0 18px 0"></p>
            <div style="display:flex;gap:10px;justify-content:flex-end">
              <button id="consolidate-cancel"
                      style="background:transparent;color:var(--text);
                             border:1px solid var(--border);border-radius:8px;
                             padding:7px 14px;font-size:13px;cursor:pointer">
                Cancel
              </button>
              <button id="consolidate-confirm"
                      style="background:#1DE9B6;color:#0a162b;border:none;
                             border-radius:8px;padding:7px 14px;font-weight:700;
                             font-size:13px;cursor:pointer">
                Fold ghosts into keeper
              </button>
            </div>
            <p id="consolidate-status" style="font-size:12px;margin:14px 0 0 0;
                          color:var(--muted)"></p>
          </div>
        </div>
        <table id="users-tbl">
          <thead><tr><th>install</th><th>first seen</th><th>last seen</th><th>sessions</th><th>open</th><th>running</th><th>actions</th><th>gestures</th><th>custom</th></tr></thead>
          <tbody></tbody>
        </table>
        <div class="total" id="users-tbl-total"></div>
        <button class="toggle" id="users-tbl-toggle" style="display:none"></button>
      </div>
    </div>
  </section>

  <!-- ========== SESSIONS ========== -->
  <section class="tab-page" id="page-sessions">
    <div class="grid">
      <div class="stat"><div class="lbl">Active installs <span id="sessions-range-stat" style="text-transform:none"></span></div><div class="val" id="sessions-active">—</div></div>
      <div class="stat"><div class="lbl">Total sessions</div><div class="val" id="sessions-total">—</div></div>
      <div class="stat"><div class="lbl">Avg sessions / install</div><div class="val" id="sessions-avg">—</div></div>
      <div class="stat"><div class="lbl">Avg actions / session</div><div class="val" id="sessions-actions-avg">—</div></div>
    </div>
    <div class="grid">
      <div class="stat"><div class="lbl">Total time on app</div><div class="val" id="duration-total">—</div></div>
      <div class="stat"><div class="lbl">Avg session length</div><div class="val" id="duration-avg">—</div></div>
      <div class="stat"><div class="lbl">Longest session</div><div class="val" id="duration-max">—</div></div>
      <div class="stat"><div class="lbl">Sessions with duration</div><div class="val" id="duration-count">—</div></div>
    </div>
    <div class="row">
      <div class="card">
        <h2>Daily total time on app (last 30 days)</h2>
        <p class="blurb">Sum of session lengths per day. Sessions that crash without a clean shutdown don't contribute.</p>
        <div id="duration-daily" class="stack"></div>
      </div>
      <div class="card">
        <h2>Session-count distribution</h2>
        <p class="blurb">Buckets installs by how many times they opened the app in this time range. A big "1 (one-time)" bar means lots of people try once and leave. Larger buckets on the right = recurring users.</p>
        <div id="session-buckets"></div>
      </div>
    </div>
    <div class="row" style="margin-top:14px">
      <div class="card" style="grid-column: 1 / -1">
        <h2>Recent sessions <span id="sessions-list-range" class="sub" style="font-size:11px"></span></h2>
        <p class="blurb">Each row is one closed session — when it ended and how long it ran. Most-recent first. Top 15 visible; click "Show all" for up to 200.</p>
        <table id="sessions-list-tbl">
          <thead><tr><th>install</th><th>ended at</th><th>duration</th></tr></thead>
          <tbody></tbody>
        </table>
        <div class="total" id="sessions-list-tbl-total"></div>
        <button class="toggle" id="sessions-list-tbl-toggle" style="display:none"></button>
      </div>
    </div>
  </section>

  <!-- ========== ERRORS ========== -->
  <section class="tab-page" id="page-errors">
    <div class="grid">
      <div class="stat"><div class="lbl">Errors <span id="errors-range-stat" style="text-transform:none"></span></div><div class="val" id="errors-total">—</div></div>
      <div class="stat"><div class="lbl">Affected installs</div><div class="val" id="errors-installs">—</div></div>
    </div>
    <div class="row">
      <div class="card" style="grid-column: 1 / -1">
        <h2>Top error types</h2>
        <p class="blurb">Each row is a (component, exception type) pair. Higher counts and broader install reach = higher priority to fix.</p>
        <table id="errors-tbl"><thead><tr><th>component</th><th>exception</th><th>count</th></tr></thead><tbody></tbody></table>
        <div class="total" id="errors-total-row"></div>
      </div>
    </div>
  </section>

  <div class="row" style="margin-top:14px">
    <div class="card" style="grid-column: 1 / -1">
      <h2>About</h2>
      <p class="sub">Use the tabs to switch sections; the time pills (24h / 7d / 30d / All) apply to most cards. Funnel + retention always show lifetime data.</p>
      <p class="sub">Bookmark this URL with <code>?token=YOUR_SHARED_SECRET</code> appended. Raw events are in the <code>events</code> D1 table — query with <code>wrangler d1 execute touchless-events --command "SELECT …"</code>.</p>
    </div>
  </div>

<script>
(function() {
  const params = new URLSearchParams(location.search);
  const token = params.get("token") || prompt("Enter your telemetry shared secret token:");
  if (!token) {
    document.getElementById("error").textContent = "Token required.";
    document.getElementById("error").style.display = "block";
    document.getElementById("sub").textContent = "";
    return;
  }

  const RANGE_LABELS = { "24h": "last 24h", "7d": "last 7 days", "30d": "last 30 days", "all": "all time" };
  let currentRange = params.get("range") || "7d";
  let currentTab = params.get("tab") || "overview";

  function fmt(n) { return Number(n || 0).toLocaleString(); }
  function pct(num, den) {
    if (!den) return "—";
    return ((num / den) * 100).toFixed(1) + "%";
  }
  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, c => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));
  }
  function shortInstall(id) {
    const s = String(id || "");
    return s.length > 12 ? s.slice(0, 8) + "…" + s.slice(-4) : s;
  }
  function fmtTime(s) {
    if (!s) return "—";
    // Server returns "YYYY-MM-DD HH:MM:SS" in UTC. Append Z so the
    // browser parses it correctly, then format with the user's locale.
    const d = new Date(String(s).replace(" ", "T") + "Z");
    if (isNaN(d.valueOf())) return String(s);
    const now = new Date();
    const diffMs = now - d;
    const diffMin = Math.round(diffMs / 60000);
    if (diffMin < 1) return "just now";
    if (diffMin < 60) return diffMin + "m ago";
    if (diffMin < 1440) return Math.round(diffMin / 60) + "h ago";
    if (diffMin < 1440 * 30) return Math.round(diffMin / 1440) + "d ago";
    return d.toLocaleDateString();
  }
  // Render a "YYYY-MM-DD" day string (already in the browser's
  // local calendar — the server applies the tz offset BEFORE
  // bucketing) using the user's locale, matching the header
  // timestamp's date format. Falls through to the raw string
  // when parsing fails.
  function fmtDay(s) {
    if (!s) return "—";
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(s).trim());
    if (!m) return String(s);
    // Construct as a LOCAL Date (year/month/day with month-1 for
    // the JS index) so toLocaleDateString renders the calendar
    // day the user expects, with no timezone shift.
    const d = new Date(parseInt(m[1], 10), parseInt(m[2], 10) - 1, parseInt(m[3], 10));
    if (isNaN(d.valueOf())) return String(s);
    return d.toLocaleDateString();
  }
  function fmtDuration(secsRaw) {
    const secs = Math.max(0, Math.round(Number(secsRaw || 0)));
    if (secs === 0) return "—";
    if (secs < 60) return secs + "s";
    if (secs < 3600) {
      const m = Math.floor(secs / 60);
      const s = secs % 60;
      return s ? (m + "m " + s + "s") : (m + "m");
    }
    if (secs < 86400) {
      const h = Math.floor(secs / 3600);
      const m = Math.floor((secs % 3600) / 60);
      return m ? (h + "h " + m + "m") : (h + "h");
    }
    const d = Math.floor(secs / 86400);
    const h = Math.floor((secs % 86400) / 3600);
    return h ? (d + "d " + h + "h") : (d + "d");
  }

  function activateTab(tab) {
    currentTab = tab;
    document.querySelectorAll(".tabs .pill").forEach(p => {
      p.classList.toggle("active", p.dataset.tab === tab);
    });
    document.querySelectorAll(".tab-page").forEach(s => {
      s.classList.toggle("active", s.id === "page-" + tab);
    });
  }

  async function load(range) {
    currentRange = range;
    document.querySelectorAll("#range-pills .pill").forEach(p => {
      p.classList.toggle("active", p.dataset.range === range);
    });
    document.getElementById("sub").textContent = "Loading " + RANGE_LABELS[range] + "…";
    document.getElementById("error").style.display = "none";
    try {
      // Browser's UTC offset in minutes EAST of UTC.
      // getTimezoneOffset() returns the WEST offset (negated), so
      // negate it for the SQL modifier ('+N minutes' shifts UTC
      // received_at into local time before DATE() buckets).
      const tzMin = -new Date().getTimezoneOffset();
      const res = await fetch("/api/stats?range=" + encodeURIComponent(range)
        + "&tz=" + encodeURIComponent(tzMin)
        + "&token=" + encodeURIComponent(token));
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      render(data);
    } catch (e) {
      document.getElementById("error").textContent = "Could not load stats: " + e.message;
      document.getElementById("error").style.display = "block";
      document.getElementById("sub").textContent = "";
    }
  }

  document.querySelectorAll("#tab-pills .pill").forEach(p => {
    p.addEventListener("click", () => activateTab(p.dataset.tab));
  });
  document.querySelectorAll("#range-pills .pill").forEach(p => {
    p.addEventListener("click", () => load(p.dataset.range));
  });

  // Refresh button + auto-refresh (60s) toggle.
  let _autoTimer = null;
  const _refreshBtn = document.getElementById("refresh-btn");
  if (_refreshBtn) {
    _refreshBtn.addEventListener("click", async () => {
      _refreshBtn.classList.add("spinning");
      try {
        await load(currentRange);
      } finally {
        _refreshBtn.classList.remove("spinning");
      }
    });
  }
  const _autoCb = document.getElementById("autorefresh-cb");
  if (_autoCb) {
    _autoCb.addEventListener("change", () => {
      if (_autoTimer !== null) { clearInterval(_autoTimer); _autoTimer = null; }
      if (_autoCb.checked) {
        _autoTimer = setInterval(() => load(currentRange), 60000);
      }
    });
  }

  // Install-id search: filter the per-install table in the Users
  // tab to rows whose install_id partially matches the query.
  // Bound below after the search input is rendered.
  let _installSearchQuery = "";
  function _applyInstallSearch() {
    const tbody = document.querySelector("#users-tbl tbody");
    if (!tbody) return;
    const q = _installSearchQuery.trim().toLowerCase();
    Array.from(tbody.querySelectorAll("tr")).forEach(tr => {
      // Match against the FULL install id (stored as data-fullid
      // on the first cell's <code>) so partial UUID searches work
      // even though the visible text is the shortened form.
      const code = tr.querySelector("code.install");
      const fullId = code ? (code.getAttribute("data-fullid") || code.textContent || "") : "";
      tr.style.display = !q || String(fullId).toLowerCase().includes(q) ? "" : "none";
    });
  }
  const _installSearch = document.getElementById("install-search");
  if (_installSearch) {
    _installSearch.addEventListener("input", e => {
      _installSearchQuery = (e.target && e.target.value) || "";
      _applyInstallSearch();
    });
  }

  // Per-user session expander: clicking a row in the Users table
  // injects a sub-row right below it containing a per-session
  // breakdown for that install_id. Click again to collapse. Uses
  // event delegation on the tbody so re-renders don't drop the
  // binding.
  const _usersTblBody = document.querySelector("#users-tbl tbody");
  function _fmtSessionDuration(sec) {
    if (sec === null || sec === undefined || isNaN(sec)) return "—";
    const s = Math.max(0, Math.round(Number(sec)));
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m " + (s % 60) + "s";
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h + "h " + m + "m";
  }
  async function _toggleSessionExpander(tr) {
    const installId = tr.getAttribute("data-installid") || "";
    if (!installId) return;
    const caret = tr.querySelector(".exp-caret");
    const next = tr.nextElementSibling;
    if (next && next.classList.contains("user-sessions-row")) {
      // Already expanded — collapse.
      next.remove();
      if (caret) caret.textContent = "▶";
      return;
    }
    if (caret) caret.textContent = "▼";
    // Insert a placeholder row so the user sees the expand happen
    // before the network round-trip completes.
    const colCount = tr.children.length;
    const placeholder = document.createElement("tr");
    placeholder.className = "user-sessions-row";
    placeholder.innerHTML = '<td colspan="' + colCount + '" class="muted" style="padding:10px 14px;background:#0a162b">Loading sessions…</td>';
    tr.parentNode.insertBefore(placeholder, tr.nextSibling);
    try {
      const res = await fetch("/api/sessions?install_id=" + encodeURIComponent(installId)
        + "&token=" + encodeURIComponent(token));
      if (!res.ok) {
        placeholder.firstChild.innerHTML = '<span style="color:#ff7e7e">Could not load sessions (HTTP ' + res.status + ').</span>';
        return;
      }
      const body = await res.json();
      const sessions = (body && body.sessions) || [];
      if (sessions.length === 0) {
        placeholder.firstChild.innerHTML = '<span class="muted">No app_session_started events for this install.</span>';
        return;
      }
      const headers = ["#", "started", "ended", "open", "running", "actions", "gestures", "errors", "version"];
      const headerHtml = '<tr>' + headers.map(h => '<th style="text-align:left;padding:6px 10px;font-size:11px;color:var(--muted);font-weight:600;text-transform:uppercase">' + esc(h) + '</th>').join('') + '</tr>';
      const bodyHtml = sessions.map((sn, i) => {
        const engineCell = (sn.engine_seconds && sn.engine_seconds > 0)
          ? esc(_fmtSessionDuration(sn.engine_seconds))
            + (sn.engine_runs > 1 ? ' <span class="muted">(' + fmt(sn.engine_runs) + 'x)</span>' : '')
          : '<span class="muted">—</span>';
        const cells = [
          String(sessions.length - i),
          esc(fmtTime(sn.started_at)),
          sn.ended_at ? esc(fmtTime(sn.ended_at)) : '<span class="muted">— (no end event)</span>',
          esc(_fmtSessionDuration(sn.duration_seconds)),
          engineCell,
          fmt(sn.actions || 0),
          fmt(sn.gestures || 0),
          (sn.errors && sn.errors > 0) ? '<span style="color:#ff7e7e">' + fmt(sn.errors) + '</span>' : fmt(0),
          sn.app_version ? esc(String(sn.app_version)) : '<span class="muted">—</span>',
        ];
        return '<tr>' + cells.map(c => '<td style="padding:5px 10px;font-size:12px;border-top:1px solid var(--border)">' + c + '</td>').join('') + '</tr>';
      }).join("");
      // "Untracked" / orphan row — events from sessions whose
      // app_session_started got dropped by bug 1 (init-time start
      // fired before user clicked Allow). Worker computes the bucket
      // as everything BEFORE the first app_session_started for this
      // install_id. Rendered as a single dim row at the bottom so the
      // dashboard math matches the row totals.
      const orphan = body && body.orphan;
      let orphanHtml = '';
      if (orphan && (orphan.gestures > 0 || orphan.actions > 0 || orphan.duration_seconds > 0)) {
        const orphanCells = [
          '<span class="muted">—</span>',
          '<span class="muted">before first start</span>',
          '<span class="muted">—</span>',
          esc(_fmtSessionDuration(orphan.duration_seconds || 0)),
          '<span class="muted">—</span>',
          fmt(orphan.actions || 0),
          fmt(orphan.gestures || 0),
          (orphan.errors && orphan.errors > 0) ? '<span style="color:#ff7e7e">' + fmt(orphan.errors) + '</span>' : fmt(0),
          '<span class="muted">pre-fix</span>',
        ];
        orphanHtml = '<tr style="opacity:0.7">'
          + orphanCells.map(c => '<td style="padding:5px 10px;font-size:12px;border-top:1px dashed var(--border);font-style:italic">' + c + '</td>').join('')
          + '</tr>';
      }
      const sessLabel = sessions.length + ' session' + (sessions.length === 1 ? '' : 's');
      const orphanLabel = orphan && (orphan.gestures > 0 || orphan.actions > 0 || orphan.duration_seconds > 0)
        ? ' + untracked events (bug 1 footprint, see bottom row)'
        : '';
      const userLabel = (window._userLabelById && window._userLabelById[String(installId || "")]) || "User ?";
      placeholder.firstChild.innerHTML =
        '<div style="padding:8px 4px 12px 4px;background:#0a162b">'
        + '<div style="font-size:11px;color:var(--muted);margin:0 10px 6px 10px">'
        +   sessLabel + orphanLabel + ' for <strong>' + esc(userLabel) + '</strong> '
        +   '<code style="font-family:ui-monospace,monospace">' + esc(installId) + '</code>'
        + '</div>'
        + '<table style="width:100%;border-collapse:collapse">'
        +   '<thead>' + headerHtml + '</thead>'
        +   '<tbody>' + bodyHtml + orphanHtml + '</tbody>'
        + '</table>'
        + '</div>';
      placeholder.firstChild.style.padding = "0";
    } catch (err) {
      placeholder.firstChild.innerHTML = '<span style="color:#ff7e7e">Network error: ' + esc(String(err)) + '</span>';
    }
  }
  if (_usersTblBody) {
    _usersTblBody.addEventListener("click", (e) => {
      // Find the closest user-row from the click target. We delegate so
      // re-renders (which replace the tbody contents) don't drop the
      // handler.
      let el = e.target;
      while (el && el !== _usersTblBody && !(el.classList && el.classList.contains("user-row"))) {
        el = el.parentNode;
      }
      if (el && el.classList && el.classList.contains("user-row")) {
        _toggleSessionExpander(el);
      }
    });
  }

  // Consolidate-installs modal: pick a "keeper" install_id, fold every
  // other install_id in the DB into it via /api/consolidate. Used to
  // clean up ghost rows from the era before the MachineGuid-derived
  // install_id (multiple settings.json wipes created multiple random
  // uuids for the same real user).
  let _allUsers = [];
  const _consolidateBtn = document.getElementById("consolidate-btn");
  const _consolidateModal = document.getElementById("consolidate-modal");
  const _consolidateTarget = document.getElementById("consolidate-target");
  const _consolidateSummary = document.getElementById("consolidate-summary");
  const _consolidateStatus = document.getElementById("consolidate-status");
  const _consolidateConfirm = document.getElementById("consolidate-confirm");
  const _consolidateCancel = document.getElementById("consolidate-cancel");

  function _openConsolidateModal() {
    if (!_consolidateModal) return;
    // Populate the dropdown with every install_id we know about, most-
    // recently-active first (mirrors the order in the users table).
    _consolidateTarget.innerHTML = "";
    for (const u of _allUsers) {
      const opt = document.createElement("option");
      opt.value = u.install_id;
      const shortId = String(u.install_id || "").slice(0, 8) + "…" + String(u.install_id || "").slice(-4);
      const seen = u.last_seen ? new Date(u.last_seen).toLocaleString() : "—";
      // Show the friendly user number first so the dropdown is
      // human-readable, then the truncated hash for unambiguous
      // identification when two users have similar activity profiles.
      const label = (window._userLabelById && window._userLabelById[String(u.install_id || "")]) || "User ?";
      opt.textContent = label + "   (" + shortId + ")   ·   last seen " + seen + "   ·   " + (u.sessions || 0) + " sessions";
      _consolidateTarget.appendChild(opt);
    }
    const ghostCount = Math.max(0, _allUsers.length - 1);
    _consolidateSummary.innerHTML =
      "Folding will rewrite events from <strong>" + ghostCount + "</strong> "
      + (ghostCount === 1 ? "ghost" : "ghosts") + " into the selected keeper.";
    _consolidateStatus.textContent = "";
    _consolidateConfirm.disabled = (_allUsers.length < 2);
    _consolidateConfirm.style.opacity = _consolidateConfirm.disabled ? "0.4" : "1";
    _consolidateModal.style.display = "flex";
  }
  function _closeConsolidateModal() {
    if (_consolidateModal) _consolidateModal.style.display = "none";
  }
  async function _runConsolidate() {
    const target = _consolidateTarget.value;
    if (!target) return;
    _consolidateStatus.textContent = "Folding…";
    _consolidateConfirm.disabled = true;
    _consolidateConfirm.style.opacity = "0.4";
    try {
      // The same SHARED_SECRET that gates the dashboard read is what
      // the worker checks on /api/consolidate. We reuse the outer-scope
      // token variable (defined at the top of the IIFE) which holds
      // either the URL ?token=... or the value the user typed into
      // the auth prompt at load time — re-reading from URL alone would
      // break the prompt-path because params.get("token") returns null
      // when the token came from prompt().
      if (!token) {
        _consolidateStatus.textContent = "Error: no auth token available.";
        _consolidateConfirm.disabled = false;
        _consolidateConfirm.style.opacity = "1";
        return;
      }
      const res = await fetch("/api/consolidate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: token,
          target_install_id: target,
          merge_all_others: true,
        }),
      });
      const body = await res.json();
      if (!res.ok || body.ok === false) {
        _consolidateStatus.textContent = "Error: " + (body.error || res.statusText);
        _consolidateConfirm.disabled = false;
        _consolidateConfirm.style.opacity = "1";
        return;
      }
      _consolidateStatus.textContent =
        "Done. " + (body.rows_updated || 0) + " events re-tagged to "
        + (target.slice(0, 8) + "…" + target.slice(-4)) + ". Refreshing…";
      // Small delay so the user sees the result before the modal closes.
      setTimeout(() => { _closeConsolidateModal(); load(currentRange); }, 1200);
    } catch (err) {
      _consolidateStatus.textContent = "Network error: " + String(err);
      _consolidateConfirm.disabled = false;
      _consolidateConfirm.style.opacity = "1";
    }
  }
  if (_consolidateBtn) _consolidateBtn.addEventListener("click", _openConsolidateModal);
  if (_consolidateCancel) _consolidateCancel.addEventListener("click", _closeConsolidateModal);
  if (_consolidateConfirm) _consolidateConfirm.addEventListener("click", _runConsolidate);
  if (_consolidateModal) {
    _consolidateModal.addEventListener("click", (e) => {
      // Click on the backdrop (the modal itself, not the inner card)
      // dismisses without acting.
      if (e.target === _consolidateModal) _closeConsolidateModal();
    });
  }

  function render(data) {
    const lbl = data.range_label || RANGE_LABELS[currentRange] || currentRange;
    const lblShort = "(" + lbl + ")";
    document.getElementById("sub").textContent =
      "Updated " + new Date().toLocaleString()
      + " · " + (Intl.DateTimeFormat().resolvedOptions().timeZone || "local")
      + " · range: " + lbl;

    // -- Overview --------------------------------------------------
    document.getElementById("installs").textContent = fmt(data.unique_installs);
    document.getElementById("events").textContent = fmt(data.total_events);
    document.getElementById("events-range-lbl").textContent = lblShort;

    const dau = data.daily_active || [];
    document.getElementById("today").textContent = fmt(dau[0]?.users || 0);
    let max7 = 0;
    for (const r of dau.slice(0, 7)) max7 = Math.max(max7, r.users || 0);
    document.getElementById("seven").textContent = fmt(max7);

    const dauEl = document.getElementById("dau");
    if (dau.length === 0) {
      dauEl.innerHTML = '<div class="sub">No data yet.</div>';
    } else {
      const max = Math.max(1, ...dau.map(r => r.users));
      dauEl.innerHTML = dau.map(r => {
        const p = Math.round((r.users / max) * 100);
        return '<div class="day-row">'
          + '<div class="day">' + esc(fmtDay(r.day)) + '</div>'
          + '<div class="bar-cell"><div class="bar" style="--w:' + p + '%"></div></div>'
          + '<div class="num">' + fmt(r.users) + '</div></div>';
      }).join("");
    }
    fillTable("top-events-tbl", data.top_events,
      r => '<td>' + esc(r.event ?? "(unknown)") + '</td><td class="num">' + fmt(r.n) + '</td>', 10);

    // -- Activation funnel + retention ----------------------------
    renderFunnel(data.funnel || {});
    renderRetention(data.retention || []);

    // -- Features --------------------------------------------------
    document.getElementById("features-range-lbl").textContent = lblShort;
    renderFeatures(data.features || {});

    // -- Sessions --------------------------------------------------
    const s = data.sessions || {};
    document.getElementById("sessions-range-stat").textContent = lblShort;
    document.getElementById("sessions-active").textContent = fmt(s.active_installs || 0);
    document.getElementById("sessions-total").textContent = fmt(s.total_sessions || 0);
    document.getElementById("sessions-avg").textContent = (s.avg_sessions ?? 0).toFixed
      ? (Number(s.avg_sessions || 0)).toFixed(1) : fmt(s.avg_sessions || 0);
    document.getElementById("sessions-actions-avg").textContent = (s.avg_actions_per_session ?? 0).toFixed
      ? (Number(s.avg_actions_per_session || 0)).toFixed(1) : fmt(s.avg_actions_per_session || 0);

    const dur = data.session_duration || {};
    document.getElementById("duration-total").textContent = fmtDuration(dur.total_seconds || 0);
    document.getElementById("duration-avg").textContent = fmtDuration(dur.avg_seconds || 0);
    document.getElementById("duration-max").textContent = fmtDuration(dur.max_seconds || 0);
    document.getElementById("duration-count").textContent = fmt(dur.sessions_with_duration || 0);
    renderDurationDaily(data.session_duration_daily || []);
    renderSessionBuckets(data.session_buckets || []);

    document.getElementById("sessions-list-range").textContent = lblShort;
    fillTable("sessions-list-tbl", data.recent_sessions, r =>
      '<td><code class="install">' + esc(shortInstall(r.install_id)) + '</code></td>'
      + '<td class="muted">' + esc(fmtTime(r.ended_at)) + '</td>'
      + '<td class="num">' + fmtDuration(r.duration) + '</td>', 15);

    // -- Users -----------------------------------------------------
    document.getElementById("users-total").textContent = fmt(data.total_users || 0);
    document.getElementById("users-24h").textContent = fmt(data.active_24h || 0);
    document.getElementById("users-7d").textContent = fmt(data.active_7d || 0);
    document.getElementById("users-30d").textContent = fmt(data.active_30d || 0);
    // Cache the full users list so the consolidate modal can populate
    // its dropdown without re-fetching.
    _allUsers = Array.isArray(data.users) ? data.users : [];
    // Build a stable install_id → "User N" mapping. Number by oldest
    // first_seen first (User 1 = earliest install in DB) so the
    // labels stay put as new users join — newcomers always get
    // higher numbers, existing ones never shift. Ghost-row install_ids
    // also get numbered so SQL cleanup can still reference them.
    const _byFirstSeen = _allUsers.slice().sort((a, b) => {
      const ta = a.first_seen ? new Date(a.first_seen).getTime() : 0;
      const tb = b.first_seen ? new Date(b.first_seen).getTime() : 0;
      return ta - tb;
    });
    window._userLabelById = {};
    _byFirstSeen.forEach((u, i) => {
      window._userLabelById[String(u.install_id || "")] = "User " + (i + 1);
    });
    function _userLabel(installId) {
      return (window._userLabelById && window._userLabelById[String(installId || "")]) || ("User ?");
    }
    // Compute the activity-dot color from (last_state_event, last_seen)
    // for each row. Three states:
    //   GREEN  = engine running (last state edge = engine_started, and
    //            last activity within ~3 min so we know the engine
    //            hasn't silently been stopped without flushing)
    //   ORANGE = app open but engine off (app_session_started or
    //            engine_stopped was the last state edge)
    //   RED    = app closed (app_session_ended last) OR no recent
    //            activity within the time window
    function _activityDot(row) {
      const lastSeenMs = row.last_seen ? Date.parse(row.last_seen) : 0;
      const lastStateAtMs = row.last_state_at ? Date.parse(row.last_state_at) : 0;
      const now = Date.now();
      const ageSec = (now - Math.max(lastSeenMs, lastStateAtMs)) / 1000;
      // 3-minute cliff: telemetry flushes every 30 s + network jitter,
      // so beyond ~3 min we treat the install as offline regardless of
      // what state it claimed last.
      const stale = !Number.isFinite(ageSec) || ageSec > 180;
      let color = '#ff5e5e';   // red default
      let label = 'closed';
      if (!stale) {
        const lse = String(row.last_state_event || '');
        if (lse === 'engine_started') { color = '#1DE9B6'; label = 'engine running'; }
        else if (lse === 'app_session_started' || lse === 'engine_stopped') { color = '#FFB347'; label = 'app open, engine off'; }
        else if (lse === 'app_session_ended') { color = '#ff5e5e'; label = 'closed'; }
        else { color = '#FFB347'; label = 'app open'; }
      }
      return '<span title="' + esc(label) + '" '
        + 'style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        + 'background:' + color + ';margin-right:8px;vertical-align:middle;'
        + 'box-shadow:0 0 6px ' + color + '99;"></span>';
    }
    fillTable("users-tbl", data.users, r =>
      // Render "User N" as the primary label with the truncated hash
      // beneath it in muted small text — keeps the SHA-256 visible for
      // SQL / consolidate use without making the dashboard look like
      // a hex dump. data-fullid carries the full hash for the search
      // filter; data-installid on the <tr> drives the session expander.
      // Activity dot prepended: green = engine actively running,
      // orange = app open but engine off, red = closed / no recent
      // activity. Title attribute shows the precise label on hover.
      '<td><span class="exp-caret">▶</span> '
        + _activityDot(r)
        + '<span class="user-num">' + esc(_userLabel(r.install_id)) + '</span>'
        + '<code class="install" data-fullid="' + esc(String(r.install_id || "")) + '">'
        + esc(shortInstall(r.install_id)) + '</code></td>'
      + '<td class="muted">' + esc(fmtTime(r.first_seen)) + '</td>'
      + '<td class="muted">' + esc(fmtTime(r.last_seen)) + '</td>'
      + '<td class="num">' + fmt(r.sessions) + '</td>'
      + '<td class="num">' + fmtDuration(r.total_seconds) + '</td>'
      + '<td class="num">' + (r.engine_seconds ? fmtDuration(r.engine_seconds) : '<span class="muted">—</span>') + '</td>'
      + '<td class="num">' + fmt(r.actions) + '</td>'
      + '<td class="num">' + fmt(r.gestures) + '</td>'
      + '<td class="num">' + ((r.custom_gestures && r.custom_gestures > 0) ? fmt(r.custom_gestures) : '<span class="muted">—</span>') + '</td>', 15);
    // Tag each freshly-rendered user row with its install id and a
    // class so the click handler below can target them.
    const _usersTbody = document.querySelector("#users-tbl tbody");
    if (_usersTbody && Array.isArray(data.users)) {
      const rowsArr = _usersTbody.querySelectorAll("tr");
      data.users.forEach((u, i) => {
        const tr = rowsArr[i];
        if (!tr) return;
        tr.classList.add("user-row");
        tr.setAttribute("data-installid", String(u.install_id || ""));
        tr.style.cursor = "pointer";
      });
    }
    // Re-apply the install-id filter every refresh so a search
    // query entered before the data reloaded isn't lost.
    if (typeof _applyInstallSearch === "function") {
      _applyInstallSearch();
    }

    // -- Details ---------------------------------------------------
    document.getElementById("details-range-actions").textContent = lblShort;
    document.getElementById("details-range-static").textContent = lblShort;
    document.getElementById("details-range-dynamic").textContent = lblShort;
    document.getElementById("details-range-voice").textContent = lblShort;

    fillTable("actions-detail-tbl", data.actions_detail,
      r => '<td>' + esc(r.action_id ?? "(unknown)") + '</td>'
         + '<td class="num">' + fmt(r.n) + '</td>'
         + '<td class="num">' + fmt(r.reach) + '</td>', 12);

    fillTable("static-gestures-tbl", data.static_gestures,
      r => '<td>' + esc(r.gesture ?? "(unknown)") + '</td>'
         + '<td>' + esc(r.handedness || "—") + '</td>'
         + '<td class="num">' + fmt(r.n) + '</td>'
         + '<td class="num">' + fmt(r.reach) + '</td>', 12);

    fillTable("dynamic-gestures-tbl", data.dynamic_gestures,
      r => '<td>' + esc(r.gesture ?? "(unknown)") + '</td>'
         + '<td>' + esc(r.handedness || "—") + '</td>'
         + '<td class="num">' + fmt(r.n) + '</td>'
         + '<td class="num">' + fmt(r.reach) + '</td>', 12);

    fillTable("voice-targets-tbl", data.voice_targets, r => {
      const successFlag = (r.success === 1 || r.success === true || r.success === "1")
        ? '<span style="color:var(--accent)">✓</span>'
        : '<span style="color:var(--warn)">✗</span>';
      return '<td>' + esc(r.target ?? "(unknown)") + '</td>'
           + '<td>' + successFlag + '</td>'
           + '<td class="num">' + fmt(r.n) + '</td>'
           + '<td class="num">' + fmt(r.reach) + '</td>';
    }, 12);

    // -- Errors ----------------------------------------------------
    document.getElementById("errors-range-stat").textContent = lblShort;
    document.getElementById("errors-total").textContent = fmt(data.errors_total || 0);
    document.getElementById("errors-installs").textContent = fmt(data.errors_affected_installs || 0);
    fillTable("errors-tbl", data.errors,
      r => '<td>' + esc(r.component ?? "(unknown)") + '</td>'
         + '<td>' + esc(r.exc_type ?? "(unknown)") + '</td>'
         + '<td class="num">' + fmt(r.n) + '</td>', 10);
  }

  function renderFunnel(f) {
    const el = document.getElementById("funnel");
    const steps = [
      { key: "launched",  name: "Launched the app",        explain: "First app_session_started event from this install." },
      { key: "engine",    name: "Started the engine",      explain: "Camera + recognition pipeline came up at least once." },
      { key: "gestured",  name: "Made a recognized gesture", explain: "App saw a stable pose or dynamic gesture." },
      { key: "fired",     name: "Triggered an action",     explain: "Voice / drawing / mouse / Spotify / volume / etc. actually did something." },
    ];
    const top = Number(f.launched || 0);
    if (top === 0) {
      el.innerHTML = '<div class="sub">No installs yet.</div>';
      return;
    }
    let html = "";
    let prev = top;
    steps.forEach((s, i) => {
      const n = Number(f[s.key] || 0);
      const pctTop = top ? (n / top) * 100 : 0;
      const pctPrev = prev ? (n / prev) * 100 : 0;
      html +=
        '<div class="funnel-step">' +
          '<div>' +
            '<div class="name">' + esc(s.name) + '</div>' +
            '<div class="sub" style="margin:0">' + esc(s.explain) + '</div>' +
            '<div class="funnel-bar"><div style="width:' + pctTop.toFixed(1) + '%"></div></div>' +
          '</div>' +
          '<div class="count">' + fmt(n) + '</div>' +
          '<div class="pct">' + pctTop.toFixed(1) + '% of top</div>' +
        '</div>';
      if (i < steps.length - 1) {
        const next = Number(f[steps[i + 1].key] || 0);
        const drop = n - next;
        const dropPct = n ? (drop / n) * 100 : 0;
        const cls = dropPct > 50 ? "arrow bad" : "arrow";
        html += '<div class="' + cls + '">↓ '
          + (n ? (((next / n) * 100).toFixed(1) + "% advance") : "no data")
          + (drop > 0 ? " · " + fmt(drop) + " dropped off" : "")
          + '</div>';
      }
      prev = n;
    });
    el.innerHTML = html;
  }

  function renderRetention(rows) {
    const tbody = document.querySelector("#retention-tbl tbody");
    const totalEl = document.getElementById("retention-total");
    if (!rows || rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="sub">No cohorts yet.</td></tr>';
      totalEl.textContent = "";
      return;
    }
    tbody.innerHTML = rows.map(r => {
      const cs = Number(r.cohort_size || 0);
      const cell = (n) => {
        const v = Number(n || 0);
        const pctVal = cs ? (v / cs) * 100 : 0;
        const cls = (cs >= 3 && pctVal < 20) ? ' class="pct weak"' : ' class="pct"';
        return '<td' + cls + '>' + fmt(v) + ' · ' + (cs ? pctVal.toFixed(0) + "%" : "—") + '</td>';
      };
      return '<tr><td>' + esc(fmtDay(r.cohort_day)) + '</td>'
           + '<td class="pct">' + fmt(cs) + '</td>'
           + cell(r.d1) + cell(r.d7) + cell(r.d30) + '</tr>';
    }).join("");
    totalEl.textContent = rows.length + " cohort" + (rows.length === 1 ? "" : "s") + " shown";
  }

  function renderFeatures(f) {
    const installs = Number(f.installs || 0);
    const features = [
      { k: "voice",   label: "Voice / Dictation" },
      { k: "drawing", label: "Drawing" },
      { k: "mouse",   label: "Mouse mode" },
      { k: "spotify", label: "Spotify" },
      { k: "volume",  label: "Volume" },
      { k: "youtube", label: "YouTube" },
      { k: "chrome",  label: "Chrome nav" },
    ];
    const el = document.getElementById("features-list");
    if (installs === 0) {
      el.innerHTML = '<div class="sub">No active installs in this range.</div>';
      return;
    }
    const max = Math.max(1, ...features.map(x => Number(f[x.k] || 0)));
    el.innerHTML = features.map(x => {
      const n = Number(f[x.k] || 0);
      const p = installs ? (n / installs) * 100 : 0;
      const wPct = (n / max) * 100;
      return '<div class="feature-row">'
        + '<div class="name">' + esc(x.label) + '</div>'
        + '<div class="bar-cell"><div class="bar" style="--w:' + wPct.toFixed(1) + '%"></div></div>'
        + '<div class="pct">' + p.toFixed(0) + '%</div>'
        + '<div class="count">' + fmt(n) + ' / ' + fmt(installs) + '</div>'
        + '</div>';
    }).join("");
  }

  function renderDurationDaily(rows) {
    const el = document.getElementById("duration-daily");
    if (!rows || rows.length === 0) {
      el.innerHTML = '<div class="sub">No completed sessions yet.</div>';
      return;
    }
    const max = Math.max(1, ...rows.map(r => Number(r.seconds || 0)));
    el.innerHTML = rows.slice().reverse().map(r => {
      const secs = Number(r.seconds || 0);
      const w = (secs / max) * 100;
      return '<div class="day-row">'
        + '<div class="day">' + esc(fmtDay(r.day)) + '</div>'
        + '<div class="bar-cell"><div class="bar" style="--w:' + w.toFixed(1) + '%"></div></div>'
        + '<div class="num">' + fmtDuration(secs) + '</div></div>';
    }).join("");
  }

  function renderSessionBuckets(rows) {
    const el = document.getElementById("session-buckets");
    if (!rows || rows.length === 0) {
      el.innerHTML = '<div class="sub">No active installs in this range.</div>';
      return;
    }
    const total = rows.reduce((s, r) => s + Number(r.installs || 0), 0) || 1;
    const max = Math.max(1, ...rows.map(r => Number(r.installs || 0)));
    el.innerHTML = rows.map(r => {
      const n = Number(r.installs || 0);
      const w = (n / max) * 100;
      const p = (n / total) * 100;
      return '<div class="feature-row">'
        + '<div class="name">' + esc(r.bucket) + ' sessions</div>'
        + '<div class="bar-cell"><div class="bar" style="--w:' + w.toFixed(1) + '%"></div></div>'
        + '<div class="pct">' + p.toFixed(0) + '%</div>'
        + '<div class="count">' + fmt(n) + '</div>'
        + '</div>';
    }).join("");
  }

  function fillTable(id, rows, renderRow, initial) {
    const tbody = document.querySelector("#" + id + " tbody");
    const toggle = document.getElementById(id.replace("-tbl", "-toggle"));
    const totalEl = document.getElementById(id.replace("-tbl", "-total"));
    const colCount = (tbody.previousElementSibling?.querySelectorAll("th")?.length) || 2;
    if (!rows || rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="' + colCount + '" class="sub">No data in this range.</td></tr>';
      if (toggle) toggle.style.display = "none";
      if (totalEl) totalEl.textContent = "";
      return;
    }
    if (totalEl) totalEl.textContent = rows.length + " unique row" + (rows.length === 1 ? "" : "s");
    const visible = Math.min(initial, rows.length);
    tbody.innerHTML = rows.map((r, i) => {
      const cls = i >= visible ? ' class="hidden"' : '';
      return '<tr' + cls + '>' + renderRow(r) + '</tr>';
    }).join("");
    if (toggle) {
      if (rows.length > initial) {
        toggle.style.display = "inline-block";
        let expanded = false;
        toggle.textContent = "Show all (" + rows.length + ")";
        toggle.onclick = () => {
          expanded = !expanded;
          tbody.querySelectorAll("tr").forEach((tr, i) => {
            tr.classList.toggle("hidden", !expanded && i >= initial);
          });
          toggle.textContent = expanded ? "Show fewer" : "Show all (" + rows.length + ")";
        };
      } else {
        toggle.style.display = "none";
      }
    }
  }

  activateTab(currentTab);
  load(currentRange);
})();
</script>
</body>
</html>`;
