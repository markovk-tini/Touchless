# Google OAuth verification — submission guide (Touchless)

Goal: move the OAuth consent screen from **Testing** to **In production /
verified** so *any* Google user can click **Connect Gmail** with no warning.
All requested scopes are **sensitive** (or narrow) — **free** verification,
**no** paid CASA security assessment.

## Scopes requested (and why) — paste these as the justifications

| Scope | Justification (use verbatim) |
|---|---|
| `…/auth/gmail.send` | Touchless sends an email the user dictates by voice or gesture. The app only sends mail; it never reads, searches, or accesses the user's mailbox. |
| `…/auth/calendar` | Touchless shows the user's upcoming events and creates new events when the user asks to schedule something by voice or gesture. |
| `…/auth/documents` | Touchless creates a new Google Doc (title + body) when the user asks it to make a document by voice or gesture. |
| `…/auth/spreadsheets` | Touchless creates a new Google Sheet (title + optional rows) when the user asks it to make a spreadsheet. |
| `…/auth/presentations` | Touchless creates a new Google Slides presentation when the user asks it to make a slideshow. |
| `…/auth/drive.file` | Touchless uploads a user-chosen local file to their Drive and lists files it created, when the user asks to save/upload a file. Uses the per-file scope only — never the user's full Drive. |

> Directions (Google Maps) is **not** part of this verification — it uses a
> Maps Platform API key, not OAuth, and is currently disabled.

> Reviewer key point to emphasize: every scope maps to an explicit
> user-invoked action; nothing runs in the background; data stays on the
> user's device (no server).

## Pre-submission checklist

1. **Privacy policy is live** on an authorized domain
   (publish `docs/PRIVACY_POLICY.md` at e.g.
   `https://touchless-control.com/privacy`).
2. **Homepage is live**: `https://touchless-control.com`.
3. **Domain verified** in [Search Console](https://search.google.com/search-console)
   (DNS TXT record via Cloudflare), and added under **Authorized domains**.
4. **OAuth consent screen** fully filled: app name `Touchless`, support
   email, developer contact, homepage, privacy-policy link, authorized
   domain.
5. **Scopes** list contains exactly the four above (Data access → Add/remove
   scopes).
6. **App is functional for each scope** (Connect Gmail works; you can send an
   email, create an event, create a doc, upload a file) — required for the
   demo video.

## Demo video (YouTube, unlisted) — script

Record your screen (with audio/captions, in English). Show, in order:

1. **App + connect:** open Touchless → click **Connect Gmail** → the Google
   consent screen appears → point out the **app name** and the **scopes**
   listed → grant consent. (This proves the client/scopes match.)
2. **gmail.send:** say/type `send an email to <addr> saying hello` → show the
   confirmation → show it sent.
3. **calendar:** `add a calendar event tomorrow at 3pm called Demo` → show
   the event created.
4. **documents:** `make a google doc titled Demo with the text hello` → show
   the doc opens/created.
5. **spreadsheets:** `make a spreadsheet titled Demo` → show it created.
6. **presentations:** `make a slideshow titled Demo` → show it created.
7. **drive.file:** `upload <some local file> to my Google Drive` → show the
   file in Drive.
8. Briefly show the **privacy policy** page.

Keep it 2–4 minutes. Put the unlisted YouTube link in the submission.

## Submit

OAuth consent screen → **Publishing status → Publish app** → then
**Prepare/Submit for verification** → paste justifications, the video link,
and confirm homepage + privacy policy → submit.

## After submitting

- Review for sensitive scopes typically takes **days to a few weeks**;
  Google may email follow-ups — answer promptly from the developer contact.
- Test users keep working throughout, so the app stays usable while you wait.
- On approval: the "unverified app" warning disappears and **any** Google
  user can connect with one click.

## Notes

- Adding scopes later (e.g. a restricted one like reading mail) would
  require **re-verification** and, for restricted scopes, the paid CASA
  assessment — avoid unless needed.
- If you ever drop a feature, remove its scope to keep the consent screen
  minimal and the review simple.
