# Touchless — Privacy Policy

_Last updated: May 26, 2026_

Touchless ("the app", "we") is a hand-gesture and voice desktop-control
application for Windows. This policy explains what data the app accesses
when you connect optional Google features, how it is used, and your
choices. **Connecting Google is entirely optional** — the core app works
without it.

## What we access (only if you connect Google)

When you click **Connect Gmail** and grant consent, Touchless requests these
Google scopes to perform actions **you explicitly ask for**:

| Permission (scope) | Why Touchless uses it |
|---|---|
| Send email on your behalf (`gmail.send`) | To send an email you compose by voice or gesture. The app can **only send** — it cannot read, search, or access your mailbox. |
| Manage your calendar (`calendar`) | To show your upcoming events and create events when you ask Touchless to schedule something. |
| Create Google Docs (`documents`) | To create a document when you ask Touchless to make one. |
| Create Google Sheets (`spreadsheets`) | To create a spreadsheet when you ask Touchless to make one. |
| Create Google Slides (`presentations`) | To create a presentation when you ask Touchless to make one. |
| Per-file Drive access (`drive.file`) | To upload a file to your Drive or list files Touchless created. This scope only ever touches files the app creates or you open with it — never your whole Drive. |

> **Directions** is provided separately via the Google Maps Platform using an
> API key (not your Google sign-in). When you ask for directions, Touchless
> sends the origin and destination you provide to Google Maps to fetch a
> route; it does not access your Google account for this.

## How we use it

We use this access **solely to carry out the specific action you request**
(send this email, create this event, make this doc, upload this file). We do
not use it for advertising, profiling, or any purpose unrelated to the
feature you invoked.

## Where your data lives

- **Your Google authorization token is stored locally on your own device**
  (`Documents/Touchless/google/token.json`). It is **not** transmitted to
  us or any third-party server.
- Touchless talks to Google's APIs **directly from your machine**. We do not
  operate a server that receives, stores, or proxies your Google data.
- We do **not** sell, rent, or share your Google user data with anyone.

## Limited Use disclosure

Touchless's use and transfer of information received from Google APIs
adheres to the
[Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy),
including the **Limited Use** requirements. We only use the data to provide
the user-facing features above, do not transfer it except as necessary to
provide those features, and do not use it for advertising or to train
generalized AI/ML models.

## Your choices

- You can decline the Google connection and still use the rest of Touchless.
- You can **revoke access** at any time at
  [myaccount.google.com/permissions](https://myaccount.google.com/permissions),
  or by deleting `Documents/Touchless/google/token.json`.

## Other data

Touchless processes camera frames and audio **locally** for gesture and
voice control; these are not uploaded by this Google integration. (See the
in-app documentation for details on the core app's processing.)

## Contact

Questions about this policy or your data: **dani@mangollc.org**.

We may update this policy; material changes will be reflected by the "Last
updated" date above.
