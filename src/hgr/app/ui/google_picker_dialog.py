"""Google Picker widget hosted inside a modal QDialog.

Why a modal (this is the one narrow inversion of the CLAUDE.md rule 6
"no blocking modal windows during live camera"): the Google Picker is
an OAuth-scoped file chooser that must own interaction until the user
picks or cancels — a partial pick would leave a stale token cached in
the Picker's in-page JS. Iris's voice loop is paused for the duration
(same pattern as mcp_picker.py's setup dialog).

Under drive.file scope, the Picker is the ONLY way for the user to
grant this app read/write access to files it did not itself create.
The chosen file's id is cached in google_picker_cache.PickerCache so
each file only needs to be picked once.

The Picker widget itself CANNOT be self-hosted — Google's
apis.google.com/js/api.js is versioned + CSP-locked to their origin.
We serve an in-memory HTML shell that pulls the script from Google,
builds a picker with the caller's OAuth token + our developer API key,
and posts the picked file back to Python via QWebChannel.

Public surface:
    open_picker_for_kind(parent, kind, oauth_token, api_key, title_hint)
      -> {"file_id","name","mime"} on pick, None on cancel.

The function blocks the calling thread by running the Qt event loop
via QDialog.exec() — MUST be called from the Qt main thread. Connector
worker threads should marshal via a signal + threading.Event.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
from typing import Optional


# Kind -> Picker ViewId identifier the JS shell selects from.
_ALLOWED_KINDS = {"sheet", "doc", "slide", "any"}


_PICKER_HTML_TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Pick a Google file</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%;
    background: #101418; color: #e6e6e6;
    font-family: 'Segoe UI', system-ui, sans-serif; }
  #status { padding: 12px 16px; font-size: 13px; }
</style>
</head><body>
  <div id="status">Loading Google Picker...</div>
  <div id="root"></div>
  <script src="https://apis.google.com/js/api.js"></script>
  <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
  <script>
    const KIND = "__KIND__";
    const TOKEN = "__TOKEN__";
    const KEY = "__KEY__";
    function setStatus(msg) {
      const el = document.getElementById('status');
      if (el) el.textContent = msg;
    }
    function reportError(msg) {
      setStatus(msg);
      try {
        new QWebChannel(qt.webChannelTransport, function(ch) {
          const py = ch.objects.pickerBridge;
          if (py && py.failed) py.failed(msg);
        });
      } catch (e) { /* channel not up yet */ }
    }
    try {
      new QWebChannel(qt.webChannelTransport, function(ch) {
        const py = ch.objects.pickerBridge;
        if (!py) { reportError('QWebChannel bridge missing'); return; }
        if (typeof gapi === 'undefined') {
          reportError('Google Picker JS did not load (network / CSP?)');
          return;
        }
        gapi.load('picker', {callback: function() {
          try {
            let viewId;
            if (KIND === 'sheet') viewId = google.picker.ViewId.SPREADSHEETS;
            else if (KIND === 'doc') viewId = google.picker.ViewId.DOCUMENTS;
            else if (KIND === 'slide') viewId = google.picker.ViewId.PRESENTATIONS;
            else viewId = google.picker.ViewId.DOCS;
            const view = new google.picker.View(viewId);
            const picker = new google.picker.PickerBuilder()
              .addView(view)
              .setOAuthToken(TOKEN)
              .setDeveloperKey(KEY)
              .setCallback(function(data) {
                if (data.action === google.picker.Action.PICKED) {
                  const f = data.docs && data.docs[0];
                  if (!f) { py.cancelled(); return; }
                  py.picked(JSON.stringify({
                    file_id: f.id, name: f.name, mime: f.mimeType
                  }));
                } else if (data.action === google.picker.Action.CANCEL) {
                  py.cancelled();
                }
              })
              .build();
            setStatus('');
            picker.setVisible(true);
          } catch (err) {
            reportError('Picker build failed: ' + err);
          }
        }, onerror: function(err) {
          reportError('gapi.load picker error: ' + err);
        }});
      });
    } catch (err) {
      reportError('bootstrap failed: ' + err);
    }
  </script>
</body></html>
"""


def _substitute_html(kind: str, token: str, api_key: str) -> str:
    # Escape token/key for a JS string literal. json.dumps handles the
    # quoting including any embedded backslashes / quotes; we strip the
    # surrounding quotes and re-insert as the JS-literal payload.
    safe_kind = json.dumps(kind).strip('"')
    safe_token = json.dumps(token).strip('"')
    safe_key = json.dumps(api_key).strip('"')
    return (_PICKER_HTML_TEMPLATE
            .replace("__KIND__", safe_kind)
            .replace("__TOKEN__", safe_token)
            .replace("__KEY__", safe_key))


def open_picker_for_kind(parent,
                         kind: str,
                         oauth_token: str,
                         api_key: str,
                         title_hint: str = "") -> Optional[dict]:
    """Open the Picker modally and return the chosen file, or None.

    kind: 'sheet' | 'doc' | 'slide' | 'any'
    oauth_token: creds.token from GoogleClient._load_creds().
    api_key: developer key with Picker API enabled (see
        google_client._picker_api_key()).
    title_hint: shown as the window title so the user knows which file
        Iris is asking for.

    Returns {"file_id","name","mime"} on pick, or None on cancel/error.
    MUST be called from the Qt main thread; the internal exec()
    re-enters the event loop.
    """
    kind = str(kind or "any").strip().lower()
    if kind not in _ALLOWED_KINDS:
        kind = "any"
    if not oauth_token or not api_key:
        return None
    try:
        from PySide6.QtCore import QObject, QUrl, Signal, Slot
        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QDialog, QVBoxLayout
    except Exception:
        # QtWebEngine missing on this build — caller must degrade to
        # "paste a file id" or ship the picker in a later build.
        return None

    class _PickerBridge(QObject):
        # Signals fired toward Python. Named distinctly from the JS-facing
        # slot methods below so the class attribute doesn't shadow itself
        # (PySide's Signal descriptor + a method of the same name would
        # overwrite the descriptor).
        pickedSignal = Signal(str)
        cancelledSignal = Signal()
        failedSignal = Signal(str)

        @Slot(str)
        def picked(self, payload_json: str) -> None:
            """JS -> Python: user picked a file."""
            self.pickedSignal.emit(payload_json)

        @Slot()
        def cancelled(self) -> None:
            """JS -> Python: user closed the Picker without choosing."""
            self.cancelledSignal.emit()

        @Slot(str)
        def failed(self, msg: str) -> None:
            """JS -> Python: bootstrap or gapi.load failed."""
            self.failedSignal.emit(msg)

    dlg = QDialog(parent)
    dlg.setWindowTitle(title_hint or "Pick a Google file")
    dlg.resize(900, 640)
    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(0, 0, 0, 0)

    view = QWebEngineView(dlg)
    layout.addWidget(view)

    bridge = _PickerBridge()
    channel = QWebChannel(view.page())
    channel.registerObject("pickerBridge", bridge)
    view.page().setWebChannel(channel)

    result: dict = {"value": None}

    def _on_picked(payload_json: str) -> None:
        try:
            result["value"] = json.loads(payload_json)
        except Exception:
            result["value"] = None
        dlg.accept()

    def _on_cancelled() -> None:
        result["value"] = None
        dlg.reject()

    def _on_failed(msg: str) -> None:
        # Log to stderr — the caller sees None and can degrade.
        try:
            import sys
            sys.stderr.write(f"[google_picker] {msg}\n")
        except Exception:
            pass

    bridge.pickedSignal.connect(_on_picked)
    bridge.cancelledSignal.connect(_on_cancelled)
    bridge.failedSignal.connect(_on_failed)

    html = _substitute_html(kind, oauth_token, api_key)
    # baseUrl must match an origin allowed on the Google Cloud client
    # so the Picker can talk to Google with the OAuth token. We use
    # the app's registered origin (touchless-control.com), the same
    # one already whitelisted for the desktop-app drive.file flow.
    view.setHtml(html, QUrl("https://touchless-control.com/picker"))

    dlg.exec()
    return result["value"]
