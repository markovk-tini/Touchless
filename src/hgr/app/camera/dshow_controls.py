"""DirectShow camera-control snapshot / diff / restore (v1.1.9.2 r18).

Reads and, ONLY to undo a change of our own, writes the driver's
IAMCameraControl (Exposure, Focus, Zoom, Pan, Tilt, Roll, Iris) and
IAMVideoProcAmp (Brightness, Contrast, Hue, Saturation, Sharpness, Gamma,
ColorEnable, WhiteBalance, BacklightCompensation, Gain) properties through
COM, including each property's Auto / Manual FLAG. OpenCV's DirectShow
backend cannot read that flag (cap.get(CAP_PROP_AUTO_EXPOSURE) is always
-1.0), which is how a "0.75 then 3.0" pair silently left cameras in
Manual for months. This module is the exact instrument for the promise
"Touchless does not change your camera settings, and if it ever does it
puts them back":

    before = snapshot_all()                # engine about to start
    ...camera session...
    after = snapshot_all()                 # capture released
    changes = diff(before, after)          # what WE changed (see rules)
    if changes: restore(before, changes)   # write back value + flag

Diff rules (so driver-owned motion is never mistaken for our change):
  * a FLAG change (Auto <-> Manual) is always a change;
  * a VALUE change counts only when the property was Manual before
    (in Auto the driver owns the value and moves it with the light).

Everything is best-effort and fully guarded: any COM failure returns an
empty snapshot / empty diff and logs one line. No exceptions escape.
Windows-only; on other platforms every function is a no-op.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional

CAMERA_PROPS = ["Pan", "Tilt", "Roll", "Zoom", "Exposure", "Iris", "Focus"]
PROCAMP_PROPS = ["Brightness", "Contrast", "Hue", "Saturation", "Sharpness", "Gamma",
                 "ColorEnable", "WhiteBalance", "BacklightCompensation", "Gain"]
FLAG_AUTO = 1
FLAG_MANUAL = 2
FLAG_NAMES = {1: "Auto", 2: "Manual", 3: "Auto|Manual"}

Snapshot = Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]
# {camera_name: {"camera_control": {prop: {...}}, "video_proc_amp": {prop: {...}}}}


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[camera-controls] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# COM plumbing (lazy: comtypes is only imported when a function is called)
# ---------------------------------------------------------------------------
_com: Dict[str, Any] = {}


def _ensure_com() -> bool:
    if _com:
        return bool(_com.get("ok"))
    _com["ok"] = False
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        from ctypes import HRESULT, POINTER, c_long
        from ctypes.wintypes import DWORD, LPCOLESTR, LPWSTR, ULONG

        import comtypes
        import comtypes.automation
        import comtypes.client
        from comtypes import COMMETHOD, GUID, IUnknown

        class IPersist(IUnknown):
            _iid_ = GUID("{0000010c-0000-0000-C000-000000000046}")
            _methods_ = [COMMETHOD([], HRESULT, "GetClassID", (["out"], POINTER(GUID), "pClassID"))]

        class IPersistStream(IPersist):
            _iid_ = GUID("{00000109-0000-0000-C000-000000000046}")
            _methods_ = [
                COMMETHOD([], HRESULT, "IsDirty"),
                COMMETHOD([], HRESULT, "Load", (["in"], ctypes.c_void_p, "pStm")),
                COMMETHOD([], HRESULT, "Save", (["in"], ctypes.c_void_p, "pStm"), (["in"], ctypes.c_int, "fClearDirty")),
                COMMETHOD([], HRESULT, "GetSizeMax", (["out"], POINTER(ctypes.c_ulonglong), "pcbSize")),
            ]

        class IMoniker(IPersistStream):
            _iid_ = GUID("{0000000f-0000-0000-C000-000000000046}")
            _methods_ = [
                COMMETHOD([], HRESULT, "BindToObject",
                          (["in"], ctypes.c_void_p, "pbc"), (["in"], ctypes.c_void_p, "pmkToLeft"),
                          (["in"], POINTER(GUID), "riidResult"), (["out"], POINTER(POINTER(IUnknown)), "ppvResult")),
                COMMETHOD([], HRESULT, "BindToStorage",
                          (["in"], ctypes.c_void_p, "pbc"), (["in"], ctypes.c_void_p, "pmkToLeft"),
                          (["in"], POINTER(GUID), "riid"), (["out"], POINTER(POINTER(IUnknown)), "ppvObj")),
                COMMETHOD([], HRESULT, "Reduce", (["in"], ctypes.c_void_p, "pbc"), (["in"], DWORD, "dwReduceHowFar"),
                          (["in"], ctypes.c_void_p, "ppmkToLeft"), (["out"], POINTER(ctypes.c_void_p), "ppmkReduced")),
                COMMETHOD([], HRESULT, "ComposeWith", (["in"], ctypes.c_void_p, "pmkRight"),
                          (["in"], ctypes.c_int, "fOnlyIfNotGeneric"), (["out"], POINTER(ctypes.c_void_p), "ppmkComposite")),
                COMMETHOD([], HRESULT, "Enum", (["in"], ctypes.c_int, "fForward"), (["out"], POINTER(ctypes.c_void_p), "ppenumMoniker")),
                COMMETHOD([], HRESULT, "IsEqual", (["in"], ctypes.c_void_p, "pmkOtherMoniker")),
                COMMETHOD([], HRESULT, "Hash", (["out"], POINTER(DWORD), "pdwHash")),
                COMMETHOD([], HRESULT, "IsRunning", (["in"], ctypes.c_void_p, "pbc"), (["in"], ctypes.c_void_p, "pmkToLeft"),
                          (["in"], ctypes.c_void_p, "pmkNewlyRunning")),
                COMMETHOD([], HRESULT, "GetTimeOfLastChange", (["in"], ctypes.c_void_p, "pbc"),
                          (["in"], ctypes.c_void_p, "pmkToLeft"), (["out"], POINTER(ctypes.c_ulonglong), "pFileTime")),
                COMMETHOD([], HRESULT, "Inverse", (["out"], POINTER(ctypes.c_void_p), "ppmk")),
                COMMETHOD([], HRESULT, "CommonPrefixWith", (["in"], ctypes.c_void_p, "pmkOther"),
                          (["out"], POINTER(ctypes.c_void_p), "ppmkPrefix")),
                COMMETHOD([], HRESULT, "RelativePathTo", (["in"], ctypes.c_void_p, "pmkOther"),
                          (["out"], POINTER(ctypes.c_void_p), "ppmkRelPath")),
                COMMETHOD([], HRESULT, "GetDisplayName", (["in"], ctypes.c_void_p, "pbc"),
                          (["in"], ctypes.c_void_p, "pmkToLeft"), (["out"], POINTER(LPWSTR), "ppszDisplayName")),
                COMMETHOD([], HRESULT, "ParseDisplayName", (["in"], ctypes.c_void_p, "pbc"),
                          (["in"], ctypes.c_void_p, "pmkToLeft"), (["in"], LPCOLESTR, "pszDisplayName"),
                          (["out"], POINTER(ULONG), "pchEaten"), (["out"], POINTER(ctypes.c_void_p), "ppmkOut")),
                COMMETHOD([], HRESULT, "IsSystemMoniker", (["out"], POINTER(DWORD), "pdwMksys")),
            ]

        class IEnumMoniker(IUnknown):
            _iid_ = GUID("{00000102-0000-0000-C000-000000000046}")
            _methods_ = [
                COMMETHOD([], HRESULT, "Next", (["in"], ULONG, "celt"), (["out"], POINTER(POINTER(IMoniker)), "rgelt"),
                          (["out"], POINTER(ULONG), "pceltFetched")),
                COMMETHOD([], HRESULT, "Skip", (["in"], ULONG, "celt")),
                COMMETHOD([], HRESULT, "Reset"),
                COMMETHOD([], HRESULT, "Clone", (["out"], POINTER(POINTER(IUnknown)), "ppenum")),
            ]

        class ICreateDevEnum(IUnknown):
            _iid_ = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
            _methods_ = [
                COMMETHOD([], HRESULT, "CreateClassEnumerator", (["in"], POINTER(GUID), "clsidDeviceClass"),
                          (["out"], POINTER(POINTER(IEnumMoniker)), "ppEnumMoniker"), (["in"], DWORD, "dwFlags")),
            ]

        class IPropertyBag(IUnknown):
            _iid_ = GUID("{55272A00-42CB-11CE-8135-00AA004BB851}")
            _methods_ = [
                COMMETHOD([], HRESULT, "Read", (["in"], LPCOLESTR, "pszPropName"),
                          (["in"], POINTER(comtypes.automation.VARIANT), "pVar"), (["in"], ctypes.c_void_p, "pErrorLog")),
                COMMETHOD([], HRESULT, "Write", (["in"], LPCOLESTR, "pszPropName"),
                          (["in"], POINTER(comtypes.automation.VARIANT), "pVar")),
            ]

        class IAMCameraControl(IUnknown):
            _iid_ = GUID("{C6E13370-30AC-11d0-A18C-00A0C9118956}")
            _methods_ = [
                COMMETHOD([], HRESULT, "GetRange", (["in"], c_long, "Property"), (["out"], POINTER(c_long), "pMin"),
                          (["out"], POINTER(c_long), "pMax"), (["out"], POINTER(c_long), "pSteppingDelta"),
                          (["out"], POINTER(c_long), "pDefault"), (["out"], POINTER(c_long), "pCapsFlags")),
                COMMETHOD([], HRESULT, "Set", (["in"], c_long, "Property"), (["in"], c_long, "lValue"),
                          (["in"], c_long, "Flags")),
                COMMETHOD([], HRESULT, "Get", (["in"], c_long, "Property"), (["out"], POINTER(c_long), "lValue"),
                          (["out"], POINTER(c_long), "Flags")),
            ]

        class IAMVideoProcAmp(IUnknown):
            _iid_ = GUID("{C6E13360-30AC-11d0-A18C-00A0C9118956}")
            _methods_ = list(IAMCameraControl._methods_)

        _com.update({
            "ctypes": ctypes, "byref": ctypes.byref, "comtypes": comtypes,
            "VARIANT": comtypes.automation.VARIANT, "client": comtypes.client,
            "CLSID_SystemDeviceEnum": GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}"),
            "CLSID_VideoInputDeviceCategory": GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}"),
            "IID_IBaseFilter": GUID("{56a86895-0ad4-11ce-b03a-0020af0ba770}"),
            "ICreateDevEnum": ICreateDevEnum, "IPropertyBag": IPropertyBag,
            "IAMCameraControl": IAMCameraControl, "IAMVideoProcAmp": IAMVideoProcAmp,
        })
        _com["ok"] = True
        return True
    except Exception as exc:
        _log(f"COM unavailable: {type(exc).__name__}: {exc}")
        return False


def _each_filter():
    """Yield (friendly_name, IUnknown filter) for every video input device."""
    c = _com
    devenum = c["client"].CreateObject(c["CLSID_SystemDeviceEnum"], interface=c["ICreateDevEnum"])
    enum = devenum.CreateClassEnumerator(c["CLSID_VideoInputDeviceCategory"], 0)
    if not enum:
        return
    while True:
        try:
            mon, fetched = enum.Next(1)
        except Exception:
            break
        if not fetched or not mon:
            break
        name = "?"
        try:
            unk = mon.BindToStorage(None, None, c["byref"](c["IPropertyBag"]._iid_))
            bag = unk.QueryInterface(c["IPropertyBag"])
            v = c["VARIANT"]()
            bag.Read("FriendlyName", c["byref"](v), None)
            name = str(v.value)
        except Exception:
            pass
        try:
            filt = mon.BindToObject(None, None, c["byref"](c["IID_IBaseFilter"]))
        except Exception:
            continue
        yield name, filt


def _read_props(iface, names: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(names):
        try:
            mn, mx, step, default, caps = iface.GetRange(i)
        except Exception:
            continue
        try:
            val, flags = iface.Get(i)
        except Exception:
            continue
        out[name] = {"index": i, "value": int(val), "flags": int(flags),
                     "min": int(mn), "max": int(mx), "default": int(default), "caps": int(caps)}
    return out


def _read_filter(filt) -> Dict[str, Dict[str, Dict[str, Any]]]:
    c = _com
    entry: Dict[str, Dict[str, Dict[str, Any]]] = {"camera_control": {}, "video_proc_amp": {}}
    try:
        entry["camera_control"] = _read_props(filt.QueryInterface(c["IAMCameraControl"]), CAMERA_PROPS)
    except Exception:
        pass
    try:
        entry["video_proc_amp"] = _read_props(filt.QueryInterface(c["IAMVideoProcAmp"]), PROCAMP_PROPS)
    except Exception:
        pass
    return entry


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def snapshot_all(only_name: Optional[str] = None) -> Snapshot:
    """Read every (or one, by friendly name) video device's controls.
    Pure reads. Returns {} if COM is unavailable or nothing matched."""
    if not _ensure_com():
        return {}
    snap: Snapshot = {}
    try:
        for name, filt in _each_filter():
            if only_name and name.strip().lower() != only_name.strip().lower():
                continue
            snap[name] = _read_filter(filt)
    except Exception as exc:
        _log(f"snapshot failed: {type(exc).__name__}: {exc}")
    return snap


def format_snapshot(snap: Snapshot) -> str:
    parts: List[str] = []
    for name, sections in snap.items():
        bits = []
        for sec in ("camera_control", "video_proc_amp"):
            for prop, v in sections.get(sec, {}).items():
                bits.append(f"{prop}={v['value']}/{FLAG_NAMES.get(v['flags'], v['flags'])}")
        parts.append(f"{name}: " + " ".join(bits))
    return "; ".join(parts) if parts else "(no cameras / COM unavailable)"


def diff(before: Snapshot, after: Snapshot) -> List[Dict[str, Any]]:
    """Changes attributable to the session. See module docstring rules."""
    changes: List[Dict[str, Any]] = []
    for name, sections in before.items():
        after_sections = after.get(name)
        if not after_sections:
            continue
        for sec in ("camera_control", "video_proc_amp"):
            for prop, b in sections.get(sec, {}).items():
                a = after_sections.get(sec, {}).get(prop)
                if a is None:
                    continue
                flag_changed = int(a["flags"]) != int(b["flags"])
                value_changed = int(a["value"]) != int(b["value"])
                if flag_changed or (value_changed and int(b["flags"]) == FLAG_MANUAL):
                    changes.append({
                        "camera": name, "section": sec, "property": prop, "index": b["index"],
                        "before_value": b["value"], "before_flags": b["flags"],
                        "after_value": a["value"], "after_flags": a["flags"],
                    })
    return changes


def format_changes(changes: List[Dict[str, Any]]) -> str:
    return ", ".join(
        f"{c['camera']}.{c['property']} {c['before_value']}/{FLAG_NAMES.get(c['before_flags'], c['before_flags'])}"
        f" -> {c['after_value']}/{FLAG_NAMES.get(c['after_flags'], c['after_flags'])}"
        for c in changes
    ) or "none"


def split_restorable(changes: List[Dict[str, Any]]):
    """Split session changes into ones that are safe to undo and ones
    that are not. Returns `(restorable, refused)`.

    The rule is directional, and it exists because "put the camera back
    exactly as we found it" is the wrong goal. A camera can be found in
    a bad state -- most often Exposure latched to a very short manual
    value by a previous crashed session, which makes the picture almost
    black. If the app un-latches that during the session and then
    faithfully restores the original on exit, the user gets a dark
    preview again on every single launch, forever.

    So: a restore may return a control to Auto, and may change a value
    while the flag stays the same, but it may never take a control that
    is currently on Auto and put it back on Manual. Undoing is only ever
    allowed to hand control back to the driver, never to take it away.
    """
    restorable: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []
    for ch in changes or []:
        try:
            now_auto = int(ch.get("after_flags", FLAG_MANUAL)) == FLAG_AUTO
            would_be_manual = int(ch.get("before_flags", FLAG_AUTO)) == FLAG_MANUAL
        except Exception:
            refused.append(ch)
            continue
        if now_auto and would_be_manual:
            refused.append(ch)
        else:
            restorable.append(ch)
    return restorable, refused


def restore(before: Snapshot, changes: List[Dict[str, Any]]) -> List[str]:
    """Write back value + flag for each listed change. Returns a list of
    human-readable results. Writes ONLY the properties in `changes`."""
    results: List[str] = []
    if not changes or not _ensure_com():
        return results
    c = _com
    wanted = {}
    for ch in changes:
        wanted.setdefault(ch["camera"], []).append(ch)
    try:
        for name, filt in _each_filter():
            todo = wanted.get(name)
            if not todo:
                continue
            ifaces = {}
            for ch in todo:
                sec = ch["section"]
                if sec not in ifaces:
                    try:
                        ifaces[sec] = filt.QueryInterface(
                            c["IAMCameraControl"] if sec == "camera_control" else c["IAMVideoProcAmp"]
                        )
                    except Exception as exc:
                        results.append(f"{name}.{ch['property']}: interface unavailable ({exc})")
                        ifaces[sec] = None
                iface = ifaces.get(sec)
                if iface is None:
                    continue
                try:
                    iface.Set(int(ch["index"]), int(ch["before_value"]), int(ch["before_flags"]))
                    time.sleep(0.05)
                    val, flags = iface.Get(int(ch["index"]))
                    ok = int(flags) == int(ch["before_flags"]) and (
                        int(ch["before_flags"]) == FLAG_AUTO or int(val) == int(ch["before_value"])
                    )
                    results.append(
                        f"{name}.{ch['property']}: restored to {ch['before_value']}/"
                        f"{FLAG_NAMES.get(ch['before_flags'], ch['before_flags'])} "
                        f"readback={val}/{FLAG_NAMES.get(int(flags), flags)} {'OK' if ok else 'NOT VERIFIED'}"
                    )
                except Exception as exc:
                    results.append(f"{name}.{ch['property']}: restore failed ({type(exc).__name__}: {exc})")
    except Exception as exc:
        results.append(f"restore aborted: {type(exc).__name__}: {exc}")
    return results

# Author: Konstantin Markov
