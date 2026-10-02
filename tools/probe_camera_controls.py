"""READ-ONLY dump of a DirectShow camera's control state (v1.1.9.2 r18).

Reads IAMCameraControl (Exposure, Focus, Zoom, Pan, Tilt, Roll, Iris) and
IAMVideoProcAmp (Brightness, Contrast, Hue, Saturation, Sharpness, Gamma,
ColorEnable, WhiteBalance, BacklightCompensation, Gain) straight from the
driver via COM, including the Auto / Manual FLAG that OpenCV's
cap.get(CAP_PROP_AUTO_EXPOSURE) cannot return on DirectShow (-1.0).

    python tools/probe_camera_controls.py            # print all cameras
    python tools/probe_camera_controls.py --json out.json

Nothing is written to the camera. Use it before and after a Touchless
session (or a Synapse change) and diff the two JSON files: that is the
ground truth for "did the app change my camera".
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from ctypes import HRESULT, POINTER, byref, c_long
from ctypes.wintypes import DWORD, LPCOLESTR, LPWSTR, ULONG

import comtypes
import comtypes.automation
import comtypes.client  # noqa: F401  (initialises COM)
from comtypes import COMMETHOD, GUID, IUnknown

# --- DirectShow GUIDs -------------------------------------------------------
CLSID_SystemDeviceEnum = GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
CLSID_VideoInputDeviceCategory = GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")
IID_IBaseFilter = GUID("{56a86895-0ad4-11ce-b03a-0020af0ba770}")

# --- minimal COM interface definitions (vtable order matters) ---------------


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
        COMMETHOD([], HRESULT, "ComposeWith", (["in"], ctypes.c_void_p, "pmkRight"), (["in"], ctypes.c_int, "fOnlyIfNotGeneric"),
                  (["out"], POINTER(ctypes.c_void_p), "ppmkComposite")),
        COMMETHOD([], HRESULT, "Enum", (["in"], ctypes.c_int, "fForward"), (["out"], POINTER(ctypes.c_void_p), "ppenumMoniker")),
        COMMETHOD([], HRESULT, "IsEqual", (["in"], ctypes.c_void_p, "pmkOtherMoniker")),
        COMMETHOD([], HRESULT, "Hash", (["out"], POINTER(DWORD), "pdwHash")),
        COMMETHOD([], HRESULT, "IsRunning", (["in"], ctypes.c_void_p, "pbc"), (["in"], ctypes.c_void_p, "pmkToLeft"),
                  (["in"], ctypes.c_void_p, "pmkNewlyRunning")),
        COMMETHOD([], HRESULT, "GetTimeOfLastChange", (["in"], ctypes.c_void_p, "pbc"), (["in"], ctypes.c_void_p, "pmkToLeft"),
                  (["out"], POINTER(ctypes.c_ulonglong), "pFileTime")),
        COMMETHOD([], HRESULT, "Inverse", (["out"], POINTER(ctypes.c_void_p), "ppmk")),
        COMMETHOD([], HRESULT, "CommonPrefixWith", (["in"], ctypes.c_void_p, "pmkOther"), (["out"], POINTER(ctypes.c_void_p), "ppmkPrefix")),
        COMMETHOD([], HRESULT, "RelativePathTo", (["in"], ctypes.c_void_p, "pmkOther"), (["out"], POINTER(ctypes.c_void_p), "ppmkRelPath")),
        COMMETHOD([], HRESULT, "GetDisplayName", (["in"], ctypes.c_void_p, "pbc"), (["in"], ctypes.c_void_p, "pmkToLeft"),
                  (["out"], POINTER(LPWSTR), "ppszDisplayName")),
        COMMETHOD([], HRESULT, "ParseDisplayName", (["in"], ctypes.c_void_p, "pbc"), (["in"], ctypes.c_void_p, "pmkToLeft"),
                  (["in"], LPCOLESTR, "pszDisplayName"), (["out"], POINTER(ULONG), "pchEaten"),
                  (["out"], POINTER(ctypes.c_void_p), "ppmkOut")),
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
        COMMETHOD([], HRESULT, "Write", (["in"], LPCOLESTR, "pszPropName"), (["in"], POINTER(comtypes.automation.VARIANT), "pVar")),
    ]


class IAMCameraControl(IUnknown):
    _iid_ = GUID("{C6E13370-30AC-11d0-A18C-00A0C9118956}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetRange", (["in"], c_long, "Property"), (["out"], POINTER(c_long), "pMin"),
                  (["out"], POINTER(c_long), "pMax"), (["out"], POINTER(c_long), "pSteppingDelta"),
                  (["out"], POINTER(c_long), "pDefault"), (["out"], POINTER(c_long), "pCapsFlags")),
        COMMETHOD([], HRESULT, "Set", (["in"], c_long, "Property"), (["in"], c_long, "lValue"), (["in"], c_long, "Flags")),
        COMMETHOD([], HRESULT, "Get", (["in"], c_long, "Property"), (["out"], POINTER(c_long), "lValue"),
                  (["out"], POINTER(c_long), "Flags")),
    ]


class IAMVideoProcAmp(IUnknown):
    _iid_ = GUID("{C6E13360-30AC-11d0-A18C-00A0C9118956}")
    _methods_ = list(IAMCameraControl._methods_)


CAMERA_PROPS = ["Pan", "Tilt", "Roll", "Zoom", "Exposure", "Iris", "Focus"]
PROCAMP_PROPS = ["Brightness", "Contrast", "Hue", "Saturation", "Sharpness", "Gamma",
                 "ColorEnable", "WhiteBalance", "BacklightCompensation", "Gain"]
FLAG_NAMES = {1: "Auto", 2: "Manual", 3: "Auto|Manual"}


def _read_props(iface, names):
    out = {}
    for i, name in enumerate(names):
        try:
            mn, mx, step, default, caps = iface.GetRange(i)
        except Exception:
            continue  # unsupported by this device
        try:
            val, flags = iface.Get(i)
        except Exception:
            val, flags = None, None
        out[name] = {
            "value": val, "flags": flags, "flags_name": FLAG_NAMES.get(flags, str(flags)),
            "min": mn, "max": mx, "step": step, "default": default, "caps": caps,
        }
    return out


def dump_all():
    devenum = comtypes.client.CreateObject(CLSID_SystemDeviceEnum, interface=ICreateDevEnum)
    enum = devenum.CreateClassEnumerator(CLSID_VideoInputDeviceCategory, 0)
    cams = []
    if not enum:
        return cams
    while True:
        try:
            mon, fetched = enum.Next(1)
        except Exception:
            break
        if not fetched or not mon:
            break
        name = "?"
        try:
            unk = mon.BindToStorage(None, None, byref(IPropertyBag._iid_))
            bag = unk.QueryInterface(IPropertyBag)
            v = comtypes.automation.VARIANT()
            bag.Read("FriendlyName", byref(v), None)
            name = str(v.value)
        except Exception as exc:
            name = f"? ({exc})"
        entry = {"name": name, "camera_control": {}, "video_proc_amp": {}}
        try:
            filt = mon.BindToObject(None, None, byref(IID_IBaseFilter))
            try:
                cc = filt.QueryInterface(IAMCameraControl)
                entry["camera_control"] = _read_props(cc, CAMERA_PROPS)
            except Exception as exc:
                entry["camera_control_error"] = str(exc)
            try:
                pa = filt.QueryInterface(IAMVideoProcAmp)
                entry["video_proc_amp"] = _read_props(pa, PROCAMP_PROPS)
            except Exception as exc:
                entry["video_proc_amp_error"] = str(exc)
        except Exception as exc:
            entry["bind_error"] = str(exc)
        cams.append(entry)
    return cams


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="also write the snapshot to this file")
    args = ap.parse_args()
    cams = dump_all()
    for c in cams:
        print(f"=== {c['name']} ===")
        for section in ("camera_control", "video_proc_amp"):
            for k, v in c.get(section, {}).items():
                print(f"  {section:16s} {k:22s} value={v['value']!s:>6} {v['flags_name']:12s} "
                      f"range=[{v['min']}..{v['max']}] default={v['default']} caps={v['caps']}")
        for k in ("camera_control_error", "video_proc_amp_error", "bind_error"):
            if k in c:
                print(f"  {k}: {c[k]}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(cams, fh, indent=2)
        print(f"snapshot written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Author: Konstantin Markov
