"""Windows taskbar Jump List for Touchless.

Adds a 'Tasks' section to the right-click menu on the Touchless
taskbar icon:
  * Pause Gestures (30 min)
  * Settings
  * Quit Touchless

Each task launches Touchless.exe with a flag. The single-instance
lock detects the flag, posts a Win32 message to the already-running
window, and exits silently. The running process's native event
filter receives the message and dispatches the action (same handler
the tray menu uses).

Built on inline comtypes IUnknown subclasses so we don't depend on
comtypes' auto-generated type-library modules (which fail to import
on some machines).

Only installed when running as the frozen PyInstaller build --
source runs use python.exe whose path makes a poor IShellLink
target (the shortcut would launch python.exe with no module).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple


ACTION_PAUSE_30 = "--touchless-pause-30"
ACTION_SETTINGS = "--touchless-settings"
ACTION_QUIT = "--touchless-quit"


def install_jumplist(*, app_user_model_id: str, exe_path: Path) -> bool:
    """Build / replace the Jump List for the current user. Safe to
    call on every launch -- CommitList atomically replaces. Returns
    True if at least one task was registered."""
    if sys.platform != "win32":
        return False
    try:
        return _install(app_user_model_id=app_user_model_id, exe_path=exe_path)
    except Exception as exc:
        try:
            import sys as _sys
            _sys.stderr.write(f"[jumplist] install failed: {exc}\n")
        except Exception:
            pass
        return False


def _install(*, app_user_model_id: str, exe_path: Path) -> bool:
    from ctypes import POINTER, byref, c_int, c_uint, c_wchar_p, c_void_p
    from comtypes import (
        GUID, IUnknown, COMMETHOD, HRESULT, CoCreateInstance, CLSCTX_INPROC_SERVER,
    )

    # -- IObjectArray -----------------------------------------------
    IID_IObjectArray = GUID("{92CA9DCD-5622-4BBA-A805-5E9F541BD8C9}")

    class IObjectArray(IUnknown):
        _iid_ = IID_IObjectArray
        _methods_ = [
            COMMETHOD([], HRESULT, "GetCount", (["out"], POINTER(c_uint), "pcObjects")),
            COMMETHOD([], HRESULT, "GetAt",
                      (["in"], c_uint, "uiIndex"),
                      (["in"], POINTER(GUID), "riid"),
                      (["out"], POINTER(c_void_p), "ppv")),
        ]

    # -- IObjectCollection (extends IObjectArray) -------------------
    IID_IObjectCollection = GUID("{5632B1A4-E38A-400A-928A-D4CD63230295}")
    CLSID_EnumerableObjectCollection = GUID("{2D3468C1-36A7-43B6-AC24-D3F02FD9607A}")

    class IObjectCollection(IObjectArray):
        _iid_ = IID_IObjectCollection
        _methods_ = [
            COMMETHOD([], HRESULT, "AddObject", (["in"], POINTER(IUnknown), "punk")),
            COMMETHOD([], HRESULT, "AddFromArray", (["in"], POINTER(IObjectArray), "psoaSource")),
            COMMETHOD([], HRESULT, "RemoveObjectAt", (["in"], c_uint, "uiIndex")),
            COMMETHOD([], HRESULT, "Clear"),
        ]

    # -- ICustomDestinationList -------------------------------------
    IID_ICustomDestinationList = GUID("{6332DEBF-87B5-4670-90C0-5E57B408A49E}")
    CLSID_DestinationList = GUID("{77F10CF0-3DB5-4966-B520-B7C54FD35ED6}")

    class ICustomDestinationList(IUnknown):
        _iid_ = IID_ICustomDestinationList
        _methods_ = [
            COMMETHOD([], HRESULT, "SetAppID", (["in"], c_wchar_p, "pszAppID")),
            COMMETHOD([], HRESULT, "BeginList",
                      (["out"], POINTER(c_uint), "pcMaxSlots"),
                      (["in"], POINTER(GUID), "riid"),
                      (["out"], POINTER(c_void_p), "ppv")),
            COMMETHOD([], HRESULT, "AppendCategory",
                      (["in"], c_wchar_p, "pszCategory"),
                      (["in"], POINTER(IObjectArray), "poa")),
            COMMETHOD([], HRESULT, "AppendKnownCategory", (["in"], c_int, "category")),
            COMMETHOD([], HRESULT, "AddUserTasks", (["in"], POINTER(IObjectArray), "poa")),
            COMMETHOD([], HRESULT, "CommitList"),
            COMMETHOD([], HRESULT, "GetRemovedDestinations",
                      (["in"], POINTER(GUID), "riid"),
                      (["out"], POINTER(c_void_p), "ppv")),
            COMMETHOD([], HRESULT, "DeleteList", (["in"], c_wchar_p, "pszAppID")),
            COMMETHOD([], HRESULT, "AbortList"),
        ]

    # -- IShellLinkW ------------------------------------------------
    IID_IShellLinkW = GUID("{000214F9-0000-0000-C000-000000000046}")
    CLSID_ShellLink = GUID("{00021401-0000-0000-C000-000000000046}")

    class IShellLinkW(IUnknown):
        _iid_ = IID_IShellLinkW
        _methods_ = [
            COMMETHOD([], HRESULT, "GetPath",
                      (["out"], c_wchar_p, "pszFile"),
                      (["in"], c_int, "cch"),
                      (["in"], c_void_p, "pfd"),
                      (["in"], c_uint, "fFlags")),
            COMMETHOD([], HRESULT, "GetIDList", (["out"], POINTER(c_void_p), "ppidl")),
            COMMETHOD([], HRESULT, "SetIDList", (["in"], c_void_p, "pidl")),
            COMMETHOD([], HRESULT, "GetDescription",
                      (["out"], c_wchar_p, "pszName"),
                      (["in"], c_int, "cch")),
            COMMETHOD([], HRESULT, "SetDescription", (["in"], c_wchar_p, "pszName")),
            COMMETHOD([], HRESULT, "GetWorkingDirectory",
                      (["out"], c_wchar_p, "pszDir"),
                      (["in"], c_int, "cch")),
            COMMETHOD([], HRESULT, "SetWorkingDirectory", (["in"], c_wchar_p, "pszDir")),
            COMMETHOD([], HRESULT, "GetArguments",
                      (["out"], c_wchar_p, "pszArgs"),
                      (["in"], c_int, "cch")),
            COMMETHOD([], HRESULT, "SetArguments", (["in"], c_wchar_p, "pszArgs")),
            COMMETHOD([], HRESULT, "GetHotkey", (["out"], POINTER(c_uint), "pwHotkey")),
            COMMETHOD([], HRESULT, "SetHotkey", (["in"], c_uint, "wHotkey")),
            COMMETHOD([], HRESULT, "GetShowCmd", (["out"], POINTER(c_int), "piShowCmd")),
            COMMETHOD([], HRESULT, "SetShowCmd", (["in"], c_int, "iShowCmd")),
            COMMETHOD([], HRESULT, "GetIconLocation",
                      (["out"], c_wchar_p, "pszIconPath"),
                      (["in"], c_int, "cch"),
                      (["out"], POINTER(c_int), "piIcon")),
            COMMETHOD([], HRESULT, "SetIconLocation",
                      (["in"], c_wchar_p, "pszIconPath"),
                      (["in"], c_int, "iIcon")),
            COMMETHOD([], HRESULT, "SetRelativePath",
                      (["in"], c_wchar_p, "pszPathRel"),
                      (["in"], c_uint, "dwReserved")),
            COMMETHOD([], HRESULT, "Resolve",
                      (["in"], c_void_p, "hwnd"),
                      (["in"], c_uint, "fFlags")),
            COMMETHOD([], HRESULT, "SetPath", (["in"], c_wchar_p, "pszFile")),
        ]

    # Construct destination list and seat the AUMID so Windows knows
    # which taskbar entry this list belongs to.
    destination_list = CoCreateInstance(
        CLSID_DestinationList,
        ICustomDestinationList,
        clsctx=CLSCTX_INPROC_SERVER,
    )
    destination_list.SetAppID(app_user_model_id)

    # BeginList returns slot count + IObjectArray of removed items
    # (items the user told Windows to forget). We don't repopulate
    # removed ones; skip introspecting the array.
    slot_count = c_uint(0)
    removed_ptr = c_void_p()
    destination_list.BeginList(byref(slot_count), byref(IID_IObjectArray), byref(removed_ptr))

    # Build a collection and add one shell-link per task.
    collection = CoCreateInstance(
        CLSID_EnumerableObjectCollection,
        IObjectCollection,
        clsctx=CLSCTX_INPROC_SERVER,
    )

    tasks: Tuple[Tuple[str, str], ...] = (
        ("Pause Gestures (30 min)", ACTION_PAUSE_30),
        ("Settings",                ACTION_SETTINGS),
        ("Quit Touchless",          ACTION_QUIT),
    )
    added = 0
    for title, argument in tasks:
        link = CoCreateInstance(
            CLSID_ShellLink,
            IShellLinkW,
            clsctx=CLSCTX_INPROC_SERVER,
        )
        link.SetPath(str(exe_path))
        link.SetArguments(argument)
        link.SetIconLocation(str(exe_path), 0)
        link.SetDescription(title)
        # Windows requires the task title via IPropertyStore PKEY_Title
        # otherwise the menu shows "(Untitled)". Set it via the
        # IPropertyStore interface on the shell link.
        try:
            _set_link_title(link, title)
        except Exception:
            pass
        collection.AddObject(link)
        added += 1

    if added == 0:
        destination_list.AbortList()
        return False

    destination_list.AppendCategory("Tasks", collection)
    destination_list.CommitList()
    return True


def _set_link_title(link, title: str) -> None:
    """Set PKEY_Title on a shell link's IPropertyStore so the Jump
    List displays our human-readable title text."""
    from ctypes import POINTER, byref, c_void_p, c_uint, c_int, Structure, Union
    from ctypes import wintypes
    from comtypes import GUID, IUnknown, COMMETHOD, HRESULT

    IID_IPropertyStore = GUID("{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}")

    class PROPERTYKEY(Structure):
        _fields_ = [("fmtid", GUID), ("pid", c_uint)]

    # PROPVARIANT is large and variant-typed. For VT_LPWSTR (31) we
    # only need vt + pwszVal pointer; everything else is left zero.
    class _PROPVARIANT_DATA(Union):
        _fields_ = [("pwszVal", c_void_p), ("pad", (c_uint * 4))]

    class PROPVARIANT(Structure):
        _fields_ = [
            ("vt", c_uint),
            ("wReserved1", c_uint),
            ("wReserved2", c_uint),
            ("wReserved3", c_uint),
            ("u", _PROPVARIANT_DATA),
            ("pad2", c_uint * 2),
        ]

    class IPropertyStore(IUnknown):
        _iid_ = IID_IPropertyStore
        _methods_ = [
            COMMETHOD([], HRESULT, "GetCount", (["out"], POINTER(c_uint), "cProps")),
            COMMETHOD([], HRESULT, "GetAt",
                      (["in"], c_uint, "iProp"),
                      (["out"], POINTER(PROPERTYKEY), "pkey")),
            COMMETHOD([], HRESULT, "GetValue",
                      (["in"], POINTER(PROPERTYKEY), "key"),
                      (["out"], POINTER(PROPVARIANT), "pv")),
            COMMETHOD([], HRESULT, "SetValue",
                      (["in"], POINTER(PROPERTYKEY), "key"),
                      (["in"], POINTER(PROPVARIANT), "pv")),
            COMMETHOD([], HRESULT, "Commit"),
        ]

    prop_store = link.QueryInterface(IPropertyStore)
    pkey_title = PROPERTYKEY(
        fmtid=GUID("{F29F85E0-4FF9-1068-AB91-08002B27B3D9}"),
        pid=2,
    )
    pv = PROPVARIANT()
    pv.vt = 31  # VT_LPWSTR
    # Allocate the string via CoTaskMemAlloc so PropVariantClear
    # is safe (Windows owns the buffer once we hand it over).
    import ctypes
    buf_len = (len(title) + 1) * 2
    buf = ctypes.windll.ole32.CoTaskMemAlloc(buf_len)
    if not buf:
        return
    ctypes.memmove(buf, ctypes.create_unicode_buffer(title), buf_len)
    pv.u.pwszVal = buf
    try:
        prop_store.SetValue(byref(pkey_title), byref(pv))
        prop_store.Commit()
    finally:
        # Windows frees the buffer when it processes the PROPVARIANT
        # via PropVariantClear, which Commit triggers internally; we
        # don't free explicitly.
        pass
