"""Pre-apply verification for downloaded update artifacts.

Two independent checks gate every apply step:

  1. SHA-256 digest match (defends against CDN corruption, MITM, and
     a quietly-swapped asset when combined with HTTPS). The expected
     digest is published in the GitHub release body as a marker
     comment (parsed by release_checker).

  2. Authenticode signature (defends against the worst case — a
     compromised GitHub PAT or release-edit permission lets an
     attacker publish a malicious URL + matching SHA-256, but they
     can't sign with the legitimate code-signing cert. We verify the
     downloaded artifact carries a valid Authenticode signature
     issued to the publisher subject we expect).

For the app-zip path, signature verification runs against
`<staged>\\Touchless.exe` after Expand-Archive (inside the apply bat)
so the file we eventually copy into place was signed. For the
full-installer path, signature runs against the .exe before
ShellExecute.

A missing SHA-256 marker is tolerated (logged warning) so legacy
releases pre-dating the marker convention can still be applied — but
signature verification is NEVER skipped: any release that fails to
pass Authenticode is refused.

Authors should publish per-release SHA-256 markers as a matter of
release hygiene. The build script can emit them automatically.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple


_log = logging.getLogger(__name__)


# Expected publisher subject substring for Authenticode verification.
# Matched case-insensitively against the cert subject's Common Name.
# Touchless's Azure Trusted Signing Individual cert is issued to
# "Konstantin Markov" — that string MUST appear in the subject for the
# binary to be considered ours.
EXPECTED_PUBLISHER_SUBSTR = "Konstantin Markov"


# Chunk size for streaming hash; 1 MB hits a sweet spot between
# Python-loop overhead and memory.
_HASH_CHUNK = 1 << 20


def compute_sha256(path: str | os.PathLike) -> str:
    """Stream the file at `path` through SHA-256, return lowercase
    hex digest. Raises OSError on read failure."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: str | os.PathLike, expected_hex: str) -> Tuple[bool, str]:
    """Return (ok, message). `expected_hex` may be empty — in that
    case we return (True, 'no published hash; verification skipped')
    and the caller is responsible for logging that this was a legacy
    release without a marker.

    A non-empty expected that doesn't match the actual digest is a
    hard reject — the caller must refuse to apply.
    """
    if not expected_hex:
        return (True, "no published hash; verification skipped (legacy release?)")
    try:
        actual = compute_sha256(path)
    except OSError as exc:
        return (False, f"couldn't hash downloaded file: {exc}")
    if actual.lower() != expected_hex.lower():
        return (
            False,
            f"SHA-256 mismatch — refusing to apply. "
            f"Expected {expected_hex.lower()}, got {actual}.",
        )
    return (True, "SHA-256 verified")


# ---- Authenticode verification ---------------------------------------------


def _is_windows() -> bool:
    return sys.platform == "win32"


def verify_authenticode(path: str | os.PathLike) -> Tuple[bool, str]:
    """Verify the file at `path` has a valid Authenticode signature
    AND the cert subject contains EXPECTED_PUBLISHER_SUBSTR.

    Returns (ok, message). The message is suitable for surfacing in a
    user-facing error dialog when ok=False.

    On non-Windows or when verification can't be performed at all
    (Authenticode is a Windows construct), returns (True, '...') so
    the caller doesn't block on platforms where the check would never
    apply. This is safe because the Updater itself only runs on
    Windows (frozen=True PyInstaller bundle); a source-run on Linux
    couldn't apply an updater anyway.
    """
    p = Path(path)
    if not p.exists():
        return (False, f"file not found for signature check: {p}")
    if not _is_windows():
        return (True, "Authenticode check skipped (non-Windows)")

    # ---- Step 1: WinVerifyTrust — is the signature valid + trusted? ----
    ok, msg = _winverifytrust(p)
    if not ok:
        return (False, msg)

    # ---- Step 2: Extract cert subject, check publisher substring ----
    subject = _cert_subject(p)
    if not subject:
        return (False, "couldn't extract signing cert subject")
    if EXPECTED_PUBLISHER_SUBSTR.lower() not in subject.lower():
        return (
            False,
            f"signed by an unexpected publisher: {subject!r}. "
            f"Expected the subject to contain {EXPECTED_PUBLISHER_SUBSTR!r}.",
        )
    return (True, f"Authenticode verified ({subject})")


def _winverifytrust(path: Path) -> Tuple[bool, str]:
    """Call WinVerifyTrust with WTD_UI_NONE to check signature validity.
    Returns (ok, message). Failure modes include: not signed, signed
    by an untrusted root, revoked cert, expired cert, signature mismatch."""
    try:
        import ctypes
        from ctypes import wintypes
    except Exception as exc:
        return (False, f"ctypes unavailable: {exc}")

    # Constants from wintrust.h
    WTD_UI_NONE = 2
    WTD_REVOKE_NONE = 0
    WTD_CHOICE_FILE = 1
    WTD_STATEACTION_VERIFY = 1
    WTD_STATEACTION_CLOSE = 2

    # WINTRUST_ACTION_GENERIC_VERIFY_V2 GUID
    # {00AAC56B-CD44-11d0-8CC2-00C04FC295EE}
    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    GENERIC_VERIFY_V2 = GUID(
        0x00AAC56B, 0xCD44, 0x11D0,
        (ctypes.c_ubyte * 8)(0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE),
    )

    class WINTRUST_FILE_INFO(ctypes.Structure):
        _fields_ = [
            ("cbStruct", ctypes.c_ulong),
            ("pcwszFilePath", wintypes.LPCWSTR),
            ("hFile", wintypes.HANDLE),
            ("pgKnownSubject", ctypes.POINTER(GUID)),
        ]

    class WINTRUST_DATA(ctypes.Structure):
        _fields_ = [
            ("cbStruct", ctypes.c_ulong),
            ("pPolicyCallbackData", ctypes.c_void_p),
            ("pSIPClientData", ctypes.c_void_p),
            ("dwUIChoice", ctypes.c_ulong),
            ("fdwRevocationChecks", ctypes.c_ulong),
            ("dwUnionChoice", ctypes.c_ulong),
            ("pFile", ctypes.POINTER(WINTRUST_FILE_INFO)),
            ("dwStateAction", ctypes.c_ulong),
            ("hWVTStateData", wintypes.HANDLE),
            ("pwszURLReference", wintypes.LPCWSTR),
            ("dwProvFlags", ctypes.c_ulong),
            ("dwUIContext", ctypes.c_ulong),
            ("pSignatureSettings", ctypes.c_void_p),
        ]

    file_info = WINTRUST_FILE_INFO(
        cbStruct=ctypes.sizeof(WINTRUST_FILE_INFO),
        pcwszFilePath=str(path),
        hFile=None,
        pgKnownSubject=None,
    )
    data = WINTRUST_DATA(
        cbStruct=ctypes.sizeof(WINTRUST_DATA),
        pPolicyCallbackData=None,
        pSIPClientData=None,
        dwUIChoice=WTD_UI_NONE,
        fdwRevocationChecks=WTD_REVOKE_NONE,
        dwUnionChoice=WTD_CHOICE_FILE,
        pFile=ctypes.pointer(file_info),
        dwStateAction=WTD_STATEACTION_VERIFY,
        hWVTStateData=None,
        pwszURLReference=None,
        dwProvFlags=0,
        dwUIContext=0,
        pSignatureSettings=None,
    )

    try:
        wintrust = ctypes.WinDLL("wintrust.dll")
        WinVerifyTrust = wintrust.WinVerifyTrust
        WinVerifyTrust.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(GUID),
            ctypes.c_void_p,
        ]
        WinVerifyTrust.restype = ctypes.c_long
        result = WinVerifyTrust(
            wintypes.HANDLE(None),
            ctypes.byref(GENERIC_VERIFY_V2),
            ctypes.byref(data),
        )
        # Always close the state, even on failure, to free leaked handles.
        data.dwStateAction = WTD_STATEACTION_CLOSE
        try:
            WinVerifyTrust(
                wintypes.HANDLE(None),
                ctypes.byref(GENERIC_VERIFY_V2),
                ctypes.byref(data),
            )
        except Exception:
            pass
    except Exception as exc:
        return (False, f"WinVerifyTrust raised: {exc}")

    if result == 0:
        return (True, "WinVerifyTrust: signature valid")
    # Map common error codes to human messages. Result is a HRESULT-style
    # LONG; common codes are in winerror.h.
    common = {
        0x800B0100: "TRUST_E_NOSIGNATURE — file is not signed",
        0x800B0109: "TRUST_E_CERT_SIGNATURE — cert signature failed verification",
        0x80092010: "CRYPT_E_REVOKED — signing cert is revoked",
        0x80096010: "TRUST_E_BAD_DIGEST — file's contents have been tampered with",
        0x800B010A: "CERT_E_CHAINING — couldn't build cert chain to a trusted root",
        0x800B0101: "CERT_E_EXPIRED — signing cert has expired (uncommon for code signing — usually still valid via counter-timestamp)",
    }
    # Treat result as unsigned for the lookup.
    code_u = result & 0xFFFFFFFF
    detail = common.get(code_u, f"WinVerifyTrust returned 0x{code_u:08X}")
    return (False, f"signature check failed: {detail}")


def _cert_subject(path: Path) -> Optional[str]:
    """Return the Subject CN of the file's signing certificate, or
    None on failure. Uses CryptQueryObject + CertGetNameStringW which
    are available on Windows 10+ without extra DLLs."""
    if not _is_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    # Constants
    CERT_QUERY_OBJECT_FILE = 0x00000001
    CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED = 1 << 7
    CERT_QUERY_FORMAT_FLAG_BINARY = 1 << 1
    CERT_NAME_SIMPLE_DISPLAY_TYPE = 4

    crypt32 = ctypes.WinDLL("crypt32.dll")
    CryptQueryObject = crypt32.CryptQueryObject
    CryptQueryObject.argtypes = [
        ctypes.c_ulong,        # dwObjectType
        ctypes.c_void_p,       # pvObject
        ctypes.c_ulong,        # dwExpectedContentTypeFlags
        ctypes.c_ulong,        # dwExpectedFormatTypeFlags
        ctypes.c_ulong,        # dwFlags (reserved)
        ctypes.POINTER(ctypes.c_ulong),  # pdwMsgAndCertEncodingType
        ctypes.POINTER(ctypes.c_ulong),  # pdwContentType
        ctypes.POINTER(ctypes.c_ulong),  # pdwFormatType
        ctypes.POINTER(wintypes.HANDLE), # phCertStore
        ctypes.POINTER(wintypes.HANDLE), # phMsg
        ctypes.POINTER(ctypes.c_void_p), # ppvContext
    ]
    CryptQueryObject.restype = wintypes.BOOL

    CryptMsgGetParam = crypt32.CryptMsgGetParam
    CryptMsgGetParam.argtypes = [
        wintypes.HANDLE, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong),
    ]
    CryptMsgGetParam.restype = wintypes.BOOL

    CertFindCertificateInStore = crypt32.CertFindCertificateInStore
    CertFindCertificateInStore.argtypes = [
        wintypes.HANDLE, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    CertFindCertificateInStore.restype = ctypes.c_void_p

    CertGetNameStringW = crypt32.CertGetNameStringW
    CertGetNameStringW.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.c_void_p, wintypes.LPWSTR, ctypes.c_ulong,
    ]
    CertGetNameStringW.restype = ctypes.c_ulong

    CertCloseStore = crypt32.CertCloseStore
    CertCloseStore.argtypes = [wintypes.HANDLE, ctypes.c_ulong]
    CertCloseStore.restype = wintypes.BOOL

    CryptMsgClose = crypt32.CryptMsgClose
    CryptMsgClose.argtypes = [wintypes.HANDLE]
    CryptMsgClose.restype = wintypes.BOOL

    CertFreeCertificateContext = crypt32.CertFreeCertificateContext
    CertFreeCertificateContext.argtypes = [ctypes.c_void_p]
    CertFreeCertificateContext.restype = wintypes.BOOL

    # CMSG_SIGNER_INFO_PARAM == 6
    CMSG_SIGNER_INFO_PARAM = 6
    # CERT_FIND_SUBJECT_CERT == 0x000B0000
    CERT_FIND_SUBJECT_CERT = 0x000B0000
    X509_ASN_ENCODING = 0x00000001
    PKCS_7_ASN_ENCODING = 0x00010000

    cert_store = wintypes.HANDLE()
    msg = wintypes.HANDLE()
    try:
        ok = CryptQueryObject(
            CERT_QUERY_OBJECT_FILE,
            ctypes.c_wchar_p(str(path)),
            CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED,
            CERT_QUERY_FORMAT_FLAG_BINARY,
            0,
            None, None, None,
            ctypes.byref(cert_store),
            ctypes.byref(msg),
            None,
        )
        if not ok:
            return None

        # Get the SIGNER_INFO so we can find the signer's cert.
        signer_size = ctypes.c_ulong(0)
        ok = CryptMsgGetParam(msg, CMSG_SIGNER_INFO_PARAM, 0, None, ctypes.byref(signer_size))
        if not ok or signer_size.value == 0:
            return None
        signer_buf = (ctypes.c_ubyte * signer_size.value)()
        ok = CryptMsgGetParam(
            msg, CMSG_SIGNER_INFO_PARAM, 0,
            ctypes.byref(signer_buf), ctypes.byref(signer_size),
        )
        if not ok:
            return None

        # CMSG_SIGNER_INFO layout: dwVersion (DWORD), Issuer (CERT_NAME_BLOB),
        # SerialNumber (CRYPT_INTEGER_BLOB), ...
        # Use CertFindCertificateInStore with the (Issuer, Serial) pair.
        class CERT_NAME_BLOB(ctypes.Structure):
            _fields_ = [("cbData", ctypes.c_ulong),
                        ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

        class CRYPT_INTEGER_BLOB(ctypes.Structure):
            _fields_ = [("cbData", ctypes.c_ulong),
                        ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

        class CERT_INFO_ID(ctypes.Structure):
            _fields_ = [("Issuer", CERT_NAME_BLOB),
                        ("SerialNumber", CRYPT_INTEGER_BLOB)]

        # Skip dwVersion (4 bytes), get Issuer + SerialNumber.
        signer_ptr = ctypes.cast(signer_buf, ctypes.c_void_p).value
        if signer_ptr is None:
            return None
        signer_info_id_ptr = signer_ptr + 4  # past dwVersion
        cert_info_id = ctypes.cast(
            signer_info_id_ptr,
            ctypes.POINTER(CERT_INFO_ID),
        ).contents

        # CertFindCertificateInStore wants a CERT_INFO with Issuer+Serial.
        class CERT_INFO(ctypes.Structure):
            _fields_ = [
                ("dwVersion", ctypes.c_ulong),
                ("SerialNumber", CRYPT_INTEGER_BLOB),
                ("SignatureAlgorithm", ctypes.c_void_p * 4),  # placeholder
                ("Issuer", CERT_NAME_BLOB),
                # (rest unused for the find call)
            ]

        find_info = CERT_INFO()
        find_info.dwVersion = 0
        find_info.SerialNumber = cert_info_id.SerialNumber
        find_info.Issuer = cert_info_id.Issuer

        cert_ctx = CertFindCertificateInStore(
            cert_store,
            X509_ASN_ENCODING | PKCS_7_ASN_ENCODING,
            0,
            CERT_FIND_SUBJECT_CERT,
            ctypes.byref(find_info),
            None,
        )
        if not cert_ctx:
            return None
        try:
            buf = ctypes.create_unicode_buffer(512)
            CertGetNameStringW(
                cert_ctx, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0,
                None, buf, 512,
            )
            return buf.value.strip() or None
        finally:
            CertFreeCertificateContext(cert_ctx)
    except Exception:
        return None
    finally:
        if msg:
            try:
                CryptMsgClose(msg)
            except Exception:
                pass
        if cert_store:
            try:
                CertCloseStore(cert_store, 0)
            except Exception:
                pass


# ---- Combined verification used by the Updater -----------------------------


def verify_full_installer(path: str | os.PathLike, expected_sha256: str) -> Tuple[bool, str]:
    """Run SHA-256 + Authenticode checks against a full installer .exe.
    Returns (ok, message)."""
    ok, msg = verify_sha256(path, expected_sha256)
    if not ok:
        return (False, msg)
    ok2, msg2 = verify_authenticode(path)
    if not ok2:
        return (False, f"{msg2} (hash check: {msg})")
    return (True, f"{msg}; {msg2}")


def verify_app_zip(path: str | os.PathLike, expected_sha256: str) -> Tuple[bool, str]:
    """Run SHA-256 against the downloaded app-zip. The Authenticode
    check on the inner Touchless.exe happens later in the apply bat
    (we can't reach into the zip from here without extracting). The
    bat verifies the staged Touchless.exe with Get-AuthenticodeSignature
    before robocopying into the install dir."""
    ok, msg = verify_sha256(path, expected_sha256)
    if not ok:
        return (False, msg)
    return (True, msg)
