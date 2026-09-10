# Touchless

Control your Windows PC with **hand gestures** and **voice**.

This GitHub tree is a **public subset**: the core app shell, camera pipeline, built-in preset gestures, and settings UI. It is not the full commercial source.

**Iris**, **custom gestures**, **profiles**, and **gesture packs** are not published here. They ship in the signed official build.

**[Download for Windows](https://touchless-control.com/downloads.html)** · **[Website](https://touchless-control.com)** · **[Microsoft Store](https://apps.microsoft.com/store/detail/XPDLMKGM1SFQG1)** · **[Product repo](https://github.com/markovk-tini/touchless-control)**

---

## What you can run from this tree

A development checkout can still launch the Windows UI and the **built-in** gesture set (fist, counts, swipes, mouse, drawing, volume, media, voice/dictation). Premium modules are replaced with no-op stubs so imports do not crash.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python run_app.py
```

For Iris, custom gesture recording, profiles, and packs, use the official installer — not this repository.

---

## License

See [LICENSE](LICENSE). The official application is distributed as a signed Windows installer.

© 2026 Konstantin Markov
