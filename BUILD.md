# Building the Windows release

Most people do not need this file. To *run* PLEAP from source see `README.md`;
to use it on Windows without Python, take the packaged build from the
[Releases](https://github.com/ProductivityLab/pleap/releases) page. This file is
for producing it.

## The packaged build (what the release carries)

```bat
build_tools\build_exe.bat
```

needs Python 3.10 or newer (64-bit) on the PATH and builds `dist\PLEAP\`:
`PLEAP.exe` and the `_internal` folder it needs (Python,
Qt, NumPy, SciPy and the example data sets). `PLEAP.exe` does not run on its
own; zip the whole `dist\PLEAP` folder (`PLEAP-v1.12-windows.zip`) and attach
that to the release, with its SHA-256
(`certutil -hashfile PLEAP-v1.12-windows.zip SHA256`).

## The installer (optional)

The rest of this file produces a conventional installer with Inno Setup
instead, which adds a Start-menu entry and an uninstaller.

## Prerequisites

1. Python 3.10 or newer (64-bit) — https://www.python.org/downloads/
   Tick "Add python.exe to PATH" during installation.
2. Inno Setup 6.3 or newer — https://jrsoftware.org/isdl.php

## One-click build

```bat
build_tools\make_installer.bat
```

## What it does

1. Creates `venv\` and installs the project with its GUI and build extras
   (`pip install -e ".[gui,build]"`, which reads `pyproject.toml`).
2. Verifies that PyQt5, numpy, pandas, scipy, statsmodels, openpyxl and
   matplotlib all import.
3. Runs PyInstaller over `src\pleap\__main__.py` → `dist\PLEAP\PLEAP.exe`.
4. Auto-locates Inno Setup (`%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe`).
5. Creates the `Output\` folder.
6. Compiles the installer → `Output\PLEAP_Setup_v1.12.exe`.

Run `pytest` before building. The installer has no way of telling you that an
estimator changed; the test suite does.

## Keeping the version in step

The version appears in four places, and all four have to agree:

| File | What to change |
|---|---|
| `src/pleap/__init__.py` | `__version__` |
| `pyproject.toml` | `version` under `[project]` |
| `CITATION.cff` | `version` and `date-released` |
| `build_tools/installer.iss` | `MyAppVersion` |

`build_tools/make_installer.bat` prints the expected output filename, so a
mismatch shows up as "ISCC reported success but `Output\...exe` not found".

## Notes on the build

- **No external solver.** Since v1.12 every linear program is solved through
  SciPy's HiGHS interface, so `pulp` and the CBC binary are gone. If you are
  updating an older spec file, drop `--collect-all pulp`.
- **`src` layout.** `--paths src` is what lets PyInstaller find the package;
  the entry point is `src\pleap\__main__.py`, not the old `pleap_main.py`.
- **Sample data.** `examples\` is bundled into the installed folder (it used
  to be `sample\`).
- **Antivirus false positives** are common for PyInstaller output. Publish the
  SHA-256 of the installer alongside it on the Releases page so users can check
  what they downloaded.

## If there is no `Output` folder and no setup file

This was fixed in v1.0.7. The older `.iss` used `ArchitecturesAllowed=x64`,
which Inno Setup 6.3+ rejects with a hard error; it now uses `x64compatible`.
`make_installer.bat` also creates `Output\` explicitly before running ISCC,
checks the error level from both steps, and resolves the absolute path to ISCC
so a `PATH` problem cannot fail silently.

If it still fails, the console window stays open and prints the exact ISCC
error level. Please open an issue with that text.
