# medaudiolab.spec
# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for MedAudioLab v2 (Windows x64, single windowed executable).

Build command:
    pyinstaller medaudiolab.spec --noconfirm --clean
or run scripts\\build_exe.bat, which also prepares the virtual environment.

Written for PyInstaller 6.x.  Differences from the PART 5.1 draft, all deliberate:

* ``cipher=`` / ``block_cipher`` and ``a.zipfiles`` are gone: PyInstaller 6 removed them.
* PySide6 is not collected wholesale (``collect_submodules('PySide6')`` would pull in
  WebEngine, Qt3D, QML ... several hundred MB the application never loads).  Only the
  Qt modules the application uses are listed; PyInstaller's PySide6 hooks then collect
  the matching Qt plugins (platforms, imageformats, styles, ...).
* ``_soundfile_data`` is collected as well: it is the package that holds libsndfile
  (``libsndfile_x64.dll``); ``collect_data_files('soundfile')`` alone misses it.
* pyqtgraph's data files are collected (the built-in colour maps, e.g. viridis, are CSV
  files that the spectrogram needs); its examples are left out.
* Hidden imports are filtered by what is installed, so a missing optional module
  (PyOpenGL, the numpy 1.x ``numpy.core`` layout, ...) is skipped instead of warning.
* UPX is allowed, but never for the Qt / Python / VC runtime DLLs, which it can corrupt.
* ``unittest`` is NOT excluded (the draft excluded it): SciPy imports it, so the frozen
  application could not even start.  Verified by running a probe inside a frozen build.
"""
import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH)
APP_NAME = "MedAudioLab"
ENTRY_POINT = ROOT / "run.py"
ICON = ROOT / "resources" / "icons" / "app_icon.ico"


def installed(module: str) -> bool:
    """True if ``module`` can be imported in the build environment."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # a parent package is missing or broken
        return False


def only_installed(names):
    return [name for name in names if installed(name)]


def without_tests(name: str) -> bool:
    """``collect_submodules`` filter: skip test packages (and anything under 'examples')."""
    return ".tests" not in name and ".testing" not in name and ".examples" not in name


# ---------------------------------------------------------------- resources
# (source, destination inside the bundle).  The application reads them through
# app.resource_path.get_resource_path(), i.e. <bundle>/resources/<relative path>.
required_resources = [
    (ROOT / "resources" / "wavelet_scales.txt", "resources"),        # CWT scale table
    (ROOT / "resources" / "styles", "resources/styles"),             # Qt stylesheet (light.qss)
    (ROOT / "resources" / "icons", "resources/icons"),               # window / executable icons
]
missing = [str(source) for source, _ in required_resources if not Path(source).exists()]
if missing or not ICON.exists() or not ENTRY_POINT.exists():
    raise SystemExit(
        "medaudiolab.spec: required files are missing:\n  "
        + "\n  ".join(missing + [str(p) for p in (ICON, ENTRY_POINT) if not p.exists()])
    )

# ------------------------------------------------------------------- imports
hidden_imports = [
    # --- the application itself (several modules are imported lazily inside functions)
    *collect_submodules("app", filter=without_tests),
    # --- PySide6: only what the application loads
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",             # pyqtgraph's SVG exporter
    "PySide6.QtOpenGL",          # pyqtgraph: optional OpenGL-accelerated views
    "PySide6.QtOpenGLWidgets",
    "shiboken6",
    # --- SciPy: the signal / fft / io parts used by the DSP code
    *collect_submodules("scipy.signal", filter=without_tests),
    *collect_submodules("scipy.fft", filter=without_tests),
    *collect_submodules("scipy.io", filter=without_tests),
    # --- audio: PortAudio and libsndfile are loaded through CFFI
    "sounddevice",
    "_sounddevice_data",
    "soundfile",
    "_soundfile_data",
    "cffi",
    "_cffi_backend",
    # --- pyserial (the Windows backend and the port enumeration)
    "serial",
    "serial.tools",
    "serial.tools.list_ports",
    "serial.tools.list_ports_windows",
    "serial.tools.list_ports_common",
    "serial.serialwin32",
    # --- pyqtgraph (without its examples)
    # (pyqtgraph.opengl and pyqtgraph.jupyter need optional packages: failing to import them is expected)
    *collect_submodules(
        "pyqtgraph",
        filter=lambda n: without_tests(n) and not n.startswith("pyqtgraph.opengl"),
        on_error="ignore",
    ),
    # --- numpy internals that are only reached dynamically
    *only_installed(["numpy.core._methods", "numpy.lib.format"]),
]

# pyqtgraph.opengl needs PyOpenGL (an optional dependency).  Bundle both if it is installed.
if installed("OpenGL"):
    hidden_imports += collect_submodules("pyqtgraph.opengl", filter=without_tests)
    hidden_imports += collect_submodules("OpenGL", filter=without_tests)

# --------------------------------------------------------------- data files
datas = [(str(source), destination) for source, destination in required_resources]
datas += collect_data_files("pyqtgraph", excludes=["**/examples/**", "**/__pycache__/**"])  # colour maps, templates
# PortAudio and libsndfile native libraries.  (sounddevice.py / soundfile.py are single
# modules; the DLLs live in these two data packages.)  PyInstaller's own sounddevice hook also
# adds the 32-bit / ARM64 / macOS PortAudio files (~3 MB); they are harmless and unused.
datas += collect_data_files("_sounddevice_data")
datas += collect_data_files("_soundfile_data", excludes=["**/__pycache__/**"])

# Software OpenGL / shader compiler for machines without a usable GPU driver.  PyInstaller's
# PySide6 hooks normally add them; listing them guarantees pyqtgraph's OpenGL views start.
binaries = collect_dynamic_libs("PySide6", search_patterns=["opengl32sw.dll", "d3dcompiler_47.dll"])

a = Analysis(
    [str(ENTRY_POINT)],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "_tkinter",                        # PySide6 replaces Tkinter
        "test",   # NOT "unittest": SciPy imports it, and the frozen application then fails to start
        "IPython", "jupyter", "pytest",
        "matplotlib",                                 # only needed for figure export, which is handed a figure
        "PyQt5", "PyQt6", "PySide2",                  # other Qt bindings must not compete with PySide6
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "concrt140.dll",
        "python3*.dll", "Qt6*.dll", "shiboken6*.dll", "pyside6*.dll", "*.pyd",
    ],
    runtime_tmpdir=None,
    console=False,  # windowed application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON),
)
