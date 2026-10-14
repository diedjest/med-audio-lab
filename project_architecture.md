# MedAudioLab v2 — Architectural Design & Implementation Roadmap

## Executive Summary

This document presents the complete architectural redesign of MedAudioLab: a commercial-grade desktop application for recording, processing, and analyzing high-frequency audio and ultrasonic signals (up to 384+ kHz). The redesign replaces the Tkinter-based prototype with a PySide6/pyqtgraph stack, introduces strict layered architecture with dependency inversion, and resolves critical bottlenecks including UI freezes during CWT computation, memory leaks, and tight coupling between DSP and presentation.

> [!IMPORTANT]
> The legacy prototype ([app.py](file:///c:/Users/dmitriy/PythonProjects/med-audio-lab/med_audio_lab/gui/app.py) — 1794 lines in a single file) conflates recording state, DSP orchestration, UI layout, and persistence into one monolithic Tkinter class. This document specifies a clean decomposition into 5 layers with 30+ focused modules.

---

## PART 1: SYSTEM ARCHITECTURE & PROJECT TOPOLOGY

### 1.1 Production Directory Structure

```text
med-audio-lab/
├── run.py                              # Entry point (unchanged)
├── requirements.txt                    # Dependency manifest
├── pyproject.toml                      # Build metadata & tool config
├── medaudiolab.spec                    # PyInstaller spec file
├── resources/
│   ├── wavelet_scales.txt              # CWT scale configuration
│   ├── icons/                          # SVG/PNG app icons
│   └── styles/
│       └── light.qss                   # Qt stylesheet (light theme)
│
├── app/
│   ├── __init__.py                     # Package root, version
│   ├── __main__.py                     # `python -m app` entry
│   ├── constants.py                    # App-wide constants, enums
│   │
│   ├── core/                           # LAYER 1: Pure domain logic (zero dependencies on Qt/IO)
│   │   ├── __init__.py
│   │   ├── dsp/
│   │   │   ├── __init__.py
│   │   │   ├── fft.py                  # FFT, Welch PSD, windowed spectral analysis
│   │   │   ├── stft.py                 # STFT spectrogram engine
│   │   │   ├── cwt.py                  # Memory-bounded Morlet CWT engine
│   │   │   ├── filters.py             # Zero-phase IIR/FIR filter pipeline (SOS)
│   │   │   ├── envelope.py            # Hilbert analytic signal & envelope extraction
│   │   │   ├── impulse_detector.py    # Adaptive threshold pulse detection
│   │   │   ├── metrics.py            # Signal statistics (RMS, peak, THD, spectral moments)
│   │   │   ├── heterodyne.py          # Ultrasonic down-shifting for audible monitoring
│   │   │   └── scales.py              # Wavelet scale loading & gap densification
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── audio_buffer.py        # Immutable AudioBuffer value object
│   │   │   ├── recording_session.py   # RecordingSession entity
│   │   │   ├── analysis_result.py     # AnalysisResult data classes
│   │   │   ├── impulse.py             # DetectedImpulse data class
│   │   │   └── device_info.py         # AudioDeviceInfo, SerialPortInfo
│   │   │
│   │   └── interfaces/
│   │       ├── __init__.py
│   │       ├── audio_engine.py        # IAudioEngine protocol (ABC)
│   │       ├── serial_engine.py       # ISerialEngine protocol (ABC)
│   │       ├── storage_repository.py  # IStorageRepository protocol (ABC)
│   │       └── export_service.py      # IExportService protocol (ABC)
│   │
│   ├── infrastructure/                 # LAYER 2: I/O, hardware, persistence
│   │   ├── __init__.py
│   │   ├── audio/
│   │   │   ├── __init__.py
│   │   │   ├── device_manager.py      # AudioDeviceManager (sounddevice enumeration)
│   │   │   ├── capture_engine.py      # CaptureEngine (PortAudio callback → ring buffer)
│   │   │   ├── playback_engine.py     # PlaybackEngine (output stream management)
│   │   │   └── ring_buffer.py         # Lock-free circular ring buffer (NumPy-backed)
│   │   │
│   │   ├── serial/
│   │   │   ├── __init__.py
│   │   │   ├── serial_engine.py       # SerialCaptureEngine (UART/ADC reader)
│   │   │   └── protocol_detector.py   # Binary vs text auto-detection
│   │   │
│   │   └── storage/
│   │       ├── __init__.py
│   │       ├── file_io.py             # WAV/CSV/TXT/BWF readers & writers
│   │       ├── sqlite_repository.py   # SQLite recording & metrics repository
│   │       └── export.py              # CSV/PNG/PDF export service
│   │
│   ├── services/                       # LAYER 3: Application orchestration
│   │   ├── __init__.py
│   │   ├── recording_service.py       # RecordingService (start/pause/stop workflow)
│   │   ├── analysis_service.py        # AnalysisService (async DSP dispatch)
│   │   ├── project_manager.py         # ProjectManager (session lifecycle, file paths)
│   │   └── settings_service.py        # SettingsService (persistent app configuration)
│   │
│   └── ui/                             # LAYER 4: Presentation (PySide6)
│       ├── __init__.py
│       ├── main_window.py             # QMainWindow shell with tab navigation
│       ├── viewmodels/
│       │   ├── __init__.py
│       │   ├── recording_vm.py        # RecordingViewModel (reactive state for Record tab)
│       │   ├── analysis_vm.py         # AnalysisViewModel (reactive state for Analysis tab)
│       │   └── library_vm.py          # LibraryViewModel (recording list, search, load)
│       │
│       ├── widgets/
│       │   ├── __init__.py
│       │   ├── waveform_widget.py     # pyqtgraph real-time waveform display
│       │   ├── spectrum_widget.py     # pyqtgraph live FFT spectrum display
│       │   ├── spectrogram_widget.py  # Matplotlib STFT/CWT scalogram canvas
│       │   ├── level_meter.py         # Peak/RMS level meter with clip indicator
│       │   ├── device_selector.py     # Audio/serial device combo with refresh
│       │   └── progress_overlay.py    # Non-blocking progress dialog for heavy DSP
│       │
│       ├── tabs/
│       │   ├── __init__.py
│       │   ├── record_tab.py          # Recording workflow tab
│       │   ├── analysis_tab.py        # Signal analysis tab
│       │   └── library_tab.py         # Recording database browser tab
│       │
│       └── workers/
│           ├── __init__.py
│           ├── capture_worker.py      # QThread wrapper for audio capture monitoring
│           ├── analysis_worker.py     # QThread wrapper for background DSP
│           └── export_worker.py       # QThread wrapper for background file export
│
├── tests/
│   ├── __init__.py
│   ├── test_core_dsp/
│   │   ├── test_fft.py
│   │   ├── test_cwt.py
│   │   ├── test_filters.py
│   │   ├── test_impulse_detector.py
│   │   └── test_metrics.py
│   ├── test_infrastructure/
│   │   ├── test_ring_buffer.py
│   │   ├── test_file_io.py
│   │   └── test_sqlite_repository.py
│   └── test_services/
│       ├── test_recording_service.py
│       └── test_analysis_service.py
│
└── scripts/
    ├── build_exe.bat
    ├── build_exe.ps1
    └── test_record_tones.py
```

### 1.2 Technology Stack & Dependency Matrix

| Category | Library | Version | Rationale |
|---|---|---|---|
| **UI Framework** | PySide6 | ≥6.6 | Qt 6 bindings — mature, fast event loop, QThread, commercial-friendly LGPL |
| **High-rate plotting** | pyqtgraph | ≥0.13 | GPU-accelerated OpenGL scatter/line plots, 60 FPS on 10M+ points |
| **Publication plots** | matplotlib | ≥3.8 | Vector PDF/SVG export, STFT/CWT colormaps, Matplotlib Qt backend |
| **Numerics** | numpy | ≥1.26 | Vectorized array ops, FFT, ring buffer backing |
| **DSP** | scipy | ≥1.12 | `signal.sosfiltfilt`, `signal.welch`, `signal.hilbert`, `signal.resample_poly` |
| **Audio I/O** | sounddevice | ≥0.5 | PortAudio bindings (WASAPI, DirectSound, ASIO, CoreAudio, ALSA) |
| **Audio files** | soundfile | ≥0.12 | libsndfile wrapper — BWF, 24-bit PCM, FLOAT_32 WAV |
| **Serial I/O** | pyserial | ≥3.5 | UART/COM port access for external ADC streams |
| **Packaging** | PyInstaller | ≥6.0 | Single-file `.exe` freeze with `--onefile --windowed` |
| **Testing** | pytest | ≥8.0 | Unit & integration testing |

#### `requirements.txt`

```text
PySide6>=6.6,<7
pyqtgraph>=0.13
matplotlib>=3.8
numpy>=1.26
scipy>=1.12
sounddevice>=0.5
soundfile>=0.12
pyserial>=3.5
pyinstaller>=6.0
pytest>=8.0
```

### 1.3 Layer Dependency Rules

```mermaid
graph TD
    UI["ui (PySide6, pyqtgraph)"] --> Services["services"]
    Services --> Core["core (models, dsp, interfaces)"]
    Services --> Infra["infrastructure (audio, serial, storage)"]
    Infra --> Core
    
    style Core fill:#dbeafe,stroke:#2563eb
    style Infra fill:#fef3c7,stroke:#d97706
    style Services fill:#d1fae5,stroke:#059669
    style UI fill:#ede9fe,stroke:#7c3aed
```

> [!WARNING]
> **Inversion of Control**: `core` defines abstract interfaces (`IAudioEngine`, `IStorageRepository`). `infrastructure` implements them. `services` depends only on abstractions. `ui` depends on services via ViewModels. No layer may import upward.

---

## PART 2: DATA FLOW & THREADING MODEL

### 2.1 Audio Capture Pipeline

```mermaid
sequenceDiagram
    participant HW as PortAudio Callback<br/>(C thread)
    participant RB as RingBuffer<br/>(shared memory)
    participant CW as CaptureWorker<br/>(QThread)
    participant VM as RecordingViewModel
    participant UI as Record Tab<br/>(Main thread)
    participant Disk as File Writer<br/>(QThread)

    HW->>RB: write(indata) — lock-free memcpy
    Note over HW,RB: Callback runs in PortAudio's<br/>real-time C thread (~5ms blocks<br/>at 192kHz, blocksize=1024)
    
    loop Every 33ms (~30 FPS)
        CW->>RB: read_latest(N) — snapshot
        CW->>VM: Signal: level_updated(peak_dbfs, rms_dbfs)
        CW->>VM: Signal: waveform_chunk(ndarray)
        CW->>VM: Signal: spectrum_updated(freqs, magnitudes, peak_hz)
    end
    
    VM->>UI: Qt property bindings update widgets
    UI->>UI: pyqtgraph.setData() — GPU render
    
    Note over CW,Disk: On stop():
    CW->>RB: drain_all() → full ndarray
    CW->>Disk: enqueue WAV write
    Disk->>Disk: sf.write() in background
    Disk->>VM: Signal: file_saved(path)
```

### 2.2 Background Analysis Pipeline (CWT/STFT)

```mermaid
sequenceDiagram
    participant UI as Analysis Tab
    participant VM as AnalysisViewModel
    participant AW as AnalysisWorker<br/>(QThread)
    participant DSP as core.dsp.cwt<br/>(pure NumPy/SciPy)
    participant PO as ProgressOverlay

    UI->>VM: request_cwt(params)
    VM->>PO: show(indeterminate=False)
    VM->>AW: start(task=CWT, data, params)
    
    loop Chunked CWT (per scale batch)
        AW->>DSP: _cwt_fft_bank(chunk)
        DSP-->>AW: partial result
        AW->>VM: Signal: progress(pct, message)
        VM->>PO: update(pct)
    end
    
    AW->>VM: Signal: result_ready(t, freqs, Wx)
    VM->>PO: hide()
    VM->>UI: update spectrogram widget
```

### 2.3 Concurrency Strategy

| Concern | Solution |
|---|---|
| **PortAudio callback (C thread)** | Lock-free ring buffer with atomic head/tail pointers. No Python objects allocated in callback. |
| **GIL contention during FFT** | NumPy/SciPy release the GIL during C-level computation (`np.fft.fft`, `scipy.signal.sosfiltfilt`). QThread workers can run in parallel. |
| **UI responsiveness** | All DSP > 50ms runs in `QThread` subclass. Progress reported via Qt signals (thread-safe cross-thread delivery). |
| **Ring buffer safety** | Single-producer (PortAudio callback), single-consumer (CaptureWorker). No mutex needed — atomic index + memory fence. Python implementation uses `threading.Lock` as conservative fallback with ~0.1μs overhead. |
| **Serial reader thread** | Dedicated `threading.Thread` (daemon) polls COM port, appends to thread-safe deque. CaptureWorker reads snapshots on timer. |
| **File I/O** | WAV writes and CSV exports delegated to `ExportWorker` (QThread) to prevent UI stalls on large files. |

### 2.4 Ring Buffer Design

```
┌─────────────────────────────────────────────────────────┐
│  NumPy ndarray (dtype=float32, shape=(capacity,))       │
│                                                         │
│  write_pos ──┐              ┌── read_pos                │
│              ▼              ▼                            │
│  [ _ _ _ _ [W W W W W W W W] R R R R _ _ _ _ _ _ _ ]   │
│                                                         │
│  capacity = 2^N (power of two for fast modulo via &)    │
│  available = (write_pos - read_pos) & (capacity - 1)    │
└─────────────────────────────────────────────────────────┘
```

- **Capacity**: `2^20` = 1,048,576 samples (~5.5s at 192 kHz). Configurable.
- **Overflow policy**: Oldest data silently overwritten (circular). No allocation.
- **Read modes**: `read_latest(n)` for live display; `drain_all()` for final capture.

---

## PART 3: DETAILED CLASS & API SPECIFICATION

### 3.1 Core Interfaces

```python
# app/core/interfaces/audio_engine.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, Callable
import numpy as np
from app.core.models.device_info import AudioDeviceInfo

class IAudioCaptureEngine(ABC):
    """Abstract audio capture engine — implemented by infrastructure."""
    
    @abstractmethod
    def list_input_devices(self) -> list[AudioDeviceInfo]: ...
    
    @abstractmethod
    def start(
        self,
        device_index: int | None,
        sample_rate: int,
        channels: int = 1,
        dtype: str = "float32",
        block_size: int = 1024,
    ) -> None: ...
    
    @abstractmethod
    def pause(self) -> None: ...
    
    @abstractmethod
    def resume(self) -> None: ...
    
    @abstractmethod
    def stop(self) -> np.ndarray: ...
    
    @abstractmethod
    def snapshot(self, max_samples: int | None = None) -> np.ndarray: ...
    
    @abstractmethod
    def peak_rms(self) -> tuple[float, float]:
        """Returns (peak_dbfs, rms_dbfs) of current buffer window."""
        ...
    
    @property
    @abstractmethod
    def is_recording(self) -> bool: ...
    
    @property
    @abstractmethod
    def is_paused(self) -> bool: ...
    
    @property
    @abstractmethod
    def sample_rate(self) -> int: ...


class IAudioPlaybackEngine(ABC):
    """Abstract playback engine."""
    
    @abstractmethod
    def list_output_devices(self) -> list[AudioDeviceInfo]: ...
    
    @abstractmethod
    def play(
        self, data: np.ndarray, sample_rate: int, device_index: int | None = None
    ) -> None: ...
    
    @abstractmethod
    def stop(self) -> None: ...
    
    @property
    @abstractmethod
    def is_playing(self) -> bool: ...
```

### 3.2 Ring Buffer

```python
# app/infrastructure/audio/ring_buffer.py
from __future__ import annotations
import threading
import numpy as np

class RingBuffer:
    """Thread-safe circular buffer backed by a pre-allocated NumPy array.
    
    Single-producer (audio callback), single-consumer (UI worker).
    Power-of-two capacity for branchless modulo.
    """
    
    __slots__ = ("_buf", "_capacity", "_mask", "_write_pos", "_read_pos", "_lock")
    
    def __init__(self, capacity: int = 1 << 20, dtype: np.dtype = np.float32) -> None:
        # Round up to next power of two
        self._capacity = 1 << max(1, (capacity - 1).bit_length())
        self._mask = self._capacity - 1
        self._buf = np.zeros(self._capacity, dtype=dtype)
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._lock = threading.Lock()
    
    @property
    def capacity(self) -> int:
        return self._capacity
    
    @property
    def available(self) -> int:
        return (self._write_pos - self._read_pos) & self._mask
    
    def write(self, data: np.ndarray) -> int:
        """Write samples into the buffer. Returns number of samples written.
        Called from audio callback thread."""
        n = data.shape[0]
        if n == 0:
            return 0
        if n >= self._capacity:
            # Only keep the last capacity samples
            data = data[-self._capacity:]
            n = self._capacity
        
        start = self._write_pos & self._mask
        end = start + n
        
        with self._lock:
            if end <= self._capacity:
                self._buf[start:end] = data
            else:
                first = self._capacity - start
                self._buf[start:] = data[:first]
                self._buf[:n - first] = data[first:]
            self._write_pos += n
        return n
    
    def read_latest(self, count: int) -> np.ndarray:
        """Read the most recent `count` samples without consuming them.
        Called from UI worker thread."""
        with self._lock:
            avail = min(count, (self._write_pos - self._read_pos) & self._mask,
                       self._write_pos)
            if avail <= 0:
                return np.zeros(0, dtype=self._buf.dtype)
            start = (self._write_pos - avail) & self._mask
            if start + avail <= self._capacity:
                return self._buf[start:start + avail].copy()
            else:
                first = self._capacity - start
                return np.concatenate([
                    self._buf[start:],
                    self._buf[:avail - first]
                ])
    
    def drain_all(self) -> np.ndarray:
        """Read and consume all available samples. Used on recording stop."""
        with self._lock:
            avail = min(self._write_pos, self._capacity)
            if avail <= 0:
                return np.zeros(0, dtype=self._buf.dtype)
            start = (self._write_pos - avail) & self._mask
            if start + avail <= self._capacity:
                result = self._buf[start:start + avail].copy()
            else:
                first = self._capacity - start
                result = np.concatenate([
                    self._buf[start:],
                    self._buf[:avail - first]
                ])
            self._read_pos = self._write_pos
            return result
    
    def reset(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._buf[:] = 0
```

### 3.3 AudioDeviceManager

```python
# app/infrastructure/audio/device_manager.py
from __future__ import annotations
from dataclasses import dataclass
import sounddevice as sd
from app.core.models.device_info import AudioDeviceInfo

class AudioDeviceManager:
    """Enumerates and validates audio hardware via PortAudio."""
    
    STANDARD_RATES = [44100, 48000, 96000, 192000, 384000, 768000]
    
    @staticmethod
    def list_input_devices() -> list[AudioDeviceInfo]:
        devices = []
        for idx, info in enumerate(sd.query_devices()):
            if int(info.get("max_input_channels", 0)) > 0:
                devices.append(AudioDeviceInfo(
                    index=idx,
                    name=str(info["name"]),
                    input_channels=int(info["max_input_channels"]),
                    output_channels=int(info.get("max_output_channels", 0)),
                    default_sample_rate=int(float(info["default_samplerate"])),
                    hostapi=str(sd.query_hostapis(info["hostapi"])["name"]),
                ))
        return devices
    
    @staticmethod
    def list_output_devices() -> list[AudioDeviceInfo]:
        devices = []
        for idx, info in enumerate(sd.query_devices()):
            if int(info.get("max_output_channels", 0)) > 0:
                devices.append(AudioDeviceInfo(
                    index=idx,
                    name=str(info["name"]),
                    input_channels=int(info.get("max_input_channels", 0)),
                    output_channels=int(info["max_output_channels"]),
                    default_sample_rate=int(float(info["default_samplerate"])),
                    hostapi=str(sd.query_hostapis(info["hostapi"])["name"]),
                ))
        return devices
    
    @staticmethod
    def supported_rates(device_index: int) -> list[int]:
        """Probe which standard sample rates the device actually supports."""
        supported = []
        for rate in AudioDeviceManager.STANDARD_RATES:
            try:
                sd.check_input_settings(device=device_index, samplerate=rate)
                supported.append(rate)
            except Exception:
                pass
        return supported
```

### 3.4 SignalProcessor (DSP Facade)

```python
# Conceptual facade — each method delegates to the appropriate core.dsp module
class SignalProcessor:
    """High-level DSP facade used by AnalysisService.
    
    All methods are pure, stateless, and accept/return NumPy arrays.
    Thread-safe by design (no mutable state).
    """
    
    def fft_spectrum(
        self, x: np.ndarray, fs: int, *, 
        window: str = "hann",
        fmin_hz: float = 0.0,
        fmax_hz: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full-resolution FFT. Returns (frequencies_hz, magnitudes)."""
        ...
    
    def welch_psd(
        self, x: np.ndarray, fs: int, *,
        nperseg: int = 4096,
        window: str = "hann",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Welch Power Spectral Density. Returns (frequencies_hz, psd)."""
        ...
    
    def stft(
        self, x: np.ndarray, fs: int, *,
        nperseg: int = 4096,
        noverlap: int = 3072,
        window: str = "hann",
        fmin_hz: float = 0.0,
        fmax_hz: float = 100_000.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """STFT spectrogram. Returns (time, frequencies, magnitude_db)."""
        ...
    
    def cwt_morlet(
        self, x: np.ndarray, fs: int, *,
        fmin_hz: float = 100.0,
        fmax_hz: float = 100_000.0,
        max_seconds: float | None = 4.0,
        memory_budget_mb: float = 800.0,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Memory-bounded Morlet CWT. Returns (time, frequencies_hz, Wx_complex)."""
        ...
    
    def filter_signal(
        self, x: np.ndarray, fs: int, *,
        filter_type: str,  # "highpass" | "lowpass" | "bandpass" | "bandstop"
        freq_hz: float | tuple[float, float],
        order: int = 4,
        design: str = "butter",
    ) -> np.ndarray:
        """Zero-phase IIR filtering via sosfiltfilt."""
        ...
    
    def detect_impulses(
        self, x: np.ndarray, fs: int, **kwargs
    ) -> list[DetectedImpulse]:
        """Adaptive threshold pulse detection."""
        ...
    
    def hilbert_envelope(self, x: np.ndarray) -> np.ndarray:
        """Analytic signal envelope via Hilbert transform."""
        ...
    
    def compute_metrics(
        self, x: np.ndarray, fs: int
    ) -> SignalMetrics:
        """Full signal statistics: RMS, peak, THD, spectral centroid, etc."""
        ...
    
    def estimate_thd(
        self, x: np.ndarray, fs: int, fundamental_hz: float, n_harmonics: int = 6
    ) -> float:
        """Total Harmonic Distortion as a ratio."""
        ...
```

### 3.5 ContinuousWaveletTransformEngine

```python
# Detailed in PART 4 — key API surface:
class CwtEngine:
    """Memory-bounded, chunked Morlet CWT with frequency-domain FFT convolution."""
    
    def __init__(
        self,
        w0: float = 2.0 * np.pi,
        memory_budget_mb: float = 800.0,
        chunk_scales: int = 8,
    ) -> None: ...
    
    def compute(
        self,
        x: np.ndarray,
        fs: int,
        *,
        fmin_hz: float = 100.0,
        fmax_hz: float = 100_000.0,
        scales: np.ndarray | None = None,
        max_seconds: float | None = None,
        target_fs: int | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> CwtResult: ...
    
    def lateral_slice(
        self, result: CwtResult, t0_s: float
    ) -> tuple[np.ndarray, np.ndarray]: ...
    
    def longitudinal_slice(
        self, result: CwtResult, f0_hz: float
    ) -> tuple[np.ndarray, np.ndarray]: ...
    
    def ridge_detection(
        self, result: CwtResult
    ) -> tuple[np.ndarray, np.ndarray]: ...
```

### 3.6 RecordingSessionManager

```python
# app/services/recording_service.py
class RecordingService(QObject):
    """Orchestrates the recording lifecycle.
    
    Signals:
        recording_started(sample_rate: int)
        recording_paused()
        recording_resumed()
        recording_stopped(session: RecordingSession)
        level_updated(peak_dbfs: float, rms_dbfs: float)
        live_data(data: np.ndarray)
        error_occurred(message: str)
    """
    
    recording_started = Signal(int)
    recording_paused = Signal()
    recording_resumed = Signal()
    recording_stopped = Signal(object)  # RecordingSession
    level_updated = Signal(float, float)
    live_data = Signal(object)  # np.ndarray
    error_occurred = Signal(str)
    
    def __init__(
        self,
        audio_engine: IAudioCaptureEngine,
        serial_engine: ISerialEngine,
        storage: IStorageRepository,
        settings: SettingsService,
    ) -> None: ...
    
    def start_audio(self, device_index: int | None, sample_rate: int) -> None: ...
    def start_serial(self, port: str, baudrate: int, analysis_rate: int) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop_and_save(self, title: str, patient: str, notes: str) -> None: ...
```

---

## PART 4: PRODUCTION-READY CORE DSP IMPLEMENTATIONS

### 4.1 CWT Engine — Memory-Bounded Morlet CWT

```python
"""
app/core/dsp/cwt.py

Memory-bounded Continuous Wavelet Transform using frequency-domain FFT convolution
with Complex Morlet wavelet. Designed for ultrasonic signals up to 384 kHz.

Key design decisions:
- Scales are processed in chunks of `chunk_scales` to bound peak memory
- Signal is zero-padded to next power-of-two for FFT efficiency
- Negative frequencies zeroed (analytic wavelet)
- Scale gap densification ensures sub-bin frequency resolution
- Progress callback enables responsive UI during long computations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None


@dataclass(frozen=True)
class CwtResult:
    """Immutable result container for CWT computation."""
    fs: int
    time: np.ndarray
    frequencies_hz: np.ndarray
    coefficients: np.ndarray  # complex64, shape (n_scales, n_time)
    scales: np.ndarray

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.coefficients)

    @property
    def power(self) -> np.ndarray:
        return np.abs(self.coefficients) ** 2

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.coefficients)


def _next_pow2(n: int) -> int:
    """Next power of two >= n."""
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def _densify_scale_gaps(
    scales: np.ndarray,
    *,
    gap_ratio: float = 1.25,
    max_freq_ratio: float = 1.08,
) -> np.ndarray:
    """Insert intermediate Morlet scales where adjacent central frequencies
    are too far apart (f ∝ 1/scale). Without this, a tone at e.g. 30 kHz
    at fs=192kHz (needing scale ≈ 6.4) might fall between CWT rows."""
    scales = np.asarray(scales, dtype=np.float64)
    if scales.size < 2:
        return scales
    out: list[float] = [float(scales[0])]
    for s0, s1 in zip(scales[:-1], scales[1:]):
        s0, s1 = float(s0), float(s1)
        ratio = s1 / s0
        if ratio > gap_ratio:
            n_insert = int(np.ceil(np.log(ratio) / np.log(max_freq_ratio))) - 1
            if n_insert > 0:
                inserted = np.geomspace(s0, s1, n_insert + 2, dtype=np.float64)[1:-1]
                out.extend(float(x) for x in inserted)
        out.append(s1)
    return np.unique(np.asarray(out, dtype=np.float64))


def load_scales(path: str | Path | None = None) -> np.ndarray:
    """Load CWT scales from a text file, with fallback to geometric default."""
    default = np.geomspace(2.0, 5986.0, 116).astype(np.float64)
    if path is None:
        # Search relative to this module
        cfg = Path(__file__).resolve().parent.parent.parent.parent / "resources" / "wavelet_scales.txt"
    else:
        cfg = Path(path)
    if not cfg.exists():
        return _densify_scale_gaps(default)
    values: list[float] = []
    with cfg.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values.append(float(line.replace(",", ".")))
    scales = np.asarray(values, dtype=np.float64)
    if scales.size == 0:
        return _densify_scale_gaps(default)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError(f"Invalid scale values in '{cfg}'. All scales must be > 0.")
    return _densify_scale_gaps(scales)


def _resample_to_target(
    x: np.ndarray, fs: int, target_fs: int
) -> tuple[np.ndarray, int]:
    """Resample signal to target sample rate using polyphase filtering."""
    if target_fs <= 0 or fs == target_fs:
        return x, fs
    if resample_poly is not None:
        y = resample_poly(x, int(target_fs), int(fs)).astype(np.float32, copy=False)
        return y, int(target_fs)
    # Fallback: linear interpolation
    n_out = max(1, int(round(x.size * target_fs / fs)))
    t_in = np.arange(x.size, dtype=np.float64) / fs
    t_out = np.arange(n_out, dtype=np.float64) / target_fs
    return np.interp(t_out, t_in, x).astype(np.float32, copy=False), int(target_fs)


class CwtEngine:
    """Memory-bounded Morlet CWT with chunked scale processing.
    
    Architecture:
    1. Signal is peak-normalized and optionally resampled
    2. Zero-padded to N_fft = next_pow2(2*N - 1) for linear convolution via FFT
    3. Morlet wavelets constructed in frequency domain: ψ̂(ω) = √s · exp(-½(sω - ω₀)²)
    4. Scales processed in batches of `chunk_scales` to limit memory to ~O(chunk × N_fft)
    5. Negative frequencies zeroed to enforce analytic wavelet
    
    Memory budget:
    - Peak allocation ≈ chunk_scales × N_fft × 8 bytes (complex64)
    - For N=1M samples, chunk=8: ~64 MB per batch (safe on 8GB systems)
    """
    
    def __init__(
        self,
        w0: float = 2.0 * np.pi,
        memory_budget_mb: float = 800.0,
        chunk_scales: int = 8,
    ) -> None:
        self._w0 = float(w0)
        self._memory_budget_mb = float(memory_budget_mb)
        self._chunk_scales = max(1, int(chunk_scales))
    
    def estimate_memory_mb(
        self, n_samples: int, n_scales: int
    ) -> float:
        """Estimate peak memory usage in MB for given dimensions."""
        n_fft = _next_pow2(2 * n_samples - 1)
        # Output matrix: n_scales × n_samples × 8 bytes (complex64)
        output_mb = n_scales * n_samples * 8 / (1024 * 1024)
        # Working set per chunk: chunk × n_fft × 16 bytes (complex128 intermediate)
        working_mb = self._chunk_scales * n_fft * 16 / (1024 * 1024)
        return output_mb + working_mb
    
    def compute(
        self,
        x: np.ndarray,
        fs: int,
        *,
        fmin_hz: float = 100.0,
        fmax_hz: float = 100_000.0,
        scales: np.ndarray | None = None,
        max_seconds: float | None = None,
        target_fs: int | None = None,
        scale_stride: int = 1,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> CwtResult:
        """Compute the Continuous Wavelet Transform.
        
        Args:
            x: Input signal (1D float32).
            fs: Sample rate in Hz.
            fmin_hz: Minimum frequency of interest.
            fmax_hz: Maximum frequency of interest.
            scales: Custom scale array. If None, loaded from config file.
            max_seconds: Truncate signal to this duration.
            target_fs: Resample to this rate before CWT.
            scale_stride: Skip every N-th scale (for memory reduction).
            progress_callback: Called with (fraction, message) during computation.
        
        Returns:
            CwtResult with time axis, frequency axis, and complex coefficients.
        
        Raises:
            ValueError: If signal is too short or no scales fall in frequency range.
            MemoryError: If estimated memory exceeds budget.
        """
        # --- Preprocessing ---
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size < 8:
            raise ValueError("Signal too short for CWT (need >= 8 samples).")
        
        # Peak normalize
        peak = float(np.max(np.abs(x)))
        if peak > 0:
            x = (x / peak).astype(np.float32, copy=False)
        
        # Resample if requested
        if target_fs is not None and target_fs > 0 and fs != target_fs:
            x, fs = _resample_to_target(x, fs, target_fs)
        
        # Truncate
        if max_seconds is not None and max_seconds > 0:
            max_n = int(max_seconds * fs)
            if max_n > 0 and x.size > max_n:
                x = x[:max_n]
        
        # --- Scale selection ---
        if scales is None:
            scales = load_scales()
        scales = np.asarray(scales, dtype=np.float64)
        
        stride = max(1, int(scale_stride))
        if stride > 1:
            scales = scales[::stride]
        
        # Convert scales to center frequencies: f = w0 * fs / (2π * s)
        freqs_hz = self._w0 * fs / (2.0 * np.pi * scales)
        
        # Filter to requested frequency band
        band = (freqs_hz >= float(fmin_hz)) & (freqs_hz <= float(fmax_hz))
        if not np.any(band):
            raise ValueError(
                f"No scales in frequency range {fmin_hz:.1f}–{fmax_hz:.1f} Hz. "
                f"Available: {freqs_hz.min():.1f}–{freqs_hz.max():.1f} Hz."
            )
        scales = scales[band]
        freqs_hz = freqs_hz[band]
        
        # --- Memory budget check ---
        mem_mb = self.estimate_memory_mb(x.size, scales.size)
        if mem_mb > self._memory_budget_mb:
            raise MemoryError(
                f"CWT would require ~{mem_mb:.0f} MB (budget: {self._memory_budget_mb:.0f} MB). "
                f"Reduce signal duration, increase scale_stride, or narrow frequency range."
            )
        
        # --- FFT of signal ---
        n = x.size
        n_fft = _next_pow2(2 * n - 1)
        X = np.fft.fft(x.astype(np.float32, copy=False), n=n_fft)
        f_fft = np.fft.fftfreq(n_fft, d=1.0 / fs)
        
        # --- Chunked wavelet convolution ---
        omega_d = 2.0 * np.pi * f_fft / float(fs)
        neg_mask = f_fft < 0
        Wx = np.empty((scales.size, n), dtype=np.complex64)
        
        total_chunks = (scales.size + self._chunk_scales - 1) // self._chunk_scales
        
        for chunk_idx, c0 in enumerate(range(0, scales.size, self._chunk_scales)):
            c1 = min(c0 + self._chunk_scales, scales.size)
            sc = scales[c0:c1]
            s_col = sc[:, np.newaxis]  # (chunk, 1)
            
            # Morlet wavelet in frequency domain:
            # ψ̂(ω) = √s · exp(-½(s·ω - ω₀)²)
            psi_hat = np.exp(
                -0.5 * ((s_col * omega_d - self._w0) ** 2),
                dtype=np.float64,
            )
            np.multiply(psi_hat, np.sqrt(s_col), out=psi_hat)
            
            # Zero negative frequencies (analytic wavelet)
            psi_hat[:, neg_mask] = 0.0
            
            # Convolution in frequency domain: W(s,τ) = IFFT[X(ω) · ψ̂*(s,ω)]
            conv = np.fft.ifft(X * np.conj(psi_hat), axis=1)
            Wx[c0:c1, :] = conv[:, :n]
            
            # Report progress
            if progress_callback is not None:
                frac = (chunk_idx + 1) / total_chunks
                progress_callback(
                    frac,
                    f"CWT: scales {c0+1}–{c1} of {scales.size} "
                    f"({frac*100:.0f}%)",
                )
        
        t = np.arange(n, dtype=np.float64) / fs
        
        return CwtResult(
            fs=fs,
            time=t,
            frequencies_hz=freqs_hz,
            coefficients=Wx,
            scales=scales,
        )
    
    def lateral_slice(
        self, result: CwtResult, t0_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Frequency spectrum at a specific time instant.
        
        Returns:
            (frequencies_hz, amplitude) arrays.
        """
        t_idx = int(np.argmin(np.abs(result.time - t0_s)))
        amp = np.abs(result.coefficients[:, t_idx])
        return result.frequencies_hz.copy(), amp.astype(np.float64)
    
    def longitudinal_slice(
        self, result: CwtResult, f0_hz: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Temporal cross-section Re(W_x) at a specific frequency.
        
        Returns:
            (time_relative_s, normalized_real_part) arrays.
        """
        f_idx = int(np.argmin(np.abs(result.frequencies_hz - f0_hz)))
        wx_row = result.coefficients[f_idx, :]
        re = np.real(wx_row.astype(np.complex128))
        denom = float(np.max(np.abs(re)))
        if denom > 0:
            re = re / denom
        # Center on peak
        t0 = float(result.time[int(np.argmax(np.abs(re)))])
        t_rel = result.time - t0
        return t_rel.astype(np.float64), re.astype(np.float64)
    
    def ridge_detection(
        self, result: CwtResult
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract the ridge line (dominant frequency vs time).
        
        Returns:
            (time, ridge_frequencies_hz) arrays.
        """
        amp = np.abs(result.coefficients)
        ridge_idx = np.argmax(amp, axis=0)
        return result.time.copy(), result.frequencies_hz[ridge_idx].copy()
```

### 4.2 Filter Pipeline — Zero-Phase IIR Filtering

```python
"""
app/core/dsp/filters.py

Zero-phase digital filtering using second-order sections (SOS).
Supports Butterworth, Chebyshev Type I, and Chebyshev Type II designs.

Design rationale:
- sosfiltfilt provides zero phase distortion (critical for impulse timing)
- SOS representation avoids numerical instability of transfer function form
- Automatic Nyquist guard prevents invalid filter specs
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from scipy.signal import sosfiltfilt, butter, cheby1, cheby2


class FilterType(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


class FilterDesign(Enum):
    BUTTERWORTH = "butter"
    CHEBYSHEV_I = "cheby1"
    CHEBYSHEV_II = "cheby2"


@dataclass(frozen=True)
class FilterSpec:
    """Immutable filter specification."""
    filter_type: FilterType
    freq_hz: float | tuple[float, float]  # scalar for LP/HP, tuple for BP/BS
    order: int = 4
    design: FilterDesign = FilterDesign.BUTTERWORTH
    ripple_db: float = 0.5  # passband ripple (Cheby I) or stopband atten (Cheby II)


class FilterPipeline:
    """Configurable, zero-phase IIR filter with automatic SOS calculation.
    
    Usage:
        pipeline = FilterPipeline()
        filtered = pipeline.apply(
            signal, fs=192000,
            spec=FilterSpec(FilterType.BANDPASS, (20000, 80000), order=6)
        )
    """
    
    @staticmethod
    def _normalize_frequency(
        freq_hz: float | tuple[float, float],
        fs: int,
        filter_type: FilterType,
    ) -> float | tuple[float, float]:
        """Normalize frequency to [0, 1) relative to Nyquist, with safety margin."""
        nyquist = fs / 2.0
        margin = 0.98  # Stay 2% below Nyquist to avoid instability
        
        if isinstance(freq_hz, tuple):
            lo = min(float(freq_hz[0]) / nyquist, margin)
            hi = min(float(freq_hz[1]) / nyquist, margin)
            if lo <= 0:
                lo = 1e-6
            if hi <= lo:
                raise ValueError(
                    f"Upper cutoff ({freq_hz[1]} Hz) must exceed lower "
                    f"cutoff ({freq_hz[0]} Hz)."
                )
            return (lo, hi)
        else:
            wn = min(float(freq_hz) / nyquist, margin)
            if wn <= 0:
                raise ValueError(f"Cutoff frequency must be > 0 Hz (got {freq_hz}).")
            return wn
    
    @staticmethod
    def design_sos(spec: FilterSpec, fs: int) -> np.ndarray:
        """Compute second-order sections for the given filter specification.
        
        Returns:
            SOS array of shape (n_sections, 6).
        """
        wn = FilterPipeline._normalize_frequency(
            spec.freq_hz, fs, spec.filter_type
        )
        btype = spec.filter_type.value
        
        if spec.design == FilterDesign.BUTTERWORTH:
            sos = butter(spec.order, wn, btype=btype, output="sos")
        elif spec.design == FilterDesign.CHEBYSHEV_I:
            sos = cheby1(spec.order, spec.ripple_db, wn, btype=btype, output="sos")
        elif spec.design == FilterDesign.CHEBYSHEV_II:
            sos = cheby2(spec.order, spec.ripple_db, wn, btype=btype, output="sos")
        else:
            raise ValueError(f"Unknown filter design: {spec.design}")
        
        return sos
    
    def apply(
        self,
        x: np.ndarray,
        fs: int,
        spec: FilterSpec,
    ) -> np.ndarray:
        """Apply zero-phase filtering to the signal.
        
        Args:
            x: Input signal (1D float32/float64).
            fs: Sample rate in Hz.
            spec: Filter specification.
        
        Returns:
            Filtered signal with same length and dtype as input.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size < 3 * spec.order:
            raise ValueError(
                f"Signal too short ({x.size} samples) for filter order {spec.order}. "
                f"Need at least {3 * spec.order} samples."
            )
        
        sos = self.design_sos(spec, fs)
        y = sosfiltfilt(sos, x).astype(x.dtype, copy=False)
        return y
    
    def apply_chain(
        self,
        x: np.ndarray,
        fs: int,
        specs: list[FilterSpec],
    ) -> np.ndarray:
        """Apply a chain of filters sequentially.
        
        Args:
            x: Input signal.
            fs: Sample rate.
            specs: List of filter specs to apply in order.
        
        Returns:
            Signal filtered through all stages.
        """
        result = np.asarray(x, dtype=np.float64).ravel()
        for spec in specs:
            result = self.apply(result, fs, spec)
        return result
    
    def frequency_response(
        self,
        spec: FilterSpec,
        fs: int,
        n_points: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the filter's frequency response magnitude.
        
        Returns:
            (frequencies_hz, magnitude_db) arrays.
        """
        from scipy.signal import sosfreqz
        sos = self.design_sos(spec, fs)
        w, h = sosfreqz(sos, worN=n_points, fs=fs)
        mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
        return np.asarray(w, dtype=np.float64), mag_db.astype(np.float64)
```

### 4.3 Impulse Detector — Adaptive Threshold Pulse Detection

```python
"""
app/core/dsp/impulse_detector.py

Robust ultrasonic pulse/impulse train detection using adaptive thresholding.
Noise floor estimated via Median Absolute Deviation (MAD) of the signal envelope.

Design rationale:
- MAD-based noise floor is robust to outliers (unlike mean/std)
- Envelope smoothing reduces false triggers from oscillation zero-crossings
- Merge gap prevents a single pulse from splitting into multiple detections
- Per-impulse FFT extracts dominant frequency of each detected burst
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DetectedImpulse:
    """Immutable data class for a single detected pulse."""
    index: int
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    duration_ms: float
    peak_time_s: float
    peak_amplitude: float
    rms: float
    dominant_freq_hz: float  # NaN if undetermined
    interval_ms: float  # NaN for the first impulse
    threshold_used: float


@dataclass
class ImpulseDetectorConfig:
    """Configuration for impulse detection."""
    threshold_db: float = -35.0
    min_duration_ms: float = 0.4
    merge_gap_ms: float = 1.5
    pre_pad_ms: float = 0.15
    post_pad_ms: float = 0.35
    envelope_ms: float = 0.35
    fmin_hz: float = 5000.0
    fmax_hz: float | None = None
    noise_factor: float = 6.0


class ImpulseDetector:
    """Adaptive threshold impulse detector for ultrasonic burst signals.
    
    Algorithm:
    1. Remove DC offset from signal
    2. Compute smoothed envelope via moving average of |x|
    3. Normalize envelope to [0, 1]
    4. Calculate adaptive threshold = median + noise_factor × 1.4826 × MAD
    5. Threshold is max(fixed_threshold, adaptive_threshold), capped at 0.85
    6. Detect active regions where envelope > threshold
    7. Merge adjacent regions closer than merge_gap_ms
    8. Extract metrics (peak, RMS, dominant frequency) for each impulse
    
    The 1.4826 factor converts MAD to an estimate of standard deviation
    for normally distributed data.
    """
    
    def __init__(self, config: ImpulseDetectorConfig | None = None) -> None:
        self._config = config or ImpulseDetectorConfig()
    
    @property
    def config(self) -> ImpulseDetectorConfig:
        return self._config
    
    @config.setter
    def config(self, value: ImpulseDetectorConfig) -> None:
        self._config = value
    
    @staticmethod
    def _remove_dc(x: np.ndarray) -> np.ndarray:
        """Remove DC offset (mean subtraction)."""
        x = x.copy()
        if x.size > 0:
            x -= float(np.mean(x))
        return x
    
    @staticmethod
    def _compute_envelope(
        x: np.ndarray, fs: int, envelope_ms: float
    ) -> np.ndarray:
        """Compute smoothed amplitude envelope."""
        abs_x = np.abs(x).astype(np.float32)
        win = max(1, int(fs * envelope_ms / 1000.0))
        if win > 1:
            kernel = np.ones(win, dtype=np.float32) / win
            return np.convolve(abs_x, kernel, mode="same").astype(np.float32)
        return abs_x
    
    @staticmethod
    def _estimate_dominant_frequency(
        segment: np.ndarray, fs: int, fmin_hz: float, fmax_hz: float
    ) -> float:
        """FFT-based dominant frequency estimation for a short segment."""
        if segment.size < 32:
            return math.nan
        
        window = np.hanning(segment.size).astype(np.float32)
        windowed = segment * window
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(segment.size, d=1.0 / fs)
        
        band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
        if not np.any(band):
            return math.nan
        
        band_spectrum = spectrum[band]
        band_freqs = freqs[band]
        
        peak_idx = int(np.argmax(band_spectrum))
        peak_freq = float(band_freqs[peak_idx])
        
        # Parabolic interpolation for sub-bin precision
        if 0 < peak_idx < band_spectrum.size - 1:
            alpha = float(band_spectrum[peak_idx - 1])
            beta = float(band_spectrum[peak_idx])
            gamma = float(band_spectrum[peak_idx + 1])
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-12:
                delta = 0.5 * (alpha - gamma) / denom
                df = float(band_freqs[1] - band_freqs[0]) if band_freqs.size > 1 else 0.0
                peak_freq += delta * df
        
        return peak_freq if peak_freq > 0 else math.nan
    
    def detect(
        self,
        x: np.ndarray,
        fs: int,
        config: ImpulseDetectorConfig | None = None,
    ) -> list[DetectedImpulse]:
        """Detect impulse bursts in the signal.
        
        Args:
            x: Input signal (1D float).
            fs: Sample rate in Hz.
            config: Override configuration. If None, uses instance config.
        
        Returns:
            List of DetectedImpulse objects, sorted by time.
        """
        cfg = config or self._config
        
        # Preprocessing
        x = np.asarray(x, dtype=np.float32).ravel()
        x = self._remove_dc(x)
        fs = max(1, int(fs))
        
        if x.size < 8:
            return []
        
        # Compute envelope
        env = self._compute_envelope(x, fs, cfg.envelope_ms)
        peak_env = float(np.max(env))
        if peak_env <= 1e-12:
            return []
        env_norm = env / peak_env
        
        # Adaptive threshold via MAD
        fixed_threshold = 10.0 ** (float(cfg.threshold_db) / 20.0)
        median_val = float(np.median(env_norm))
        mad_val = float(np.median(np.abs(env_norm - median_val)))
        adaptive_threshold = median_val + float(cfg.noise_factor) * 1.4826 * mad_val
        threshold = min(0.85, max(fixed_threshold, adaptive_threshold))
        
        # Find active regions
        active = env_norm >= threshold
        if not np.any(active):
            return []
        
        # Detect edges
        padded = np.concatenate(([0], active.astype(np.int8), [0]))
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        
        # Merge close regions
        merge_gap = max(0, int(fs * float(cfg.merge_gap_ms) / 1000.0))
        merged: list[tuple[int, int]] = []
        for start, end in zip(starts, ends):
            si, ei = int(start), int(end)
            if merged and si - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], ei)
            else:
                merged.append((si, ei))
        
        # Extract impulses
        min_len = max(1, int(fs * float(cfg.min_duration_ms) / 1000.0))
        pre = max(0, int(fs * float(cfg.pre_pad_ms) / 1000.0))
        post = max(0, int(fs * float(cfg.post_pad_ms) / 1000.0))
        fmax = min(float(cfg.fmax_hz) if cfg.fmax_hz else fs / 2.0, fs / 2.0)
        
        impulses: list[DetectedImpulse] = []
        last_peak_s: float | None = None
        
        for raw_start, raw_end in merged:
            if raw_end - raw_start < min_len:
                continue
            
            start_i = max(0, raw_start - pre)
            end_i = min(x.size, raw_end + post)
            if end_i <= start_i:
                continue
            
            segment = x[start_i:end_i]
            peak_rel = int(np.argmax(np.abs(segment)))
            peak_i = start_i + peak_rel
            rms = float(np.sqrt(np.mean(np.square(segment))))
            
            dom_hz = self._estimate_dominant_frequency(
                segment, fs, cfg.fmin_hz, fmax
            )
            
            peak_s = peak_i / fs
            interval = math.nan if last_peak_s is None else (peak_s - last_peak_s) * 1000.0
            last_peak_s = peak_s
            
            impulses.append(DetectedImpulse(
                index=len(impulses) + 1,
                start_sample=start_i,
                end_sample=end_i,
                start_s=start_i / fs,
                end_s=end_i / fs,
                duration_ms=(end_i - start_i) / fs * 1000.0,
                peak_time_s=peak_s,
                peak_amplitude=float(np.max(np.abs(segment))),
                rms=rms,
                dominant_freq_hz=dom_hz,
                interval_ms=interval,
                threshold_used=threshold,
            ))
        
        return impulses
```

---

## PART 5: PYINSTALLER PACKAGING & COMMERCIAL FREEZE SPECIFICATION

### 5.1 PyInstaller `.spec` File

```python
# medaudiolab.spec
# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for MedAudioLab v2.
Build command:
    pyinstaller medaudiolab.spec --noconfirm --clean
"""
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None
ROOT = Path(SPECPATH)

# Hidden imports for dynamic imports that PyInstaller can't detect
hidden_imports = [
    # PySide6
    *collect_submodules('PySide6'),
    # scipy submodules used dynamically
    'scipy.signal',
    'scipy.fft',
    'scipy.io',
    'scipy.io.wavfile',
    # sounddevice / PortAudio
    'sounddevice',
    '_sounddevice_data',
    # pyserial
    'serial.tools.list_ports',
    'serial.tools.list_ports_windows',
    'serial.tools.list_ports_common',
    # pyqtgraph
    *collect_submodules('pyqtgraph'),
    # numpy internals
    'numpy.core._methods',
    'numpy.lib.format',
]

# Data files to bundle
datas = [
    # Wavelet scales configuration
    (str(ROOT / 'resources' / 'wavelet_scales.txt'), 'resources'),
    # Qt stylesheet
    (str(ROOT / 'resources' / 'styles'), 'resources/styles'),
    # App icons
    (str(ROOT / 'resources' / 'icons'), 'resources/icons'),
    # sounddevice PortAudio DLLs
    *collect_data_files('sounddevice'),
    *collect_data_files('_sounddevice_data'),
    # soundfile libsndfile DLLs
    *collect_data_files('soundfile'),
]

a = Analysis(
    [str(ROOT / 'run.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', '_tkinter',  # No longer needed (PySide6 replaces Tkinter)
        'test', 'unittest',
        'IPython', 'jupyter',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MedAudioLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windowed application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / 'resources' / 'icons' / 'app_icon.ico'),
)
```

### 5.2 Runtime Resource Locator

```python
# app/resource_path.py
"""
Runtime resource locator supporting both development mode and frozen PyInstaller bundles.

In development: resources are at project_root/resources/
In frozen mode: resources are at sys._MEIPASS/resources/
"""
from __future__ import annotations

import sys
from pathlib import Path


def _get_base_path() -> Path:
    """Get the base path for resource resolution."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        return Path(sys._MEIPASS)
    else:
        # Running in development
        return Path(__file__).resolve().parent.parent


def get_resource_path(relative_path: str) -> Path:
    """Resolve a resource path that works in both dev and frozen modes.
    
    Args:
        relative_path: Path relative to the resources/ directory.
                       Example: "wavelet_scales.txt" or "styles/light.qss"
    
    Returns:
        Absolute Path to the resource file.
    
    Raises:
        FileNotFoundError: If the resource doesn't exist.
    """
    base = _get_base_path()
    full_path = base / "resources" / relative_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"Resource not found: {full_path}\n"
            f"Base path: {base}\n"
            f"Frozen: {getattr(sys, 'frozen', False)}"
        )
    return full_path


def get_resource_dir() -> Path:
    """Get the resources directory root."""
    return _get_base_path() / "resources"


def is_frozen() -> bool:
    """Check if running as a PyInstaller bundle."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
```

### 5.3 Platform Compatibility Guidelines

| Concern | Windows 10/11 x64 | macOS | Linux |
|---|---|---|---|
| **Audio Backend** | WASAPI (default), DirectSound, ASIO (if DLL present) | CoreAudio | ALSA, PulseAudio |
| **PortAudio DLL** | Bundled via `sounddevice` data files | Bundled `.dylib` | System `libportaudio.so` |
| **libsndfile** | Bundled via `soundfile` data files | Bundled `.dylib` | System `libsndfile.so` |
| **Qt Platform Plugin** | `platforms/qwindows.dll` (auto-bundled by PyInstaller) | `platforms/libqcocoa.dylib` | `platforms/libqxcb.so` |
| **High DPI** | `QApplication.setHighDpiScaleFactorRoundingPolicy` | Automatic Retina support | `QT_AUTO_SCREEN_SCALE_FACTOR=1` |
| **COM/Serial** | `COMx` ports via `pyserial` | `/dev/tty.*` | `/dev/ttyUSB*`, `/dev/ttyACM*` |
| **ASIO Support** | Requires separate ASIO SDK DLL (not redistributable). User must install ASIO4ALL or vendor driver. | N/A | N/A |

> [!TIP]
> **WASAPI Exclusive Mode**: For lowest-latency recording at 192+ kHz on Windows, configure `sounddevice` with `extra_settings=sd.WasapiSettings(exclusive=True)`. This bypasses the Windows audio mixer but locks the device.
