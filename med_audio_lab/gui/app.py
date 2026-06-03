from __future__ import annotations

import subprocess
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
import numpy as np

# Импортируем из наших новых пакетов:
from ..backend import analysis
from ..backend.storage import (
    APP_DIR,
    RECORDS_DIR,
    RecordingInfo,
    add_recording,
    ensure_storage,
    export_report_csv,
    get_recording,
    get_save_dir,
    list_recordings,
    save_metrics,
    save_signal_csv,
    save_wav,
    set_setting,
)
from ..dsp.wavelet import save_longitudinal_slice_txt
from ..utils.io import read_wav
from ..utils.txt_converter import txt_to_wav
from ..hardware.microphone import AudioUnavailableError, Player, Recorder, first_input_samplerate, input_devices
from ..hardware.serial_port import SerialRecorder, list_serial_ports

# Импортируем локальную логику графики
from .plotting import AMBER, BLUE, GREEN, RED, VIOLET, CanvasPlot, plot_longitudinal_slice_morlet_re


class MedAudioLabApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        ensure_storage()
        self.title("MedAudioLab - запись и анализ звука")
        self.geometry("1280x820")
        self.minsize(1060, 680)

        self.recorder = Recorder()
        self.serial_recorder = SerialRecorder()
        self.player = Player()
        self.source_var = tk.StringVar(value="microphone")
        self.current_data = np.zeros(0, dtype=np.float32)
        self.current_raw_data = np.zeros(0, dtype=np.float32)
        self.current_samplerate = 44100
        self.current_record_id: int | None = None
        self.current_path: Path | None = None
        self.record_tone_hz: float | None = None
        self.current_metrics: dict[str, str] = {}
        self.current_Wx: np.ndarray | None = None
        self.current_cwt_t: np.ndarray | None = None
        self.current_cwt_freqs: np.ndarray | None = None
        self._cwt_cache: dict | None = None
        self.device_by_label = {}
        self.serial_by_label = {}
        self.save_dir = get_save_dir()
        self._bg_busy = False

        self._configure_style()
        self._build_ui()
        self.refresh_devices()
        self.refresh_serial_ports()
        self.refresh_library()
        self.after(120, self._live_tick)

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=("Segoe UI", 10), background="#f6f8fb", foreground="#1f2937")
        style.configure("TFrame", background="#f6f8fb")
        style.configure("TLabelframe", background="#f6f8fb")
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), background="#f6f8fb")
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 18), background="#f6f8fb")
        style.configure("Muted.TLabel", foreground="#64748b", background="#f6f8fb")
        style.configure("Strong.TLabel", font=("Segoe UI Semibold", 10), background="#f6f8fb")
        style.configure("PlotTitle.TLabel", font=("Segoe UI Semibold", 12), background="#f6f8fb")
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10), padding=(14, 8))
        style.configure("Danger.TButton", font=("Segoe UI Semibold", 10), padding=(14, 8))
        style.configure("Wide.TButton", padding=(12, 7))
        style.map("Primary.TButton", background=[("active", "#dbeafe")])
        style.map("Danger.TButton", background=[("active", "#fee2e2")])
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 9))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="MedAudioLab", style="Header.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="запись, прослушивание, хранение и понятный спектральный анализ",
            style="Muted.TLabel",
        ).pack(side="left", padx=(12, 0), pady=(7, 0))

        self.status = tk.StringVar(value=f"Готово. База данных: {APP_DIR}")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, pady=(12, 8))

        self.record_tab = ttk.Frame(self.notebook, padding=10)
        self.analysis_tab = ttk.Frame(self.notebook, padding=10)
        self.library_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.record_tab, text="Запись")
        self.notebook.add(self.analysis_tab, text="Анализ")
        self.notebook.add(self.library_tab, text="База записей")

        self._build_record_tab()
        self._build_analysis_tab()
        self._build_library_tab()

        ttk.Label(root, textvariable=self.status, style="Muted.TLabel").pack(fill="x")

    def _build_record_tab(self) -> None:
        action_bar = ttk.LabelFrame(self.record_tab, text="Основные действия", padding=10)
        action_bar.pack(fill="x", pady=(0, 10))

        self.record_btn = ttk.Button(action_bar, text="Начать запись", command=self.start_recording, style="Primary.TButton")
        self.record_btn.pack(side="left", padx=(0, 8))
        self.pause_btn = ttk.Button(action_bar, text="Пауза", command=self.toggle_pause, state="disabled", style="Wide.TButton")
        self.pause_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ttk.Button(action_bar, text="Остановить и сохранить", command=self.stop_recording, state="disabled", style="Danger.TButton")
        self.stop_btn.pack(side="left", padx=(0, 16))
        self.play_btn = ttk.Button(action_bar, text="Прослушать", command=self.play_current, state="disabled", style="Wide.TButton")
        self.play_btn.pack(side="left", padx=(0, 8))
        ttk.Button(action_bar, text="Остановить звук", command=self.stop_playback, style="Wide.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(action_bar, text="Загрузить файл", command=self.open_wav, style="Primary.TButton").pack(side="left")

        level_box = ttk.Frame(action_bar)
        level_box.pack(side="right", fill="x", expand=True, padx=(16, 0))
        self.record_state = tk.StringVar(value="Ожидание")
        ttk.Label(level_box, textvariable=self.record_state, style="Strong.TLabel").pack(anchor="e")
        self.level_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(level_box, variable=self.level_var, maximum=100).pack(fill="x", pady=(4, 0))

        body = ttk.Frame(self.record_tab)
        body.pack(fill="both", expand=True)
        left_holder = ttk.Frame(body, width=390)
        left_holder.pack(side="left", fill="y", padx=(0, 12))
        left_holder.pack_propagate(False)
        left_canvas = tk.Canvas(left_holder, background="#f6f8fb", highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_holder, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)
        left = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _resize_left_panel(_event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfigure(left_window, width=left_canvas.winfo_width())

        left.bind("<Configure>", _resize_left_panel)
        left_canvas.bind("<Configure>", _resize_left_panel)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        source_box = ttk.LabelFrame(left, text="Источник сигнала", padding=10)
        source_box.pack(fill="x")
        ttk.Radiobutton(
            source_box,
            text="Обычный микрофон",
            variable=self.source_var,
            value="microphone",
            command=self.on_source_changed,
        ).pack(anchor="w")
        ttk.Radiobutton(
            source_box,
            text="COM-порт / внешний датчик",
            variable=self.source_var,
            value="serial",
            command=self.on_source_changed,
        ).pack(anchor="w", pady=(4, 0))
        self.source_hint = tk.StringVar(value="Будет записываться звук с выбранного микрофона.")
        ttk.Label(source_box, textvariable=self.source_hint, style="Muted.TLabel", wraplength=320).pack(fill="x", pady=(8, 0))

        device_box = ttk.LabelFrame(left, text="Микрофон", padding=10)
        device_box.pack(fill="x", pady=(10, 0))
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_box, textvariable=self.device_var, width=42, state="readonly")
        self.device_combo.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(device_box)
        row.pack(fill="x")
        ttk.Button(row, text="Обновить", command=self.refresh_devices).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(row, text="Проверить", command=self.check_microphone, style="Primary.TButton").pack(side="left", fill="x", expand=True)
        self.device_status = tk.StringVar(value="Микрофон еще не проверен")
        ttk.Label(device_box, textvariable=self.device_status, style="Muted.TLabel", wraplength=320).pack(fill="x", pady=(8, 0))

        serial_box = ttk.LabelFrame(left, text="COM-порт", padding=10)
        serial_box.pack(fill="x", pady=(10, 0))
        self.serial_var = tk.StringVar()
        self.serial_combo = ttk.Combobox(serial_box, textvariable=self.serial_var, width=42, state="readonly")
        self.serial_combo.pack(fill="x", pady=(0, 8))
        serial_row = ttk.Frame(serial_box)
        serial_row.pack(fill="x")
        ttk.Button(serial_row, text="Обновить", command=self.refresh_serial_ports).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(serial_row, text="Проверить", command=self.check_serial_port, style="Primary.TButton").pack(side="left", fill="x", expand=True)
        baud_row = ttk.Frame(serial_box)
        baud_row.pack(fill="x", pady=(8, 0))
        ttk.Label(baud_row, text="Скорость").pack(side="left")
        self.baud_var = tk.StringVar(value="921600")
        ttk.Combobox(
            baud_row,
            textvariable=self.baud_var,
            values=("9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"),
            width=10,
        ).pack(side="left", padx=(8, 14))
        ttk.Label(baud_row, text="Гц для анализа").pack(side="left")
        self.serial_rate_var = tk.StringVar(value="200000")
        ttk.Entry(baud_row, textvariable=self.serial_rate_var, width=8).pack(side="left", padx=(8, 0))
        self.serial_status = tk.StringVar(value="COM-порт еще не проверен")
        ttk.Label(serial_box, textvariable=self.serial_status, style="Muted.TLabel", wraplength=320).pack(fill="x", pady=(8, 0))

        save_box = ttk.LabelFrame(left, text="Куда сохранять WAV", padding=10)
        save_box.pack(fill="x", pady=(10, 0))
        self.save_dir_var = tk.StringVar(value=str(self.save_dir))
        save_entry = ttk.Entry(save_box, textvariable=self.save_dir_var)
        save_entry.pack(fill="x", pady=(0, 8))
        save_row = ttk.Frame(save_box)
        save_row.pack(fill="x")
        ttk.Button(save_row, text="Выбрать папку", command=self.choose_save_dir).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(save_row, text="Открыть", command=lambda: self.open_folder(self.save_dir)).pack(side="left", fill="x", expand=True)

        meta_box = ttk.LabelFrame(left, text="Карточка записи", padding=10)
        meta_box.pack(fill="x", pady=(10, 0))
        ttk.Label(meta_box, text="Название").pack(anchor="w")
        self.title_entry = ttk.Entry(meta_box)
        self.title_entry.insert(0, "Аудиозапись")
        self.title_entry.pack(fill="x", pady=(0, 8))
        ttk.Label(meta_box, text="Пациент / образец").pack(anchor="w")
        self.patient_entry = ttk.Entry(meta_box)
        self.patient_entry.pack(fill="x", pady=(0, 8))
        ttk.Label(meta_box, text="Комментарий").pack(anchor="w")
        self.notes_text = tk.Text(meta_box, height=4, wrap="word", font=("Segoe UI", 10), relief="solid", bd=1)
        self.notes_text.pack(fill="x")

        hints = ttk.LabelFrame(left, text="Подсказки", padding=10)
        hints.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(
            hints,
            text=(
                "1. Проверьте микрофон.\n"
                "2. Выберите папку сохранения.\n"
                "3. Нажмите «Начать запись».\n"
                "4. После остановки файл появится в базе."
            ),
            style="Muted.TLabel",
            justify="left",
            wraplength=320,
        ).pack(anchor="nw")

        self.record_plot = CanvasPlot(right, "Живая форма сигнала")
        self.record_plot.pack(fill="both", expand=True)
        self.record_plot.clear("Живая форма сигнала", "Нажмите «Начать запись»")

    def _build_analysis_tab(self) -> None:
        file_row = ttk.Frame(self.analysis_tab)
        file_row.pack(fill="x", pady=(0, 6))
        ttk.Button(file_row, text="Загрузить файл", command=self.open_wav, style="Primary.TButton").pack(side="left")
        ttk.Button(file_row, text="TXT → WAV", command=self.on_txt_to_wav, style="Wide.TButton").pack(side="left", padx=(8, 0))
        ttk.Label(file_row, text="Morlet f (Гц):", style="Muted.TLabel").pack(side="left", padx=(16, 4))
        self.morlet_target_var = tk.StringVar(value="10000")
        ttk.Entry(file_row, textvariable=self.morlet_target_var, width=10).pack(side="left")
        ttk.Label(file_row, text="f от (Гц):", style="Muted.TLabel").pack(side="left", padx=(12, 4))
        self.morlet_fmin_var = tk.StringVar(value="")
        ttk.Entry(file_row, textvariable=self.morlet_fmin_var, width=8).pack(side="left")
        ttk.Label(file_row, text="f до (Гц):", style="Muted.TLabel").pack(side="left", padx=(8, 4))
        self.morlet_fmax_var = tk.StringVar(value="")
        ttk.Entry(file_row, textvariable=self.morlet_fmax_var, width=8).pack(side="left")

        top = ttk.Frame(self.analysis_tab)
        top.pack(fill="x")
        for text, command in [
            ("Форма", self.show_waveform),
            ("Фурье", self.show_fft),
            ("АЧХ", self.show_response),
            ("Спектрограмма (STFT)", self.show_spectrogram),
            ("Вейвлет Морле", self.show_wavelet),
            ("Срез вейвлета", self.show_wavelet_slice),
            ("Автокорреляция", self.show_autocorrelation),
            ("Огибающая", self.show_envelope),
        ]:
            ttk.Button(top, text=text, command=command, style="Wide.TButton").pack(side="left", padx=(0, 6))

        body = ttk.Frame(self.analysis_tab)
        body.pack(fill="both", expand=True, pady=(10, 0))
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right_holder = ttk.Frame(body, width=360)
        right_holder.pack(side="left", fill="y")
        right_holder.pack_propagate(False)
        right_canvas = tk.Canvas(right_holder, background="#f6f8fb", highlightthickness=0)
        right_scroll = ttk.Scrollbar(right_holder, orient="vertical", command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_scroll.pack(side="right", fill="y")
        right_canvas.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(right_canvas)
        right_window = right_canvas.create_window((0, 0), window=right, anchor="nw")

        def _resize_analysis_side(_event=None):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
            right_canvas.itemconfigure(right_window, width=right_canvas.winfo_width())

        right.bind("<Configure>", _resize_analysis_side)
        right_canvas.bind("<Configure>", _resize_analysis_side)

        self.analysis_plot = CanvasPlot(left, "Анализ")
        self.analysis_plot.pack(fill="both", expand=True)
        self.analysis_plot.clear("Анализ", "Запишите или откройте файл")
        self.analysis_plot.set_spec_click_handler(self._on_spectrogram_click)

        metrics_box = ttk.LabelFrame(right, text="Показатели", padding=8)
        metrics_box.pack(fill="x")
        self.metrics_tree = ttk.Treeview(metrics_box, columns=("value",), show="tree headings", height=8)
        self.metrics_tree.heading("#0", text="Параметр")
        self.metrics_tree.heading("value", text="Значение")
        self.metrics_tree.column("#0", width=165, anchor="w")
        self.metrics_tree.column("value", width=140, anchor="w")
        self.metrics_tree.pack(fill="both", expand=True)

        transform_box = ttk.LabelFrame(right, text="Преобразования", padding=8)
        transform_box.pack(fill="x", pady=(10, 0))
        buttons = [
            ("Нормализовать", lambda: self.apply_transform("normalize")),
            ("Убрать DC", lambda: self.apply_transform("dc")),
            ("Шумовой порог", lambda: self.apply_transform("gate")),
            ("Обрезать тишину", lambda: self.apply_transform("trim")),
            ("НЧ-фильтр", lambda: self.apply_transform("lowpass")),
            ("ВЧ-фильтр", lambda: self.apply_transform("highpass")),
            ("Полосовой фильтр", lambda: self.apply_transform("bandpass")),
        ]
        for i, (text, command) in enumerate(buttons):
            ttk.Button(transform_box, text=text, command=command).grid(row=i // 2, column=i % 2, sticky="ew", padx=3,
                                                                       pady=3)
        transform_box.columnconfigure(0, weight=1)
        transform_box.columnconfigure(1, weight=1)
        ttk.Button(transform_box, text="Вернуть исходный", command=self.restore_original).grid(row=8, column=0,
                                                                                               columnspan=2,
                                                                                               sticky="ew", padx=3,
                                                                                               pady=3)
        ttk.Button(transform_box, text="Обрезать по выделению", command=self.crop_to_selection,
                   style="Primary.TButton").grid(row=7, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        ttk.Button(transform_box, text="Сохранить как новую запись", command=self.save_current_as_new,
                   style="Primary.TButton").grid(row=4, column=0, columnspan=2, sticky="ew", padx=3, pady=(8, 3))
        ttk.Button(transform_box, text="Экспорт отчета CSV", command=self.export_report).grid(row=5, column=0,
                                                                                              columnspan=2, sticky="ew",
                                                                                              padx=3, pady=3)
        ttk.Button(transform_box, text="Экспорт гребня CWT", command=self.export_cwt_ridge).grid(row=6, column=0,
                                                                                                 columnspan=2,
                                                                                                 sticky="ew", padx=3,
                                                                                                 pady=3)

    def _build_library_tab(self) -> None:
        top = ttk.Frame(self.library_tab)
        top.pack(fill="x")
        ttk.Button(top, text="Обновить", command=self.refresh_library).pack(side="left")
        ttk.Button(top, text="Загрузить выбранную", command=self.load_selected_recording, style="Primary.TButton").pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Текущая папка сохранения", command=lambda: self.open_folder(self.save_dir)).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Папка базы", command=lambda: self.open_folder(APP_DIR)).pack(side="left", padx=(8, 0))

        columns = ("date", "title", "patient", "duration", "samplerate", "path")
        self.library_tree = ttk.Treeview(self.library_tab, columns=columns, show="headings")
        headings = {
            "date": "Дата",
            "title": "Название",
            "patient": "Пациент/образец",
            "duration": "Длительность",
            "samplerate": "Гц",
            "path": "Файл",
        }
        widths = {"date": 140, "title": 190, "patient": 160, "duration": 95, "samplerate": 70, "path": 450}
        for col in columns:
            self.library_tree.heading(col, text=headings[col])
            self.library_tree.column(col, width=widths[col], anchor="w")
        self.library_tree.pack(fill="both", expand=True, pady=(10, 0))
        self.library_tree.bind("<Double-1>", lambda _event: self.load_selected_recording())

    def choose_save_dir(self) -> None:
        path = filedialog.askdirectory(title="Выберите папку для WAV-записей", initialdir=str(self.save_dir))
        if not path:
            return
        self.save_dir = Path(path)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir_var.set(str(self.save_dir))
        set_setting("save_dir", str(self.save_dir))
        self.status.set(f"Папка сохранения изменена: {self.save_dir}")

    def on_source_changed(self) -> None:
        if self.source_var.get() == "serial":
            self.source_hint.set("Будут читаться числовые отсчеты из COM-порта. Каждая строка должна содержать число.")
            self.play_btn.configure(state="normal" if self.current_data.size else "disabled")
        else:
            self.source_hint.set("Будет записываться звук с выбранного микрофона.")

    def refresh_serial_ports(self) -> None:
        try:
            ports = list_serial_ports()
        except Exception as exc:
            self.serial_combo["values"] = []
            self.serial_status.set(f"Не удалось получить список COM-портов: {exc}")
            return
        self.serial_by_label = {port.label: port for port in ports}
        labels = list(self.serial_by_label.keys())
        self.serial_combo["values"] = labels
        if labels and not self.serial_var.get():
            self.serial_var.set(labels[0])
        self.serial_status.set(f"Найдено COM-портов: {len(labels)}" if labels else "COM-порты не найдены")

    def selected_serial_port(self):
        label = self.serial_var.get()
        return self.serial_by_label.get(label)

    def check_serial_port(self) -> None:
        port = self.selected_serial_port()
        if not port:
            messagebox.showwarning("COM-порт", "COM-порт не выбран или не найден.")
            return
        try:
            baudrate = int(self.baud_var.get())
            samplerate = int(float(self.serial_rate_var.get()))
            if baudrate <= 0 or samplerate <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("COM-порт", "Проверьте скорость COM-порта и частоту анализа.")
            return
        self.serial_status.set(
            f"Выбран {port.device}; UART {baudrate} бод; анализ {samplerate} Гц\n"
            "STM32: бинарный поток uint16 LE или текстовые строки с числом"
        )
        self.status.set("COM-порт выбран (STM32 / USB-UART)")

    def refresh_devices(self) -> None:
        try:
            devices = input_devices()
        except Exception as exc:
            self.device_combo["values"] = []
            self.device_status.set(f"Не удалось получить список микрофонов: {exc}")
            self.status.set("Аудиоустройства недоступны")
            return
        self.device_by_label = {dev.label: dev for dev in devices}
        labels = list(self.device_by_label.keys())
        self.device_combo["values"] = labels
        if labels and not self.device_var.get():
            self.device_var.set(labels[0])
            self.current_samplerate = first_input_samplerate(devices)
        self.device_status.set(f"Найдено микрофонов: {len(labels)}" if labels else "Микрофоны не найдены")
        self.status.set("Список аудиоустройств обновлен")

    def selected_device(self):
        label = self.device_var.get()
        return self.device_by_label.get(label)

    def check_microphone(self) -> None:
        dev = self.selected_device()
        if not dev:
            messagebox.showwarning("Микрофон", "Микрофон не выбран или не найден.")
            return
        self.current_samplerate = dev.default_samplerate or 44100
        self.device_status.set(
            f"Микрофон виден: {dev.name}\n"
            f"Каналов: {dev.input_channels}; частота: {self.current_samplerate} Гц"
        )
        self.status.set("Микрофон виден и выбран")

    def start_recording(self) -> None:
        self.save_dir = Path(self.save_dir_var.get()).expanduser()
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            set_setting("save_dir", str(self.save_dir))
            if self.source_var.get() == "serial":
                port = self.selected_serial_port()
                if not port:
                    messagebox.showwarning("Запись", "Сначала выберите COM-порт.")
                    return
                baudrate = int(self.baud_var.get())
                self.current_samplerate = int(float(self.serial_rate_var.get()))
                if baudrate <= 0 or self.current_samplerate <= 0:
                    raise ValueError("Скорость и частота анализа должны быть положительными.")
                self.serial_recorder.start(port.device, baudrate)
                self.record_state.set("Идет запись с COM")
                self.status.set(f"Идет чтение COM-порта {port.device}")
            else:
                dev = self.selected_device()
                if not dev:
                    messagebox.showwarning("Запись", "Сначала выберите микрофон.")
                    return
                self.current_samplerate = dev.default_samplerate or 44100
                self.recorder.start(dev.index, self.current_samplerate)
                self.record_state.set("Идет запись")
                self.status.set("Идет запись")
        except Exception as exc:
            messagebox.showerror("Запись", f"Не удалось начать запись:\n{exc}")
            return
        self.current_data = np.zeros(0, dtype=np.float32)
        self.current_raw_data = np.zeros(0, dtype=np.float32)
        self.record_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal", text="Пауза")
        self.stop_btn.configure(state="normal")
        self.play_btn.configure(state="disabled")
        self.record_plot.clear("Живая форма сигнала", "Запись началась")

    def toggle_pause(self) -> None:
        active = self.serial_recorder if self.serial_recorder.is_recording else self.recorder
        if not active.is_recording:
            return
        if active.is_paused:
            active.resume()
            self.pause_btn.configure(text="Пауза")
            self.record_state.set("Идет запись")
            self.status.set("Запись продолжена")
        else:
            active.pause()
            self.pause_btn.configure(text="Продолжить")
            self.record_state.set("Пауза")
            self.status.set("Запись на паузе")

    def stop_recording(self) -> None:
        is_serial = self.serial_recorder.is_recording
        raw_data = self.serial_recorder.stop() if is_serial else self.recorder.stop()

        if is_serial:
            actual = self.serial_recorder.actual_samplerate
            self.serial_status.set(f"Скорость USB: ~{actual} Гц | Частота для анализа: {self.current_samplerate} Гц")

        self.record_btn.configure(state="normal")
        self.pause_btn.configure(state="disabled", text="Пауза")
        self.stop_btn.configure(state="disabled")
        self.level_var.set(0)
        self.record_state.set("Ожидание")

        min_samples = 3 if is_serial else max(256, self.current_samplerate // 20)
        if raw_data.size < min_samples:
            self.status.set("Запись остановлена, данных слишком мало для сохранения")
            messagebox.showinfo("Запись", "Данных слишком мало, файл не сохранен.")
            return

        data = analysis.normalize(raw_data) if is_serial else raw_data
        self.current_data = data
        self.current_raw_data = raw_data
        title = self.title_entry.get().strip() or "Аудиозапись"
        patient = self.patient_entry.get().strip()
        notes = self.notes_text.get("1.0", "end").strip()

        if is_serial:
            csv_path = save_signal_csv(raw_data, self.current_samplerate, title, directory=self.save_dir,
                                       suffix="com_raw")
            notes = f"{notes}\nCOM samplerate: {self.current_samplerate} Гц\nCOM CSV: {csv_path}".strip()

        path = save_wav(data, self.current_samplerate, title, directory=self.save_dir)
        duration = data.size / self.current_samplerate
        record_id = add_recording(path, title, patient, notes, self.current_samplerate, duration)

        self.current_path = path
        self.current_record_id = record_id
        self.record_tone_hz = analysis.resolve_record_tone_hz(path, data, self.current_samplerate)
        if self.record_tone_hz is not None:
            thz = self.record_tone_hz
            self.morlet_target_var.set(str(int(thz)) if abs(thz - round(thz)) < 1.0 else f"{thz:.1f}")
        self.current_metrics = analysis.metrics(data, self.current_samplerate)
        save_metrics(record_id, self.current_metrics)

        self.update_metrics_view()
        self.record_plot.draw_waveform(data, self.current_samplerate, "Сохраненная форма сигнала")
        self.analysis_plot.draw_waveform(data, self.current_samplerate, "Форма сигнала")
        self.play_btn.configure(state="normal")
        self.refresh_library()
        if self.record_tone_hz is not None:
            self.status.set(f"Запись сохранена: {path} | тон {self.record_tone_hz:.0f} Гц")
        else:
            self.status.set(f"Запись сохранена: {path}")
        self.notebook.select(self.analysis_tab)

    def play_current(self) -> None:
        if self.current_data.size == 0:
            messagebox.showinfo("Прослушивание", "Нет текущей записи.")
            return
        try:
            self.player.play(self.current_data, self.current_samplerate)
        except (AudioUnavailableError, Exception) as exc:
            messagebox.showerror("Прослушивание", f"Не удалось воспроизвести звук:\n{exc}")
            return
        self.status.set("Воспроизведение")

    def stop_playback(self) -> None:
        self.player.stop()
        self.status.set("Воспроизведение остановлено")

    @staticmethod
    def _format_error(exc: BaseException) -> str:
        text = str(exc).strip()
        if text and not text.startswith("<Axes:"):
            return text
        return f"{type(exc).__name__}: сбой отрисовки (попробуйте «Форма» и снова построить график)"

    def _destroy_span_selector(self) -> None:
        span = getattr(self, "span", None)
        if span is None:
            return
        try:
            span.disconnect_events()
        except Exception:
            pass
        self.span = None

    def _recover_analysis_plot(self) -> None:
        try:
            self._destroy_span_selector()
            self.analysis_plot._rebuild_axes()
        except Exception:
            pass

    def _safe_analysis_plot(self, action) -> None:
        try:
            self._destroy_span_selector()
            action()
        except Exception as exc:
            traceback.print_exc()
            self._recover_analysis_plot()
            messagebox.showerror("Построение графика", self._format_error(exc))
            self.status.set("Ошибка построения графика.")

    def _run_background(self, status_text: str, worker, on_success) -> None:
        if self._bg_busy:
            messagebox.showinfo("Подождите", "Уже выполняется тяжёлый расчёт. Дождитесь окончания.")
            return
        self._bg_busy = True
        self.status.set(status_text)
        self.update_idletasks()

        def thread_main() -> None:
            err: Exception | None = None
            result = None
            try:
                result = worker()
            except Exception as exc:
                err = exc

            def finish() -> None:
                self._bg_busy = False
                if err is not None:
                    traceback.print_exc()
                    messagebox.showerror("Ошибка", self._format_error(err))
                    self.status.set("Ошибка расчёта (программа продолжает работу).")
                    return
                try:
                    on_success(result)
                except Exception as exc:
                    traceback.print_exc()
                    self._recover_analysis_plot()
                    messagebox.showerror("Ошибка отображения", self._format_error(exc))
                    self.status.set("Ошибка отображения графика.")

            self.after(0, finish)

        threading.Thread(target=thread_main, daemon=True).start()

    def open_wav(self) -> None:
        path = filedialog.askopenfilename(
            title="Загрузить файл",
            initialdir=str(self.save_dir if self.save_dir.exists() else RECORDS_DIR),
            filetypes=[("Аудио и Данные", "*.wav *.txt *.csv"), ("WAV", "*.wav"), ("TXT/CSV", "*.txt *.csv"),
                       ("Все файлы", "*.*")],
        )
        if not path:
            return
        self.load_wav_path(Path(path))

    def load_wav_path(self, path: Path, record_id: int | None = None) -> None:
        try:
            if path.suffix.lower() in [".txt", ".csv"]:
                samplerate, data = analysis.load_txt_as_audio(str(path))
            else:
                samplerate, data = read_wav(path)
        except Exception as exc:
            messagebox.showerror("Открыть файл", f"Не удалось открыть файл:\n{exc}")
            return

        self.current_samplerate = samplerate
        self.current_data = data
        self.original_samplerate = samplerate
        self.original_data = data.copy()

        self.current_path = path
        self.current_record_id = record_id
        self.record_tone_hz = analysis.resolve_record_tone_hz(path, data, samplerate)
        if self.record_tone_hz is not None:
            thz = self.record_tone_hz
            self.morlet_target_var.set(str(int(thz)) if abs(thz - round(thz)) < 1.0 else f"{thz:.1f}")
        self._cwt_cache = None
        self.current_Wx = None
        self.current_cwt_t = None
        self.current_cwt_freqs = None
        self.current_metrics = analysis.metrics(data, samplerate)
        if record_id:
            save_metrics(record_id, self.current_metrics)
        self.update_metrics_view()
        self.play_btn.configure(state="normal")
        self.record_plot.draw_waveform(data, samplerate, "Загруженная форма сигнала")
        self.analysis_plot.draw_waveform(data, samplerate, "Форма сигнала")
        if self.record_tone_hz is not None:
            self.status.set(f"Загружен файл: {path} | тон {self.record_tone_hz:.0f} Гц")
        else:
            self.status.set(f"Загружен файл: {path}")
        self.notebook.select(self.analysis_tab)

    def require_audio(self) -> bool:
        if self.current_data.size == 0:
            messagebox.showinfo("Анализ", "Сначала запишите или откройте WAV-файл.")
            return False
        return True

    def show_waveform(self) -> None:
        if not self.require_audio():
            return

        metrics = analysis.metrics(self.current_data, self.current_samplerate)
        peak = float(np.max(np.abs(self.current_data))) if self.current_data.size else 0.0
        peak_idx = int(np.argmax(np.abs(self.current_data))) if self.current_data.size else 0
        peak_t = peak_idx / self.current_samplerate
        note = f"RMS: {metrics.get('RMS', '-')}\nПик: {peak:.4f}"
        annotations = [{"x": peak_t, "y": self.current_data[peak_idx], "label": f"пик {peak:.3f}", "color": RED}]

        self.analysis_plot.draw_waveform(self.current_data, self.current_samplerate,
                                         "Форма сигнала (выделите фрагмент мышью)")

        def on_select(xmin, xmax):
            self.selected_range = (xmin, xmax)
            self.status.set(f"Выделен фрагмент: {xmin:.4f} с — {xmax:.4f} с. Нажмите «Обрезать по выделению».")

        self.span = SpanSelector(
            self.analysis_plot.ax, on_select, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='red'), interactive=True
        )

    def crop_to_selection(self) -> None:
        if not self.require_audio():
            return

        if not hasattr(self, 'selected_range') or self.selected_range is None:
            messagebox.showinfo("Обрезка", "Сначала выделите нужный фрагмент левой кнопкой мыши на графике «Форма».")
            return

        xmin, xmax = self.selected_range
        start_idx = max(0, int(xmin * self.current_samplerate))
        end_idx = min(self.current_data.size, int(xmax * self.current_samplerate))

        if end_idx <= start_idx:
            return

        self.current_data = self.current_data[start_idx:end_idx]
        self.selected_range = None

        self.current_metrics = analysis.metrics(self.current_data, self.current_samplerate)
        self.update_metrics_view()
        self.show_waveform()
        self.status.set(f"Сигнал успешно обрезан до интервала {xmin:.4f} - {xmax:.4f} с.")

    def restore_original(self) -> None:
        if not hasattr(self, 'original_data') or self.original_data is None:
            return

        self.current_data = self.original_data.copy()
        self.current_samplerate = self.original_samplerate
        self.selected_range = None

        self.current_metrics = analysis.metrics(self.current_data, self.current_samplerate)
        self.update_metrics_view()
        self.show_waveform()
        self.status.set("Восстановлен исходный файл (отменены все обрезки и фильтры).")

    def show_fft(self) -> None:
        if not self.require_audio():
            return

        def _draw() -> None:
            freqs, amp = analysis.fft_spectrum(self.current_data, self.current_samplerate)
            nyqist = self.current_samplerate / 2
            mask = (freqs >= 5000) & (freqs <= nyqist)
            freqs_plot = freqs[mask]
            amp_plot = amp[mask]
            annotations = []
            note = f"FFT (Chmi): окно Ханнинга, 5000–{int(nyqist)} Гц"
            if amp_plot.size > 1:
                peak_idx = int(np.argmax(amp_plot))
                peak_f = float(freqs_plot[peak_idx])
                peak_a = float(amp_plot[peak_idx])
                note += f"\nПик: {peak_f:.1f} Гц"
                annotations = [{"x": peak_f, "y": peak_a, "label": f"{peak_f:.1f} Гц", "color": RED}]
            self.analysis_plot.draw_xy(
                freqs_plot,
                amp_plot,
                f"Спектр Фурье (5000–{int(nyqist)} Гц)",
                "Частота (Гц)",
                "Амплитуда",
                GREEN,
                annotations=annotations,
                note=note,
            )

        self._safe_analysis_plot(_draw)

    def show_response(self) -> None:
        if not self.require_audio():
            return

        def _draw() -> None:
            freqs, amp = analysis.fft_spectrum(self.current_data, self.current_samplerate)
            mask = (freqs >= 5000) & (freqs <= (self.current_samplerate / 2))
            freqs_plot = freqs[mask]
            amp_plot = amp[mask]
            norm = amp_plot / max(float(np.max(amp_plot)), 1e-12)
            db = 20 * np.log10(np.maximum(norm, 1e-6))
            annotations = []
            note = "АЧХ нормирована\n0 dB = максимум записи"
            if db.size > 1:
                peak_idx = int(np.argmax(db))
                peak_f = float(freqs_plot[peak_idx])
                annotations.append({"x": peak_f, "y": float(db[peak_idx]), "label": f"0 dB @ {peak_f:.1f} Гц", "color": RED})
                active = np.where(db >= -3.0)[0]
                if active.size:
                    low_f = float(freqs_plot[int(active[0])])
                    high_f = float(freqs_plot[int(active[-1])])
                    annotations.append({"x": low_f, "y": -3.0, "label": f"-3 dB {low_f:.1f} Гц", "color": AMBER})
                    annotations.append({"x": high_f, "y": -3.0, "label": f"-3 dB {high_f:.1f} Гц", "color": AMBER})
                    note += f"\nПолоса -3 dB: {low_f:.1f}-{high_f:.1f} Гц"
            self.analysis_plot.draw_xy(
                freqs_plot, db, "АЧХ записи", "частота, Гц", "уровень, dB", BLUE,
                annotations=annotations, note=note,
            )

        self._safe_analysis_plot(_draw)

    def _render_stft_result(self, result: tuple) -> None:
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("STFT вернул некорректный результат.")
        self._destroy_span_selector()
        t, freqs_hz, S = result
        note = "Chmi: nperseg=2048, noverlap=1536\nЛКМ — боковой срез, ПКМ — продольный"
        if self._cwt_cache is not None:
            note += "\n(CWT в памяти — можно «Срез вейвлета»)"
        self.analysis_plot.draw_interactive_spec(
            t,
            freqs_hz,
            S,
            title="STFT спектрограмма",
            y_label="Частота (Гц)",
            spec_kind="STFT",
            log_freq=False,
            colorbar_label="dB",
            note=note,
        )
        self.analysis_plot.set_spec_click_handler(self._on_spectrogram_click)
        self.status.set("STFT построена.")

    def show_spectrogram(self) -> None:
        if not self.require_audio():
            return
        data = self.current_data.copy()
        fs = int(self.current_samplerate)

        def worker():
            return analysis.stft_spectrogram(
                data,
                fs,
                nperseg=2048,
                noverlap=1536,
                fmin_hz=5000.0,
                fmax_hz=100000.0,
                db=True,
            )

        self._run_background("STFT: расчёт...", worker, self._render_stft_result)

    def _morlet_tone_hz(self) -> float | None:
        """Частота тона записи: авто при загрузке, иначе поле Morlet f."""
        if self.record_tone_hz is not None and self.record_tone_hz > 0:
            return float(self.record_tone_hz)
        try:
            v = float(self.morlet_target_var.get().strip().replace(",", "."))
            return v if v > 0 else None
        except ValueError:
            return None

    def _morlet_display_freq_bounds(self) -> tuple[float | None, float | None]:
        def _parse(var: tk.StringVar) -> float | None:
            raw = var.get().strip().replace(",", ".")
            if not raw:
                return None
            try:
                v = float(raw)
            except ValueError:
                return None
            return v if v > 0 else None

        fmin = _parse(self.morlet_fmin_var)
        fmax = _parse(self.morlet_fmax_var)
        if fmin is not None and fmax is not None and fmin > fmax:
            fmin, fmax = fmax, fmin
        return fmin, fmax

    def _crop_cwt_for_display(
        self,
        freqs_hz: np.ndarray,
        Wx: np.ndarray,
        amp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float] | None]:
        freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        Wx = np.asarray(Wx)
        amp = np.asarray(amp, dtype=np.float32)
        fmin, fmax = self._morlet_display_freq_bounds()
        mask = np.ones(freqs_hz.shape, dtype=bool)
        if fmin is not None:
            mask &= freqs_hz >= fmin
        if fmax is not None:
            mask &= freqs_hz <= fmax
        if not np.any(mask):
            lo = fmin if fmin is not None else float(np.min(freqs_hz[freqs_hz > 0]))
            hi = fmax if fmax is not None else float(np.max(freqs_hz))
            raise ValueError(
                f"В диапазоне {lo:.0f}–{hi:.0f} Гц нет строк CWT. "
                "Ослабьте фильтр или пересчитайте вейвлет."
            )
        freqs_disp = freqs_hz[mask]
        Wx_disp = Wx[mask, :]
        amp_disp = amp[mask, :]
        pos = freqs_disp > 0
        if not np.any(pos):
            raise ValueError("Нет положительных частот в выбранном диапазоне.")
        y0 = float(fmin) if fmin is not None else float(np.min(freqs_disp[pos]))
        y1 = float(fmax) if fmax is not None else float(np.max(freqs_disp[pos]))
        freq_ylim = (y0, y1) if y0 < y1 else None
        return freqs_disp, Wx_disp, amp_disp, freq_ylim

    def _store_cwt_cache(self, t: np.ndarray, freqs_hz: np.ndarray, Wx: np.ndarray) -> None:
        self.current_Wx = Wx
        self.current_cwt_t = t
        self.current_cwt_freqs = freqs_hz
        self._cwt_cache = {
            "t": np.asarray(t, dtype=np.float64),
            "freqs_hz": np.asarray(freqs_hz, dtype=np.float64),
            "Wx": Wx,
        }

    def _get_cwt_cache(self) -> dict | None:
        if self._cwt_cache is not None:
            return self._cwt_cache
        if self.current_Wx is not None and self.current_Wx.size > 0:
            return {
                "t": self.current_cwt_t,
                "freqs_hz": self.current_cwt_freqs,
                "Wx": self.current_Wx,
            }
        return None

    def _render_cwt_result(self, result: tuple) -> None:
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("CWT вернул некорректный результат.")
        self._destroy_span_selector()
        t, freqs_hz, Wx = result
        self._store_cwt_cache(t, freqs_hz, Wx)
        amp = np.abs(Wx)
        try:
            freqs_disp, Wx_disp, amp_disp, freq_ylim = self._crop_cwt_for_display(freqs_hz, Wx, amp)
        except ValueError as exc:
            messagebox.showerror("Morlet CWT", str(exc))
            self.status.set(str(exc))
            return
        tone_hz = self._morlet_tone_hz()
        if tone_hz is None and self.current_data.size:
            tone_hz = analysis.estimate_dominant_tone_hz(
                self.current_data, self.current_samplerate
            )
        self.analysis_plot.draw_interactive_spec(
            t,
            freqs_disp,
            amp_disp,
            title="Morlet CWT скалограмма |Wx|",
            y_label="Частота (Гц)",
            Wx=Wx_disp,
            spec_kind="Morlet CWT",
            log_freq=True,
            target_hz_line=tone_hz,
            freq_ylim=freq_ylim,
            colorbar_label="Амплитуда",
            note="ЛКМ — боковой срез, ПКМ — продольный\nМожно затем STFT / «Срез вейвлета»",
        )
        self.analysis_plot.set_spec_click_handler(self._on_spectrogram_click)
        bounds = self._morlet_display_freq_bounds()
        if bounds[0] is not None or bounds[1] is not None:
            lo = bounds[0] if bounds[0] is not None else "…"
            hi = bounds[1] if bounds[1] is not None else "…"
            self.status.set(f"Morlet CWT построен ({lo}–{hi} Гц). Полный CWT в памяти.")
        else:
            self.status.set("Morlet CWT построен. CWT сохранён в памяти.")

    def show_wavelet(self) -> None:
        if not self.require_audio():
            return

        info = analysis.estimate_cwt_load(self.current_data, self.current_samplerate)
        if info["too_heavy"]:
            if not messagebox.askyesno(
                "Вейвлет Морле",
                f"Сигнал большой ({info['message']}).\n"
                "Будут использованы первые ~4 с и укрупнение шкал.\n\nПродолжить?",
            ):
                self.status.set("CWT отменён.")
                return

        data = self.current_data.copy()
        fs = int(self.current_samplerate)

        def worker():
            return analysis.cwt_scalogram(
                data,
                fs,
                fmin_hz=5000.0,
                fmax_hz=100000.0,
                max_seconds=4.0,
                downsample=1,
                target_fs=192000,
            )

        self._run_background("Morlet CWT: расчёт (не закрывайте окно)...", worker, self._render_cwt_result)

    def show_autocorrelation(self) -> None:
        if not self.require_audio():
            return

        def _draw() -> None:
            lags, corr = analysis.autocorrelation(self.current_data, self.current_samplerate)
            annotations = []
            note = "Пик возле 0 мс ожидаем"
            if corr.size > 2:
                search = corr.copy()
                search[:2] = 0.0
                idx = int(np.argmax(search))
                lag = float(lags[idx])
                freq = 1000.0 / lag if lag > 0 else 0.0
                annotations.append({"x": lag, "y": float(corr[idx]), "label": f"период {lag:.2f} мс", "color": RED})
                if freq:
                    note += f"\nОценка частоты: {freq:.1f} Гц"
            self.analysis_plot.draw_xy(
                lags, corr, "Автокорреляция", "задержка, мс", "коэффициент", GREEN,
                annotations=annotations, note=note,
            )

        self._safe_analysis_plot(_draw)

    def show_wavelet_slice(self) -> None:
        if not self.require_audio():
            return
        cache = self._get_cwt_cache()
        if cache is None:
            messagebox.showinfo("Срез вейвлета", "Сначала постройте «Вейвлет Морле».")
            return

        def _draw() -> None:
            t = np.asarray(cache["t"], dtype=np.float64)
            freqs = np.asarray(cache["freqs_hz"], dtype=np.float64)
            Wx = cache["Wx"]
            image = np.abs(Wx)
            if image.size == 0:
                raise ValueError("Пустая скалограмма CWT.")
            r, _c = np.unravel_index(int(np.argmax(image)), image.shape)
            f_hz = float(freqs[r])
            t_rel, y_vals = analysis.get_wavelet_slice(t, Wx[r, :], f_hz=f_hz)
            half_w_s = max(abs(float(t_rel[0])), abs(float(t_rel[-1]))) if t_rel.size else 0.0
            if half_w_s >= 0.1:
                t_plot, xl = t_rel, "Время относительно центра (с)"
            elif half_w_s >= 1e-3:
                t_plot, xl = t_rel * 1e3, "Время относительно центра (мс)"
            else:
                t_plot, xl = t_rel * 1e6, "Время относительно центра (мкс)"
            note = f"Chmi продольный срез Re(Wx)\nf ≈ {f_hz:.1f} Гц"
            self.analysis_plot.draw_xy(
                t_plot,
                y_vals,
                "Продольный срез: Re(Wx), Morlet",
                xl,
                "Норм. амплитуда",
                BLUE,
                note=note,
            )
            self.status.set("Срез вейвлета построен.")

        self._safe_analysis_plot(_draw)

    def export_cwt_ridge(self) -> None:
        if not self.require_audio():
            return
        cache = self._get_cwt_cache()
        if cache is None:
            messagebox.showinfo("Экспорт CWT", "Сначала постройте «Вейвлет Морле».")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить гребень CWT",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("TXT", "*.txt")],
            initialfile="cwt_ridge.csv",
        )
        if not path:
            return
        try:
            analysis.save_cwt_ridge_txt(cache["t"], cache["freqs_hz"], cache["Wx"], path)
            self.status.set(f"Гребень сохранен: {path}")
            messagebox.showinfo("Экспорт CWT", f"Гребень частот сохранен:\n{path}")
        except Exception as exc:
            messagebox.showerror("Экспорт CWT", str(exc))

    def on_txt_to_wav(self) -> None:
        try:
            input_txt = filedialog.askopenfilename(
                title="Выберите TXT/CSV",
                filetypes=[("Text/CSV", "*.txt *.csv"), ("Все файлы", "*.*")],
                initialdir=str(self.save_dir),
            )
            if not input_txt:
                return
            default_out = str(Path(input_txt).with_suffix(".wav"))
            output_wav = filedialog.asksaveasfilename(
                title="Сохранить WAV",
                defaultextension=".wav",
                initialfile=Path(default_out).name,
                initialdir=str(Path(default_out).parent),
                filetypes=[("WAV", "*.wav"), ("Все файлы", "*.*")],
            )
            if not output_wav:
                return
            sample_rate = simpledialog.askinteger(
                "TXT → WAV",
                "Sample rate (Гц). Для 2-колоночного TXT можно Cancel.",
                minvalue=1,
                parent=self,
            )
            amp_col = simpledialog.askinteger(
                "TXT → WAV",
                "Колонка амплитуды (по умолчанию -1):",
                initialvalue=-1,
                parent=self,
            )
            if amp_col is None:
                amp_col = -1
            normalize = messagebox.askyesno("TXT → WAV", "Нормировать в [-1, 1]?")
            fs, n = txt_to_wav(
                input_txt,
                output_wav,
                sample_rate=sample_rate,
                amplitude_col=amp_col,
                normalize=normalize,
            )
            self.load_wav_path(Path(output_wav))
            messagebox.showinfo("TXT → WAV", f"Готово:\n{output_wav}\nfs={fs} Гц, samples={n}")
        except Exception as exc:
            messagebox.showerror("TXT → WAV", str(exc))

    def _on_spectrogram_click(self, event, current_spec: dict | None) -> None:
        try:
            if event.inaxes != self.analysis_plot.ax or current_spec is None:
                return
            if event.button not in (1, 3):
                return

            t = current_spec["t"]
            freqs_hz = current_spec["freqs_hz"]
            spec = current_spec["spec"]
            title = current_spec["title"]
            y_label = current_spec["y_label"]
            Wx = current_spec.get("Wx")

            if event.button == 1:
                if event.xdata is None:
                    return
                ti = int(np.argmin(np.abs(t - float(event.xdata))))
                t_sel = float(t[ti])
                slice_vals = np.asarray(spec[:, ti], dtype=np.float64)
                order = np.argsort(freqs_hz)
                freqs_plot = freqs_hz[order]
                slice_plot = slice_vals[order]
                fig, ax = plt.subplots(figsize=(6, 6))
                if title == "Morlet CWT":
                    pos = freqs_plot > 0
                    if not np.any(pos):
                        messagebox.showinfo("Срез", "Нет положительных частот для лог. шкалы.")
                        plt.close(fig)
                        return
                    ax.plot(freqs_plot[pos], slice_plot[pos], color="#1f77b4", linewidth=1.5)
                    ax.set_xscale("log")
                    ax.set_xlim(float(np.min(freqs_plot[pos])), float(np.max(freqs_plot[pos])))
                else:
                    ax.plot(freqs_plot, slice_plot, color="#1f77b4", linewidth=1.5)
                    ax.set_xlim(float(freqs_plot[0]), float(freqs_plot[-1]))
                ax.set_title(f"Боковой срез: t={t_sel:.6f} с\n{title}")
                ax.set_xlabel("Частота (Гц)")
                ax.set_ylabel(y_label)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.show()
            else:
                if event.ydata is None:
                    return
                fi = int(np.argmin(np.abs(freqs_hz - event.ydata)))
                f_sel = float(freqs_hz[fi])
                fig, ax = plt.subplots(figsize=(8, 4))
                if Wx is not None and title == "Morlet CWT":
                    plot_longitudinal_slice_morlet_re(t, Wx[fi, :], f_hz=f_sel, ax=ax, show=False)
                    ax.set_title(
                        f"Продольный срез: Re(Wx), f ≈ {f_sel:.1f} Гц\n({title})",
                        fontweight="bold",
                    )
                    if self.current_path:
                        stem = self.current_path.stem
                        safe_f = f"{f_sel:.0f}".replace(".", "p")
                        out_txt = self.save_dir / f"{stem}_longitudinal_slice_f{safe_f}Hz.txt"
                        save_longitudinal_slice_txt(t, Wx[fi, :], path=str(out_txt), f_hz=f_sel)
                else:
                    slice_vals = spec[fi, :]
                    ax.plot(t, slice_vals, color="#1f77b4", linewidth=1.5)
                    ax.set_title(f"Продольный срез: f={f_sel:.2f} Гц\n{title}")
                    ax.set_xlabel("Время (с)")
                    ax.set_ylabel(y_label)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(float(t[0]), float(t[-1]))
                fig.tight_layout()
                fig.show()
        except Exception as exc:
            messagebox.showerror("Срез по клику", str(exc))

    def show_envelope(self) -> None:
        if not self.require_audio():
            return

        def _draw() -> None:
            env = analysis.envelope(self.current_data, samplerate=self.current_samplerate)
            t = np.arange(env.size, dtype=np.float32) / self.current_samplerate
            annotations = []
            note = ""
            if env.size:
                idx = int(np.argmax(env))
                annotations.append({"x": float(t[idx]), "y": float(env[idx]), "label": f"макс {env[idx]:.3f}", "color": RED})
                note = f"Максимум огибающей:\n{t[idx]:.2f} с"
            self.analysis_plot.draw_xy(
                t, env, "Огибающая сигнала", "время, с", "амплитуда", RED,
                annotations=annotations, note=note,
            )

        self._safe_analysis_plot(_draw)

    def apply_transform(self, name: str) -> None:
        if not self.require_audio():
            return
        try:
            if name == "normalize":
                self.current_data = analysis.normalize(self.current_data)
                label = "нормализация"
            elif name == "dc":
                self.current_data = analysis.remove_dc(self.current_data)
                label = "удаление DC"
            elif name == "gate":
                threshold = simpledialog.askfloat("Шумовой порог", "Порог, dBFS:", initialvalue=-45.0, parent=self)
                if threshold is None:
                    return
                self.current_data = analysis.noise_gate(self.current_data, threshold)
                label = f"шумовой порог {threshold:g} dBFS"
            elif name == "trim":
                self.current_data = analysis.trim_silence(self.current_data, samplerate=self.current_samplerate)
                label = "обрезка тишины"
            elif name == "lowpass":
                high = simpledialog.askfloat("НЧ-фильтр", "Пропускать частоты ниже, Гц:", initialvalue=2000.0, parent=self)
                if high is None:
                    return
                self.current_data = analysis.fft_filter(self.current_data, self.current_samplerate, high_hz=high)
                label = f"НЧ-фильтр до {high:g} Гц"
            elif name == "highpass":
                low = simpledialog.askfloat("ВЧ-фильтр", "Пропускать частоты выше, Гц:", initialvalue=80.0, parent=self)
                if low is None:
                    return
                self.current_data = analysis.fft_filter(self.current_data, self.current_samplerate, low_hz=low)
                label = f"ВЧ-фильтр от {low:g} Гц"
            elif name == "bandpass":
                low = simpledialog.askfloat("Полосовой фильтр", "Нижняя граница, Гц:", initialvalue=80.0, parent=self)
                if low is None:
                    return
                high = simpledialog.askfloat("Полосовой фильтр", "Верхняя граница, Гц:", initialvalue=3000.0, parent=self)
                if high is None:
                    return
                self.current_data = analysis.fft_filter(self.current_data, self.current_samplerate, low_hz=low, high_hz=high)
                label = f"полоса {low:g}-{high:g} Гц"
            else:
                return
        except Exception as exc:
            messagebox.showerror("Преобразование", f"Не удалось выполнить преобразование:\n{exc}")
            return
        self.current_metrics = analysis.metrics(self.current_data, self.current_samplerate)
        self.update_metrics_view()
        self.analysis_plot.draw_waveform(self.current_data, self.current_samplerate, f"После преобразования: {label}")
        self.status.set(f"Выполнено преобразование: {label}")

    def save_current_as_new(self) -> None:
        if not self.require_audio():
            return
        base = self.title_entry.get().strip() or "Аудиозапись"
        title = simpledialog.askstring("Сохранить", "Название новой записи:", initialvalue=f"{base}_processed", parent=self)
        if not title:
            return
        patient = self.patient_entry.get().strip()
        notes = self.notes_text.get("1.0", "end").strip()
        path = save_wav(self.current_data, self.current_samplerate, title, directory=self.save_dir)
        record_id = add_recording(path, title, patient, notes, self.current_samplerate, self.current_data.size / self.current_samplerate)
        self.current_path = path
        self.current_record_id = record_id
        self.current_metrics = analysis.metrics(self.current_data, self.current_samplerate)
        save_metrics(record_id, self.current_metrics)
        self.refresh_library()
        self.status.set(f"Новая запись сохранена: {path}")

    def export_report(self) -> None:
        if not self.require_audio():
            return
        record = get_recording(self.current_record_id) if self.current_record_id else None
        if record is None:
            path = self.current_path or Path("unsaved.wav")
            record = RecordingInfo(
                id=0,
                created_at="не сохранено",
                title=path.stem,
                patient=self.patient_entry.get().strip(),
                notes=self.notes_text.get("1.0", "end").strip(),
                samplerate=self.current_samplerate,
                duration=self.current_data.size / self.current_samplerate,
                file_path=str(path),
            )
        metrics = self.current_metrics or analysis.metrics(self.current_data, self.current_samplerate)
        report = export_report_csv(record, metrics)
        self.status.set(f"Отчет сохранен: {report}")
        messagebox.showinfo("Экспорт отчета", f"CSV-отчет сохранен:\n{report}")

    def update_metrics_view(self) -> None:
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        for key, value in self.current_metrics.items():
            self.metrics_tree.insert("", "end", text=key, values=(value,))

    def refresh_library(self) -> None:
        self.library_tree.delete(*self.library_tree.get_children())
        for rec in list_recordings():
            self.library_tree.insert(
                "",
                "end",
                iid=str(rec.id),
                values=(
                    rec.created_at,
                    rec.title,
                    rec.patient,
                    f"{rec.duration:.2f} с",
                    rec.samplerate,
                    rec.file_path,
                ),
            )

    def load_selected_recording(self) -> None:
        selected = self.library_tree.selection()
        if not selected:
            messagebox.showinfo("База записей", "Выберите запись в таблице.")
            return
        record_id = int(selected[0])
        rec = get_recording(record_id)
        if not rec:
            messagebox.showerror("База записей", "Запись не найдена в базе.")
            return
        path = Path(rec.file_path)
        if not path.exists():
            messagebox.showerror("База записей", f"Файл не найден:\n{path}")
            return
        self.title_entry.delete(0, "end")
        self.title_entry.insert(0, rec.title)
        self.patient_entry.delete(0, "end")
        self.patient_entry.insert(0, rec.patient)
        self.notes_text.delete("1.0", "end")
        self.notes_text.insert("1.0", rec.notes)
        self.load_wav_path(path, record_id)

    def open_folder(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["explorer", str(path)])
        except Exception as exc:
            messagebox.showerror("Папка", f"Не удалось открыть папку:\n{exc}")

    def _live_tick(self) -> None:
        if self.recorder.is_recording or self.serial_recorder.is_recording:
            active = self.serial_recorder if self.serial_recorder.is_recording else self.recorder
            snapshot = active.snapshot(max_samples=self.current_samplerate * 8)
            if snapshot.size:
                plot_data = analysis.normalize(snapshot) if self.serial_recorder.is_recording else snapshot
                title = "Живой сигнал COM-порта" if self.serial_recorder.is_recording else "Живая форма сигнала"
                self.record_plot.draw_waveform(plot_data, self.current_samplerate, title)
            if self.serial_recorder.is_recording:
                good, bad = self.serial_recorder.line_stats
                self.record_state.set(f"COM: принято {good}, ошибок {bad}")
                level = min(100.0, active.level * 100.0)
            else:
                level = min(100.0, active.level * 650.0)
            self.level_var.set(level)
        self.after(160, self._live_tick)


def main() -> None:
    app = MedAudioLabApp()
    app.mainloop()

if __name__ == "__main__":
    main()