"""
NeuroStat Pipeline GUI - PyQt6 Version
A modern, user-friendly interface for EEG preprocessing, source localization,
connectivity analysis, and statistical testing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Callable, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox, QTextEdit,
    QFileDialog, QMessageBox, QProgressBar, QFrame, QSplitter,
    QScrollArea, QSpinBox, QDoubleSpinBox, QTabWidget, QGridLayout,
    QSizePolicy, QToolTip, QStatusBar, QRadioButton, QButtonGroup,
    QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor

import json

from pli_pipeline import (
    LOGGER,
    LOG_FORMAT,
    StudyDesign,
    PipelineConfig,
    StatsConfig,
    run_preprocessing_for_design,
    run_source_and_connectivity_for_design,
    aggregate_csv_table,
    run_stats_for_design,
)


# ============================================================================
# STYLESHEET - Modern Dark Theme
# ============================================================================
STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
}

QGroupBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #89b4fa;
    font-size: 14px;
}

QLabel {
    color: #cdd6f4;
    padding: 2px;
}

QLabel[class="hint"] {
    color: #6c7086;
    font-style: italic;
    font-size: 11px;
}

QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #45475a;
    border: 1px solid #585b70;
    border-radius: 6px;
    padding: 8px 12px;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 2px solid #89b4fa;
}

QLineEdit:disabled {
    background-color: #313244;
    color: #6c7086;
}

QPushButton {
    background-color: #45475a;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    color: #cdd6f4;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #585b70;
}

QPushButton:pressed {
    background-color: #313244;
}

QPushButton:disabled {
    background-color: #313244;
    color: #6c7086;
}

QPushButton[class="primary"] {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QPushButton[class="primary"]:hover {
    background-color: #b4befe;
}

QPushButton[class="primary"]:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QPushButton[class="success"] {
    background-color: #a6e3a1;
    color: #1e1e2e;
}

QPushButton[class="success"]:hover {
    background-color: #94e2d5;
}

QPushButton[class="browse"] {
    padding: 8px 16px;
    min-width: 80px;
}

QCheckBox {
    spacing: 8px;
    color: #cdd6f4;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #585b70;
    background-color: #45475a;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QCheckBox::indicator:hover {
    border-color: #89b4fa;
}

QTextEdit {
    background-color: #11111b;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px;
    color: #a6adc8;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
}

QProgressBar {
    background-color: #45475a;
    border: none;
    border-radius: 6px;
    height: 24px;
    text-align: center;
    color: #1e1e2e;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 6px;
}

QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 8px;
    background-color: #1e1e2e;
}

QTabBar::tab {
    background-color: #313244;
    border: 1px solid #45475a;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 10px 20px;
    margin-right: 2px;
    color: #6c7086;
}

QTabBar::tab:selected {
    background-color: #1e1e2e;
    color: #89b4fa;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #45475a;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QSplitter::handle {
    background-color: #45475a;
    width: 2px;
}

QStatusBar {
    background-color: #11111b;
    color: #6c7086;
    border-top: 1px solid #45475a;
}

QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px;
}
"""


# ============================================================================
# WORKER THREAD - Keeps UI responsive during long operations
# ============================================================================
class PipelineWorker(QThread):
    """Worker thread for running pipeline operations without blocking UI."""

    progress = pyqtSignal(str)  # Log messages
    step_progress = pyqtSignal(int, int, str)  # current, total, description
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(
        self,
        operation: str,
        base_path: Path,
        design: StudyDesign,
        config: PipelineConfig,
        steps: Optional[dict] = None
    ):
        super().__init__()
        self.operation = operation
        self.base_path = base_path
        self.design = design
        self.config = config
        self.steps = steps or {}
        self._abort = False
        self._is_running = False

    def abort(self):
        self._abort = True

    def is_running(self) -> bool:
        return self._is_running

    def run(self):
        self._is_running = True
        try:
            if self.operation == "preprocessing":
                self._run_preprocessing()
            elif self.operation == "source":
                self._run_source()
            elif self.operation == "csv":
                self._run_csv()
            elif self.operation == "stats":
                self._run_stats()
            elif self.operation == "all":
                self._run_all()
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.progress.emit(f"[ERROR] {error_msg}")
            self.finished.emit(False, str(e))
        finally:
            self._is_running = False
            # Clean up GPU memory if available
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass

    def _emit_progress(self, msg: str):
        self.progress.emit(msg)

    def _run_preprocessing(self):
        self.step_progress.emit(0, 1, "Preprocessing...")
        summaries = run_preprocessing_for_design(
            self.base_path, self.design, self.config,
            progress_cb=self._emit_progress
        )
        n = sum(1 for s in summaries if s.get("preprocessed_fif"))
        self.finished.emit(True, f"Preprocessing complete. {n} files ready.")

    def _run_source(self):
        self.step_progress.emit(0, 1, "Source localization + connectivity...")
        try:
            summaries = run_source_and_connectivity_for_design(
                self.base_path, self.design, self.config,
                progress_cb=self._emit_progress
            )
            m = sum(len(s.get("pli_outputs", {})) for s in summaries)
            self.finished.emit(True, f"Source/Connectivity complete. Saved {m} band CSVs.")
        except Exception as e:
            import traceback
            self._emit_progress(f"[ERROR] Source/Connectivity failed: {e}")
            self._emit_progress(traceback.format_exc())
            self.finished.emit(False, f"Source/Connectivity failed: {e}")

    def _run_csv(self):
        self.step_progress.emit(0, 1, "Aggregating PLI data...")
        table = aggregate_csv_table(self.design, self.config, progress_cb=self._emit_progress)
        self.finished.emit(True, f"Saved table ({len(table)} rows)")

    def _run_stats(self):
        self.step_progress.emit(0, 1, "Running statistical tests...")
        run_stats_for_design(self.design, self.config, progress_cb=self._emit_progress)
        self.finished.emit(True, "Statistics complete.")

    def _run_all(self):
        steps = []
        if self.steps.get("preprocessing"):
            steps.append(("preprocessing", "Preprocessing"))
        if self.steps.get("source"):
            steps.append(("source", "Source + Connectivity"))
        if self.steps.get("csv"):
            steps.append(("csv", "CSV Export"))
        if self.steps.get("stats"):
            steps.append(("stats", "Statistics"))

        total = len(steps)
        errors = []

        for i, (step_id, step_name) in enumerate(steps):
            if self._abort:
                self.finished.emit(False, "Operation aborted by user.")
                return

            self.step_progress.emit(i, total, f"Step {i+1}/{total}: {step_name}")
            self._emit_progress(f"\n{'='*50}")
            self._emit_progress(f"[RUN ALL] Starting {step_name}...")

            try:
                if step_id == "preprocessing":
                    summaries = run_preprocessing_for_design(
                        self.base_path, self.design, self.config,
                        progress_cb=self._emit_progress
                    )
                    n = sum(1 for s in summaries if s.get("preprocessed_fif"))
                    self._emit_progress(f"[RUN ALL] Preprocessing complete. {n} files ready.")

                elif step_id == "source":
                    summaries = run_source_and_connectivity_for_design(
                        self.base_path, self.design, self.config,
                        progress_cb=self._emit_progress
                    )
                    m = sum(len(s.get("pli_outputs", {})) for s in summaries)
                    self._emit_progress(f"[RUN ALL] Source/Connectivity complete. {m} band CSVs.")

                elif step_id == "csv":
                    table = aggregate_csv_table(
                        self.design, self.config, progress_cb=self._emit_progress
                    )
                    self._emit_progress(f"[RUN ALL] Saved table ({len(table)} rows)")

                elif step_id == "stats":
                    run_stats_for_design(
                        self.design, self.config, progress_cb=self._emit_progress
                    )
                    self._emit_progress("[RUN ALL] Statistics complete.")

            except Exception as e:
                errors.append(f"{step_name}: {e}")
                self._emit_progress(f"[ERROR] {step_name}: {e}")

        self.step_progress.emit(total, total, "Complete")

        if errors:
            self.finished.emit(False, f"Completed with {len(errors)} error(s):\n" + "\n".join(errors))
        else:
            self.finished.emit(True, "All steps completed successfully!")


# ============================================================================
# MAIN WINDOW
# ============================================================================
class NeuroStatApp(QMainWindow):
    """Main application window for NeuroStat Pipeline."""

    def __init__(self):
        super().__init__()
        self.worker: Optional[PipelineWorker] = None
        self.preview_pixmap: Optional[QPixmap] = None
        self.brain_pixmap: Optional[QPixmap] = None
        self._init_ui()

    def closeEvent(self, event):
        """Handle window close - ensure worker thread is properly terminated."""
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Operation Running",
                "A pipeline operation is still running. Do you want to abort and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.abort()
                self.worker.quit()
                # Wait up to 5 seconds for the thread to finish
                if not self.worker.wait(5000):
                    self.worker.terminate()
                    self.worker.wait(2000)
                event.accept()
            else:
                event.ignore()
                return
        else:
            event.accept()

        # Clean up GPU memory
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _init_ui(self):
        self.setWindowTitle("NeuroStat Pipeline for RCTs")
        self.setMinimumSize(1300, 800)
        self.resize(1400, 900)

        # Central widget with main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header
        header = self._create_header()
        main_layout.addWidget(header)

        # Main content area with tabs
        tabs = QTabWidget()
        tabs.addTab(self._create_setup_tab(), "Study Setup")
        tabs.addTab(self._create_pipeline_tab(), "Pipeline")
        tabs.addTab(self._create_logs_tab(), "Logs & Output")
        main_layout.addWidget(tabs, 1)

        # Bottom action bar
        action_bar = self._create_action_bar()
        main_layout.addWidget(action_bar)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Apply stylesheet
        self.setStyleSheet(STYLESHEET)

    def _create_header(self) -> QWidget:
        """Create the application header."""
        header = QFrame()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 8)

        # Title and subtitle
        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)

        title = QLabel("NeuroStat Pipeline")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #89b4fa;")
        title_layout.addWidget(title)

        subtitle = QLabel("EEG Preprocessing, Source Localization, Connectivity & Statistics for RCTs")
        subtitle.setStyleSheet("color: #6c7086; font-size: 13px;")
        title_layout.addWidget(subtitle)

        layout.addLayout(title_layout)
        layout.addStretch()

        # Brain image (if available)
        self.brain_label = QLabel()
        self.brain_label.setFixedSize(100, 60)
        self._load_brain_image()
        layout.addWidget(self.brain_label)

        return header

    def _load_brain_image(self):
        """Load and display the brain image."""
        paths = [Path(__file__).parent / "BrainImage.png", Path.cwd() / "BrainImage.png"]
        for p in paths:
            if p.exists():
                pixmap = QPixmap(str(p))
                if not pixmap.isNull():
                    self.brain_pixmap = pixmap
                    scaled = pixmap.scaled(100, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.brain_label.setPixmap(scaled)
                    return

    def _create_setup_tab(self) -> QWidget:
        """Create the Study Setup tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)

        # Study Design section
        design_group = QGroupBox("Study Design")
        design_layout = QGridLayout(design_group)
        design_layout.setSpacing(12)

        # Base folder
        design_layout.addWidget(QLabel("Base Folder:"), 0, 0)
        self.base_input = QLineEdit()
        self.base_input.setPlaceholderText("Select folder containing group subfolders...")
        self.base_input.setToolTip("Root folder containing your study data organized as: base/Group/Session/*.set")
        design_layout.addWidget(self.base_input, 0, 1)

        browse_base = QPushButton("Browse...")
        browse_base.setProperty("class", "browse")
        browse_base.clicked.connect(self._browse_base)
        design_layout.addWidget(browse_base, 0, 2)

        # Groups
        design_layout.addWidget(QLabel("Groups:"), 1, 0)
        self.groups_input = QLineEdit("GroupA, GroupB")
        self.groups_input.setToolTip("Comma-separated list of group names (e.g., Treatment, Control)")
        design_layout.addWidget(self.groups_input, 1, 1, 1, 2)

        # Sessions
        design_layout.addWidget(QLabel("Sessions:"), 2, 0)
        self.sessions_input = QLineEdit("pre, post")
        self.sessions_input.setToolTip("Comma-separated list of session names (e.g., baseline, followup)")
        design_layout.addWidget(self.sessions_input, 2, 1, 1, 2)

        # Output folder
        design_layout.addWidget(QLabel("Output Folder:"), 3, 0)
        self.output_input = QLineEdit(str(Path.cwd() / "processed"))
        self.output_input.setToolTip("Where to save processed data and results")
        design_layout.addWidget(self.output_input, 3, 1)

        browse_out = QPushButton("Browse...")
        browse_out.setProperty("class", "browse")
        browse_out.clicked.connect(self._browse_output)
        design_layout.addWidget(browse_out, 3, 2)

        # Hint
        hint = QLabel("Expected data layout: base / GroupName / SessionName / subject.set")
        hint.setProperty("class", "hint")
        design_layout.addWidget(hint, 4, 0, 1, 3)

        content_layout.addWidget(design_group)

        # Preprocessing Settings
        preproc_group = QGroupBox("Preprocessing Settings")
        preproc_layout = QGridLayout(preproc_group)
        preproc_layout.setSpacing(12)

        # Bandpass filter
        preproc_layout.addWidget(QLabel("Bandpass Filter (Hz):"), 0, 0)
        filter_layout = QHBoxLayout()

        self.l_freq_input = QDoubleSpinBox()
        self.l_freq_input.setRange(0.1, 100)
        self.l_freq_input.setValue(1.0)
        self.l_freq_input.setDecimals(1)
        self.l_freq_input.setToolTip("Low-frequency cutoff for bandpass filter")
        filter_layout.addWidget(self.l_freq_input)

        filter_layout.addWidget(QLabel("to"))

        self.h_freq_input = QDoubleSpinBox()
        self.h_freq_input.setRange(1, 500)
        self.h_freq_input.setValue(40.0)
        self.h_freq_input.setDecimals(1)
        self.h_freq_input.setToolTip("High-frequency cutoff for bandpass filter")
        filter_layout.addWidget(self.h_freq_input)

        filter_layout.addStretch()
        preproc_layout.addLayout(filter_layout, 0, 1)

        # Notch filter
        preproc_layout.addWidget(QLabel("Notch Filter (Hz):"), 1, 0)
        self.notch_input = QLineEdit("50, 100")
        self.notch_input.setToolTip("Comma-separated frequencies to notch out (powerline noise)")
        preproc_layout.addWidget(self.notch_input, 1, 1)

        content_layout.addWidget(preproc_group)

        # Preprocessing Pipeline Selection (Mutually Exclusive)
        pipeline_group = QGroupBox("Preprocessing Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)
        pipeline_layout.setSpacing(12)

        # Radio button group for mutually exclusive selection
        self.pipeline_button_group = QButtonGroup(self)

        # GEDAI Pipeline Option
        gedai_frame = QFrame()
        gedai_frame.setStyleSheet("""
            QFrame {
                background-color: #252536;
                border: 2px solid #45475a;
                border-radius: 8px;
                padding: 8px;
            }
            QFrame[selected="true"] {
                border-color: #89b4fa;
            }
        """)
        gedai_frame_layout = QVBoxLayout(gedai_frame)
        gedai_frame_layout.setSpacing(8)

        # GEDAI radio button and description
        self.gedai_radio = QRadioButton("GEDAI Pipeline")
        self.gedai_radio.setStyleSheet("font-weight: bold; font-size: 14px; color: #89b4fa;")
        self.gedai_radio.setToolTip(
            "GEDAI (Generalized Eigenvalue De-Artifacting Instrument) is a complete\n"
            "preprocessing pipeline that uses leadfield-based denoising to separate\n"
            "brain signals from artifacts. Recommended for heavily contaminated data."
        )
        self.pipeline_button_group.addButton(self.gedai_radio, 0)
        gedai_frame_layout.addWidget(self.gedai_radio)

        gedai_desc = QLabel(
            "Complete pipeline using Generalized Eigenvalue Decomposition with EEG forward model.\n"
            "Separates brain signals from artifacts using leadfield-derived spatial patterns."
        )
        gedai_desc.setWordWrap(True)
        gedai_desc.setStyleSheet("color: #6c7086; font-size: 11px; margin-left: 20px;")
        gedai_frame_layout.addWidget(gedai_desc)

        # GEDAI parameters
        gedai_params = QHBoxLayout()
        gedai_params.setContentsMargins(20, 0, 0, 0)
        gedai_params.addWidget(QLabel("Threshold:"))
        self.gedai_thresh_input = QDoubleSpinBox()
        self.gedai_thresh_input.setRange(1.0, 5.0)
        self.gedai_thresh_input.setValue(3.0)
        self.gedai_thresh_input.setDecimals(1)
        self.gedai_thresh_input.setSingleStep(0.5)
        self.gedai_thresh_input.setToolTip("Artifact rejection threshold (2=aggressive, 3=balanced, 4=conservative)")
        self.gedai_thresh_input.setFixedWidth(80)
        gedai_params.addWidget(self.gedai_thresh_input)
        gedai_params.addStretch()
        gedai_frame_layout.addLayout(gedai_params)

        pipeline_layout.addWidget(gedai_frame)

        # Traditional Pipeline Option
        traditional_frame = QFrame()
        traditional_frame.setStyleSheet("""
            QFrame {
                background-color: #252536;
                border: 2px solid #45475a;
                border-radius: 8px;
                padding: 8px;
            }
            QFrame[selected="true"] {
                border-color: #a6e3a1;
            }
        """)
        traditional_frame_layout = QVBoxLayout(traditional_frame)
        traditional_frame_layout.setSpacing(8)

        # Traditional radio button and description
        self.traditional_radio = QRadioButton("Traditional Pipeline")
        self.traditional_radio.setChecked(True)  # Default selection
        self.traditional_radio.setStyleSheet("font-weight: bold; font-size: 14px; color: #a6e3a1;")
        self.traditional_radio.setToolTip(
            "Traditional preprocessing using ASR for transient artifact removal\n"
            "and ICLabel for ICA component classification. Well-established methods."
        )
        self.pipeline_button_group.addButton(self.traditional_radio, 1)
        traditional_frame_layout.addWidget(self.traditional_radio)

        traditional_desc = QLabel(
            "Combines ASR (Artifact Subspace Reconstruction) and ICLabel (ICA Component Classification).\n"
            "ASR removes transient artifacts, ICLabel uses deep learning to identify artifact components."
        )
        traditional_desc.setWordWrap(True)
        traditional_desc.setStyleSheet("color: #6c7086; font-size: 11px; margin-left: 20px;")
        traditional_frame_layout.addWidget(traditional_desc)

        # Traditional parameters - ASR
        asr_row = QHBoxLayout()
        asr_row.setContentsMargins(20, 0, 0, 0)
        self.asr_checkbox = QCheckBox("ASR (Artifact Subspace Reconstruction)")
        self.asr_checkbox.setChecked(True)
        self.asr_checkbox.setToolTip("Removes large transient artifacts using principal component analysis")
        asr_row.addWidget(self.asr_checkbox)
        asr_row.addStretch()
        traditional_frame_layout.addLayout(asr_row)

        # Traditional parameters - ICLabel
        iclabel_row = QHBoxLayout()
        iclabel_row.setContentsMargins(20, 0, 0, 0)
        self.iclabel_checkbox = QCheckBox("ICLabel (ICA Classification)")
        self.iclabel_checkbox.setChecked(True)
        self.iclabel_checkbox.setToolTip("Uses deep learning to classify ICA components as brain/artifact")
        iclabel_row.addWidget(self.iclabel_checkbox)

        iclabel_row.addWidget(QLabel("Brain Threshold:"))
        self.iclabel_thresh_input = QDoubleSpinBox()
        self.iclabel_thresh_input.setRange(0.0, 1.0)
        self.iclabel_thresh_input.setValue(0.85)
        self.iclabel_thresh_input.setDecimals(2)
        self.iclabel_thresh_input.setSingleStep(0.05)
        self.iclabel_thresh_input.setToolTip("Components with brain probability above this are kept")
        self.iclabel_thresh_input.setFixedWidth(80)
        iclabel_row.addWidget(self.iclabel_thresh_input)
        iclabel_row.addStretch()
        traditional_frame_layout.addLayout(iclabel_row)

        pipeline_layout.addWidget(traditional_frame)

        # Connect radio buttons to update UI state
        self.gedai_radio.toggled.connect(self._on_pipeline_changed)
        self.traditional_radio.toggled.connect(self._on_pipeline_changed)

        content_layout.addWidget(pipeline_group)

        # Advanced Preprocessing Options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QGridLayout(advanced_group)
        advanced_layout.setSpacing(12)

        # Bad channel detection
        self.bad_channel_checkbox = QCheckBox("Auto-detect & Interpolate Bad Channels")
        self.bad_channel_checkbox.setChecked(True)
        self.bad_channel_checkbox.setToolTip("Automatically detect and interpolate bad channels based on variance")
        advanced_layout.addWidget(self.bad_channel_checkbox, 0, 0)

        advanced_layout.addWidget(QLabel("Z-score threshold:"), 0, 1)
        self.bad_channel_thresh = QDoubleSpinBox()
        self.bad_channel_thresh.setRange(2.0, 10.0)
        self.bad_channel_thresh.setValue(4.0)
        self.bad_channel_thresh.setDecimals(1)
        self.bad_channel_thresh.setToolTip("Channels with variance z-score above this are marked bad")
        advanced_layout.addWidget(self.bad_channel_thresh, 0, 2)

        # ICA method
        advanced_layout.addWidget(QLabel("ICA Method:"), 1, 0)
        self.ica_method_input = QLineEdit("fastica")
        self.ica_method_input.setToolTip("ICA algorithm: fastica, infomax, or picard")
        advanced_layout.addWidget(self.ica_method_input, 1, 1, 1, 2)

        # ICA components
        advanced_layout.addWidget(QLabel("ICA Components:"), 2, 0)
        self.ica_components_input = QSpinBox()
        self.ica_components_input.setRange(0, 100)
        self.ica_components_input.setValue(0)
        self.ica_components_input.setSpecialValueText("Auto")
        self.ica_components_input.setToolTip("Number of ICA components (0 = auto, typically 25)")
        advanced_layout.addWidget(self.ica_components_input, 2, 1, 1, 2)

        content_layout.addWidget(advanced_group)

        # Statistics Settings
        stats_group = QGroupBox("Statistics Settings")
        stats_layout = QGridLayout(stats_group)
        stats_layout.setSpacing(12)

        # GPU option
        self.gpu_checkbox = QCheckBox("Use GPU Acceleration (Experimental)")
        self.gpu_checkbox.setChecked(False)  # Disabled by default for stability
        self.gpu_checkbox.setToolTip(
            "Enable GPU acceleration for PLI computation using CUDA/CuPy.\n"
            "WARNING: GPU mode may be unstable on some systems.\n"
            "Leave unchecked for reliable CPU computation."
        )
        self.gpu_checkbox.setStyleSheet("color: #f9e2af;")  # Yellow warning color
        stats_layout.addWidget(self.gpu_checkbox, 0, 0, 1, 2)

        # Permutations
        stats_layout.addWidget(QLabel("Permutations:"), 1, 0)
        self.perms_input = QSpinBox()
        self.perms_input.setRange(100, 100000)
        self.perms_input.setValue(1000)
        self.perms_input.setSingleStep(100)
        self.perms_input.setToolTip("Number of permutations for cluster-based testing")
        stats_layout.addWidget(self.perms_input, 1, 1)

        # Cluster threshold
        stats_layout.addWidget(QLabel("Cluster Threshold (t):"), 2, 0)
        self.cluster_thresh_input = QLineEdit()
        self.cluster_thresh_input.setPlaceholderText("Auto (leave empty)")
        self.cluster_thresh_input.setToolTip("T-value threshold for cluster formation (empty = automatic)")
        stats_layout.addWidget(self.cluster_thresh_input, 2, 1)

        content_layout.addWidget(stats_group)
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

        return tab

    def _create_pipeline_tab(self) -> QWidget:
        """Create the Pipeline execution tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Step selection
        steps_group = QGroupBox("Pipeline Steps")
        steps_layout = QVBoxLayout(steps_group)

        info_label = QLabel("Select which steps to run. Each step can be run individually or as part of the full pipeline.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #6c7086;")
        steps_layout.addWidget(info_label)

        # Step checkboxes with descriptions
        steps_grid = QGridLayout()
        steps_grid.setSpacing(16)

        # Step 1: Preprocessing
        self.step1_check = QCheckBox("1. Preprocessing")
        self.step1_check.setChecked(True)
        self.step1_check.setStyleSheet("font-weight: bold; font-size: 14px;")
        steps_grid.addWidget(self.step1_check, 0, 0)
        step1_desc = QLabel("Filter, reference, ICA artifact removal")
        step1_desc.setStyleSheet("color: #6c7086; margin-left: 24px;")
        steps_grid.addWidget(step1_desc, 0, 1)
        step1_btn = QPushButton("Run Step 1")
        step1_btn.clicked.connect(self._run_preprocessing)
        steps_grid.addWidget(step1_btn, 0, 2)

        # Step 2: Source + Connectivity
        self.step2_check = QCheckBox("2. Source Localization + Connectivity")
        self.step2_check.setChecked(True)
        self.step2_check.setStyleSheet("font-weight: bold; font-size: 14px;")
        steps_grid.addWidget(self.step2_check, 1, 0)
        step2_desc = QLabel("MNE inverse, PLI connectivity matrices")
        step2_desc.setStyleSheet("color: #6c7086; margin-left: 24px;")
        steps_grid.addWidget(step2_desc, 1, 1)
        step2_btn = QPushButton("Run Step 2")
        step2_btn.clicked.connect(self._run_source)
        steps_grid.addWidget(step2_btn, 1, 2)

        # Step 3: CSV Export
        self.step3_check = QCheckBox("3. CSV/Excel Export")
        self.step3_check.setChecked(True)
        self.step3_check.setStyleSheet("font-weight: bold; font-size: 14px;")
        steps_grid.addWidget(self.step3_check, 2, 0)
        step3_desc = QLabel("Aggregate PLI values for network analysis")
        step3_desc.setStyleSheet("color: #6c7086; margin-left: 24px;")
        steps_grid.addWidget(step3_desc, 2, 1)
        step3_btn = QPushButton("Run Step 3")
        step3_btn.clicked.connect(self._run_csv)
        steps_grid.addWidget(step3_btn, 2, 2)

        # Step 4: Statistics
        self.step4_check = QCheckBox("4. Statistical Analysis")
        self.step4_check.setChecked(True)
        self.step4_check.setStyleSheet("font-weight: bold; font-size: 14px;")
        steps_grid.addWidget(self.step4_check, 3, 0)
        step4_desc = QLabel("Cluster-based permutation tests")
        step4_desc.setStyleSheet("color: #6c7086; margin-left: 24px;")
        steps_grid.addWidget(step4_desc, 3, 1)
        step4_btn = QPushButton("Run Step 4")
        step4_btn.clicked.connect(self._run_stats)
        steps_grid.addWidget(step4_btn, 3, 2)

        steps_layout.addLayout(steps_grid)
        layout.addWidget(steps_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        # Overall progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Current operation label
        self.operation_label = QLabel("No operation running")
        self.operation_label.setStyleSheet("color: #6c7086;")
        progress_layout.addWidget(self.operation_label)

        layout.addWidget(progress_group)

        # Results section with before/after and metrics
        results_group = QGroupBox("Preprocessing Results")
        results_layout = QHBoxLayout(results_group)
        results_layout.setSpacing(16)

        # Left side: Visualization tabs
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)

        self.viz_tabs = QTabWidget()
        self.viz_tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 6px 12px;
                font-size: 11px;
            }
        """)

        # Tab 1: EEG Traces
        traces_tab = QWidget()
        traces_layout = QVBoxLayout(traces_tab)
        self.traces_label = QLabel("Run preprocessing to see EEG traces comparison")
        self.traces_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.traces_label.setStyleSheet("color: #6c7086;")
        self.traces_label.setMinimumHeight(180)
        self.traces_label.setScaledContents(False)
        traces_layout.addWidget(self.traces_label)
        self.viz_tabs.addTab(traces_tab, "EEG Traces")

        # Tab 2: Power Spectrum
        psd_tab = QWidget()
        psd_layout = QVBoxLayout(psd_tab)
        self.psd_label = QLabel("Power spectrum comparison will appear here")
        self.psd_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.psd_label.setStyleSheet("color: #6c7086;")
        self.psd_label.setMinimumHeight(180)
        psd_layout.addWidget(self.psd_label)
        self.viz_tabs.addTab(psd_tab, "Power Spectrum")

        # Tab 3: Topography
        topo_tab = QWidget()
        topo_layout = QVBoxLayout(topo_tab)
        self.topo_label = QLabel("Scalp topography comparison will appear here")
        self.topo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.topo_label.setStyleSheet("color: #6c7086;")
        self.topo_label.setMinimumHeight(180)
        topo_layout.addWidget(self.topo_label)
        self.viz_tabs.addTab(topo_tab, "Topography")

        # Tab 4: Global Field Power
        gfp_tab = QWidget()
        gfp_layout = QVBoxLayout(gfp_tab)
        self.gfp_label = QLabel("Global field power comparison will appear here")
        self.gfp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gfp_label.setStyleSheet("color: #6c7086;")
        self.gfp_label.setMinimumHeight(180)
        gfp_layout.addWidget(self.gfp_label)
        self.viz_tabs.addTab(gfp_tab, "GFP")

        # Tab 5: Alpha PLI Connectivity Circle
        pli_tab = QWidget()
        pli_layout = QVBoxLayout(pli_tab)
        self.pli_circle_label = QLabel("Run Source/Connectivity to see Alpha band PLI circle plot")
        self.pli_circle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pli_circle_label.setStyleSheet("color: #6c7086;")
        self.pli_circle_label.setMinimumHeight(180)
        pli_layout.addWidget(self.pli_circle_label)
        self.viz_tabs.addTab(pli_tab, "Alpha PLI")

        viz_layout.addWidget(self.viz_tabs)
        results_layout.addWidget(viz_widget, stretch=2)

        # Right side: Cleaning Metrics Panel
        metrics_widget = QFrame()
        metrics_widget.setStyleSheet("""
            QFrame {
                background-color: #252536;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        metrics_widget.setFixedWidth(280)
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setSpacing(8)

        metrics_title = QLabel("Cleaning Metrics")
        metrics_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #89b4fa; border: none;")
        metrics_layout.addWidget(metrics_title)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #45475a;")
        metrics_layout.addWidget(sep)

        # Pipeline used
        self.metric_pipeline = QLabel("Pipeline: --")
        self.metric_pipeline.setStyleSheet("color: #cdd6f4; border: none;")
        metrics_layout.addWidget(self.metric_pipeline)

        # Signal quality score (SENSAI for GEDAI)
        self.metric_quality = QLabel("Signal Quality: --")
        self.metric_quality.setStyleSheet("color: #cdd6f4; border: none;")
        metrics_layout.addWidget(self.metric_quality)

        # Variance reduction
        self.metric_variance = QLabel("Variance Reduction: --")
        self.metric_variance.setStyleSheet("color: #cdd6f4; border: none;")
        metrics_layout.addWidget(self.metric_variance)

        # SNR improvement
        self.metric_snr = QLabel("SNR Improvement: --")
        self.metric_snr.setStyleSheet("color: #cdd6f4; border: none;")
        metrics_layout.addWidget(self.metric_snr)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: #45475a;")
        metrics_layout.addWidget(sep2)

        # Components section
        comp_title = QLabel("Component Analysis")
        comp_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #a6e3a1; border: none;")
        metrics_layout.addWidget(comp_title)

        self.metric_ica_rejected = QLabel("ICA Components Rejected: --")
        self.metric_ica_rejected.setStyleSheet("color: #cdd6f4; border: none;")
        metrics_layout.addWidget(self.metric_ica_rejected)

        self.metric_bad_channels = QLabel("Bad Channels Interpolated: --")
        self.metric_bad_channels.setStyleSheet("color: #cdd6f4; border: none;")
        metrics_layout.addWidget(self.metric_bad_channels)

        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet("background-color: #45475a;")
        metrics_layout.addWidget(sep3)

        # Frequency band power
        band_title = QLabel("Band Power Changes")
        band_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #f9e2af; border: none;")
        metrics_layout.addWidget(band_title)

        self.metric_delta = QLabel("Delta (1-4 Hz): --")
        self.metric_delta.setStyleSheet("color: #cdd6f4; font-size: 11px; border: none;")
        metrics_layout.addWidget(self.metric_delta)

        self.metric_theta = QLabel("Theta (4-8 Hz): --")
        self.metric_theta.setStyleSheet("color: #cdd6f4; font-size: 11px; border: none;")
        metrics_layout.addWidget(self.metric_theta)

        self.metric_alpha = QLabel("Alpha (8-13 Hz): --")
        self.metric_alpha.setStyleSheet("color: #cdd6f4; font-size: 11px; border: none;")
        metrics_layout.addWidget(self.metric_alpha)

        self.metric_beta = QLabel("Beta (13-30 Hz): --")
        self.metric_beta.setStyleSheet("color: #cdd6f4; font-size: 11px; border: none;")
        metrics_layout.addWidget(self.metric_beta)

        metrics_layout.addStretch()

        results_layout.addWidget(metrics_widget)

        layout.addWidget(results_group, 1)

        # Keep reference to preview_label for backward compatibility
        self.preview_label = self.traces_label

        return tab

    def _create_logs_tab(self) -> QWidget:
        """Create the Logs tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # Log display
        log_group = QGroupBox("Pipeline Log")
        log_layout = QVBoxLayout(log_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText("Pipeline output will appear here...")
        log_layout.addWidget(self.log_display)

        # Log controls
        controls = QHBoxLayout()

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_display.clear)
        controls.addWidget(clear_btn)

        save_btn = QPushButton("Save Log...")
        save_btn.clicked.connect(self._save_log)
        controls.addWidget(save_btn)

        controls.addStretch()
        log_layout.addLayout(controls)

        layout.addWidget(log_group)

        return tab

    def _create_action_bar(self) -> QWidget:
        """Create the bottom action bar."""
        bar = QFrame()
        bar.setStyleSheet("""
            QFrame {
                background-color: #313244;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 12, 16, 12)

        # Run All button
        self.run_all_btn = QPushButton("Run Selected Steps")
        self.run_all_btn.setProperty("class", "success")
        self.run_all_btn.setMinimumWidth(200)
        self.run_all_btn.setMinimumHeight(44)
        self.run_all_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.run_all_btn.clicked.connect(self._run_all)
        layout.addWidget(self.run_all_btn)

        layout.addStretch()

        # Abort button (initially hidden)
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #1e1e2e;
            }
            QPushButton:hover {
                background-color: #eba0ac;
            }
        """)
        self.abort_btn.setMinimumHeight(44)
        self.abort_btn.clicked.connect(self._abort_operation)
        self.abort_btn.hide()
        layout.addWidget(self.abort_btn)

        # Status indicator
        self.status_label = QLabel("Ready to run")
        self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        layout.addWidget(self.status_label)

        return bar

    # ========================================================================
    # ACTIONS
    # ========================================================================

    def _browse_base(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Base Folder")
        if folder:
            self.base_input.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_input.setText(folder)

    def _get_design(self) -> StudyDesign:
        groups = [g.strip() for g in self.groups_input.text().split(",") if g.strip()]
        sessions = [s.strip() for s in self.sessions_input.text().split(",") if s.strip()]
        return StudyDesign(groups=groups, sessions=sessions)

    def _on_pipeline_changed(self):
        """Handle pipeline selection change - enable/disable relevant controls."""
        is_gedai = self.gedai_radio.isChecked()

        # Enable/disable GEDAI controls
        self.gedai_thresh_input.setEnabled(is_gedai)

        # Enable/disable Traditional controls
        self.asr_checkbox.setEnabled(not is_gedai)
        self.iclabel_checkbox.setEnabled(not is_gedai)
        self.iclabel_thresh_input.setEnabled(not is_gedai and self.iclabel_checkbox.isChecked())

    def _get_config(self) -> PipelineConfig:
        cfg = PipelineConfig()

        # Preprocessing - Basic filtering
        cfg.preprocessing.l_freq = self.l_freq_input.value()
        cfg.preprocessing.h_freq = self.h_freq_input.value()

        notch_text = self.notch_input.text().strip()
        if notch_text:
            try:
                cfg.preprocessing.notch_freqs = tuple(
                    float(x) for x in notch_text.split(",") if x.strip()
                )
            except ValueError:
                pass

        # Artifact removal methods based on pipeline selection
        if self.gedai_radio.isChecked():
            # GEDAI Pipeline - use GEDAI only, disable ASR/ICLabel
            cfg.preprocessing.use_ged = True
            cfg.preprocessing.ged_threshold = self.gedai_thresh_input.value()
            cfg.preprocessing.use_asr = False
            cfg.preprocessing.use_iclabel = False
        else:
            # Traditional Pipeline - use ASR and/or ICLabel
            cfg.preprocessing.use_ged = False
            cfg.preprocessing.use_asr = self.asr_checkbox.isChecked()
            cfg.preprocessing.use_iclabel = self.iclabel_checkbox.isChecked()
            cfg.preprocessing.iclabel_brain_threshold = self.iclabel_thresh_input.value()

        # Advanced options
        cfg.preprocessing.interpolate_bad_channels = self.bad_channel_checkbox.isChecked()
        cfg.preprocessing.bad_channel_threshold = self.bad_channel_thresh.value()

        # ICA settings
        ica_method = self.ica_method_input.text().strip()
        if ica_method:
            cfg.preprocessing.ica_method = ica_method

        ica_components = self.ica_components_input.value()
        cfg.preprocessing.ica_n_components = ica_components if ica_components > 0 else None

        # Connectivity
        cfg.connectivity.use_gpu = self.gpu_checkbox.isChecked()

        # Stats
        if not hasattr(cfg, "stats") or cfg.stats is None:
            cfg.stats = StatsConfig()
        cfg.stats.n_permutations = self.perms_input.value()

        cluster_text = self.cluster_thresh_input.text().strip()
        if cluster_text:
            try:
                cfg.stats.cluster_threshold = float(cluster_text)
            except ValueError:
                cfg.stats.cluster_threshold = None
        else:
            cfg.stats.cluster_threshold = None

        # Output
        try:
            cfg.output_root = Path(self.output_input.text()).expanduser()
            cfg.output_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Output folder error: {e}")

        return cfg

    def _validate_inputs(self) -> Optional[tuple]:
        base = self.base_input.text().strip()
        if not base:
            QMessageBox.warning(self, "Missing Input", "Please select a base folder first.")
            return None

        if not Path(base).exists():
            QMessageBox.warning(self, "Invalid Path", "The base folder does not exist.")
            return None

        design = self._get_design()
        if not design.groups or not design.sessions:
            QMessageBox.warning(
                self, "Design Error",
                "Please provide at least one group and one session."
            )
            return None

        config = self._get_config()
        return Path(base), design, config

    def _log(self, msg: str):
        """Append message to log display."""
        self.log_display.append(msg)
        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_running(self, running: bool, operation: str = ""):
        """Update UI state for running/stopped."""
        self.run_all_btn.setEnabled(not running)
        self.abort_btn.setVisible(running)

        if running:
            self.status_label.setText(f"Running: {operation}")
            self.status_label.setStyleSheet("color: #89b4fa; font-weight: bold;")
            self.progress_bar.setFormat(operation)
        else:
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            self.progress_bar.setFormat("Complete")

    def _on_progress(self, msg: str):
        """Handle progress message from worker."""
        self._log(msg)

    def _on_step_progress(self, current: int, total: int, description: str):
        """Handle step progress from worker."""
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
        self.operation_label.setText(description)
        self.progress_bar.setFormat(description)

    def _on_finished(self, success: bool, message: str):
        """Handle worker completion."""
        self._set_running(False)
        self.worker = None

        if success:
            self.progress_bar.setValue(100)
            self._log(f"\n[SUCCESS] {message}")
            QMessageBox.information(self, "Complete", message)
            self._update_preview()
        else:
            self._log(f"\n[ERROR] {message}")
            QMessageBox.warning(self, "Error", message)

        self.status_bar.showMessage("Ready")

    def _run_operation(self, operation: str, steps: Optional[dict] = None):
        """Start a pipeline operation in a worker thread."""
        validated = self._validate_inputs()
        if not validated:
            return

        base_path, design, config = validated

        self._set_running(True, operation)
        self.progress_bar.setValue(0)

        self.worker = PipelineWorker(operation, base_path, design, config, steps)
        self.worker.progress.connect(self._on_progress)
        self.worker.step_progress.connect(self._on_step_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _run_preprocessing(self):
        self._run_operation("preprocessing")

    def _run_source(self):
        self._run_operation("source")

    def _run_csv(self):
        self._run_operation("csv")

    def _run_stats(self):
        self._run_operation("stats")

    def _run_all(self):
        """Run all selected steps."""
        steps = {
            "preprocessing": self.step1_check.isChecked(),
            "source": self.step2_check.isChecked(),
            "csv": self.step3_check.isChecked(),
            "stats": self.step4_check.isChecked(),
        }

        if not any(steps.values()):
            QMessageBox.warning(
                self, "No Steps Selected",
                "Please select at least one step to run."
            )
            return

        self._run_operation("all", steps)

    def _abort_operation(self):
        """Abort the running operation."""
        if self.worker and self.worker.isRunning():
            self.worker.abort()
            self._log("\n[ABORT] Aborting operation...")
            self.status_label.setText("Aborting...")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")

    def _update_preview(self):
        """Update all visualizations and metrics after preprocessing."""
        design = self._get_design()
        config = self._get_config()

        # Find the first subject's output folder
        subj_folder = None
        for g in design.groups:
            for s in design.sessions:
                folder = config.output_root / g / s
                if not folder.exists():
                    continue
                for subj in sorted(folder.glob("*")):
                    if subj.is_dir():
                        subj_folder = subj
                        break
                if subj_folder:
                    break
            if subj_folder:
                break

        if not subj_folder:
            self.traces_label.setText("No preview available")
            return

        # Load and display visualizations
        self._load_visualization(subj_folder / "traces_before_after.png", self.traces_label)
        self._load_visualization(subj_folder / "psd_before_after.png", self.psd_label)
        self._load_visualization(subj_folder / "topomap_before_after.png", self.topo_label)
        self._load_visualization(subj_folder / "gfp_before_after.png", self.gfp_label)

        # Load Alpha PLI circle plot if available
        self._load_visualization(subj_folder / "alpha_circle.png", self.pli_circle_label)

        # Load and display metrics
        self._load_metrics(subj_folder)

    def _load_visualization(self, img_path: Path, label: QLabel):
        """Load an image into a label widget."""
        if img_path.exists():
            pixmap = QPixmap(str(img_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    label.width() - 20 if label.width() > 40 else 600,
                    label.height() - 20 if label.height() > 40 else 250,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                label.setPixmap(scaled)
                return
        label.setText("Image not available")

    def _load_metrics(self, subj_folder: Path):
        """Load preprocessing metrics from JSON file."""
        metrics_path = subj_folder / "preprocessing_metrics.json"

        # Determine pipeline type from config
        if self.gedai_radio.isChecked():
            self.metric_pipeline.setText("Pipeline: GEDAI")
            self.metric_pipeline.setStyleSheet("color: #89b4fa; font-weight: bold; border: none;")
        else:
            self.metric_pipeline.setText("Pipeline: Traditional (ASR + ICLabel)")
            self.metric_pipeline.setStyleSheet("color: #a6e3a1; font-weight: bold; border: none;")

        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                # Signal quality (SENSAI score for GEDAI or computed SNR)
                quality = metrics.get('signal_quality', metrics.get('sensai_score', None))
                if quality is not None:
                    color = "#a6e3a1" if quality > 70 else "#f9e2af" if quality > 40 else "#f38ba8"
                    self.metric_quality.setText(f"Signal Quality: {quality:.1f}%")
                    self.metric_quality.setStyleSheet(f"color: {color}; font-weight: bold; border: none;")
                else:
                    self.metric_quality.setText("Signal Quality: --")

                # Variance reduction
                var_reduction = metrics.get('variance_reduction', None)
                if var_reduction is not None:
                    self.metric_variance.setText(f"Variance Reduction: {var_reduction:.1f}%")
                else:
                    self.metric_variance.setText("Variance Reduction: --")

                # SNR improvement
                snr_improvement = metrics.get('snr_improvement_db', None)
                if snr_improvement is not None:
                    sign = "+" if snr_improvement > 0 else ""
                    self.metric_snr.setText(f"SNR Improvement: {sign}{snr_improvement:.1f} dB")
                else:
                    self.metric_snr.setText("SNR Improvement: --")

                # ICA components
                ica_rejected = metrics.get('ica_components_rejected', None)
                ica_total = metrics.get('ica_components_total', None)
                if ica_rejected is not None:
                    if ica_total:
                        self.metric_ica_rejected.setText(f"ICA Components Rejected: {ica_rejected}/{ica_total}")
                    else:
                        self.metric_ica_rejected.setText(f"ICA Components Rejected: {ica_rejected}")
                else:
                    self.metric_ica_rejected.setText("ICA Components Rejected: --")

                # Bad channels
                bad_channels = metrics.get('bad_channels', [])
                if isinstance(bad_channels, list):
                    n_bad = len(bad_channels)
                    self.metric_bad_channels.setText(f"Bad Channels Interpolated: {n_bad}")
                else:
                    self.metric_bad_channels.setText(f"Bad Channels Interpolated: {bad_channels}")

                # Band power changes
                band_changes = metrics.get('band_power_changes', {})
                self._update_band_metric(self.metric_delta, "Delta (1-4 Hz)", band_changes.get('delta'))
                self._update_band_metric(self.metric_theta, "Theta (4-8 Hz)", band_changes.get('theta'))
                self._update_band_metric(self.metric_alpha, "Alpha (8-13 Hz)", band_changes.get('alpha'))
                self._update_band_metric(self.metric_beta, "Beta (13-30 Hz)", band_changes.get('beta'))

            except Exception as e:
                self._log(f"Error loading metrics: {e}")
                self._reset_metrics()
        else:
            self._reset_metrics()

    def _update_band_metric(self, label: QLabel, band_name: str, change_value):
        """Update a band power change metric label."""
        if change_value is not None:
            sign = "+" if change_value > 0 else ""
            # Green for reduction (negative), yellow for small increase, red for large increase
            if change_value < -10:
                color = "#a6e3a1"  # Good - noise reduction
            elif change_value < 10:
                color = "#cdd6f4"  # Neutral
            else:
                color = "#f9e2af"  # Warning - might be losing signal
            label.setText(f"{band_name}: {sign}{change_value:.1f}%")
            label.setStyleSheet(f"color: {color}; font-size: 11px; border: none;")
        else:
            label.setText(f"{band_name}: --")
            label.setStyleSheet("color: #cdd6f4; font-size: 11px; border: none;")

    def _reset_metrics(self):
        """Reset all metric labels to default state."""
        self.metric_quality.setText("Signal Quality: --")
        self.metric_quality.setStyleSheet("color: #cdd6f4; border: none;")
        self.metric_variance.setText("Variance Reduction: --")
        self.metric_snr.setText("SNR Improvement: --")
        self.metric_ica_rejected.setText("ICA Components Rejected: --")
        self.metric_bad_channels.setText("Bad Channels Interpolated: --")
        self.metric_delta.setText("Delta (1-4 Hz): --")
        self.metric_theta.setText("Theta (4-8 Hz): --")
        self.metric_alpha.setText("Alpha (8-13 Hz): --")
        self.metric_beta.setText("Beta (13-30 Hz): --")

    def _save_log(self):
        """Save the log to a file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "pipeline_log.txt", "Text Files (*.txt)"
        )
        if filename:
            try:
                with open(filename, "w") as f:
                    f.write(self.log_display.toPlainText())
                QMessageBox.information(self, "Saved", f"Log saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save log: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================
def launch_app() -> None:
    """Launch the NeuroStat Pipeline application."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = NeuroStatApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    launch_app()
