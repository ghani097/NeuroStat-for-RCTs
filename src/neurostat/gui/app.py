"""PyQt6 view layer for the NeuroStat v2 workflow GUI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from neurostat.atlas import DK68_LABELS
from neurostat.contracts import ProgressEvent, StageName
from neurostat.gui.controller import ControllerWorkflowError, WorkflowController
from neurostat.gui.discovery import (
    DiscoveredRecord,
    Layout,
    detect_layout,
    discover_records,
)
from neurostat.gui.fsaverage import MNEResourcePaths, provision_fsaverage
from neurostat.gui.graphvar_setup import (
    NODE_SET_CUSTOM,
    NODE_SET_DK68,
    NODE_SET_TRIPLE,
    BrainNetSettings,
    GraphVarSettings,
    node_set_labels,
)
from neurostat.gui.networks import PRESET_NETWORKS
from neurostat.gui.study_builder import (
    AnalysisProfilePair,
    StudyForm,
    default_frequency_bands,
    recommended_profile_pair,
    validate_form,
)
from neurostat.workflow import StageOutcome, WorkflowResult

_STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("ingest", "Ingest"),
    ("scouts", "Scouts"),
    ("connectivity", "Connectivity"),
    ("network_models", "Network models"),
    ("graphvar", "GraphVar"),
    ("brainnet", "BrainNet"),
)

_LAYOUT_HINT = (
    "Layouts detected automatically: Group/Session/file, "
    "Group/Subject/Session/file, Group/Session/Subject/file. "
    "Files: EEGLAB .set, MNE .fif, FieldTrip .mat, BioSemi .bdf, EDF/EDF+, "
    "BrainVision .vhdr, TMSi .Poly5, LSL .xdf and WiBCI .csv"
)

_CHANNEL_MAPS: tuple[tuple[str, str], ...] = (
    ("Auto-detect", "auto"),
    ("Keep file labels", "none"),
    ("TMSi 64 (ExG1-64, drop M1/M2)", "tmsi64"),
    ("BioSemi 32 (A1-B32)", "biosemi32"),
    ("BioSemi 64 (A1-B32)", "biosemi64"),
)

_MNE_LABEL = "MNE fsaverage + continuous multitaper PLI"
_FIXED_LABEL = "Fixed linear operator + evoked average Hilbert PLI"


def _qt():
    from PyQt6 import QtCore, QtWidgets

    return QtCore, QtWidgets


QtCore, QtWidgets = _qt()


class _WorkflowWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(object)
    outcome = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str, object, str)
    validation_failed = QtCore.pyqtSignal(str)
    fsaverage_ready = QtCore.pyqtSignal(object)

    def __init__(
        self,
        controller: WorkflowController,
        stages: tuple[str, ...],
        *,
        form: StudyForm | None = None,
        project: str | None = None,
        auto_fsaverage: bool = False,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._stages = stages
        self._form = form
        self._project = project
        self._auto_fsaverage = auto_fsaverage

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            if self._form is not None:
                form = self._resolved_form()
                result = self._controller.run_study(
                    form,
                    self._stages,
                    progress=self.progress.emit,
                    outcome=self.outcome.emit,
                )
            elif self._project is not None:
                result = self._controller.run(
                    self._project,
                    self._stages,
                    progress=self.progress.emit,
                    outcome=self.outcome.emit,
                )
            else:
                raise ValueError("No study form or project YAML was supplied.")
        except ControllerWorkflowError as exc:
            self.failed.emit(exc.stage.value, exc.partial, exc.message)
        except ValueError as exc:
            self.validation_failed.emit(str(exc))
        except RuntimeError as exc:
            self.validation_failed.emit(str(exc))
        else:
            self.finished.emit(result)

    def _resolved_form(self) -> StudyForm:
        if self._form is None:
            raise ValueError("No study form was supplied.")
        form = self._form
        if (
            self._auto_fsaverage
            and form.profile_pair is AnalysisProfilePair.MNE_FSAVERAGE_CONTINUOUS
        ):
            resources = provision_fsaverage(progress=self._emit_fsaverage_progress)
            self.fsaverage_ready.emit(resources)
            form = replace(
                form,
                mne_subjects_dir=resources.subjects_dir,
                mne_subject=resources.subject,
                mne_trans=resources.trans,
                mne_source_space=resources.source_space,
                mne_bem=resources.bem,
            )
        return form

    def _emit_fsaverage_progress(self, message: str) -> None:
        # ProgressEvent validates its stage, so a raw string aborts the run.
        # fsaverage is provisioned for the source stage that consumes it.
        self.progress.emit(ProgressEvent(StageName.SCOUTS, str(message)))


class MainWindow(QtWidgets.QMainWindow):
    run_finished = QtCore.pyqtSignal()

    def __init__(self, controller: WorkflowController | None = None) -> None:
        super().__init__()
        self._controller = controller or WorkflowController()
        self._thread: object | None = None
        self._worker: object | None = None
        self._run_mode = "study"
        self._last_records: tuple[DiscoveredRecord, ...] = ()
        self._custom_network_labels: tuple[str, ...] = ()
        self._graphvar_custom_labels: tuple[str, ...] = ()
        self._result_rows: dict[str, int] = {}
        self.setWindowTitle("NeuroStat v2 workflow")
        self.resize(1180, 780)
        self._build_ui()
        self._connect()
        self._refresh_profile_controls()
        self._refresh_preprocessing_visibility()
        self._refresh_network_counts()
        self._refresh_stage_availability()
        self._refresh_plan()

    def selected_stages(self) -> tuple[str, ...]:
        if self.all_stages.isChecked():
            return self._available_stage_names()
        return tuple(
            stage
            for stage, checkbox in self.stage_checks.items()
            if checkbox.isEnabled() and checkbox.isChecked()
        )

    def set_project_path(self, path: str) -> None:
        self.project_path.setText(path)
        self._run_mode = "yaml"

    def set_selected_stages(self, stages: Sequence[str]) -> None:
        stage_set = set(stages)
        self.all_stages.setChecked("all" in stage_set)
        for stage, checkbox in self.stage_checks.items():
            checkbox.setChecked(stage in stage_set)
        self._refresh_stage_availability()
        self._refresh_plan()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        tabs = QtWidgets.QTabWidget()
        self.tabs = tabs
        tabs.addTab(self._create_study_setup_tab(), "Study Setup")
        tabs.addTab(self._create_pipeline_tab(), "Pipeline")
        tabs.addTab(self._create_logs_tab(), "Logs & Output")
        layout.addWidget(tabs, 1)

        layout.addWidget(self._create_action_bar())
        self.setCentralWidget(central)
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self._set_status("Ready")

    def _create_study_setup_tab(self) -> object:
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setSpacing(12)

        content_layout.addWidget(self._create_study_design_group())
        content_layout.addWidget(self._create_data_state_group())
        content_layout.addWidget(self._create_input_format_group())
        self.preprocessing_settings_group = self._create_preprocessing_settings_group()
        self.preprocessing_pipeline_group = self._create_preprocessing_pipeline_group()
        self.advanced_options_group = self._create_advanced_options_group()
        content_layout.addWidget(self.preprocessing_settings_group)
        content_layout.addWidget(self.preprocessing_pipeline_group)
        content_layout.addWidget(self.advanced_options_group)
        content_layout.addWidget(self._create_analysis_group())
        content_layout.addWidget(self._create_networks_group())
        content_layout.addStretch(1)

        scroll.setWidget(content)
        tab_layout.addWidget(scroll)
        return tab

    def _create_study_design_group(self) -> object:
        group = QtWidgets.QGroupBox("Study Design")
        layout = QtWidgets.QGridLayout(group)
        layout.setSpacing(10)

        layout.addWidget(QtWidgets.QLabel("Base Folder:"), 0, 0)
        self.base_input = QtWidgets.QLineEdit()
        self.base_input.setPlaceholderText(
            "Select folder containing group subfolders..."
        )
        layout.addWidget(self.base_input, 0, 1)
        self.base_browse_button = QtWidgets.QPushButton("Browse...")
        layout.addWidget(self.base_browse_button, 0, 2)

        layout.addWidget(QtWidgets.QLabel("Groups:"), 1, 0)
        self.groups_input = QtWidgets.QLineEdit("GroupA, GroupB")
        layout.addWidget(self.groups_input, 1, 1, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Sessions:"), 2, 0)
        self.sessions_input = QtWidgets.QLineEdit("pre, post")
        layout.addWidget(self.sessions_input, 2, 1, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Conditions:"), 3, 0)
        self.conditions_input = QtWidgets.QLineEdit("rest")
        layout.addWidget(self.conditions_input, 3, 1, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Study ID:"), 4, 0)
        self.study_id_input = QtWidgets.QLineEdit("study")
        layout.addWidget(self.study_id_input, 4, 1, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Output Folder:"), 5, 0)
        self.output_input = QtWidgets.QLineEdit(str(Path.cwd() / "processed"))
        layout.addWidget(self.output_input, 5, 1)
        self.output_browse_button = QtWidgets.QPushButton("Browse...")
        layout.addWidget(self.output_browse_button, 5, 2)

        hint = QtWidgets.QLabel(_LAYOUT_HINT)
        hint.setWordWrap(True)
        layout.addWidget(hint, 6, 0, 1, 3)
        return group

    def _create_data_state_group(self) -> object:
        group = QtWidgets.QGroupBox("Data state")
        layout = QtWidgets.QHBoxLayout(group)
        self.already_cleaned_checkbox = QtWidgets.QCheckBox(
            "My data is already cleaned"
        )
        self.already_cleaned_checkbox.setChecked(True)
        layout.addWidget(self.already_cleaned_checkbox)
        note = QtWidgets.QLabel(
            "v2 ingestion converts already-cleaned data into the canonical contract; "
            "it never cleans raw data."
        )
        note.setWordWrap(True)
        layout.addWidget(note, 1)
        return group

    def _create_input_format_group(self) -> object:
        group = QtWidgets.QGroupBox("Input files")
        layout = QtWidgets.QGridLayout(group)
        layout.addWidget(QtWidgets.QLabel("Channel labels:"), 0, 0)
        self.channel_map_combo = QtWidgets.QComboBox()
        for label, value in _CHANNEL_MAPS:
            self.channel_map_combo.addItem(label, value)
        layout.addWidget(self.channel_map_combo, 0, 1)
        self.channel_map_browse_button = QtWidgets.QPushButton("Load map file...")
        layout.addWidget(self.channel_map_browse_button, 0, 2)
        layout.addWidget(QtWidgets.QLabel("WiBCI CSV rate (Hz):"), 1, 0)
        self.csv_rate_input = QtWidgets.QDoubleSpinBox()
        self.csv_rate_input.setRange(1.0, 100000.0)
        self.csv_rate_input.setDecimals(1)
        self.csv_rate_input.setValue(250.0)
        layout.addWidget(self.csv_rate_input, 1, 1)
        layout.addWidget(QtWidgets.QLabel("XDF stream name:"), 2, 0)
        self.xdf_stream_input = QtWidgets.QLineEdit()
        self.xdf_stream_input.setPlaceholderText("blank = the single EEG stream")
        layout.addWidget(self.xdf_stream_input, 2, 1)
        self.check_files_button = QtWidgets.QPushButton("Check files")
        self.check_files_button.setToolTip(
            "Read the first file of each group and report channels, rate, "
            "duration, events and unmatched labels."
        )
        layout.addWidget(self.check_files_button, 2, 2)
        note = QtWidgets.QLabel(
            "Formats are detected from the file extension. Raw files that v1 "
            "cannot read are converted to .set under <output>/staged_raw first."
        )
        note.setWordWrap(True)
        layout.addWidget(note, 3, 0, 1, 3)
        return group

    def _create_preprocessing_settings_group(self) -> object:
        group = QtWidgets.QGroupBox("Preprocessing Settings")
        layout = QtWidgets.QGridLayout(group)
        layout.addWidget(QtWidgets.QLabel("Bandpass Filter (Hz):"), 0, 0)
        bandpass = QtWidgets.QHBoxLayout()
        self.l_freq_input = QtWidgets.QDoubleSpinBox()
        self.l_freq_input.setRange(0.1, 100.0)
        self.l_freq_input.setValue(1.0)
        self.l_freq_input.setDecimals(1)
        bandpass.addWidget(self.l_freq_input)
        bandpass.addWidget(QtWidgets.QLabel("to"))
        self.h_freq_input = QtWidgets.QDoubleSpinBox()
        self.h_freq_input.setRange(1.0, 500.0)
        self.h_freq_input.setValue(40.0)
        self.h_freq_input.setDecimals(1)
        bandpass.addWidget(self.h_freq_input)
        bandpass.addStretch(1)
        layout.addLayout(bandpass, 0, 1)

        layout.addWidget(QtWidgets.QLabel("Notch Filter (Hz):"), 1, 0)
        self.notch_input = QtWidgets.QLineEdit("50, 100")
        layout.addWidget(self.notch_input, 1, 1)
        return group

    def _create_preprocessing_pipeline_group(self) -> object:
        group = QtWidgets.QGroupBox("Preprocessing Pipeline")
        layout = QtWidgets.QVBoxLayout(group)
        self.pipeline_button_group = QtWidgets.QButtonGroup(self)

        self.traditional_radio = QtWidgets.QRadioButton("Traditional Pipeline")
        self.traditional_radio.setChecked(True)
        self.pipeline_button_group.addButton(self.traditional_radio)
        layout.addWidget(self.traditional_radio)

        traditional_options = QtWidgets.QHBoxLayout()
        traditional_options.setContentsMargins(20, 0, 0, 0)
        self.asr_checkbox = QtWidgets.QCheckBox("ASR")
        self.asr_checkbox.setChecked(True)
        traditional_options.addWidget(self.asr_checkbox)
        self.iclabel_checkbox = QtWidgets.QCheckBox("ICLabel")
        self.iclabel_checkbox.setChecked(True)
        traditional_options.addWidget(self.iclabel_checkbox)
        traditional_options.addWidget(QtWidgets.QLabel("Brain Threshold:"))
        self.iclabel_thresh_input = QtWidgets.QDoubleSpinBox()
        self.iclabel_thresh_input.setRange(0.0, 1.0)
        self.iclabel_thresh_input.setSingleStep(0.05)
        self.iclabel_thresh_input.setValue(0.85)
        traditional_options.addWidget(self.iclabel_thresh_input)
        traditional_options.addStretch(1)
        layout.addLayout(traditional_options)

        self.gedai_radio = QtWidgets.QRadioButton("GEDAI Pipeline")
        self.pipeline_button_group.addButton(self.gedai_radio)
        layout.addWidget(self.gedai_radio)

        ged_options = QtWidgets.QHBoxLayout()
        ged_options.setContentsMargins(20, 0, 0, 0)
        ged_options.addWidget(QtWidgets.QLabel("Threshold:"))
        self.gedai_thresh_input = QtWidgets.QDoubleSpinBox()
        self.gedai_thresh_input.setRange(1.0, 5.0)
        self.gedai_thresh_input.setSingleStep(0.5)
        self.gedai_thresh_input.setValue(3.0)
        ged_options.addWidget(self.gedai_thresh_input)
        ged_options.addStretch(1)
        layout.addLayout(ged_options)
        return group

    def _create_advanced_options_group(self) -> object:
        group = QtWidgets.QGroupBox("Advanced Options")
        layout = QtWidgets.QGridLayout(group)
        self.bad_channel_checkbox = QtWidgets.QCheckBox(
            "Auto-detect & Interpolate Bad Channels"
        )
        self.bad_channel_checkbox.setChecked(True)
        layout.addWidget(self.bad_channel_checkbox, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Z-score threshold:"), 0, 1)
        self.bad_channel_thresh = QtWidgets.QDoubleSpinBox()
        self.bad_channel_thresh.setRange(2.0, 10.0)
        self.bad_channel_thresh.setValue(4.0)
        self.bad_channel_thresh.setDecimals(1)
        layout.addWidget(self.bad_channel_thresh, 0, 2)

        layout.addWidget(QtWidgets.QLabel("ICA Method:"), 1, 0)
        self.ica_method_input = QtWidgets.QLineEdit("fastica")
        layout.addWidget(self.ica_method_input, 1, 1, 1, 2)

        layout.addWidget(QtWidgets.QLabel("ICA Components:"), 2, 0)
        self.ica_components_input = QtWidgets.QSpinBox()
        self.ica_components_input.setRange(0, 100)
        self.ica_components_input.setValue(0)
        self.ica_components_input.setSpecialValueText("Auto")
        layout.addWidget(self.ica_components_input, 2, 1, 1, 2)
        return group

    def _create_analysis_group(self) -> object:
        group = QtWidgets.QGroupBox("Analysis")
        layout = QtWidgets.QGridLayout(group)
        layout.setSpacing(10)

        layout.addWidget(QtWidgets.QLabel("Source / Connectivity:"), 0, 0)
        self.profile_pair_combo = QtWidgets.QComboBox()
        self.profile_pair_combo.addItem(
            _MNE_LABEL,
            AnalysisProfilePair.MNE_FSAVERAGE_CONTINUOUS.value,
        )
        self.profile_pair_combo.addItem(
            _FIXED_LABEL,
            AnalysisProfilePair.FIXED_EVOKED.value,
        )
        layout.addWidget(self.profile_pair_combo, 0, 1, 1, 3)

        self.auto_fsaverage_checkbox = QtWidgets.QCheckBox("Auto (fetch fsaverage)")
        self.auto_fsaverage_checkbox.setChecked(True)
        layout.addWidget(self.auto_fsaverage_checkbox, 1, 1, 1, 3)

        self.subjects_dir_input = self._path_row(
            layout, 2, "subjects_dir", directory=True
        )
        self.trans_input = self._path_row(layout, 3, "trans")
        self.source_space_input = self._path_row(layout, 4, "source space")
        self.bem_input = self._path_row(layout, 5, "bem")

        self.fixed_operator_input = self._path_row(layout, 6, "operator .mat")
        layout.addWidget(QtWidgets.QLabel("sensor labels:"), 7, 0)
        self.fixed_sensor_labels_input = QtWidgets.QLineEdit()
        layout.addWidget(self.fixed_sensor_labels_input, 7, 1, 1, 3)
        layout.addWidget(QtWidgets.QLabel("scout labels:"), 8, 0)
        self.fixed_scout_labels_input = QtWidgets.QLineEdit()
        layout.addWidget(self.fixed_scout_labels_input, 8, 1, 1, 3)

        self.frequency_table = QtWidgets.QTableWidget(0, 3)
        self.frequency_table.setHorizontalHeaderLabels(("Band", "Low", "High"))
        self.frequency_table.verticalHeader().setVisible(False)
        self.frequency_table.horizontalHeader().setStretchLastSection(True)
        self.frequency_table.setMinimumHeight(160)
        for name, (low, high) in default_frequency_bands().items():
            row = self.frequency_table.rowCount()
            self.frequency_table.insertRow(row)
            self.frequency_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.frequency_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(low)))
            self.frequency_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(high)))
        layout.addWidget(QtWidgets.QLabel("Frequency Bands:"), 9, 0)
        layout.addWidget(self.frequency_table, 9, 1, 1, 3)

        self.graphvar_executable_input = self._path_row(layout, 10, "GraphVar MATLAB")
        self.graphvar_root_input = self._path_row(
            layout, 11, "GraphVar root", directory=True
        )
        layout.addWidget(QtWidgets.QLabel("GraphVar regions:"), 12, 0)
        regions = QtWidgets.QHBoxLayout()
        self.graphvar_node_set_combo = QtWidgets.QComboBox()
        self.graphvar_node_set_combo.addItem("All 68 DK regions", NODE_SET_DK68)
        self.graphvar_node_set_combo.addItem(
            "Triple network (SN, CEN, DMN: 28)", NODE_SET_TRIPLE
        )
        self.graphvar_node_set_combo.addItem("Custom set...", NODE_SET_CUSTOM)
        regions.addWidget(self.graphvar_node_set_combo)
        self.graphvar_custom_name_input = QtWidgets.QLineEdit()
        self.graphvar_custom_name_input.setPlaceholderText("Set name, for example DAN")
        regions.addWidget(self.graphvar_custom_name_input)
        self.graphvar_node_count = QtWidgets.QLabel()
        regions.addWidget(self.graphvar_node_count)
        layout.addLayout(regions, 12, 1, 1, 2)
        self.graphvar_regions_button = QtWidgets.QPushButton("Choose regions...")
        layout.addWidget(self.graphvar_regions_button, 12, 3)
        self.graphvar_viewer_config_input = self._path_row(
            layout, 13, "GraphVar viewer config"
        )
        self.brainnet_executable_input = self._path_row(layout, 14, "BrainNet MATLAB")
        self.brainnet_root_input = self._path_row(
            layout, 15, "BrainNet root", directory=True
        )
        self.brainnet_surface_input = self._path_row(layout, 16, "BrainNet surface")
        self.brainnet_viewer_config_input = self._path_row(
            layout, 17, "BrainNet viewer config"
        )
        layout.addWidget(QtWidgets.QLabel("GraphVar permutations:"), 18, 0)
        counts = QtWidgets.QHBoxLayout()
        self.graphvar_permutations_input = QtWidgets.QSpinBox()
        self.graphvar_permutations_input.setRange(100, 100000)
        self.graphvar_permutations_input.setSingleStep(1000)
        self.graphvar_permutations_input.setValue(5000)
        counts.addWidget(self.graphvar_permutations_input)
        counts.addWidget(QtWidgets.QLabel("BrainNet links per figure:"))
        self.brainnet_top_k_input = QtWidgets.QSpinBox()
        self.brainnet_top_k_input.setRange(1, 200)
        self.brainnet_top_k_input.setValue(8)
        counts.addWidget(self.brainnet_top_k_input)
        counts.addStretch(1)
        layout.addLayout(counts, 18, 1, 1, 3)
        note = QtWidgets.QLabel(
            "The GraphVar atlas workbook is written for you from fsaverage MNI "
            "centroids. BrainNet draws only bands whose component passes the "
            "node-level permutation test."
        )
        note.setWordWrap(True)
        layout.addWidget(note, 19, 0, 1, 4)
        self._refresh_graphvar_node_set()
        return group

    def _create_networks_group(self) -> object:
        group = QtWidgets.QGroupBox("Networks")
        layout = QtWidgets.QGridLayout(group)
        layout.setSpacing(8)
        self.network_checks: dict[str, object] = {}
        self.network_count_labels: dict[str, object] = {}
        for row, name in enumerate(("SN", "CEN", "DMN")):
            checkbox = QtWidgets.QCheckBox(name)
            checkbox.setChecked(True)
            self.network_checks[name] = checkbox
            layout.addWidget(checkbox, row, 0)
            count = QtWidgets.QLabel()
            self.network_count_labels[name] = count
            layout.addWidget(count, row, 1)

        self.custom_network_check = QtWidgets.QCheckBox("Custom...")
        self.network_checks["Custom"] = self.custom_network_check
        layout.addWidget(self.custom_network_check, 3, 0)
        self.custom_network_count = QtWidgets.QLabel("0 labels")
        self.network_count_labels["Custom"] = self.custom_network_count
        layout.addWidget(self.custom_network_count, 3, 1)
        self.custom_network_button = QtWidgets.QPushButton("Choose labels...")
        layout.addWidget(self.custom_network_button, 3, 2)
        layout.setColumnStretch(3, 1)
        return group

    def _path_row(
        self,
        layout: QtWidgets.QGridLayout,
        row: int,
        label: str,
        *,
        directory: bool = False,
    ) -> QtWidgets.QLineEdit:
        layout.addWidget(QtWidgets.QLabel(f"{label}:"), row, 0)
        line_edit = QtWidgets.QLineEdit()
        layout.addWidget(line_edit, row, 1, 1, 2)
        button = QtWidgets.QPushButton("Browse...")
        button.clicked.connect(
            lambda _checked=False, field=line_edit, use_dir=directory: (
                self._browse_path(
                    field,
                    use_dir,
                )
            )
        )
        layout.addWidget(button, row, 3)
        line_edit._browse_button = button
        return line_edit

    def _create_pipeline_tab(self) -> object:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(12)

        discovery_group = QtWidgets.QGroupBox("Discovered Records")
        discovery_layout = QtWidgets.QVBoxLayout(discovery_group)
        self.scan_button = QtWidgets.QPushButton("Scan")
        discovery_layout.addWidget(self.scan_button)
        self.records_table = QtWidgets.QTableWidget(0, 7)
        self.records_table.setHorizontalHeaderLabels(
            ("Record", "Participant", "Arm", "Session", "Condition", "Format", "Path")
        )
        self.records_table.horizontalHeader().setStretchLastSection(True)
        self.records_table.verticalHeader().setVisible(False)
        discovery_layout.addWidget(self.records_table)
        layout.addWidget(discovery_group, 1)

        stage_group = QtWidgets.QGroupBox("Stages")
        stage_layout = QtWidgets.QVBoxLayout(stage_group)
        check_row = QtWidgets.QHBoxLayout()
        self.all_stages = QtWidgets.QCheckBox("all")
        self.all_stages.setChecked(True)
        check_row.addWidget(self.all_stages)
        self.stage_checks: dict[str, object] = {}
        for stage, label in _STAGE_LABELS:
            checkbox = QtWidgets.QCheckBox(label)
            self.stage_checks[stage] = checkbox
            check_row.addWidget(checkbox)
        check_row.addStretch(1)
        stage_layout.addLayout(check_row)
        self.plan_list = QtWidgets.QListWidget()
        stage_layout.addWidget(self.plan_list)
        layout.addWidget(stage_group)

        progress_group = QtWidgets.QGroupBox("Progress")
        progress_layout = QtWidgets.QVBoxLayout(progress_group)
        self.stage_progress = QtWidgets.QProgressBar()
        self.stage_progress.setMinimum(0)
        self.stage_progress.setValue(0)
        self.stage_progress.setFormat("0 of 0 stages")
        progress_layout.addWidget(self.stage_progress)
        self.progress_list = QtWidgets.QListWidget()
        progress_layout.addWidget(self.progress_list)
        layout.addWidget(progress_group, 1)

        self.results = QtWidgets.QTableWidget(0, 4)
        self.results.setHorizontalHeaderLabels(("Stage", "State", "Run ID", "Reused"))
        self.results.horizontalHeader().setStretchLastSection(True)
        self.results.verticalHeader().setVisible(False)
        layout.addWidget(self._grouped("Results", self.results), 1)
        return tab

    def _create_logs_tab(self) -> object:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        group = QtWidgets.QGroupBox("Pipeline Log")
        group_layout = QtWidgets.QVBoxLayout(group)
        self.log_display = QtWidgets.QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText("Pipeline output will appear here...")
        group_layout.addWidget(self.log_display)
        controls = QtWidgets.QHBoxLayout()
        self.clear_log_button = QtWidgets.QPushButton("Clear Log")
        controls.addWidget(self.clear_log_button)
        self.save_log_button = QtWidgets.QPushButton("Save Log...")
        controls.addWidget(self.save_log_button)
        controls.addStretch(1)
        group_layout.addLayout(controls)
        layout.addWidget(group)
        return tab

    def _create_action_bar(self) -> object:
        bar = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout(bar)
        self.run_button = QtWidgets.QPushButton("Run Selected Steps")
        layout.addWidget(self.run_button)
        self.open_project_button = QtWidgets.QPushButton(
            "Open existing project.yaml..."
        )
        layout.addWidget(self.open_project_button)
        self.project_path = QtWidgets.QLineEdit()
        self.project_path.setVisible(False)
        layout.addWidget(self.project_path)
        layout.addStretch(1)
        self.status = QtWidgets.QLabel()
        layout.addWidget(self.status)
        return bar

    def _grouped(self, title: str, widget: object) -> object:
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _connect(self) -> None:
        self.base_browse_button.clicked.connect(self._browse_base)
        self.output_browse_button.clicked.connect(self._browse_output)
        self.already_cleaned_checkbox.toggled.connect(
            self._refresh_preprocessing_visibility
        )
        self.traditional_radio.toggled.connect(self._refresh_preprocessing_controls)
        self.gedai_radio.toggled.connect(self._refresh_preprocessing_controls)
        self.iclabel_checkbox.toggled.connect(self._refresh_preprocessing_controls)
        self.profile_pair_combo.currentIndexChanged.connect(
            self._refresh_profile_controls
        )
        self.auto_fsaverage_checkbox.toggled.connect(self._refresh_profile_controls)
        for field in (
            self.graphvar_executable_input,
            self.graphvar_root_input,
            self.graphvar_viewer_config_input,
            self.brainnet_executable_input,
            self.brainnet_root_input,
            self.brainnet_surface_input,
            self.brainnet_viewer_config_input,
        ):
            field.textChanged.connect(self._refresh_stage_availability)
        for checkbox in self.network_checks.values():
            checkbox.toggled.connect(self._refresh_network_counts)
        self.custom_network_check.toggled.connect(self._custom_network_toggled)
        self.graphvar_node_set_combo.currentIndexChanged.connect(
            self._graphvar_node_set_changed
        )
        self.graphvar_regions_button.clicked.connect(self._choose_graphvar_regions)
        self.custom_network_button.clicked.connect(self._choose_custom_network)
        self.scan_button.clicked.connect(self._scan_records)
        self.check_files_button.clicked.connect(self._check_files)
        self.channel_map_browse_button.clicked.connect(self._browse_channel_map)
        self.run_button.clicked.connect(self._start_run)
        self.open_project_button.clicked.connect(self._choose_project)
        self.clear_log_button.clicked.connect(self.log_display.clear)
        self.save_log_button.clicked.connect(self._save_log)
        self.all_stages.toggled.connect(self._refresh_plan)
        for checkbox in self.stage_checks.values():
            checkbox.toggled.connect(self._stage_toggled)

    def _browse_base(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Base Folder",
        )
        if folder:
            self.base_input.setText(folder)

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
        )
        if folder:
            self.output_input.setText(folder)

    def _browse_path(self, field: QtWidgets.QLineEdit, directory: bool) -> None:
        if directory:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose folder")
        else:
            path, _selected = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Choose file",
                "",
                "All files (*)",
            )
        if path:
            field.setText(path)

    def _choose_project(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose project YAML",
            "",
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.set_project_path(path)
            self._set_status(f"Selected existing project YAML: {path}")
            self._log(f"Selected existing project YAML: {path}")

    def _stage_toggled(self) -> None:
        if self.sender() is not None and self.sender() is not self.all_stages:
            self.all_stages.setChecked(False)
        self._refresh_plan()

    def _refresh_plan(self) -> None:
        self.plan_list.clear()
        try:
            plan = self._controller.resolve_plan(self.selected_stages())
        except ValueError as exc:
            self._set_status(str(exc))
            return
        for stage in plan:
            self.plan_list.addItem(stage.value)
        self._set_status(f"{len(plan)} stage(s) selected")

    def _available_stage_names(self) -> tuple[str, ...]:
        return tuple(
            stage
            for stage, checkbox in self.stage_checks.items()
            if checkbox.isEnabled()
        )

    def _refresh_stage_availability(self) -> None:
        graphvar_ready = self._graphvar_resources_supplied()
        brainnet_ready = graphvar_ready and self._brainnet_resources_supplied()
        self._set_stage_available(
            "graphvar",
            graphvar_ready,
            "Requires MATLAB, GraphVar root and viewer config. Configure under "
            "Analysis.",
        )
        self._set_stage_available(
            "brainnet",
            brainnet_ready,
            "Requires MATLAB, GraphVar resources, BrainNet root, surface file, "
            "and viewer config. Configure under Analysis.",
        )
        self._refresh_plan()

    def _set_stage_available(
        self,
        stage: str,
        available: bool,
        disabled_tooltip: str,
    ) -> None:
        checkbox = self.stage_checks[stage]
        checkbox.setEnabled(available)
        checkbox.setToolTip("" if available else disabled_tooltip)
        if not available:
            checkbox.setChecked(False)

    def _graphvar_resources_supplied(self) -> bool:
        return all(
            self._has_text(field)
            for field in (
                self.graphvar_executable_input,
                self.graphvar_root_input,
                self.graphvar_viewer_config_input,
            )
        )

    def _brainnet_resources_supplied(self) -> bool:
        return all(
            self._has_text(field)
            for field in (
                self.brainnet_executable_input,
                self.brainnet_root_input,
                self.brainnet_surface_input,
                self.brainnet_viewer_config_input,
            )
        )

    def _refresh_network_counts(self) -> None:
        for name, labels in PRESET_NETWORKS.items():
            self.network_count_labels[name].setText(f"{len(labels)} labels")
        self.custom_network_count.setText(f"{len(self._custom_network_labels)} labels")
        self._refresh_plan()

    def _custom_network_toggled(self, checked: bool) -> None:
        if checked and not self._custom_network_labels:
            self._choose_custom_network()

    def _choose_custom_network(self) -> None:
        chosen = self._pick_dk68_labels(
            "Custom network labels", self._custom_network_labels
        )
        if chosen is not None:
            self._custom_network_labels = chosen
            self.custom_network_check.setChecked(bool(self._custom_network_labels))
        elif not self._custom_network_labels:
            self.custom_network_check.setChecked(False)
        self._refresh_network_counts()

    def _graphvar_node_set_changed(self) -> None:
        if (
            self.graphvar_node_set_combo.currentData() == NODE_SET_CUSTOM
            and not self._graphvar_custom_labels
        ):
            self._choose_graphvar_regions()
        self._refresh_graphvar_node_set()

    def _choose_graphvar_regions(self) -> None:
        chosen = self._pick_dk68_labels(
            "GraphVar custom regions", self._graphvar_custom_labels
        )
        if chosen is not None:
            self._graphvar_custom_labels = chosen
            index = self.graphvar_node_set_combo.findData(NODE_SET_CUSTOM)
            self.graphvar_node_set_combo.setCurrentIndex(index)
        self._refresh_graphvar_node_set()

    def _refresh_graphvar_node_set(self) -> None:
        node_set = str(self.graphvar_node_set_combo.currentData())
        custom = node_set == NODE_SET_CUSTOM
        self.graphvar_custom_name_input.setEnabled(custom)
        self.graphvar_regions_button.setEnabled(custom)
        settings = GraphVarSettings(
            Path("matlab"),
            Path("."),
            Path("."),
            node_set=node_set,
            custom_labels=self._graphvar_custom_labels,
        )
        try:
            count = len(node_set_labels(settings))
        except ValueError:
            count = 0
        self.graphvar_node_count.setText(f"{count} regions")

    def _pick_dk68_labels(
        self, title: str, current: tuple[str, ...]
    ) -> tuple[str, ...] | None:
        """Show the 68-region checklist; return the choice, or None on cancel."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dialog)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(content)
        chosen = set(current)
        boxes: list[tuple[str, QtWidgets.QCheckBox]] = []
        for row, index in enumerate(range(0, len(DK68_LABELS), 2)):
            left = DK68_LABELS[index]
            right = DK68_LABELS[index + 1]
            parcel = left.removesuffix("-lh")
            grid.addWidget(QtWidgets.QLabel(parcel), row, 0)
            for column, label in enumerate((left, right), 1):
                checkbox = QtWidgets.QCheckBox(label)
                checkbox.setChecked(label in chosen)
                boxes.append((label, checkbox))
                grid.addWidget(checkbox, row, column)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return tuple(label for label, checkbox in boxes if checkbox.isChecked())
        return None

    def _refresh_preprocessing_visibility(self) -> None:
        visible = not self.already_cleaned_checkbox.isChecked()
        self.preprocessing_settings_group.setVisible(visible)
        self.preprocessing_pipeline_group.setVisible(visible)
        self.advanced_options_group.setVisible(visible)
        self._refresh_preprocessing_controls()

    def _refresh_preprocessing_controls(self) -> None:
        traditional = self.traditional_radio.isChecked()
        self.asr_checkbox.setEnabled(traditional)
        self.iclabel_checkbox.setEnabled(traditional)
        self.iclabel_thresh_input.setEnabled(
            traditional and self.iclabel_checkbox.isChecked()
        )
        self.gedai_thresh_input.setEnabled(self.gedai_radio.isChecked())

    def _refresh_profile_controls(self) -> None:
        profile = self._selected_profile_pair()
        is_mne = profile is AnalysisProfilePair.MNE_FSAVERAGE_CONTINUOUS
        auto_mne = is_mne and self.auto_fsaverage_checkbox.isChecked()
        self.auto_fsaverage_checkbox.setEnabled(is_mne)
        for field in (
            self.subjects_dir_input,
            self.trans_input,
            self.source_space_input,
            self.bem_input,
        ):
            field.setEnabled(is_mne and not auto_mne)
            button = field._browse_button
            button.setEnabled(is_mne and not auto_mne)
        for field in (
            self.fixed_operator_input,
            self.fixed_sensor_labels_input,
            self.fixed_scout_labels_input,
        ):
            field.setEnabled(not is_mne)
        fixed_button = self.fixed_operator_input._browse_button
        fixed_button.setEnabled(not is_mne)

    def _selected_profile_pair(self) -> AnalysisProfilePair:
        value = self.profile_pair_combo.currentData()
        return AnalysisProfilePair(str(value))

    def _selected_networks(self) -> dict[str, tuple[str, ...]]:
        networks: dict[str, tuple[str, ...]] = {}
        for name, labels in PRESET_NETWORKS.items():
            if self.network_checks[name].isChecked():
                networks[name] = tuple(labels)
        if self.custom_network_check.isChecked():
            networks["Custom"] = self._custom_network_labels
        return networks

    def _scan_records(self) -> None:
        try:
            records, layout = self._discover_from_widgets()
        except ValueError as exc:
            self._set_status(str(exc))
            self._log(str(exc))
            return
        self._fill_records_table(records)
        self._last_records = records
        if records:
            message = f"Detected layout {layout.value}; found {len(records)} record(s)."
            message += self._align_profile_pair(records)
        else:
            message = (
                "No EEG records found. Supported layouts: Group/Session/file, "
                "Group/Subject/Session/file, Group/Session/Subject/file."
            )
        self._set_status(message)
        self._log(message)

    def _align_profile_pair(self, records: tuple[DiscoveredRecord, ...]) -> str:
        """Select the only source pairing the scanned data can use.

        Epoched FieldTrip data cannot go through the MNE fsaverage backend, and
        continuous raw cannot go through the fixed operator. Choosing here means
        the mismatch is caught at scan time rather than after a long ingest.
        """
        recommended = recommended_profile_pair(records)
        if recommended is None or recommended is self._selected_profile_pair():
            return ""
        label = (
            _MNE_LABEL
            if recommended is AnalysisProfilePair.MNE_FSAVERAGE_CONTINUOUS
            else _FIXED_LABEL
        )
        index = self.profile_pair_combo.findText(label)
        if index < 0:
            return ""
        self.profile_pair_combo.setCurrentIndex(index)
        self._refresh_profile_controls()
        return f" Source set to {label} to match the discovered data."

    def _discover_from_widgets(self) -> tuple[tuple[DiscoveredRecord, ...], Layout]:
        base = Path(self.base_input.text().strip()).expanduser()
        arms = self._csv_values(self.groups_input.text())
        sessions = self._csv_values(self.sessions_input.text())
        conditions = self._csv_values(self.conditions_input.text())
        if not base:
            raise ValueError("Select a base folder before scanning.")
        layout = detect_layout(base, sessions)
        records = discover_records(base, arms, sessions, conditions, layout)
        return records, layout

    def _fill_records_table(self, records: tuple[DiscoveredRecord, ...]) -> None:
        self.records_table.setRowCount(0)
        for record in records:
            row = self.records_table.rowCount()
            self.records_table.insertRow(row)
            values = (
                record.record_id,
                record.participant_id,
                record.arm,
                record.session,
                record.condition,
                record.format,
                str(record.path),
            )
            for column, value in enumerate(values):
                self.records_table.setItem(
                    row,
                    column,
                    QtWidgets.QTableWidgetItem(value),
                )

    def _start_run(self) -> None:
        stages = self.selected_stages()
        try:
            plan = self._controller.resolve_plan(stages)
        except ValueError as exc:
            self._set_status(str(exc))
            self._log(str(exc))
            return

        if self._run_mode == "yaml":
            project = self.project_path.text().strip()
            if not project:
                self._set_status("choose a project YAML before running")
                return
            self._start_worker(project=project, stages=stages)
            return

        try:
            form = self._build_form()
            records = discover_records(
                form.base_folder,
                form.arms,
                form.sessions,
                form.conditions,
            )
            # Fields can change after a scan, so align the source pairing against
            # what is actually on disk and rebuild the form from the result.
            if self._align_profile_pair(records):
                form = self._build_form()
        except ValueError as exc:
            self._report_refusal(str(exc))
            return
        self._fill_records_table(records)
        self._last_records = records
        problems = validate_form(form, records, plan)
        if problems:
            self._report_refusal(
                "Study setup is not valid: " + "; ".join(problems),
            )
            return
        self._start_worker(
            form=form,
            stages=stages,
            auto_fsaverage=self.auto_fsaverage_checkbox.isChecked(),
        )

    def _report_refusal(self, message: str) -> None:
        """Report a run that was refused before any worker started.

        `run_finished` still has to fire, otherwise anything waiting on the run
        waits forever for a run that will never happen.
        """
        self._set_status(message)
        self._log(message)
        self.run_button.setEnabled(True)
        self.run_finished.emit()

    def _build_form(self) -> StudyForm:
        profile_pair = self._selected_profile_pair()
        return StudyForm(
            study_id=self.study_id_input.text().strip() or "study",
            base_folder=Path(self.base_input.text().strip()).expanduser(),
            arms=self._csv_values(self.groups_input.text()),
            sessions=self._csv_values(self.sessions_input.text()),
            conditions=self._csv_values(self.conditions_input.text()),
            output_root=Path(self.output_input.text().strip()).expanduser(),
            already_cleaned=self.already_cleaned_checkbox.isChecked(),
            frequency_bands=self._frequency_bands(),
            profile_pair=profile_pair,
            fixed_operator=self._optional_path(self.fixed_operator_input.text()),
            fixed_sensor_labels=self._csv_values(self.fixed_sensor_labels_input.text()),
            fixed_scout_labels=self._csv_values(self.fixed_scout_labels_input.text()),
            mne_subjects_dir=self._optional_path(self.subjects_dir_input.text()),
            mne_trans=self._optional_path(self.trans_input.text()),
            mne_source_space=self._optional_path(self.source_space_input.text()),
            mne_bem=self._optional_path(self.bem_input.text()),
            preprocessing_settings=self._preprocessing_settings(),
            input_channel_map=str(self.channel_map_combo.currentData() or "auto"),
            xdf_stream=self.xdf_stream_input.text().strip() or None,
            csv_sampling_rate=(
                None
                if self.csv_rate_input.value() == 250.0
                else self.csv_rate_input.value()
            ),
            graphvar=self._graphvar_settings(),
            brainnet=self._brainnet_settings(),
            networks=self._selected_networks(),
        )

    def _graphvar_settings(self) -> GraphVarSettings | None:
        if not self._graphvar_resources_supplied():
            return None
        return GraphVarSettings(
            matlab_executable=Path(self.graphvar_executable_input.text().strip()).expanduser(),
            graphvar_root=Path(self.graphvar_root_input.text().strip()).expanduser(),
            viewer_config=Path(
                self.graphvar_viewer_config_input.text().strip()
            ).expanduser(),
            node_set=str(self.graphvar_node_set_combo.currentData()),
            custom_name=self.graphvar_custom_name_input.text().strip() or "Custom",
            custom_labels=self._graphvar_custom_labels,
            permutations=self.graphvar_permutations_input.value(),
        )

    def _brainnet_settings(self) -> BrainNetSettings | None:
        if not self._brainnet_resources_supplied():
            return None
        return BrainNetSettings(
            brainnet_root=Path(self.brainnet_root_input.text().strip()).expanduser(),
            surface=Path(self.brainnet_surface_input.text().strip()).expanduser(),
            viewer_config=Path(
                self.brainnet_viewer_config_input.text().strip()
            ).expanduser(),
            matlab_executable=self._optional_path(self.brainnet_executable_input.text()),
            top_k=self.brainnet_top_k_input.value(),
        )

    def _browse_channel_map(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select channel map (two columns: file label, new label)",
            "",
            "Channel maps (*.tsv *.csv *.txt)",
        )
        if not path:
            return
        index = self.channel_map_combo.findData(path)
        if index < 0:
            self.channel_map_combo.addItem(f"File: {Path(path).name}", path)
            index = self.channel_map_combo.count() - 1
        self.channel_map_combo.setCurrentIndex(index)

    def _check_files(self) -> None:
        from neurostat.io.staging import preview_records

        try:
            records, _layout = self._discover_from_widgets()
            form = self._build_form()
        except ValueError as exc:
            self._set_status(str(exc))
            self._log(str(exc))
            return
        if not records:
            self._set_status("no EEG files found to check")
            return
        problems = 0
        for record, summary, error in preview_records(
            records, reader_options=form.reader_options()
        ):
            if error is not None:
                problems += 1
                self._log(f"{record.path.name}: cannot read: {error}")
                continue
            assert summary is not None
            self._log(
                f"{record.path.name} [{summary['format']}]: {summary['channels']} channels, "
                f"{summary['sfreq']:g} Hz, {summary['duration_seconds']:.1f} s, "
                f"{summary['events']} events, labels {summary['channel_map']}"
            )
            if summary["unmatched_labels"]:
                problems += 1
                self._log(
                    "  no standard position for: "
                    + ", ".join(summary["unmatched_labels"][:12])
                )
            for note in summary["notes"]:
                self._log(f"  note: {note}")
        self._set_status(
            "files look readable" if not problems else f"{problems} file problem(s); see log"
        )

    def _frequency_bands(self) -> Mapping[str, tuple[float, float]]:
        bands: dict[str, tuple[float, float]] = {}
        for row in range(self.frequency_table.rowCount()):
            name_item = self.frequency_table.item(row, 0)
            low_item = self.frequency_table.item(row, 1)
            high_item = self.frequency_table.item(row, 2)
            if name_item is None or low_item is None or high_item is None:
                continue
            name = name_item.text().strip()
            if not name:
                continue
            try:
                bands[name] = (float(low_item.text()), float(high_item.text()))
            except ValueError as exc:
                raise ValueError(
                    f"Frequency band {name!r} must use numeric limits."
                ) from exc
        return bands

    def _preprocessing_settings(self) -> Mapping[str, Any]:
        ica_components = self.ica_components_input.value()
        return {
            "l_freq": self.l_freq_input.value(),
            "h_freq": self.h_freq_input.value(),
            "notch_freqs": self._float_csv_values(self.notch_input.text()),
            "use_asr": self.traditional_radio.isChecked()
            and self.asr_checkbox.isChecked(),
            "use_iclabel": self.traditional_radio.isChecked()
            and self.iclabel_checkbox.isChecked(),
            "iclabel_brain_threshold": self.iclabel_thresh_input.value(),
            "use_ged": self.gedai_radio.isChecked(),
            "ged_threshold": self.gedai_thresh_input.value(),
            "bad_channel_threshold": self.bad_channel_thresh.value(),
            "interpolate_bad_channels": self.bad_channel_checkbox.isChecked(),
            "ica_method": self.ica_method_input.text().strip() or "fastica",
            "ica_n_components": None if ica_components == 0 else ica_components,
        }

    def _start_worker(
        self,
        *,
        stages: tuple[str, ...],
        form: StudyForm | None = None,
        project: str | None = None,
        auto_fsaverage: bool = False,
    ) -> None:
        plan = self._controller.resolve_plan(stages)
        self.run_button.setEnabled(False)
        self._prepopulate_results(plan)
        self.progress_list.clear()
        self.stage_progress.setMaximum(len(plan))
        self.stage_progress.setValue(0)
        self.stage_progress.setStyleSheet("")
        self.stage_progress.setFormat("%v of %m stages - pending")
        self._set_status("running")
        self._log("Workflow started.")

        thread = QtCore.QThread(self)
        worker = _WorkflowWorker(
            self._controller,
            stages,
            form=form,
            project=project,
            auto_fsaverage=auto_fsaverage,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._append_progress)
        worker.outcome.connect(self._append_outcome)
        worker.fsaverage_ready.connect(self._apply_fsaverage_paths)
        worker.finished.connect(self._run_completed)
        worker.failed.connect(self._run_failed)
        worker.validation_failed.connect(self._validation_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.validation_failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _prepopulate_results(self, plan: tuple[StageName, ...]) -> None:
        self.results.setRowCount(0)
        self._result_rows = {}
        for stage in plan:
            row = self.results.rowCount()
            self.results.insertRow(row)
            self._result_rows[stage.value] = row
            values = (stage.value, "pending", "", "")
            for column, value in enumerate(values):
                self.results.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def _append_progress(self, event: ProgressEvent) -> None:
        stage = event.stage.value if hasattr(event.stage, "value") else str(event.stage)
        text = f"{stage}: {event.message}"
        if event.completed is not None and event.total is not None:
            text += f" ({event.completed}/{event.total})"
        if str(event.message).endswith("started"):
            self.stage_progress.setFormat(f"%v of %m stages - Running {stage}...")
        self.progress_list.addItem(text)
        self._log(text)

    def _append_outcome(self, outcome: StageOutcome) -> None:
        row = self._result_rows.get(outcome.stage.value)
        if row is None:
            row = self.results.rowCount()
            self.results.insertRow(row)
            self._result_rows[outcome.stage.value] = row
        values = (
            outcome.stage.value,
            outcome.state.value,
            outcome.run_id or "",
            "" if outcome.reused is None else str(outcome.reused),
        )
        for column, value in enumerate(values):
            self.results.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self._update_stage_progress(outcome.stage.value)

    def _update_stage_progress(self, current_stage: str) -> None:
        completed = 0
        failed = False
        for row in range(self.results.rowCount()):
            item = self.results.item(row, 1)
            state = "" if item is None else item.text()
            if state in {"succeeded", "failed"}:
                completed += 1
            if state == "failed":
                failed = True
        self.stage_progress.setValue(completed)
        if failed:
            self.stage_progress.setStyleSheet(
                "QProgressBar::chunk { background-color: #b3261e; }"
            )
            self.stage_progress.setFormat(
                f"%v of %m stages - failed at {current_stage}"
            )
        else:
            self.stage_progress.setFormat(f"%v of %m stages - {current_stage}")

    def _apply_fsaverage_paths(self, resources: MNEResourcePaths) -> None:
        self.subjects_dir_input.setText(str(resources.subjects_dir))
        self.trans_input.setText(str(resources.trans))
        self.source_space_input.setText(str(resources.source_space))
        self.bem_input.setText(str(resources.bem))

    def _run_completed(self, result: WorkflowResult) -> None:
        message = "workflow succeeded" if result.success else "workflow ended"
        self.stage_progress.setValue(self.stage_progress.maximum())
        self.stage_progress.setFormat("%v of %m stages - complete")
        self._set_status(message)
        self.run_button.setEnabled(True)
        if self._run_mode == "study":
            project_file = self._build_form().resolved_project_file
            self._log(f"Generated project YAML: {project_file}")
        self._log(message)
        self.run_finished.emit()

    def _run_failed(self, stage: str, partial: WorkflowResult, message: str) -> None:
        succeeded = [
            outcome.stage.value
            for outcome in partial.outcomes
            if outcome.state.value == "succeeded"
        ]
        prefix = f"{stage} failed"
        if succeeded:
            prefix += f"; succeeded: {', '.join(succeeded)}"
        status = f"{prefix}; {message}"
        self.stage_progress.setStyleSheet(
            "QProgressBar::chunk { background-color: #b3261e; }"
        )
        self.stage_progress.setFormat(f"%v of %m stages - failed at {stage}")
        self._set_status(status)
        self._log(status)
        self.run_button.setEnabled(True)
        self.run_finished.emit()

    def _validation_failed(self, message: str) -> None:
        self._set_status(message)
        self._log(message)
        self.run_button.setEnabled(True)
        self.run_finished.emit()

    def _clear_worker(self) -> None:
        self._thread = None
        self._worker = None

    def _set_status(self, message: str) -> None:
        self.status.setText(message)
        self.status_bar.showMessage(message)

    def _log(self, message: str) -> None:
        self.log_display.append(message)
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _save_log(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Log",
            "pipeline_log.txt",
            "Text Files (*.txt);;All files (*)",
        )
        if path:
            Path(path).write_text(self.log_display.toPlainText(), encoding="utf-8")

    @staticmethod
    def _csv_values(text: str) -> tuple[str, ...]:
        return tuple(value.strip() for value in text.split(",") if value.strip())

    @staticmethod
    def _float_csv_values(text: str) -> tuple[float, ...]:
        return tuple(float(value.strip()) for value in text.split(",") if value.strip())

    @staticmethod
    def _optional_path(text: str) -> Path | None:
        value = text.strip()
        return Path(value).expanduser() if value else None

    @staticmethod
    def _has_text(field: QtWidgets.QLineEdit) -> bool:
        return bool(field.text().strip())

    def closeEvent(self, event: object) -> None:
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(3000)
        super().closeEvent(event)


def main(argv: Sequence[str] | None = None) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(list(argv or []))
    window = MainWindow()
    window.show()
    return app.exec()


__all__ = ["MainWindow", "main"]
