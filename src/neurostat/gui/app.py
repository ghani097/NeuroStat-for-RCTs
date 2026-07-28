"""PyQt6 view layer for the NeuroStat v2 workflow GUI."""

from __future__ import annotations

from collections.abc import Sequence

from neurostat.contracts import ProgressEvent
from neurostat.gui.controller import ControllerWorkflowError, WorkflowController
from neurostat.workflow import StageOutcome, WorkflowResult

_STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("ingest", "Ingest"),
    ("scouts", "Scouts"),
    ("connectivity", "Connectivity"),
    ("network_models", "Network models"),
    ("graphvar", "GraphVar"),
    ("brainnet", "BrainNet"),
)


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

    def __init__(
        self,
        controller: WorkflowController,
        project: str,
        stages: tuple[str, ...],
    ) -> None:
        super().__init__()
        self._controller = controller
        self._project = project
        self._stages = stages

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            result = self._controller.run(
                self._project,
                self._stages,
                progress=self.progress.emit,
                outcome=self.outcome.emit,
            )
        except ControllerWorkflowError as exc:
            self.failed.emit(exc.stage.value, exc.partial, exc.message)
        except ValueError as exc:
            self.validation_failed.emit(str(exc))
        except RuntimeError as exc:
            self.validation_failed.emit(str(exc))
        else:
            self.finished.emit(result)


class MainWindow(QtWidgets.QMainWindow):
    run_finished = QtCore.pyqtSignal()

    def __init__(self, controller: WorkflowController | None = None) -> None:
        super().__init__()
        self._controller = controller or WorkflowController()
        self._thread: object | None = None
        self._worker: object | None = None
        self.setWindowTitle("NeuroStat v2 workflow")
        self.resize(900, 620)
        self._build_ui()
        self._connect()
        self._refresh_plan()

    def selected_stages(self) -> tuple[str, ...]:
        if self.all_stages.isChecked():
            return ("all",)
        return tuple(
            stage
            for stage, checkbox in self.stage_checks.items()
            if checkbox.isChecked()
        )

    def set_project_path(self, path: str) -> None:
        self.project_path.setText(path)

    def set_selected_stages(self, stages: Sequence[str]) -> None:
        stage_set = set(stages)
        self.all_stages.setChecked("all" in stage_set)
        for stage, checkbox in self.stage_checks.items():
            checkbox.setChecked(stage in stage_set)
        self._refresh_plan()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        project_row = QtWidgets.QHBoxLayout()
        self.project_path = QtWidgets.QLineEdit()
        self.project_path.setPlaceholderText("Project YAML")
        self.browse_button = QtWidgets.QPushButton("Browse")
        project_row.addWidget(self.project_path, 1)
        project_row.addWidget(self.browse_button)
        layout.addLayout(project_row)

        stage_box = QtWidgets.QGroupBox("Stages")
        stage_layout = QtWidgets.QHBoxLayout(stage_box)
        self.all_stages = QtWidgets.QCheckBox("all")
        self.all_stages.setChecked(True)
        stage_layout.addWidget(self.all_stages)
        self.stage_checks: dict[str, object] = {}
        for stage, label in _STAGE_LABELS:
            checkbox = QtWidgets.QCheckBox(label)
            self.stage_checks[stage] = checkbox
            stage_layout.addWidget(checkbox)
        stage_layout.addStretch(1)
        layout.addWidget(stage_box)

        split = QtWidgets.QSplitter()
        self.plan_list = QtWidgets.QListWidget()
        self.progress_list = QtWidgets.QListWidget()
        plan_group = self._grouped("Resolved plan", self.plan_list)
        progress_group = self._grouped("Progress", self.progress_list)
        split.addWidget(plan_group)
        split.addWidget(progress_group)
        layout.addWidget(split, 1)

        self.results = QtWidgets.QTableWidget(0, 4)
        self.results.setHorizontalHeaderLabels(("Stage", "State", "Run ID", "Reused"))
        self.results.horizontalHeader().setStretchLastSection(True)
        self.results.verticalHeader().setVisible(False)
        layout.addWidget(self.results, 1)

        footer = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel()
        self.run_button = QtWidgets.QPushButton("Run")
        footer.addWidget(self.status, 1)
        footer.addWidget(self.run_button)
        layout.addLayout(footer)
        self.setCentralWidget(central)

    def _grouped(self, title: str, widget: object) -> object:
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _connect(self) -> None:
        self.browse_button.clicked.connect(self._choose_project)
        self.run_button.clicked.connect(self._start_run)
        self.all_stages.toggled.connect(self._refresh_plan)
        for checkbox in self.stage_checks.values():
            checkbox.toggled.connect(self._stage_toggled)

    def _choose_project(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose project YAML",
            "",
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.project_path.setText(path)
            self._refresh_plan()

    def _stage_toggled(self) -> None:
        if self.sender() is not None and self.sender() is not self.all_stages:
            self.all_stages.setChecked(False)
        self._refresh_plan()

    def _refresh_plan(self) -> None:
        self.plan_list.clear()
        try:
            plan = self._controller.resolve_plan(self.selected_stages())
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        for stage in plan:
            self.plan_list.addItem(stage.value)
        self.status.setText(f"{len(plan)} stage(s) selected")

    def _start_run(self) -> None:
        project = self.project_path.text().strip()
        if not project:
            self.status.setText("choose a project YAML before running")
            return
        stages = self.selected_stages()
        try:
            self._controller.resolve_plan(stages)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.run_button.setEnabled(False)
        self.results.setRowCount(0)
        self.progress_list.clear()
        self.status.setText("running")

        thread = QtCore.QThread(self)
        worker = _WorkflowWorker(self._controller, project, stages)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._append_progress)
        worker.outcome.connect(self._append_outcome)
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

    def _append_progress(self, event: ProgressEvent) -> None:
        text = f"{event.stage.value}: {event.message}"
        if event.completed is not None and event.total is not None:
            text += f" ({event.completed}/{event.total})"
        self.progress_list.addItem(text)

    def _append_outcome(self, outcome: StageOutcome) -> None:
        row = self.results.rowCount()
        self.results.insertRow(row)
        values = (
            outcome.stage.value,
            outcome.state.value,
            outcome.run_id or "",
            "" if outcome.reused is None else str(outcome.reused),
        )
        for column, value in enumerate(values):
            self.results.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def _run_completed(self, result: WorkflowResult) -> None:
        self.status.setText(
            "workflow succeeded" if result.success else "workflow ended"
        )
        self.run_button.setEnabled(True)
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
        self.status.setText(f"{prefix}; {message}")
        self.run_button.setEnabled(True)
        self.run_finished.emit()

    def _validation_failed(self, message: str) -> None:
        self.status.setText(message)
        self.run_button.setEnabled(True)
        self.run_finished.emit()

    def _clear_worker(self) -> None:
        self._thread = None
        self._worker = None

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
