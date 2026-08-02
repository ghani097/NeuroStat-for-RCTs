"""Offscreen tests for the v1-style study setup view.

The view exists so a user never has to hand-author a project file, so these
tests drive the widgets the way a person would: type the study details, scan a
folder, and run.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from neurostat.contracts import StageName, StageResult, StageState
from neurostat.gui.controller import WorkflowController
from neurostat.workflow import StageOutcome, WorkflowResult

# The QApplication must outlive every widget built from it. Holding it here
# keeps it alive across tests; letting it fall out of scope crashes Qt with
# "Must construct a QApplication before a QWidget".
_APPLICATION: object | None = None


def _window(
    monkeypatch: pytest.MonkeyPatch,
    controller: WorkflowController | None = None,
):
    """Build an offscreen main window."""
    global _APPLICATION

    pytest.importorskip("PyQt6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets

    from neurostat.gui import app as gui_app

    if _APPLICATION is None:
        _APPLICATION = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return gui_app.MainWindow(controller)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    return path


def _study_tree(root: Path) -> Path:
    """Build a two-arm, two-session tree in the subject-then-session shape."""
    for arm in ("GroupA", "GroupB"):
        for subject in ("Subject 1", "Subject 2"):
            for session in ("pre", "post"):
                _touch(root / arm / subject / session / "data_ft.mat")
    return root


def test_defaults_match_the_v1_form(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _window(monkeypatch)

    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "Study Setup",
        "Pipeline",
        "Logs & Output",
    ]
    assert window.groups_input.text() == "GroupA, GroupB"
    assert window.sessions_input.text() == "pre, post"
    assert window.conditions_input.text() == "rest"
    assert window.l_freq_input.value() == pytest.approx(1.0)
    assert window.h_freq_input.value() == pytest.approx(40.0)
    assert window.notch_input.text() == "50, 100"
    assert window.traditional_radio.isChecked()
    assert window.asr_checkbox.isChecked()
    assert window.iclabel_checkbox.isChecked()
    assert window.iclabel_thresh_input.value() == pytest.approx(0.85)
    assert window.bad_channel_thresh.value() == pytest.approx(4.0)
    assert window.ica_method_input.text() == "fastica"
    assert window.ica_components_input.value() == 0
    assert window.gedai_thresh_input.value() == pytest.approx(3.0)
    # v2's normal case, and fsaverage is provisioned the way v1 did it.
    assert window.already_cleaned_checkbox.isChecked()
    assert window.auto_fsaverage_checkbox.isChecked()


def test_already_cleaned_toggle_shows_and_hides_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _window(monkeypatch)

    window.already_cleaned_checkbox.setChecked(True)
    assert window.preprocessing_settings_group.isHidden()
    assert window.preprocessing_pipeline_group.isHidden()
    assert window.advanced_options_group.isHidden()

    window.already_cleaned_checkbox.setChecked(False)
    assert not window.preprocessing_settings_group.isHidden()
    assert not window.preprocessing_pipeline_group.isHidden()
    assert not window.advanced_options_group.isHidden()


def test_scan_fills_the_records_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _study_tree(tmp_path / "study")
    window = _window(monkeypatch)
    window.base_input.setText(str(base))
    window.groups_input.setText("GroupA, GroupB")
    window.sessions_input.setText("pre, post")

    window.scan_button.click()

    headers = [
        window.records_table.horizontalHeaderItem(index).text()
        for index in range(window.records_table.columnCount())
    ]
    assert headers == [
        "Record",
        "Participant",
        "Arm",
        "Session",
        "Condition",
        "Format",
        "Path",
    ]
    assert window.records_table.rowCount() == 8
    arms = {
        window.records_table.item(row, 2).text()
        for row in range(window.records_table.rowCount())
    }
    assert arms == {"GroupA", "GroupB"}
    # Participants are namespaced, so `Subject 1` in each arm stays distinct.
    participants = {
        window.records_table.item(row, 1).text()
        for row in range(window.records_table.rowCount())
    }
    assert participants == {
        "groupa-subject-1",
        "groupa-subject-2",
        "groupb-subject-1",
        "groupb-subject-2",
    }


def test_scan_reports_an_unrecognised_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "flat"
    _touch(base / "loose.set")
    window = _window(monkeypatch)
    window.base_input.setText(str(base))

    window.scan_button.click()

    assert window.records_table.rowCount() == 0
    assert window.status.text()


def test_invalid_study_never_starts_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2 requires exactly two arms; three must be refused before any work."""
    base = _study_tree(tmp_path / "study")
    started: list[str] = []

    def explode(*_args: object, **_kwargs: object) -> WorkflowResult:
        started.append("run_study")
        raise AssertionError("run_study must not be reached for an invalid setup")

    controller = WorkflowController()
    controller.run_study = explode  # type: ignore[method-assign]
    window = _window(monkeypatch, controller)
    window.base_input.setText(str(base))
    window.groups_input.setText("GroupA, GroupB, GroupC")
    window.sessions_input.setText("pre, post")
    window.already_cleaned_checkbox.setChecked(True)

    window.run_button.click()

    assert started == []
    assert window.run_button.isEnabled()


def test_fsaverage_progress_carries_a_real_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ProgressEvent validates its stage, so a raw string aborts the whole run.

    This fired for every auto-fsaverage run and surfaced only as the opaque
    message "stage must be a StageName".
    """
    pytest.importorskip("PyQt6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from neurostat.gui import app as gui_app

    seen: list[object] = []
    worker = gui_app._WorkflowWorker.__new__(gui_app._WorkflowWorker)
    worker.progress = type("Signal", (), {"emit": staticmethod(seen.append)})()

    gui_app._WorkflowWorker._emit_fsaverage_progress(worker, "Fetching fsaverage")

    assert len(seen) == 1
    event = seen[0]
    assert isinstance(event.stage, StageName)
    assert event.message == "Fetching fsaverage"


def test_run_uses_the_study_form_and_fills_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PyQt6 import QtCore

    base = _study_tree(tmp_path / "study")
    seen: dict[str, object] = {}
    ui_thread = threading.get_ident()

    def fake_run_study(form, stages, *, progress=None, outcome=None):
        seen["form"] = form
        seen["stages"] = tuple(stages)
        seen["thread"] = threading.get_ident()
        result = WorkflowResult(
            (
                StageOutcome(
                    StageResult(StageName.INGEST, StageState.SUCCEEDED),
                    None,
                ),
            ),
        )
        if outcome is not None:
            for item in result.outcomes:
                outcome(item)
        return result

    controller = WorkflowController()
    controller.run_study = fake_run_study  # type: ignore[method-assign]
    window = _window(monkeypatch, controller)
    window.base_input.setText(str(base))
    window.groups_input.setText("GroupA, GroupB")
    window.sessions_input.setText("pre, post")
    window.output_input.setText(str(tmp_path / "outputs"))
    window.set_selected_stages(("ingest",))

    window.run_button.click()

    loop = QtCore.QEventLoop()
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    window.run_finished.connect(loop.quit)
    timer.start(5000)
    loop.exec()
    assert timer.isActive(), window.status.text()
    timer.stop()

    form = seen["form"]
    assert form.arms == ("GroupA", "GroupB")
    assert form.sessions == ("pre", "post")
    assert form.already_cleaned is True
    assert Path(form.base_folder) == base
    # The engine must never run on the UI thread.
    assert seen["thread"] != ui_thread
    assert window.results.rowCount() == 1
    assert window.results.item(0, 0).text() == "ingest"
