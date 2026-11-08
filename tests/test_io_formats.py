"""Round-trip tests for every multi-format reader in neurostat.io."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from io_writers import write_brainvision, write_edf, write_poly5, write_wibci_csv, write_xdf  # noqa: E402

from neurostat.io.channel_maps import TMSI64, ChannelMapError, auto_map, resolve_channel_map  # noqa: E402
from neurostat.io.formats import FORMATS, data_stem, format_for_path  # noqa: E402
from neurostat.io.poly5 import Poly5Error, read_poly5  # noqa: E402
from neurostat.io.reader import FormatReadError, load_recording  # noqa: E402

LABELS = ["Fp1", "Fz", "Cz", "Pz", "O1", "O2"]
SFREQ = 256


def _signal(n_channels: int = len(LABELS), seconds: int = 4, amplitude_uv: float = 100.0) -> np.ndarray:
    times = np.arange(seconds * SFREQ) / SFREQ
    return np.vstack([amplitude_uv * np.sin(2 * np.pi * (5 + i) * times) for i in range(n_channels)])


def test_registry_lists_every_requested_format():
    assert {"bdf", "edf", "brainvision", "poly5", "xdf", "wibci_csv", "set", "fif", "fieldtrip_mat"} <= set(FORMATS)
    assert format_for_path("a/b/rest.BDF") == "bdf"
    assert format_for_path("x.Poly5") == "poly5"
    assert format_for_path("x.fif.gz") == "fif"
    assert format_for_path("x.vmrk") is None and format_for_path("x.eeg") is None
    assert data_stem("Sub1_rest.fif.gz") == "Sub1_rest" and data_stem("A.Poly5") == "A"


def test_generic_csv_is_not_mistaken_for_wibci(tmp_path):
    other = tmp_path / "Sub1_PLI.csv"
    other.write_text("roi,value\nA,1\n", encoding="utf-8")
    assert format_for_path(other) is None
    wibci = tmp_path / "Sub1.csv"
    write_wibci_csv(wibci, _signal(), LABELS)
    assert format_for_path(wibci) == "wibci_csv"


@pytest.mark.parametrize("bdf", [False, True])
def test_edf_and_bdf_read_in_volts(tmp_path, bdf):
    path = tmp_path / ("rec.bdf" if bdf else "rec.edf")
    data = _signal()
    status = np.zeros(data.shape[1], dtype=int)
    status[300:310] = 7
    write_edf(path, data, LABELS, SFREQ, bdf=bdf, status=status if bdf else None)
    loaded = load_recording(path)
    assert loaded.format == ("bdf" if bdf else "edf")
    assert loaded.raw.ch_names == LABELS
    tolerance = 0.2e-6 if not bdf else 0.05e-6
    np.testing.assert_allclose(loaded.raw.get_data(), data * 1e-6, atol=tolerance)
    if bdf:
        assert "Status" in loaded.dropped_channels
        assert [a["description"] for a in loaded.raw.annotations] == ["Status/7"]
        assert loaded.raw.annotations[0]["onset"] == pytest.approx(300 / SFREQ)
    assert not loaded.unmatched_labels


def test_biosemi_labels_are_renamed_by_auto_map(tmp_path):
    path = tmp_path / "bio.bdf"
    labels = [f"A{i}" for i in range(1, 33)]
    write_edf(path, _signal(32, 2), labels, SFREQ, bdf=True, status=np.zeros(2 * SFREQ, dtype=int))
    loaded = load_recording(path)
    assert loaded.channel_map == "biosemi32"
    assert loaded.raw.ch_names[:3] == ["Fp1", "AF3", "F7"]


def test_edf_prefixed_labels_are_cleaned():
    name, mapping = auto_map(["EEG Fp1-REF", "EEG Cz-REF", "EEG O2-REF"])
    assert name == "strip-prefixes" and mapping["EEG Cz-REF"] == "Cz"


def test_brainvision_reads_markers_and_companions(tmp_path):
    from neurostat.io.reader import companion_files

    path = tmp_path / "rec.vhdr"
    data = _signal()
    write_brainvision(path, data, LABELS, SFREQ, [(512, "S  1")])
    loaded = load_recording(path)
    np.testing.assert_allclose(loaded.raw.get_data(), data * 1e-6, rtol=1e-6, atol=1e-12)
    assert any("S  1" in item["description"] for item in loaded.raw.annotations)
    assert {p.suffix for p in companion_files(path, "brainvision")} == {".eeg", ".vmrk"}


def test_poly5_tmsi64_map_matches_published_conversion(tmp_path):
    names = [f"ExG{i}" for i in range(1, 65)] + ["BIP65", "Digi"]
    units = ["\xb5V"] * 64 + ["mV", "BIT"]
    data = np.vstack([_signal(64, 2), np.zeros((1, 2 * SFREQ)), np.full((1, 2 * SFREQ), 255.0)])
    data[-1, 100:104] = 254.0
    path = tmp_path / "A00101.Poly5"
    write_poly5(path, data.astype(np.float32), names, units, SFREQ)
    loaded = load_recording(path)
    expected = [label for label in TMSI64 if label not in {"M1", "M2"}]
    assert loaded.channel_map == "tmsi64" and loaded.raw.ch_names == expected
    keep = [i for i, label in enumerate(TMSI64) if label not in {"M1", "M2"}]
    np.testing.assert_allclose(loaded.raw.get_data(), data[keep].astype(np.float32).astype(float) * 1e-6)
    assert [a["description"] for a in loaded.raw.annotations] == ["Digi/254"]
    assert {"ExG13", "ExG19", "BIP65", "Digi"} <= set(loaded.dropped_channels)


def test_poly5_rejects_other_versions_and_truncation_warns(tmp_path):
    bad = tmp_path / "bad.Poly5"
    bad.write_bytes(b"POLY SAMPLE FILEversion 2.04" + bytes(400))
    with pytest.raises(Poly5Error, match="2.04"):
        read_poly5(bad)
    good = tmp_path / "cut.Poly5"
    write_poly5(good, _signal(2, 1).astype(np.float32), ["ExG1", "ExG2"], ["uV", "uV"], SFREQ)
    good.write_bytes(good.read_bytes()[:-100])
    with pytest.warns(UserWarning, match="complete blocks"):
        poly = read_poly5(good)
    assert poly.blocks_read == poly.blocks_expected - 1


@pytest.mark.skipif(importlib.util.find_spec("pyxdf") is None, reason="pyxdf not installed")
def test_xdf_reads_eeg_stream_and_markers(tmp_path):
    path = tmp_path / "rec.xdf"
    data = _signal()
    write_xdf(path, data, LABELS, float(SFREQ), [(1.0, "go"), (2.5, "stop")])
    loaded = load_recording(path)
    assert loaded.raw.ch_names == LABELS
    assert loaded.raw.info["sfreq"] == SFREQ
    np.testing.assert_allclose(loaded.raw.get_data(), data.astype(np.float32) * 1e-6, rtol=1e-5, atol=1e-12)
    assert [a["description"] for a in loaded.raw.annotations] == ["go", "stop"]
    assert loaded.raw.annotations[1]["onset"] == pytest.approx(2.5, abs=1 / SFREQ)
    with pytest.raises(FormatReadError, match="no stream named"):
        load_recording(path, stream="Nope")


def test_wibci_csv_decodes_deltas_fills_gaps_and_finds_events(tmp_path):
    path = tmp_path / "Sub1Rest.csv"
    data = _signal(4, 4, 50.0) + 2000.0
    write_wibci_csv(path, data, LABELS[:4], drop_rows=(400, 401), events={600: 1, 800: 120})
    loaded = load_recording(path)
    assert loaded.format == "wibci_csv" and loaded.raw.info["sfreq"] == 250.0
    assert loaded.raw.ch_names == LABELS[:4]
    assert loaded.missing_fraction == pytest.approx(2 / data.shape[1])
    got = loaded.raw.get_data() * 1e6
    keep = np.r_[0:400, 402:data.shape[1]]
    np.testing.assert_allclose(got[:, keep], data[:, keep], atol=1e-6)
    np.testing.assert_allclose(got[:, 400:402], data[:, 400:402], atol=2.0)
    assert [a["description"] for a in loaded.raw.annotations] == ["hardA/1", "softA/120"]
    assert load_recording(path, sampling_rate=500).raw.info["sfreq"] == 500


def test_option_misuse_is_refused(tmp_path):
    path = tmp_path / "rec.edf"
    write_edf(path, _signal(), LABELS, SFREQ)
    with pytest.raises(FormatReadError, match="sampling_rate"):
        load_recording(path, sampling_rate=100)
    with pytest.raises(FormatReadError, match="data_unit"):
        load_recording(path, data_unit="uV")
    with pytest.raises(ChannelMapError, match="matches none"):
        load_recording(path, channel_map={"X1": "Fp1"})


def test_channel_map_file(tmp_path):
    mapping = tmp_path / "map.tsv"
    mapping.write_text("from\tto\nCh1\tFp1\nCh2\t\n", encoding="utf-8")
    name, resolved = resolve_channel_map(str(mapping), ["Ch1", "Ch2"])
    assert name == "file:map.tsv" and resolved == {"Ch1": "Fp1", "Ch2": ""}
