"""Qt-free GraphVar and BrainNet setup for GUI studies.

The GUI picks one node set (all 68 Desikan-Killiany regions, the triple
network, or a named custom set chosen from the 68), and these helpers write
the GraphVar atlas workbook and the ``graphvar``/``brainnet`` project blocks.
"""

from __future__ import annotations

import csv
import re
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from neurostat.atlas import DK68_LABELS

from .networks import PRESET_NETWORKS

#: fsaverage ``aparc`` label centres of mass in MNI space (mm), from
#: ``Label.center_of_mass`` and ``mne.vertex_to_mni``.
DK68_MNI: Mapping[str, tuple[float, float, float]] = {
    "bankssts-lh": (-49.1, -43.7, 4.0),
    "bankssts-rh": (45.0, -39.5, 6.7),
    "caudalanteriorcingulate-lh": (-5.1, 19.2, 26.4),
    "caudalanteriorcingulate-rh": (5.7, 19.7, 26.6),
    "caudalmiddlefrontal-lh": (-40.7, 14.6, 48.5),
    "caudalmiddlefrontal-rh": (41.6, 15.7, 47.8),
    "cuneus-lh": (-3.7, -76.3, 20.2),
    "cuneus-rh": (4.3, -79.1, 21.8),
    "entorhinal-lh": (-23.5, -7.4, -33.5),
    "entorhinal-rh": (24.8, -6.0, -33.7),
    "frontalpole-lh": (-8.0, 63.1, -8.5),
    "frontalpole-rh": (8.7, 62.7, -10.2),
    "fusiform-lh": (-35.3, -42.8, -21.5),
    "fusiform-rh": (35.3, -42.9, -20.4),
    "inferiorparietal-lh": (-40.9, -69.7, 32.1),
    "inferiorparietal-rh": (45.5, -63.8, 30.6),
    "inferiortemporal-lh": (-54.0, -37.7, -26.2),
    "inferiortemporal-rh": (54.3, -33.8, -25.3),
    "insula-lh": (-36.5, -5.6, 3.2),
    "insula-rh": (37.4, -5.7, 2.4),
    "isthmuscingulate-lh": (-8.3, -48.2, 16.9),
    "isthmuscingulate-rh": (8.8, -46.9, 16.6),
    "lateraloccipital-lh": (-28.7, -92.7, -5.8),
    "lateraloccipital-rh": (29.8, -92.3, -2.5),
    "lateralorbitofrontal-lh": (-26.7, 28.6, -13.8),
    "lateralorbitofrontal-rh": (22.2, 28.1, -12.1),
    "lingual-lh": (-12.7, -71.5, -3.4),
    "lingual-rh": (9.8, -69.2, -2.4),
    "medialorbitofrontal-lh": (-5.1, 37.2, -20.3),
    "medialorbitofrontal-rh": (7.2, 38.5, -18.4),
    "middletemporal-lh": (-63.6, -31.4, -12.9),
    "middletemporal-rh": (65.6, -28.2, -13.0),
    "paracentral-lh": (-5.7, -29.9, 53.8),
    "paracentral-rh": (6.2, -28.1, 53.3),
    "parahippocampal-lh": (-24.2, -32.6, -19.2),
    "parahippocampal-rh": (25.5, -32.3, -17.5),
    "parsopercularis-lh": (-52.5, 16.0, 12.7),
    "parsopercularis-rh": (54.0, 15.0, 14.3),
    "parsorbitalis-lh": (-41.8, 39.9, -13.3),
    "parsorbitalis-rh": (44.1, 40.7, -13.3),
    "parstriangularis-lh": (-45.6, 29.9, -0.4),
    "parstriangularis-rh": (50.5, 28.3, 1.6),
    "pericalcarine-lh": (-13.8, -82.2, 10.2),
    "pericalcarine-rh": (15.7, -81.0, 9.2),
    "postcentral-lh": (-50.8, -17.8, 50.8),
    "postcentral-rh": (50.4, -15.8, 51.0),
    "posteriorcingulate-lh": (-4.1, -17.5, 38.0),
    "posteriorcingulate-rh": (4.4, -16.7, 38.0),
    "precentral-lh": (-48.0, -9.8, 48.7),
    "precentral-rh": (45.9, -7.0, 51.8),
    "precuneus-lh": (-7.3, -58.6, 36.4),
    "precuneus-rh": (9.0, -58.1, 37.3),
    "rostralanteriorcingulate-lh": (-7.0, 36.3, 0.0),
    "rostralanteriorcingulate-rh": (7.4, 35.8, 0.4),
    "rostralmiddlefrontal-lh": (-34.5, 45.4, 18.8),
    "rostralmiddlefrontal-rh": (33.6, 44.8, 19.8),
    "superiorfrontal-lh": (-6.8, 30.4, 52.3),
    "superiorfrontal-rh": (6.8, 32.3, 52.7),
    "superiorparietal-lh": (-23.2, -62.1, 52.3),
    "superiorparietal-rh": (20.8, -58.8, 53.1),
    "superiortemporal-lh": (-61.4, -12.9, 0.4),
    "superiortemporal-rh": (62.6, -7.2, -1.8),
    "supramarginal-lh": (-57.3, -38.4, 36.9),
    "supramarginal-rh": (57.7, -32.8, 37.3),
    "temporalpole-lh": (-29.7, 12.6, -36.8),
    "temporalpole-rh": (30.9, 12.2, -35.1),
    "transversetemporal-lh": (-44.5, -23.7, 8.1),
    "transversetemporal-rh": (44.9, -22.1, 7.8),
}

NODE_SET_DK68 = "dk68"
NODE_SET_TRIPLE = "triple_network"
NODE_SET_CUSTOM = "custom"
NODE_SETS = (NODE_SET_DK68, NODE_SET_TRIPLE, NODE_SET_CUSTOM)
TRIPLE_NETWORKS = ("SN", "CEN", "DMN")
_RELEASE = re.compile(r"R?(20\d\d[ab])", re.IGNORECASE)
_CUSTOM_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]*")


@dataclass(frozen=True, slots=True)
class GraphVarSettings:
    """GraphVar choices collected by the GUI."""

    matlab_executable: Path
    graphvar_root: Path
    viewer_config: Path
    node_set: str = NODE_SET_DK68
    custom_name: str = "Custom"
    custom_labels: tuple[str, ...] = ()
    permutations: int = 5000
    component_alpha: float = 0.05
    familywise_alpha: float = 0.05
    seed: int = 20260724
    matlab_release: str | None = None
    timeout_seconds: int = 7200

    @property
    def release(self) -> str:
        """MATLAB release such as ``2025b``, read from the executable path."""
        if self.matlab_release:
            return self.matlab_release
        for part in reversed(Path(self.matlab_executable).parts):
            match = _RELEASE.fullmatch(part)
            if match:
                return match.group(1).lower()
        raise ValueError(
            "Cannot tell the MATLAB release from the executable path; "
            "enter it (for example 2025b)."
        )

    @property
    def node_set_name(self) -> str:
        if self.node_set == NODE_SET_CUSTOM:
            return self.custom_name.strip() or "Custom"
        return {NODE_SET_DK68: "DK68", NODE_SET_TRIPLE: "Triple network"}[self.node_set]


@dataclass(frozen=True, slots=True)
class BrainNetSettings:
    """BrainNet Viewer resources collected by the GUI."""

    brainnet_root: Path
    surface: Path
    viewer_config: Path
    matlab_executable: Path | None = None
    top_k: int = 8
    width_pixels: int = 2000
    height_pixels: int = 1500
    dpi: int = 300
    timeout_seconds: int = 900


@dataclass(frozen=True, slots=True)
class GraphNode:
    label: str
    atlas_index_1based: int
    display_label: str
    coordinates: tuple[float, float, float]
    module: int = 1


def display_label(label: str) -> str:
    """``superiorfrontal-lh`` becomes ``L.superiorfrontal``."""
    name, _, hemi = label.rpartition("-")
    return ("L." if hemi == "lh" else "R.") + name


def node_set_labels(settings: GraphVarSettings) -> tuple[str, ...]:
    """Return the selected labels in DK68 order."""
    if settings.node_set == NODE_SET_DK68:
        return tuple(DK68_LABELS)
    if settings.node_set == NODE_SET_TRIPLE:
        chosen = {label for name in TRIPLE_NETWORKS for label in PRESET_NETWORKS[name]}
    elif settings.node_set == NODE_SET_CUSTOM:
        chosen = set(settings.custom_labels)
        unknown = sorted(chosen - set(DK68_LABELS))
        if unknown:
            raise ValueError(
                "Custom GraphVar regions are not DK68 labels: " + ", ".join(unknown)
            )
    else:
        raise ValueError(f"Unknown GraphVar node set {settings.node_set!r}")
    return tuple(label for label in DK68_LABELS if label in chosen)


def graph_nodes(settings: GraphVarSettings) -> tuple[GraphNode, ...]:
    labels = node_set_labels(settings)
    if len(labels) < 3:
        raise ValueError(
            f"GraphVar needs at least 3 regions; {settings.node_set_name} has {len(labels)}."
        )
    modules = _modules(settings)
    return tuple(
        GraphNode(
            label,
            DK68_LABELS.index(label) + 1,
            display_label(label),
            DK68_MNI[label],
            modules.get(label, 1),
        )
        for label in labels
    )


def settings_problems(
    graphvar: GraphVarSettings | None,
    brainnet: BrainNetSettings | None,
    want_brainnet: bool,
) -> list[str]:
    """Actionable problems with the GraphVar and BrainNet choices."""
    if graphvar is None:
        return ["GraphVar needs its MATLAB, GraphVar root and viewer config paths."]
    problems = _missing(
        (
            ("GraphVar MATLAB", graphvar.matlab_executable, False),
            ("GraphVar root", graphvar.graphvar_root, True),
            ("GraphVar viewer config", graphvar.viewer_config, False),
        )
    )
    try:
        graphvar.release
    except ValueError as exc:
        problems.append(str(exc))
    try:
        graph_nodes(graphvar)
    except ValueError as exc:
        problems.append(str(exc))
    if graphvar.node_set == NODE_SET_CUSTOM and not _CUSTOM_NAME.fullmatch(
        graphvar.custom_name.strip()
    ):
        problems.append(
            "Give the custom region set a short name (letters, digits, spaces, _ . -)."
        )
    if want_brainnet:
        if brainnet is None:
            problems.append("BrainNet needs its root, surface and viewer config paths.")
        else:
            problems.extend(
                _missing(
                    (
                        ("BrainNet root", brainnet.brainnet_root, True),
                        ("BrainNet surface", brainnet.surface, False),
                        ("BrainNet viewer config", brainnet.viewer_config, False),
                    )
                )
            )
    return problems


def write_atlas_workbook(path: str | Path, nodes: Sequence[GraphNode]) -> Path:
    """Write the GraphVar atlas: all 68 rows, column A = 1 for selected rows.

    Columns are selected flag, display label, atlas index and MNI x, y, z.
    """
    selected = {node.label for node in nodes}
    rows = []
    for index, label in enumerate(DK68_LABELS, 1):
        x, y, z = DK68_MNI[label]
        flag = 1 if label in selected else 0
        rows.append(
            f'<row r="{index}"><c r="A{index}"><v>{flag}</v></c>'
            f'<c r="B{index}" t="inlineStr"><is><t>{escape(display_label(label))}</t></is></c>'
            f'<c r="C{index}"><v>{index}</v></c><c r="D{index}"><v>{x}</v></c>'
            f'<c r="E{index}"><v>{y}</v></c><c r="F{index}"><v>{z}</v></c></row>'
        )
    main = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    relationships = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    parts = (
        (
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.'
            'openxmlformats.org/package/2006/content-types"><Default Extension="xml" '
            'ContentType="application/xml"/><Default Extension="rels" ContentType='
            '"application/vnd.openxmlformats-package.relationships+xml"/><Override '
            'PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/'
            'worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.spreadsheetml.worksheet+xml"/></Types>',
        ),
        (
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.'
            'openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" '
            f'Type="{relationships}/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        ),
        (
            "xl/workbook.xml",
            f'<?xml version="1.0" encoding="UTF-8"?><workbook {main} xmlns:r="{relationships}">'
            '<sheets><sheet name="Atlas" sheetId="1" r:id="rId1"/></sheets></workbook>',
        ),
        (
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.'
            'openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" '
            f'Type="{relationships}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        ),
        (
            "xl/worksheets/sheet1.xml",
            f'<?xml version="1.0" encoding="UTF-8"?><worksheet {main}><sheetData>'
            f'{"".join(rows)}</sheetData></worksheet>',
        ),
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in parts:
            # A fixed timestamp keeps the workbook byte-identical across runs, so
            # the GraphVar lock (which hashes it) is stable and reruns reuse it.
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, text)
    return target


def family_name(condition: str) -> str:
    return f"{_slug(condition)}-graphvar"


def workspace_names(
    conditions: Sequence[str], bands: Iterable[str], sessions: Sequence[str]
) -> list[str]:
    first = sessions[0]
    bands = list(bands)
    return [
        f"{family_name(condition)}__{band}__{first}-{session}"
        for condition in conditions
        for band in bands
        for session in sessions[1:]
    ]


def build_graphvar_block(
    settings: GraphVarSettings,
    *,
    arms: Sequence[str],
    sessions: Sequence[str],
    conditions: Sequence[str],
    bands: Iterable[str],
    workbook: Path,
) -> dict[str, Any]:
    """Build the strict ``graphvar`` project block (first session vs each later one)."""
    nodes = graph_nodes(settings)
    first = sessions[0]
    pairs = [
        {"name": f"{first}-{session}", "first_session": first, "second_session": session}
        for session in sessions[1:]
    ]
    bands = list(bands)
    return {
        "atlas": {
            "workbook": str(workbook),
            "brain_viewer_config": str(settings.viewer_config),
            "nodes": [
                {
                    "connectivity_label": node.label,
                    "atlas_index_1based": node.atlas_index_1based,
                    "display_label": node.display_label,
                    "coordinates": list(node.coordinates),
                }
                for node in nodes
            ],
        },
        "families": [
            {
                "name": family_name(condition),
                "holm_family": f"{_slug(condition)}-components",
                "condition": condition,
                "bands": bands,
                "session_pairs": pairs,
                "effect": {"numerator_arm": arms[0], "denominator_arm": arms[1]},
                "component_forming_alpha": settings.component_alpha,
                "familywise_alpha": settings.familywise_alpha,
                "permutations": settings.permutations,
                "seed": {"policy": "sha256_workspace", "base": settings.seed},
                "matrix_field_template": "OutPLI_{band}",
                "effect_label": f"arm_{arms[0]}-arm_{arms[1]}=0",
            }
            for condition in conditions
        ],
        "runtime": {
            "executable": str(settings.matlab_executable),
            "graphvar_root": str(settings.graphvar_root),
            "matlab_release": settings.release,
            "timeout_seconds": settings.timeout_seconds,
        },
    }


def supported_workspaces(graphvar_run_dir: str | Path, top_k: int) -> list[str]:
    """Workspaces, in GraphVar order, whose node-pFWE-supported component has at least ``top_k`` links."""
    path = Path(graphvar_run_dir) / "graphvar" / "components-all.csv"
    order: list[str] = []
    supported: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            workspace = row["workspace"]
            if workspace not in order:
                order.append(workspace)
            if (
                row["primary_node_pfwe_supported"].strip().lower() == "true"
                and int(row["component_link_count"]) >= top_k
            ):
                supported.add(workspace)
    return [workspace for workspace in order if workspace in supported]


def build_brainnet_block(
    settings: BrainNetSettings,
    graphvar: GraphVarSettings,
    workspaces: Sequence[str],
) -> dict[str, Any]:
    """Build the strict ``brainnet`` block for an already-chosen workspace list."""
    executable = settings.matlab_executable or graphvar.matlab_executable
    return {
        "resources": {
            "surface": str(settings.surface),
            "viewer_config": str(settings.viewer_config),
        },
        "selection": {
            "workspaces": list(workspaces),
            "policy": "largest_primary_node_pfwe",
            "top_k": settings.top_k,
        },
        "nodes": {
            "styles": [
                {
                    "connectivity_label": node.label,
                    "display_label": node.display_label,
                    "color": node.module,
                    "size": 3,
                }
                for node in graph_nodes(graphvar)
            ]
        },
        "render": {
            "width_pixels": settings.width_pixels,
            "height_pixels": settings.height_pixels,
            "dpi": settings.dpi,
        },
        "runtime": {
            "executable": str(executable),
            "brainnet_root": str(settings.brainnet_root),
            "matlab_release": graphvar.release,
            "timeout_seconds": settings.timeout_seconds,
        },
    }


def _modules(settings: GraphVarSettings) -> dict[str, int]:
    """BrainNet colour index: one per network for the triple network, else by hemisphere."""
    if settings.node_set == NODE_SET_TRIPLE:
        return {
            label: index
            for index, name in enumerate(TRIPLE_NETWORKS, 1)
            for label in PRESET_NETWORKS[name]
        }
    return {label: 1 if label.endswith("-lh") else 2 for label in DK68_LABELS}


def _missing(items: Iterable[tuple[str, Path, bool]]) -> list[str]:
    problems = []
    for name, path, directory in items:
        exists = Path(path).is_dir() if directory else Path(path).is_file()
        if not exists:
            problems.append(f"{name} not found: {path}")
    return problems


def _slug(value: str) -> str:
    from .discovery import slugify

    return slugify(value)


__all__ = [
    "BrainNetSettings",
    "DK68_MNI",
    "GraphNode",
    "GraphVarSettings",
    "NODE_SETS",
    "NODE_SET_CUSTOM",
    "NODE_SET_DK68",
    "NODE_SET_TRIPLE",
    "build_brainnet_block",
    "build_graphvar_block",
    "display_label",
    "family_name",
    "graph_nodes",
    "node_set_labels",
    "settings_problems",
    "supported_workspaces",
    "workspace_names",
    "write_atlas_workbook",
]
