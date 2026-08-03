"""Explicit-axis streamed extraction of GraphVar v7.3 permutation results."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .workspace import GraphVarError

INFERENCE_SCOPE = (
    "Member link is descriptive and uncorrected; corrected inference applies "
    "only to the complete component."
)


@dataclass(frozen=True, slots=True)
class ExtractedWorkspace:
    status: dict[str, Any]
    components: tuple[dict[str, Any], ...]
    links: tuple[dict[str, Any], ...]
    null_node_max: np.ndarray
    null_edge_max: np.ndarray


def conservative_upper_tail(values: np.ndarray) -> np.ndarray:
    """Empirical P(null >= value), assigning every tie its last rank."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or not len(array):
        raise GraphVarError("null edge values must be a finite nonempty vector")
    order = np.argsort(-array, kind="mergesort")
    sorted_values = array[order]
    probabilities = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and sorted_values[end] == sorted_values[start]:
            end += 1
        probabilities[order[start:end]] = end / len(array)
        start = end
    return probabilities


def connected_components(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise GraphVarError("component mask must be square")
    if not np.array_equal(value, value.T) or np.any(np.diag(value)):
        raise GraphVarError("component mask must be symmetric with a zero diagonal")
    unseen = set(range(value.shape[0]))
    components = []
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            neighbors = [
                int(item) for item in np.flatnonzero(value[node]) if int(item) in unseen
            ]
            for neighbor in neighbors:
                unseen.remove(neighbor)
                stack.append(neighbor)
        if len(nodes) >= 2 and np.any(value[np.ix_(nodes, nodes)]):
            components.append(tuple(sorted(nodes)))
    return tuple(components)


def holm_adjust(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if (
        array.ndim != 1
        or not np.isfinite(array).all()
        or np.any((array < 0) | (array > 1))
    ):
        raise GraphVarError("Holm values must be finite probabilities")
    if not len(array):
        return array
    order = np.argsort(array, kind="mergesort")
    scaled = (len(array) - np.arange(len(array))) * array[order]
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(array)
    adjusted[order] = adjusted_sorted
    return adjusted


def _decode_reference(file: Any, reference: Any) -> str:
    import h5py

    if not isinstance(reference, h5py.Reference) or not reference:
        raise GraphVarError("MATLAB label is not a valid HDF5 object reference")
    dataset = file[reference]
    value = np.asarray(dataset[()])
    matlab_class = dataset.attrs.get("MATLAB_class")
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode("ascii", errors="strict")
    if matlab_class != "char" or value.dtype != np.uint16 or value.ndim != 2:
        raise GraphVarError("MATLAB label reference is not a uint16 char matrix")
    return "".join(chr(int(code)) for code in value.ravel(order="F") if int(code))


def _labels(file: Any, effects: int) -> tuple[str, ...]:
    dataset = file["LAB"]
    references = np.asarray(dataset[()])
    if references.dtype.kind != "O" or references.size != effects:
        raise GraphVarError("LAB must contain exactly one MATLAB reference per effect")
    return tuple(
        _decode_reference(file, reference) for reference in references.ravel(order="F")
    )


def _scalar(file: Any, name: str) -> float:
    value = np.asarray(file[name][()], dtype=np.float64)
    if value.size != 1 or not np.isfinite(value).all():
        raise GraphVarError(f"{name} must be one finite scalar")
    return float(value.ravel()[0])


def _orientation(
    label: str,
    numerator: str,
    denominator: str,
    explicit_multiplier: Any = None,
) -> int:
    direct = f"arm_{numerator}-arm_{denominator}=0"
    reverse = f"arm_{denominator}-arm_{numerator}=0"
    if label == direct:
        return 1
    if label == reverse:
        return -1
    if explicit_multiplier in (-1, 1) and type(explicit_multiplier) is int:
        return explicit_multiplier
    raise GraphVarError(
        "effect label is not the exact configured arm contrast or its exact reverse"
    )


def _validate_observed(name: str, value: np.ndarray) -> None:
    if (
        not np.isfinite(value).all()
        or not np.allclose(value, value.T, atol=1e-12, rtol=0)
        or not np.allclose(np.diag(value), 0, atol=1e-12, rtol=0)
    ):
        raise GraphVarError(f"{name} observed matrix is malformed")


def extract_workspace_result(
    result_path: str | Path,
    request: dict[str, Any],
    node_map: tuple[dict[str, Any], ...],
) -> ExtractedWorkspace:
    """Extract all candidate components without loading the 4-D null arrays."""
    import h5py

    result_path = Path(result_path)
    required = {"B", "F", "P", "NP", "NF", "NB", "LAB", "J", "DDF", "N"}
    try:
        file = h5py.File(result_path, "r")
    except OSError as exc:
        raise GraphVarError(
            f"GraphVar result is not valid v7.3/HDF5: {result_path}"
        ) from exc
    with file:
        if not required.issubset(file):
            raise GraphVarError(
                f"GraphVar result fields are missing: {sorted(required - set(file))}"
            )
        nodes = len(node_map)
        observed_shape = tuple(file["B"].shape)
        if len(observed_shape) != 3 or observed_shape[1:] != (nodes, nodes):
            raise GraphVarError(
                f"observed physical axes must be (effects,nodes,nodes), got {observed_shape}"
            )
        effects = observed_shape[0]
        for name in ("F", "P", "NP"):
            if tuple(file[name].shape) != observed_shape:
                raise GraphVarError(f"{name} physical axes differ from B")
        permutations = int(request["permutations"])
        null_shape = (permutations, effects, nodes, nodes)
        for name in ("NF", "NB"):
            if tuple(file[name].shape) != null_shape:
                raise GraphVarError(
                    f"{name} physical axes must be (permutations,effects,nodes,nodes), "
                    f"got {tuple(file[name].shape)}"
                )
        labels = _labels(file, effects)
        expected_label = str(request["effect_label"])
        matches = [
            index for index, label in enumerate(labels) if label == expected_label
        ]
        if len(matches) != 1:
            raise GraphVarError(
                f"configured effect label must identify exactly one effect; labels={labels!r}"
            )
        effect = matches[0]
        sign = _orientation(
            labels[effect],
            str(request["numerator_arm"]),
            str(request["denominator_arm"]),
            request.get("effect_orientation_multiplier"),
        )
        observed = {
            name: np.asarray(file[name][effect, :, :], dtype=np.float64)
            for name in ("B", "F", "P", "NP")
        }
        observed["B"] *= sign
        for name, matrix in observed.items():
            _validate_observed(name, matrix)
        if np.any((observed["P"] < 0) | (observed["P"] > 1)) or np.any(
            (observed["NP"] < 0) | (observed["NP"] > 1)
        ):
            raise GraphVarError("observed P/NP values must be probabilities")
        if np.any(observed["F"] < 0):
            raise GraphVarError("observed F values must be nonnegative")
        j_values = np.asarray(file["J"][()], dtype=np.float64).ravel(order="F")
        if (
            len(j_values) != effects
            or not np.isfinite(j_values).all()
            or np.any(j_values <= 0)
            or np.any(j_values != np.floor(j_values))
        ):
            raise GraphVarError(
                "J must contain one positive integer numerator DF per effect"
            )
        observation_n = _scalar(file, "N")
        denominator_df = _scalar(file, "DDF")
        if (
            observation_n <= 0
            or not float(observation_n).is_integer()
            or denominator_df <= 0
            or not float(denominator_df).is_integer()
        ):
            raise GraphVarError("N and DDF must be positive integers")
        if observation_n != int(request["observations"]):
            raise GraphVarError(
                "GraphVar N differs from the workspace observation rows"
            )

        alpha = float(request["component_forming_alpha"])
        observed_mask = np.triu(observed["NP"] <= alpha, 1)
        observed_mask |= observed_mask.T
        # NF is commonly chunked along permutations. Reading every edge as
        # NF[:,effect,i,j] decompresses the same chunks hundreds of times.
        # Spool the selected effect once in chunk-aligned blocks, then rank
        # edges from a bounded local memmap.
        with tempfile.TemporaryDirectory(
            prefix="neurostat-graphvar-null-"
        ) as temporary:
            spool_path = Path(temporary) / "nf-effect.float64"
            spool = np.memmap(
                spool_path,
                mode="w+",
                dtype=np.float64,
                shape=(permutations, nodes, nodes),
            )
            chunk_permutations = (
                int(file["NF"].chunks[0])
                if file["NF"].chunks is not None
                else min(permutations, 128)
            )
            block = max(1, chunk_permutations * 32)
            for start in range(0, permutations, block):
                end = min(permutations, start + block)
                nf_block = np.asarray(
                    file["NF"][start:end, effect, :, :], dtype=np.float64
                )
                nb_block = np.asarray(
                    file["NB"][start:end, effect, :, :], dtype=np.float64
                )
                if (
                    not np.isfinite(nf_block).all()
                    or not np.isfinite(nb_block).all()
                    or np.any(nf_block < 0)
                    or not np.allclose(
                        nf_block, np.swapaxes(nf_block, 1, 2), atol=1e-12, rtol=0
                    )
                    or not np.allclose(
                        nb_block, np.swapaxes(nb_block, 1, 2), atol=1e-12, rtol=0
                    )
                    or not np.allclose(
                        np.diagonal(nf_block, axis1=1, axis2=2),
                        0,
                        atol=1e-12,
                        rtol=0,
                    )
                    or not np.allclose(
                        np.diagonal(nb_block, axis1=1, axis2=2),
                        0,
                        atol=1e-12,
                        rtol=0,
                    )
                ):
                    raise GraphVarError(
                        "selected-effect NF/NB null matrices are malformed"
                    )
                spool[start:end, :, :] = nf_block
            spool.flush()
            null_mask = np.zeros((permutations, nodes, nodes), dtype=bool)
            for first in range(nodes - 1):
                for second in range(first + 1, nodes):
                    selected = (
                        conservative_upper_tail(
                            np.asarray(spool[:, first, second], dtype=np.float64)
                        )
                        <= alpha
                    )
                    null_mask[:, first, second] = selected
                    null_mask[:, second, first] = selected
            del spool

            null_node_max = np.ones(permutations, dtype=np.int32)
            null_edge_max = np.zeros(permutations, dtype=np.int32)
            for permutation in range(permutations):
                mask = null_mask[permutation]
                for component in connected_components(mask):
                    null_node_max[permutation] = max(
                        null_node_max[permutation], len(component)
                    )
                    null_edge_max[permutation] = max(
                        null_edge_max[permutation],
                        int(
                            np.count_nonzero(
                                np.triu(mask[np.ix_(component, component)], 1)
                            )
                        ),
                    )
    candidates = sorted(
        connected_components(observed_mask), key=lambda item: (-len(item), min(item))
    )
    components = []
    links = []
    for component_number, component in enumerate(candidates, 1):
        component_id = f"{request['workspace_id']}_C{component_number:02d}"
        submask = observed_mask[np.ix_(component, component)]
        local_first, local_second = np.where(np.triu(submask, 1))
        edges = tuple(
            (component[int(first)], component[int(second)])
            for first, second in zip(local_first, local_second, strict=True)
        )
        node_exceed = int(np.count_nonzero(null_node_max >= len(component)))
        edge_exceed = int(np.count_nonzero(null_edge_max >= len(edges)))
        primary_tail = node_exceed / permutations
        primary_plus_one = (node_exceed + 1) / (permutations + 1)
        nbs_tail = edge_exceed / permutations
        nbs_plus_one = (edge_exceed + 1) / (permutations + 1)
        row = {
            "holm_family": request["holm_family"],
            "family": request["family"],
            "condition": request["condition"],
            "workspace": request["workspace_id"],
            "band": request["band"],
            "session_pair": request["pair"],
            "component_id": component_id,
            "effect_label": labels[effect],
            "effect_orientation_multiplier": sign,
            "component_node_count": len(component),
            "component_link_count": len(edges),
            "member_rois": "; ".join(
                node_map[index]["display_label"] for index in component
            ),
            "component_forming_alpha": alpha,
            "permutations": permutations,
            "primary_node_exceedances": node_exceed,
            "primary_node_pfwe_graphvar_tail": primary_tail,
            "primary_node_pfwe_plus_one": primary_plus_one,
            "primary_node_pfwe_supported": primary_plus_one
            <= float(request["familywise_alpha"]),
            "posthoc_nbs_edge_exceedances": edge_exceed,
            "posthoc_nbs_edge_pfwe_graphvar_tail": nbs_tail,
            "posthoc_nbs_edge_pfwe_plus_one": nbs_plus_one,
            "posthoc_nbs_edge_pfwe_supported": nbs_plus_one
            <= float(request["familywise_alpha"]),
            "graphvar_observation_rows": int(observation_n),
            "graphvar_numerator_df": float(j_values[effect]),
            "graphvar_denominator_df": denominator_df,
            "inference_scope": (
                "Primary study-wide support is based only on participating-node "
                "component pFWE plus its condition-level Holm correction; edge "
                "extent is a separate post-hoc NBS family."
            ),
        }
        components.append(row)
        for first, second in edges:
            first_node = node_map[first]
            second_node = node_map[second]
            effect_value = float(observed["B"][first, second])
            links.append(
                {
                    "holm_family": request["holm_family"],
                    "condition": request["condition"],
                    "workspace": request["workspace_id"],
                    "band": request["band"],
                    "session_pair": request["pair"],
                    "component_id": component_id,
                    "component_node_count": len(component),
                    "component_link_count": len(edges),
                    "roi1_result_index_1based": first + 1,
                    "roi2_result_index_1based": second + 1,
                    "roi1_atlas_index_1based": first_node["atlas_index_1based"],
                    "roi2_atlas_index_1based": second_node["atlas_index_1based"],
                    "roi1_connectivity_label": first_node["connectivity_label"],
                    "roi2_connectivity_label": second_node["connectivity_label"],
                    "roi1": first_node["display_label"],
                    "roi2": second_node["display_label"],
                    "effect_label": labels[effect],
                    "paired_change_contrast_b": effect_value,
                    "f_statistic": float(observed["F"][first, second]),
                    "parametric_p_uncorrected": float(observed["P"][first, second]),
                    "permutation_p_uncorrected": float(observed["NP"][first, second]),
                    "effect_direction": (
                        f"Positive: {request['numerator_arm']} has a more positive "
                        f"second-minus-first change than {request['denominator_arm']}"
                        if effect_value > 0
                        else f"Negative: {request['numerator_arm']} has a more negative "
                        f"second-minus-first change than {request['denominator_arm']}"
                        if effect_value < 0
                        else "Zero configured paired-change contrast"
                    ),
                    "inference_scope": INFERENCE_SCOPE,
                }
            )
    status = {
        "workspace": request["workspace_id"],
        "holm_family": request["holm_family"],
        "condition": request["condition"],
        "band": request["band"],
        "session_pair": request["pair"],
        "status": "complete_no_component" if not components else "complete",
        "candidate_components": len(components),
        "permutations": permutations,
        "effect_label": labels[effect],
        "effect_orientation_multiplier": sign,
        "graphvar_observation_rows": int(observation_n),
        "graphvar_denominator_df": denominator_df,
    }
    return ExtractedWorkspace(
        status, tuple(components), tuple(links), null_node_max, null_edge_max
    )


def apply_condition_holm(
    components: list[dict[str, Any]], familywise_alpha_by_family: dict[str, float]
) -> None:
    """Apply independent all-candidate primary and NBS Holm families in place."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in components:
        grouped.setdefault(str(row["holm_family"]), []).append(row)
    for family, rows in grouped.items():
        primary = holm_adjust(row["primary_node_pfwe_plus_one"] for row in rows)
        nbs = holm_adjust(row["posthoc_nbs_edge_pfwe_plus_one"] for row in rows)
        alpha = familywise_alpha_by_family[family]
        for row, primary_value, nbs_value in zip(rows, primary, nbs, strict=True):
            row["primary_node_holm_all_candidates"] = float(primary_value)
            row["primary_studywide_supported"] = bool(
                row["primary_node_pfwe_supported"] and primary_value <= alpha
            )
            row["posthoc_nbs_edge_holm_all_candidates"] = float(nbs_value)
            row["posthoc_nbs_studywide_supported"] = bool(
                row["posthoc_nbs_edge_pfwe_supported"] and nbs_value <= alpha
            )
