"""Bone-name-independent semantic decomposition for generated quadruped rigs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class SemanticRigError(ValueError):
    pass


@dataclass(frozen=True)
class QuadrupedSemantics:
    root: str
    axial: tuple[str, ...]
    head_chain: tuple[str, ...]
    tail_chain: tuple[str, ...]
    front_side_negative: tuple[str, ...]
    front_side_positive: tuple[str, ...]
    hind_side_negative: tuple[str, ...]
    hind_side_positive: tuple[str, ...]
    auxiliary_branches: tuple[tuple[str, ...], ...]
    foot_leaves: tuple[str, ...]

    def chains(self) -> dict[str, tuple[str, ...]]:
        result = {
            "axial": self.axial,
            "head": self.head_chain,
            "tail": self.tail_chain,
            "front_side_negative": self.front_side_negative,
            "front_side_positive": self.front_side_positive,
            "hind_side_negative": self.hind_side_negative,
            "hind_side_positive": self.hind_side_positive,
        }
        result.update(
            {
                f"auxiliary_{index}": branch
                for index, branch in enumerate(self.auxiliary_branches)
            }
        )
        return result

    def all_bones(self) -> tuple[str, ...]:
        ordered = []
        for chain in self.chains().values():
            for name in chain:
                if name not in ordered:
                    ordered.append(name)
        return tuple(ordered)


def quadruped_semantic_labels(
    semantics: QuadrupedSemantics,
    records: Iterable[Mapping],
    *,
    bbox_min: Sequence[float],
    bbox_extent: Sequence[float],
    front_axis: str,
    low_leaf_height_fraction: float = 0.22,
) -> dict[str, str]:
    """Return exhaustive labels with neutral shared junctions and paw controls.

    ``QuadrupedSemantics.chains()`` deliberately preserves the literal tree:
    disconnected exporter controls remain auxiliary branches.  Weight QA,
    however, must know that a low ``hoof`` controller on the animal's negative
    side belongs to the same locomotion limb as the nearest deform chain.  This
    helper also keeps a shared pelvic/tail-base junction neutral instead of
    assigning it to one of the distal branches.  Both associations leave chain
    order unchanged, which keeps retargeting topology and weight sanitation
    concerns separate.
    """
    axis_components = {
        "positive-x": (0, 1.0, 1),
        "negative-x": (0, -1.0, 1),
        "positive-y": (1, 1.0, 0),
        "negative-y": (1, -1.0, 0),
    }
    if front_axis not in axis_components:
        raise SemanticRigError(f"unsupported front axis: {front_axis}")
    records = [dict(record) for record in records]
    by_name = {record.get("name"): record for record in records}
    if None in by_name or len(by_name) != len(records):
        raise SemanticRigError("bone names must be present and unique")
    labels: dict[str, str] = {}
    core_names = {
        "axial",
        "head",
        "tail",
        "front_side_negative",
        "front_side_positive",
        "hind_side_negative",
        "hind_side_positive",
    }
    branch_names = core_names - {"axial", "head"}
    core_occurrences: dict[
        str,
        list[tuple[str, int, tuple[str, ...]]],
    ] = {}
    for label, chain in semantics.chains().items():
        if label.startswith("auxiliary_"):
            continue
        if label not in core_names:
            raise SemanticRigError(f"unknown semantic chain label: {label}")
        chain = tuple(chain)
        seen_in_chain: set[str] = set()
        for index, name in enumerate(chain):
            if name not in by_name:
                raise SemanticRigError(
                    f"semantic chain {label} references missing bone: {name}"
                )
            if name in seen_in_chain:
                raise SemanticRigError(
                    f"semantic chain {label} repeats bone: {name}"
                )
            seen_in_chain.add(name)
            if (
                index
                and by_name[name].get("parent") != chain[index - 1]
            ):
                raise SemanticRigError(
                    f"semantic chain {label} is not parent-child contiguous at "
                    f"{name}"
                )
            core_occurrences.setdefault(name, []).append((label, index, chain))

    for name, occurrences in core_occurrences.items():
        if len(occurrences) == 1:
            labels[name] = occurrences[0][0]
            continue
        occurrence_labels = {label for label, _index, _chain in occurrences}
        suffixes = [chain[index + 1 :] for _label, index, chain in occurrences]
        if (
            not occurrence_labels.issubset(branch_names)
            or any(not suffix for suffix in suffixes)
            or len(set(suffixes)) != len(suffixes)
        ):
            raise SemanticRigError(f"bone appears in multiple chains: {name}")
        # Some generated one-root rigs place a pelvic or tail-base junction
        # below the nominal axial path.  Literal root-to-leaf paths then share
        # that proximal junction across two limbs, or across a limb and the
        # tail.  It is not owned by either distal domain: treating it as axial
        # preserves its neutral support weights while the distinct non-empty
        # suffixes above prove that this is a real branch, not a duplicated
        # semantic chain.
        labels[name] = "axial"

    forward_index, sign, lateral_index = axis_components[front_axis]
    extent = [float(value) for value in bbox_extent]
    if len(extent) != 3 or extent[forward_index] <= 0.0 or extent[lateral_index] <= 0.0:
        raise SemanticRigError("mesh cardinal extent must be positive")
    floor = float(bbox_min[2])
    low_limit = floor + float(low_leaf_height_fraction) * extent[2]
    limb_chains = {
        "front_side_negative": semantics.front_side_negative,
        "front_side_positive": semantics.front_side_positive,
        "hind_side_negative": semantics.hind_side_negative,
        "hind_side_positive": semantics.hind_side_positive,
    }
    foot_points = {
        label: _point(by_name[chain[-1]], "head_world")
        for label, chain in limb_chains.items()
    }

    for index, branch in enumerate(semantics.auxiliary_branches):
        points = [_point(by_name[name], "head_world") for name in branch]
        branch_label = f"auxiliary_{index}"
        if min(point[2] for point in points) <= low_limit:
            representative = min(points, key=lambda point: point[2])

            def distance(item: tuple[str, tuple[float, float, float]]):
                _label, point = item
                forward_delta = (
                    sign * (representative[forward_index] - point[forward_index])
                    / extent[forward_index]
                )
                lateral_delta = (
                    representative[lateral_index] - point[lateral_index]
                ) / extent[lateral_index]
                return (forward_delta * forward_delta + lateral_delta * lateral_delta, _label)

            branch_label = min(foot_points.items(), key=distance)[0]
        for name in branch:
            if name in labels:
                raise SemanticRigError(f"bone appears in multiple chains: {name}")
            labels[name] = branch_label
    if set(labels) != set(by_name):
        raise SemanticRigError(
            "semantic labels must cover every bone: "
            f"missing={sorted(set(by_name) - set(labels))}"
        )
    return labels


def _point(record: Mapping, key: str) -> tuple[float, float, float]:
    value = record.get(key)
    if not isinstance(value, Sequence) or len(value) != 3:
        raise SemanticRigError(f"invalid {key} for bone {record.get('name')}")
    return tuple(float(component) for component in value)


def _leaf_floor_probe(record: Mapping) -> float:
    """Return the lower authored endpoint of a leaf-bone segment.

    A generated rest pose may lift a paw enough that the leaf head is above
    the global low-band while its tail still ends inside the paw.  Using the
    lower of head and tail admits that valid limb.  Extra hoof/controller
    leaves are still resolved by the existing four-quadrant clustering and
    articulated-path score, so this does not weaken the four-paw gate.
    """
    return min(_point(record, "head_world")[2], _point(record, "tail_world")[2])


def _path_to_root(name: str, by_name: Mapping[str, Mapping]) -> list[str]:
    path = []
    seen = set()
    current = name
    while current is not None:
        if current in seen or current not in by_name:
            raise SemanticRigError("bone hierarchy is cyclic or references a missing parent")
        seen.add(current)
        path.append(current)
        current = by_name[current].get("parent")
    path.reverse()
    return path


def _common_prefix(first: Sequence[str], second: Sequence[str]) -> list[str]:
    result = []
    for left, right in zip(first, second):
        if left != right:
            break
        result.append(left)
    return result


def _binary_cluster(values: Sequence[float], *, label: str) -> list[int]:
    """Split a cardinal coordinate into deterministic low/high clusters.

    Exported quadruped rigs sometimes carry a disconnected hoof controller in
    addition to the deforming lower-leg chain.  Both branches terminate near
    the floor, so a literal low-leaf count sees eight feet.  Clustering the
    forward and lateral coordinates independently lets us recover the four
    anatomical paw locations without relying on bone names.
    """
    if len(values) < 2:
        raise SemanticRigError(f"{label} clustering needs at least two values")
    lower = min(float(value) for value in values)
    upper = max(float(value) for value in values)
    if upper - lower <= 1.0e-9:
        raise SemanticRigError(f"{label} low-limb coordinates are degenerate")
    assignments = [0] * len(values)
    for _iteration in range(32):
        updated = [
            0 if abs(float(value) - lower) <= abs(float(value) - upper) else 1
            for value in values
        ]
        if not any(index == 0 for index in updated) or not any(
            index == 1 for index in updated
        ):
            raise SemanticRigError(f"{label} low-limb clustering collapsed")
        new_lower = sum(
            float(value) for value, index in zip(values, updated) if index == 0
        ) / updated.count(0)
        new_upper = sum(
            float(value) for value, index in zip(values, updated) if index == 1
        ) / updated.count(1)
        if updated == assignments and abs(new_lower - lower) <= 1.0e-12 and abs(
            new_upper - upper
        ) <= 1.0e-12:
            break
        assignments = updated
        lower, upper = new_lower, new_upper
    return assignments


def _select_anatomical_foot_leaves(
    low_leaves: Sequence[str],
    by_name: Mapping[str, Mapping],
    *,
    forward_index: int,
    forward_sign: float,
    lateral_index: int,
    low_limit: float,
) -> list[str]:
    """Choose one deforming limb endpoint for each of four paw locations.

    A candidate with a deeper root path wins within a paw cluster.  This keeps
    the articulated upper/lower-leg chain and leaves disconnected IK/hoof/end
    controls as auxiliary branches.  The final lexical tie-break is only for
    deterministic output; it is never used to decide the paw quadrant.
    """
    if len(low_leaves) < 4:
        raise SemanticRigError(
            "expected exactly four anatomical low limb groups, "
            f"but found only {list(low_leaves)}"
        )
    if len(low_leaves) == 4:
        return list(low_leaves)

    forward_values = [
        forward_sign * _point(by_name[name], "head_world")[forward_index]
        for name in low_leaves
    ]
    lateral_values = [
        _point(by_name[name], "head_world")[lateral_index]
        for name in low_leaves
    ]
    forward_groups = _binary_cluster(forward_values, label="front/hind")
    lateral_groups = _binary_cluster(lateral_values, label="left/right")
    quadrants: dict[tuple[int, int], list[str]] = {}
    for name, forward_group, lateral_group in zip(
        low_leaves, forward_groups, lateral_groups
    ):
        quadrants.setdefault((forward_group, lateral_group), []).append(name)
    expected = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if set(quadrants) != expected:
        raise SemanticRigError(
            "expected exactly four anatomical low limb groups, got "
            f"{ {key: sorted(value) for key, value in quadrants.items()} }"
        )

    def candidate_score(name: str) -> tuple[int, int, float, str]:
        path = _path_to_root(name, by_name)
        heads = [_point(by_name[item], "head_world")[2] for item in path]
        vertical_span = max(heads) - min(heads)
        articulated_high_bones = sum(
            value > low_limit for value in heads[1:]
        )
        return (articulated_high_bones, len(path), vertical_span, name)

    return [
        max(quadrants[key], key=candidate_score)
        for key in sorted(quadrants)
    ]


def _collect_auxiliary_branches(
    by_name: Mapping[str, Mapping], covered: set[str]
) -> tuple[tuple[str, ...], ...]:
    """Return residual subtrees in deterministic parent-first order.

    TokenRig may add high branches for ears, muzzle, jaw, or similar anatomy.
    They are not extra locomotion chains, but they must remain represented so
    the retargeter can make them follow their nearest semantic ancestor.
    """
    missing = set(by_name) - covered
    if not missing:
        return ()
    roots = sorted(
        name for name in missing if by_name[name].get("parent") not in missing
    )
    if not roots:
        raise SemanticRigError("auxiliary bone subtrees have no covered attachment")
    visited: set[str] = set()
    branches: list[tuple[str, ...]] = []

    def visit(name: str, ordered: list[str]) -> None:
        if name in visited:
            raise SemanticRigError(f"auxiliary bone subtree repeats {name}")
        visited.add(name)
        ordered.append(name)
        for child in sorted(by_name[name].get("children", [])):
            if child in missing:
                visit(child, ordered)

    for root in roots:
        parent = by_name[root].get("parent")
        if parent not in covered:
            raise SemanticRigError(
                f"auxiliary branch {root} does not attach to a semantic bone"
            )
        ordered: list[str] = []
        visit(root, ordered)
        branches.append(tuple(ordered))
    if visited != missing:
        raise SemanticRigError(
            "auxiliary bone coverage is incomplete: "
            f"missing={sorted(missing - visited)}"
        )
    return tuple(branches)


def infer_quadruped_semantics(
    records: Iterable[Mapping],
    *,
    bbox_min: Sequence[float],
    bbox_extent: Sequence[float],
    front_axis: str,
    low_leaf_height_fraction: float = 0.22,
) -> QuadrupedSemantics:
    """Decompose a one-root quadruped hierarchy using geometry and topology.

    Both endpoints of a leaf segment are considered for the low-band because
    image-to-3D rest poses often contain a lifted far-side paw.  Potential
    extrapolated hoof/controller tails remain safe: four-quadrant clustering
    and articulated-path scoring select one deforming chain per paw.
    """
    axis_components = {
        "positive-x": (0, 1.0, 1),
        "negative-x": (0, -1.0, 1),
        "positive-y": (1, 1.0, 0),
        "negative-y": (1, -1.0, 0),
    }
    if front_axis not in axis_components:
        raise SemanticRigError(f"unsupported front axis: {front_axis}")
    records = [dict(record) for record in records]
    by_name = {record.get("name"): record for record in records}
    if None in by_name or len(by_name) != len(records):
        raise SemanticRigError("bone names must be present and unique")
    roots = [name for name, record in by_name.items() if record.get("parent") is None]
    if len(roots) != 1:
        raise SemanticRigError(f"quadruped rig needs one root, got {roots}")
    root = roots[0]
    floor = float(bbox_min[2])
    height = float(bbox_extent[2])
    if height <= 0.0:
        raise SemanticRigError("mesh height must be positive")
    low_limit = floor + float(low_leaf_height_fraction) * height
    leaves = [
        name for name, record in by_name.items() if not record.get("children", [])
    ]
    low_leaves = [
        name for name in leaves if _leaf_floor_probe(by_name[name]) <= low_limit
    ]
    forward_index, sign, lateral_index = axis_components[front_axis]
    foot_leaves = _select_anatomical_foot_leaves(
        low_leaves,
        by_name,
        forward_index=forward_index,
        forward_sign=sign,
        lateral_index=lateral_index,
        low_limit=low_limit,
    )
    # Exclude only the four selected anatomical feet from head/tail inference.
    # A breed-correct tail can legitimately curve into the same low band as a
    # paw (the generated Labrador is one such case).  Extra low hoof controls
    # remain safe here: the anatomical head and tail are selected by the
    # reviewed forward extremes, while residual control branches are retained
    # as auxiliary coverage below.
    non_foot_leaves = [name for name in leaves if name not in foot_leaves]
    if len(non_foot_leaves) < 2:
        raise SemanticRigError("quadruped rig needs distinct head and tail leaves")

    forward = (
        lambda name: sign
        * _point(by_name[name], "head_world")[forward_index]
    )
    head_leaf = max(non_foot_leaves, key=forward)
    tail_leaf = min(non_foot_leaves, key=forward)
    if head_leaf == tail_leaf:
        raise SemanticRigError("head and tail leaf inference collapsed")
    head_path = _path_to_root(head_leaf, by_name)
    tail_path = _path_to_root(tail_leaf, by_name)
    head_tail_common = _common_prefix(head_path, tail_path)
    if not head_tail_common:
        raise SemanticRigError("head and tail chains do not share the root")
    tail_chain = tuple(tail_path[len(head_tail_common) :])
    if not tail_chain:
        raise SemanticRigError("tail chain is empty")

    limb_entries = []
    axial_attachment_indices = []
    for leaf in foot_leaves:
        path = _path_to_root(leaf, by_name)
        common = _common_prefix(path, head_path)
        if not common:
            raise SemanticRigError(f"foot path {leaf} does not share the axial root")
        chain = tuple(path[len(common) :])
        if not chain:
            raise SemanticRigError(f"foot chain {leaf} is empty")
        attachment = common[-1]
        axial_attachment_indices.append(head_path.index(attachment))
        lateral = sum(
            _point(by_name[name], "head_world")[lateral_index] for name in chain
        ) / len(chain)
        limb_entries.append(
            {
                "leaf": leaf,
                "chain": chain,
                "attachment": attachment,
                "forward": forward(leaf),
                "lateral": lateral,
            }
        )

    # The two most forward paws are forelimbs; the remaining two are hindlimbs.
    ordered = sorted(limb_entries, key=lambda item: item["forward"], reverse=True)
    front = ordered[:2]
    hind = ordered[2:]
    for label, pair in (("front", front), ("hind", hind)):
        if len(pair) != 2:
            raise SemanticRigError(f"{label} limb pair is incomplete")
        pair.sort(key=lambda item: item["lateral"])
        if not pair[0]["lateral"] < pair[1]["lateral"]:
            raise SemanticRigError(f"{label} limb lateral ordering is degenerate")

    last_limb_attachment = max(axial_attachment_indices)
    axial = tuple(head_path[: last_limb_attachment + 1])
    head_chain = tuple(head_path[last_limb_attachment + 1 :])
    if not axial or not head_chain:
        raise SemanticRigError("axial or head chain is empty")
    core_chains = (
        axial,
        head_chain,
        tail_chain,
        front[0]["chain"],
        front[1]["chain"],
        hind[0]["chain"],
        hind[1]["chain"],
    )
    core_covered = {name for chain in core_chains for name in chain}
    auxiliary_branches = _collect_auxiliary_branches(by_name, core_covered)
    semantics = QuadrupedSemantics(
        root=root,
        axial=axial,
        head_chain=head_chain,
        tail_chain=tail_chain,
        front_side_negative=front[0]["chain"],
        front_side_positive=front[1]["chain"],
        hind_side_negative=hind[0]["chain"],
        hind_side_positive=hind[1]["chain"],
        auxiliary_branches=auxiliary_branches,
        foot_leaves=tuple(sorted(foot_leaves)),
    )
    if set(semantics.all_bones()) != set(by_name):
        missing = sorted(set(by_name) - set(semantics.all_bones()))
        extra = sorted(set(semantics.all_bones()) - set(by_name))
        raise SemanticRigError(
            f"semantic decomposition must cover every bone; missing={missing} extra={extra}"
        )
    return semantics
