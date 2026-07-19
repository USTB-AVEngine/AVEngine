#!/usr/bin/env python3
"""Build a small local review page for optional SPEAR/UE room renders.

The page is intentionally a viewer, not a release manifest. It reads the one
evidence JSON beside each run and links generated media without computing or
freezing another layer of hashes.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY / "tmp/m6y/REVIEW_INDEX.html"


def _load_document(
    run_directory: Path, *, filename: str, owner: str
) -> tuple[Path, dict[str, Any]]:
    root = run_directory.resolve()
    document_path = root / filename
    if not root.is_dir() or not document_path.is_file():
        raise ValueError(f"{owner} document is missing: {document_path}")
    try:
        value = json.loads(document_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {owner} document: {document_path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{owner} document root must be an object")
    return root, value


def _load_evidence(run_directory: Path, *, owner: str) -> tuple[Path, dict[str, Any]]:
    return _load_document(run_directory, filename="evidence.json", owner=owner)


def _status(value: Any) -> str:
    return str(value) if isinstance(value, str) and value else "unknown"


def _url(path: Path, *, output: Path) -> str:
    relative = Path(os.path.relpath(path.resolve(), output.parent.resolve()))
    return quote(relative.as_posix(), safe="/._-")


def _media_card(
    *,
    title: str,
    media: Mapping[str, Any],
    base_directory: Path,
    output: Path,
) -> str:
    path_value = media.get("path")
    if not isinstance(path_value, str) or not path_value:
        return ""
    media_path = (base_directory / path_value).resolve()
    exists = media_path.is_file()
    state = _status(media.get("status")) if exists else "missing"
    dimensions = ""
    if isinstance(media.get("width"), int) and isinstance(media.get("height"), int):
        dimensions = f"{media['width']}x{media['height']}"
    frames = media.get("frame_count")
    details = " · ".join(
        item
        for item in (dimensions, f"{frames} frames" if isinstance(frames, int) else "")
        if item
    )
    link = _url(media_path, output=output)
    video = (
        f'<video controls preload="metadata" src="{link}"></video>'
        if exists and media_path.suffix.casefold() == ".mp4"
        else ""
    )
    return (
        '<article class="media-card">'
        f"<h3>{escape(title)}</h3>"
        f'<p><span class="status {escape(state)}">{escape(state)}</span>'
        f" {escape(details)}</p>"
        f"{video}"
        f'<p><a href="{link}">{escape(media_path.name)}</a></p>'
        "</article>"
    )


def _evidence_link(path: Path, *, output: Path) -> str:
    return f'<a href="{_url(path, output=output)}">evidence.json</a>'


def _habitat_apartment_section(
    run: tuple[Path, Mapping[str, Any]] | None, output: Path
) -> str:
    if run is None:
        return ""
    root, value = run
    variants = value.get("variants")
    if not isinstance(variants, list):
        raise ValueError("Habitat Apartment bundle has no variants list")
    representative = next(
        (
            item
            for item in variants
            if isinstance(item, Mapping)
            and item.get("scenario_id") == "S3"
            and item.get("variant_id") == "A"
        ),
        None,
    )
    if not isinstance(representative, Mapping):
        raise ValueError("Habitat Apartment bundle lacks S3/A review media")
    clip = value.get("clip") if isinstance(value.get("clip"), Mapping) else {}
    profile = (
        value.get("review_visual_profile")
        if isinstance(value.get("review_visual_profile"), Mapping)
        else {}
    )
    frame_count = clip.get("frame_count")
    cards: list[str] = []
    for key, label, width, height in (
        ("clean_video", "S3 clean + binaural", 1280, 720),
        ("diagnostic_video", "S3 main view + Topdown/binaural", 1280, 480),
    ):
        path = representative.get(key)
        if isinstance(path, str):
            cards.append(
                _media_card(
                    title=label,
                    media={
                        "path": path,
                        "status": representative.get("status"),
                        "width": width,
                        "height": height,
                        "frame_count": frame_count,
                    },
                    base_directory=root,
                    output=output,
                )
            )
    state = _status(value.get("status"))
    bundle_index = root / "REVIEW_INDEX.html"
    return (
        "<section><h2>Habitat Apartment natural-light review</h2>"
        f'<p><span class="status {escape(state)}">{escape(state)}</span> · '
        f'<a href="{_url(bundle_index, output=output)}">full S0–S5 review</a> · '
        f'<a href="{_url(root / "bundle_manifest.json", output=output)}">'
        "bundle_manifest.json</a></p>"
        f"<p>Profile <code>{escape(str(profile.get('profile_id', 'unknown')))}</code> "
        "uses a shallow neutral window key, a cool-neutral bounce and a weaker "
        "warm practical while retaining HBAO. It redistributes contrast rather "
        "than applying one global darkening factor. The fixed-camera exterior "
        "is still a direction-projected panel, not UE glass, HDRIBackdrop, "
        "reflection capture or Lumen.</p>"
        f'<div class="media-grid">{"".join(cards)}</div></section>'
    )


def _apartment_section(root: Path, value: Mapping[str, Any], output: Path) -> str:
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Apartment evidence has no scenarios list")
    by_id = {
        item.get("scenario_id"): item
        for item in scenarios
        if isinstance(item, Mapping) and isinstance(item.get("scenario_id"), str)
    }
    cards: list[str] = []
    rows: list[str] = []
    for scenario_id in ("S0", "S3", "S4"):
        scenario = by_id.get(scenario_id)
        if not isinstance(scenario, Mapping):
            rows.append(
                f'<tr><td>{scenario_id}</td><td><span class="status missing">missing</span></td></tr>'
            )
            continue
        state = _status(scenario.get("status"))
        rows.append(
            f'<tr><td>{scenario_id}</td><td><span class="status {escape(state)}">{escape(state)}</span></td></tr>'
        )
        media = scenario.get("media")
        if not isinstance(media, Mapping):
            continue
        scenario_root = root / scenario_id
        for key, label in (
            ("ue_clean_binaural", "clean + Habitat-native binaural"),
            ("ue_topdown_binaural", "UE + Habitat-native Topdown/binaural"),
        ):
            record = media.get(key)
            if isinstance(record, Mapping):
                cards.append(
                    _media_card(
                        title=f"{scenario_id}: {label}",
                        media=record,
                        base_directory=scenario_root,
                        output=output,
                    )
                )
    suite_state = _status(value.get("status"))
    return (
        "<section><h2>Native SPEAR Apartment</h2>"
        f'<p>Suite <span class="status {escape(suite_state)}">'
        f"{escape(suite_state)}</span> · "
        f"{_evidence_link(root / 'evidence.json', output=output)}</p>"
        "<p>UE owns only comparison pixels. Timeline, source logic, audio, "
        "Topdown, flags and metadata remain Habitat-native authority.</p>"
        "<table><thead><tr><th>Scenario</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f'<div class="media-grid">{"".join(cards)}</div></section>'
    )


def _mp3d_section(root: Path, value: Mapping[str, Any], output: Path) -> str:
    media = value.get("media")
    if not isinstance(media, Mapping):
        raise ValueError("MP3D evidence has no media object")
    cards = []
    for key, label in (
        ("ue_clean_binaural", "UE clean + Habitat-native binaural"),
        ("ue_topdown_binaural", "UE + Habitat-native Topdown/binaural"),
        (
            "ue_habitat_topdown_triptych_binaural",
            "UE | Habitat | Topdown comparison",
        ),
    ):
        record = media.get(key)
        if isinstance(record, Mapping):
            cards.append(
                _media_card(
                    title=label,
                    media=record,
                    base_directory=root,
                    output=output,
                )
            )
    clock = value.get("clock") if isinstance(value.get("clock"), Mapping) else {}
    exposure = (
        value.get("exposure_qa")
        if isinstance(value.get("exposure_qa"), Mapping)
        else {}
    )
    run_state = _status(value.get("status"))
    exposure_state = _status(exposure.get("status"))
    color = (
        value.get("color_fidelity_qa")
        if isinstance(value.get("color_fidelity_qa"), Mapping)
        else {}
    )
    color_state = _status(color.get("status"))
    mean_ratio = color.get("mean_chroma_ratio_ue_to_habitat")
    ratio_text = f"{mean_ratio:.3f}" if isinstance(mean_ratio, (int, float)) else "?"
    return (
        "<section><h2>MP3D 17DRP5sb8fy</h2>"
        f'<p>Run <span class="status {escape(run_state)}">'
        f"{escape(run_state)}</span> · "
        f"{_evidence_link(root / 'evidence.json', output=output)}</p>"
        f"<p>{escape(str(clock.get('frame_count', '?')))} frames; "
        "Timeline v2 applicable: "
        f"{escape(str(clock.get('timeline_v2_applicable', 'unknown')).lower())}; "
        f'exposure QA: <span class="status {escape(exposure_state)}">'
        f"{escape(exposure_state)}</span>; color QA: "
        f'<span class="status {escape(color_state)}">{escape(color_state)}</span> '
        f"(UE/Habitat mean-chroma ratio {escape(ratio_text)}).</p>"
        "<p>The white/grey result was an import-color-space error, not an "
        "inherent property of the scan: MP3D reuses each source image for "
        "base color and occlusion, while UE needs an sRGB base-color view and "
        "a linear AO view. The corrected importer keeps both views and uses "
        "neutral output gain. Scan textures still contain baked illumination; "
        "the weak review lights do not reconstruct Matterport capture lights. "
        "Remaining holes are scan geometry. The retained route moves each root "
        "only 1.1 m in 18 seconds and is not a normal-speed result.</p>"
        f'<div class="media-grid">{"".join(cards)}</div></section>'
    )


def _replicacad_section(
    run: tuple[Path, Mapping[str, Any]] | None, output: Path
) -> str:
    if run is None:
        return (
            "<section><h2>ReplicaCAD apt_0</h2>"
            '<p><span class="status pending">pending</span></p>'
            "<p>The import plan exists, but no successful UE import/reload, runtime "
            "readback or comparison video is claimed yet.</p></section>"
        )
    root, value = run
    media = value.get("media")
    cards = []
    if isinstance(media, Mapping):
        for key, label in (
            ("ue_clean_binaural", "UE clean + Habitat-native binaural"),
            ("ue_topdown_binaural", "UE + Habitat-native Topdown/binaural"),
            (
                "ue_habitat_topdown_triptych_binaural",
                "UE | Habitat | Topdown comparison",
            ),
            ("ue_visual_only", "UE visual only"),
        ):
            record = media.get(key)
            if isinstance(record, Mapping) and isinstance(record.get("path"), str):
                cards.append(
                    _media_card(
                        title=label,
                        media=record,
                        base_directory=root,
                        output=output,
                    )
                )
    state = _status(value.get("status"))
    clock = value.get("clock") if isinstance(value.get("clock"), Mapping) else {}
    runtime = value.get("runtime") if isinstance(value.get("runtime"), Mapping) else {}
    scene = (
        runtime.get("scene_and_lighting_readback")
        if isinstance(runtime.get("scene_and_lighting_readback"), Mapping)
        else {}
    )
    lighting = (
        scene.get("lighting_profile_application")
        if isinstance(scene.get("lighting_profile_application"), Mapping)
        else {}
    )
    exposure = (
        value.get("exposure_qa")
        if isinstance(value.get("exposure_qa"), Mapping)
        else {}
    )
    claim_boundary = value.get("claim_boundary")
    claim_text = claim_boundary if isinstance(claim_boundary, str) else ""
    active_ids = lighting.get("active_positive_light_ids")
    excluded_ids = lighting.get("excluded_positive_light_ids")
    active_text = (
        ", ".join(str(item) for item in active_ids)
        if isinstance(active_ids, list)
        else "?"
    )
    excluded_text = (
        ", ".join(str(item) for item in excluded_ids)
        if isinstance(excluded_ids, list)
        else "?"
    )
    return (
        "<section><h2>ReplicaCAD apt_0</h2>"
        f'<p><span class="status {escape(state)}">{escape(state)}</span> · '
        f"{_evidence_link(root / 'evidence.json', output=output)}</p>"
        f"<p>{escape(str(clock.get('frame_count', '?')))} frames; "
        "Timeline v2 applicable: "
        f"{escape(str(clock.get('timeline_v2_applicable', 'unknown')).lower())}; "
        f"{escape(str(scene.get('tagged_comparison_visual_actor_count', '?')))} "
        "tagged scene actors; "
        f"{escape(str(scene.get('active_positive_point_light_count', '?')))} "
        f"active room-local lights (IDs {escape(active_text)}; excluded "
        f"{escape(excluded_text)}; UE scale "
        f"{escape(str(lighting.get('ue_intensity_scale', '?')))}); exposure QA: "
        f'<span class="status {escape(_status(exposure.get("status")))}">'
        f"{escape(_status(exposure.get('status')))}</span>.</p>"
        "<p>The two strongest positive dataset lights sit outside the open "
        "stage-shell bounds. The review profile sets those two to zero, keeps "
        "the three indoor source lights, and neither moves nor adds lights. "
        "A real Habitat point-light comparison was darker and less natural, "
        "so the maintained Habitat presentation remains <code>no_lights + "
        "HBAO</code>; its point-light realization is research-only.</p>"
        f"<p>{escape(claim_text)}</p>"
        f'<div class="media-grid">{"".join(cards)}</div></section>'
    )


def build_page(
    *,
    habitat_apartment: tuple[Path, Mapping[str, Any]] | None,
    apartment: tuple[Path, Mapping[str, Any]],
    mp3d: tuple[Path, Mapping[str, Any]],
    replicacad: tuple[Path, Mapping[str, Any]] | None,
    output: Path,
) -> str:
    apartment_root, apartment_value = apartment
    mp3d_root, mp3d_value = mp3d
    body = "".join(
        (
            _habitat_apartment_section(habitat_apartment, output),
            _apartment_section(apartment_root, apartment_value, output),
            _mp3d_section(mp3d_root, mp3d_value, output),
            _replicacad_section(replicacad, output),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AVEngine M6.x/M6.y Lighting Review</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 1500px; margin: auto; padding: 2rem; background: #11151a; color: #e8edf2; }}
    h1, h2 {{ color: #fff; }}
    section {{ margin: 2rem 0; padding: 1.25rem; background: #1a2028; border-radius: 12px; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }}
    .media-card {{ background: #0d1117; padding: 1rem; border-radius: 8px; }}
    video {{ width: 100%; background: #000; }}
    a {{ color: #77bdfb; }}
    table {{ border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ border: 1px solid #46515e; padding: .45rem .75rem; text-align: left; }}
    .status {{ display: inline-block; border-radius: 999px; padding: .12rem .55rem; background: #4a5562; }}
    .status.pass {{ background: #176b3a; }}
    .status.pending, .status.blocked {{ background: #7a5617; }}
    .status.fail, .status.missing {{ background: #8b2834; }}
  </style>
</head>
<body>
  <h1>AVEngine M6.x/M6.y Lighting and Comparison Review</h1>
  <p>Habitat-native AVEngine owns episode state, navigation, source centers,
  source logic, binaural audio, Topdown, flags and metadata. SPEAR/UE is a
  <code>comparison_visual</code> backend only.</p>
  {body}
</body>
</html>
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--habitat-apartment-bundle", type=Path)
    parser.add_argument("--apartment-suite", type=Path, required=True)
    parser.add_argument("--mp3d-run", type=Path, required=True)
    parser.add_argument("--replicacad-run", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    habitat_apartment = (
        _load_document(
            args.habitat_apartment_bundle,
            filename="bundle_manifest.json",
            owner="Habitat Apartment",
        )
        if args.habitat_apartment_bundle is not None
        else None
    )
    apartment = _load_evidence(args.apartment_suite, owner="Apartment")
    mp3d = _load_evidence(args.mp3d_run, owner="MP3D")
    replicacad = (
        _load_evidence(args.replicacad_run, owner="ReplicaCAD")
        if args.replicacad_run is not None
        else None
    )
    page = build_page(
        habitat_apartment=habitat_apartment,
        apartment=apartment,
        mp3d=mp3d,
        replicacad=replicacad,
        output=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(f"M6Y_REVIEW_INDEX_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
