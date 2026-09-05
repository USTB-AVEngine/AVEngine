#!/usr/bin/env python3
"""Create an isolated SPEAR Editor stage using current source and installed binaries.

The installed binary source revision must have identical checked-in native UE
source. Content is copied into the new project, never hard-linked; no old
Editor stage is modified. Generated maps and imports belong to this new stage.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[2]


def assemble(source_stage: Path, output: Path, binary_source_revision: str) -> dict:
    source_stage = source_stage.resolve()
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace editor stage: {output}")
    native = REPO / "native/spear/unreal"
    required = [source_stage / "SpearSim/Binaries/Linux/UnrealEditor.modules",
                source_stage / "SpearSim/Content",
                source_stage / "plugins/SpContent/Content",
                native / "SpearSim/SpearSim.uproject"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    subprocess.run(["git", "cat-file", "-e", binary_source_revision + "^{commit}"],
                   cwd=REPO, check=True)
    check = subprocess.run(["git", "diff", "--quiet", binary_source_revision,
                            "HEAD", "--", "native/spear/unreal"], cwd=REPO)
    if check.returncode:
        raise ValueError("native source differs from installed binary origin; build new binaries")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "native/spear/unreal"], cwd=REPO).returncode:
        raise ValueError("uncommitted native changes require their own binary build")
    output.mkdir(parents=True)
    for name in ("SpearSim", "plugins"):
        shutil.copytree(native / name, output / name)
    for relative in ("SpearSim/Binaries", "SpearSim/Content", "plugins/SpContent/Content"):
        source = source_stage / relative
        target = output / relative
        if target.exists():
            # The repository may carry small content seeds. Copy only to this
            # fresh destination; installed content remains a read-only input.
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copytree(source, target)
    project = output / "SpearSim/SpearSim.uproject"
    config = json.loads(project.read_text())
    plugins = config.setdefault("Plugins", [])
    for name in ("USDImporter", "SpContent", "PythonScriptPlugin", "EditorScriptingUtilities"):
        matches = [v for v in plugins if v.get("Name") == name]
        if matches:
            matches[0]["Enabled"] = True
        else:
            plugins.append({"Name": name, "Enabled": True})
    project.write_text(json.dumps(config, indent=2) + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    result = {"status": "assembled", "research_only": True, "qualification_claim": False,
              "avengine_source": str(REPO), "avengine_source_commit": commit,
              "binary_source_revision": binary_source_revision,
              "native_source_comparison": "git_diff_equal",
              "installed_binary_and_content_source": str(source_stage),
              "content_copy_policy": "independent_files_no_hardlinks",
              "uproject": str(project), "native_execution": "not_run"}
    (output / "STAGE_PROVENANCE.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-source-revision", required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.source_stage, args.output, args.binary_source_revision), indent=2))


if __name__ == "__main__":
    main()
