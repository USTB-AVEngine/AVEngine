#!/usr/bin/env python3
"""Reseal and re-pin example evidence bindings after a legitimate content change.

Three binding kinds, all discovered from the validators themselves:
  1. file pins   {"path": rel, ["byte_size": n,] "sha256": file_bytes_hash}
                 {"location": {"path": rel}, "sha256": ..., ["byte_size": ...]}
  2. authority   {"authority_uri": "repo://rel", "authority_file_sha256": ...,
                  ["authority_manifest_content_sha256": target's own seal]}
  3. self-seals  any key ending in content_sha256 (not authority_*):
                 canonical_json_sha256 of the containing dict minus the key.

Iterate to a fixpoint because fixing a child's bytes invalidates parents.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
from avengine.contracts.json_io import canonical_json_sha256, sha256_file  # noqa: E402

SEAL_KEYS = {"manifest_content_sha256", "program_content_sha256",
             "registry_content_sha256", "record_content_sha256",
             "trajectory_content_sha256", "request_content_sha256"}
PIN_MARKERS = {"path", "authority_uri", "location", "uri"}
files = [Path(p) for p in subprocess.run(
    ["git", "ls-files", "examples"], capture_output=True, text=True, check=True
).stdout.splitlines() if p.endswith(".json")]


def fix_doc(doc, changes):
    if isinstance(doc, dict):
        for v in doc.values():
            fix_doc(v, changes)
        # 1. file pins
        target = None
        if isinstance(doc.get("path"), str) and "sha256" in doc:
            target = Path(doc["path"])
        elif (isinstance(doc.get("location"), dict)
              and isinstance(doc["location"].get("path"), str) and "sha256" in doc):
            target = Path(doc["location"]["path"])
        if target is not None and target.exists() and target.is_file():
            actual = sha256_file(target)
            if doc.get("sha256") != actual:
                changes.append(f"pin sha256 -> {target}")
                doc["sha256"] = actual
            if "byte_size" in doc and doc["byte_size"] != target.stat().st_size:
                changes.append(f"pin byte_size -> {target}")
                doc["byte_size"] = target.stat().st_size
        # 2. authority pins
        uri = doc.get("authority_uri")
        if isinstance(uri, str) and uri.startswith("repo://") and "authority_file_sha256" in doc:
            t = Path(uri[len("repo://"):])
            if t.exists():
                actual = sha256_file(t)
                if doc["authority_file_sha256"] != actual:
                    changes.append(f"authority file -> {t}")
                    doc["authority_file_sha256"] = actual
                if "authority_manifest_content_sha256" in doc:
                    try:
                        seal = json.loads(t.read_text(encoding="utf-8")).get(
                            "manifest_content_sha256")
                    except json.JSONDecodeError:
                        seal = None
                    if seal and doc["authority_manifest_content_sha256"] != seal:
                        changes.append(f"authority seal -> {t}")
                        doc["authority_manifest_content_sha256"] = seal
        # 3. self-seals (confirmed keys only, never on pin stubs)
        if not (PIN_MARKERS & doc.keys()):
            for key in list(doc):
                if key in SEAL_KEYS and isinstance(doc[key], str) and len(doc[key]) == 64:
                    stored = doc.pop(key)
                    actual = canonical_json_sha256(doc)
                    doc[key] = actual
                    if actual != stored:
                        changes.append(f"seal {key}")
    elif isinstance(doc, list):
        for v in doc:
            fix_doc(v, changes)


for iteration in range(1, 8):
    total = 0
    for p in sorted(files):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        changes: list[str] = []
        fix_doc(doc, changes)
        if changes:
            p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
            total += len(changes)
            print(f"[{iteration}] {p}: {len(changes)} fix(es)")
    print(f"[iteration {iteration}] {total} change(s)")
    if total == 0:
        break
print("[done]")
