#!/usr/bin/env python3
"""Derive sealed audio programs for Gate B twin points (qa-v3 pilot).

Gate B 孪生沿用主点的事件时间与槽位序列(音频锚不变),但:
- 外观孪生(twA)交换了两只狗的资产 → endpoint 绑定随槽位×资产翻转,
  program 的候选与事件 endpoint 必须按孪生 actor_selection 重新解析;
- 路线孪生(twR)资产不换 → endpoint 原样,仅 program_id 换名。

每个孪生生成自己的 program(+plan 拷贝),用引擎 canonical_json_sha256
重新密封;渲染调度器由此回归"每点用自己的 program"的统一约定。

endpoint 解析走登记表(与渲染链 derive_slot_bindings 同口径:
entity_instance_id × entity_asset_id × emitter_anchor_id 唯一匹配),
不手写映射。输出 no-clobber:目标文件已存在则跳过(幂等),内容不一致
则失败。research_candidate。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from avengine.contracts.json_io import canonical_json_sha256  # noqa: E402


def resolve_slot_endpoints(selection: dict, asset_registry: dict,
                           endpoint_registry: dict) -> dict[str, str]:
    assets = {a["asset_id"]: a for a in asset_registry["assets"]}
    endpoints = endpoint_registry["source_endpoints"]
    out: dict[str, str] = {}
    for actor in selection["actors"]:
        slot = actor["source_slot_id"]
        asset_id = actor["asset_id"]
        instance = actor["legacy_timeline_actor_id"]
        anchor = assets[asset_id]["default_emitter_anchor_id"]
        matches = [e for e in endpoints
                   if e["binding"]["entity_asset_id"] == asset_id
                   and e["binding"]["entity_instance_id"] == instance
                   and e["binding"]["emitter_anchor_id"] == anchor]
        if len(matches) != 1:
            raise SystemExit(
                f"FAIL: expected exactly one endpoint for {instance}/"
                f"{asset_id}/{anchor}, found {len(matches)}")
        out[slot] = matches[0]["source_endpoint_id"]
    if set(out) != {"source1", "source2"}:
        raise SystemExit("FAIL: twin selection must bind source1+source2")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design-root", required=True, type=Path)
    parser.add_argument("--endpoint-registry", required=True, type=Path)
    parser.add_argument("--asset-registry", required=True, type=Path)
    args = parser.parse_args(argv)

    programs_dir = args.design_root / "programs"
    asset_reg = json.loads(args.asset_registry.read_text())
    ep_reg = json.loads(args.endpoint_registry.read_text())

    twins = []
    for pdir in sorted(args.design_root.iterdir()):
        if not pdir.is_dir() or "_tw" not in pdir.name:
            continue
        spec = json.loads((pdir / "spec.json").read_text())
        if spec.get("twin_of"):
            twins.append((pdir, spec))
    made = skipped = 0
    for pdir, spec in twins:
        twin_id, parent = pdir.name, spec["twin_of"]
        parent_prog_matches = sorted(
            programs_dir.glob(f"qa_v3_*_{parent}_rand_v1.json"))
        if len(parent_prog_matches) != 1:
            print(f"FAIL: parent program for {twin_id} ({parent}): "
                  f"{len(parent_prog_matches)} matches", file=sys.stderr)
            return 1
        parent_prog = json.loads(parent_prog_matches[0].read_text())
        parent_plan = json.loads(
            (programs_dir / (parent_prog_matches[0].stem + ".plan.json"))
            .read_text())

        selection = json.loads((pdir / "actor_selection.json").read_text())
        slot_eps = resolve_slot_endpoints(selection, asset_reg, ep_reg)
        old_candidates = parent_prog["candidate_source_endpoint_ids"]
        old_to_slot = {old_candidates[0]: "source1",
                       old_candidates[1]: "source2"}

        doc = {k: v for k, v in parent_prog.items()
               if k != "program_content_sha256"}
        pair_kind = parent_prog["program_id"].split("_")[2]  # qa_v3_<kind>_...
        doc["program_id"] = f"qa_v3_{pair_kind}_{twin_id}_rand_v1"
        doc["candidate_source_endpoint_ids"] = [slot_eps["source1"],
                                                slot_eps["source2"]]
        doc["events"] = [
            dict(e, source_endpoint_id=slot_eps[old_to_slot[
                e["source_endpoint_id"]]])
            for e in parent_prog["events"]]
        doc["program_content_sha256"] = canonical_json_sha256(doc)

        prog_path = programs_dir / f"qa_v3_{pair_kind}_{twin_id}_rand_v1.json"
        plan_path = programs_dir / f"qa_v3_{pair_kind}_{twin_id}_rand_v1.plan.json"
        if prog_path.exists():
            existing = json.loads(prog_path.read_text())
            if existing != doc:
                print(f"FAIL: {prog_path} exists with different content",
                      file=sys.stderr)
                return 1
            skipped += 1
            continue
        prog_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        plan_doc = dict(parent_plan, derived_for_twin=twin_id,
                        twin_of=parent)
        plan_path.write_text(json.dumps(plan_doc, ensure_ascii=False,
                                        indent=1))
        made += 1
        print(f"ok {twin_id}: candidates={doc['candidate_source_endpoint_ids']}")
    print(f"twin programs made={made} skipped_identical={skipped} "
          f"twins={len(twins)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
