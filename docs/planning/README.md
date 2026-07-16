# Habitat-Native Planning Authority

This directory preserves the planning inputs for the Habitat-native AVEngine
restructure. The files were copied byte-for-byte from `/data/jzy/code` on
2026-07-16 because that parent directory is not a Git repository.

Read them in this order:

1. `CODEX_ENTRY_PROMPT_HABITAT_NATIVE_AVENGINE.md`
2. `CODEX_MASTER_PLAN_HABITAT_NATIVE_AVENGINE_20260716.md`
3. `../reviews/AVEngine_review_20260715_zh.md`
4. `../../schemas/avengine_timeline_v2.schema.json`

The timeline v2 schema is immutable. Any semantic change requires a new schema
version, an ADR, and a migration path. Older plans under `docs/superpowers/`
describe the legacy UE/SPEAR route and remain historical migration evidence;
they are not the authority for the Habitat-native architecture.

## Import verification

| File | SHA-256 |
|---|---|
| `CODEX_ENTRY_PROMPT_HABITAT_NATIVE_AVENGINE.md` | `3d1710c68f2a0389e98b539308a8675c7e719b317d1a0743f1cb7c9ae9655b48` |
| `CODEX_MASTER_PLAN_HABITAT_NATIVE_AVENGINE_20260716.md` | `6f0f309baa13d51d73e96c87263560b19e2339677da3d20692e6a76d783d3ba3` |
| `../reviews/AVEngine_review_20260715_zh.md` | `a4066feefc807dccc6842bfc9e92f155e89bb554cdc9d80738cd4c96c966786d` |
| `../../schemas/avengine_timeline_v2.schema.json` | `4e1624cc17d1117d1ef046c13700c5c9c4a0357265ba2fbfb2859e9929bd6773` |
