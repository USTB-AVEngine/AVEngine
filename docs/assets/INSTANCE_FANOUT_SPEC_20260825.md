# Instance fan-out spec (2026-08-25)

Which instance-level axes a variant batch varies, and what each row costs.  This
document selects from
[`GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md`](GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md);
it does not redefine it.

## Decision

| Axis | Status | Why |
| --- | --- | --- |
| `size` | varied | A reviewer can confirm it in a frame, and it is realized by a runtime scale. |
| `coat_profile` | varied | A reviewer can confirm it in a frame, within that breed's reviewed three-level domain. |
| `body_build` | pinned to the source asset's value | slim versus stocky cannot be confirmed by looking at frames. |
| `life_stage` | pinned to the source asset's value | adult versus senior cannot be confirmed by looking at frames. |

The project requires visual claims to be self-certified by looking at rendered
frames.  An attribute a reviewer cannot confirm that way cannot carry a QA gold
label: a wrong value would be undetectable downstream, and the model would score
against an unverifiable answer.  Pinning is therefore about label integrity, not
about saving work.

Both pinned axes stay in the appearance contract and in every registered entry.
Nothing is deleted: the L9 orthogonal array, `APPEARANCE_AXES`,
`CANONICAL_DOMAINS` and `COAT_PROFILE_DOMAINS` are untouched.  A future batch that
wants an orthogonal sweep over all four axes still can; it simply does not use
this spec.

## Everything is derived, nothing is authored per breed

`examples/assets/instance_fanout_axes_v1.json` names the enabled and pinned axes
and nothing else.  `tools/assets/plan_instance_variants.py` then reads:

- axis domains from `avengine.appearance.contracts` — `CANONICAL_DOMAINS` for
  `size`, `COAT_PROFILE_DOMAINS[(species_id, breed_id, profile_id)]` for the coat;
- realized attributes from `examples/runtime/source_asset_runtime_profiles.json`;
- the per-axis realization operation from the contract's axis-to-operation table.

No species, breed, coat value or asset id appears in the tool, the spec or the
tests.  A breed whose reviewed coat domain is not registered in the contract
fails closed with that instruction, which is the same rule the contract states
for adding a breed.

**Derived ids observe the source id rather than rebuilding it.**  Registered ids
follow `<prefix><coat token>_<size>_<body_build>_<life_stage><tail>`, but the
prefix is not always the full breed id — at least one registered asset
abbreviates it.  The rule therefore locates the attribute triple inside the
asset's own id, keeps the prefix and tail verbatim, and substitutes the varying
tokens.  The unit tests round-trip this against every appearance-realized asset
in the registry: each plan must contain its own source id exactly once.

## Cost model

The plan marks every row, using the contract's operation table:

- `uniform_actor_scale_v1` changes no geometry and no material, so a row that
  varies only `size` is a **runtime-only derivation**: write the scale into the
  runtime profile, import nothing.
- every other operation needs a **new UE asset**, exactly as the registered
  material variants each got their own gate directory.

Rows that agree on all non-scale axis values share one UE import; the group that
matches the source asset reuses what is already imported.  A three-value coat
domain crossed with three sizes therefore yields nine rows, two UE imports and
six runtime-only derivations.

Two engineering limits apply when executing a plan: the Studio sweep cap is 64
tasks per batch, and a package stage is assembled wholesale — batch the rows that
need a UE import instead of running them one at a time.

## Running it

```bash
$PY tools/assets/plan_instance_variants.py \
    --asset-id <registered source asset id> \
    --output <fresh path>/instance_variant_plan.json
```

Omit `--asset-id` to plan every eligible asset; entries whose ids do not encode
their attributes are reported as skipped rather than silently dropped.  The
output path is never overwritten.

Planning is free and offline.  Executing the rows that need a UE import still
requires the SPEAR-side appearance realization, a UE import and a stage
assembly, and registry insertion still needs the owner's nod.
