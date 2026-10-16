# Legacy implementation boundary

The former UE/SPEAR-first AVEngine is a migration source, not the default
runtime. No module below this boundary may be imported by the default CLI,
fast tests or central dataset admission. Historical source locations and
decisions are documented under `docs/legacy/` and `docs/migration/`.
