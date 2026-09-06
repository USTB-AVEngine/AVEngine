# P7 prepared speech audio

1. 改了哪些文件（路径），提交号。

- `src/avengine/assets/sound_prepare.py`: speech-band filter/crop, source activity measurement, metadata bridge, class activity profiles, prepared-set builder, no-clobber reader and listening sample selector.
- `tools/assets/prepare_speech_audio.py`: P7-only entry point.
- `tests/unit/test_assets_sound_prepare.py`: 9 focused P7 tests; the original 6 tests remain.
- `docs/roadmap/codex_reports_20260906/P7_prepared_audio.md`: this report.
- Implementation baseline: `ca70486`; this report is committed with P7 implementation. Resolve its commit with `git log -1 --format=%H -- docs/roadmap/codex_reports_20260906/P7_prepared_audio.md`. The machine preparation is complete; required human listening remains pending.

2. 跑了哪些测试，各自的通过、失败、跳过计数；有失败就贴错误原文。

- Final `tests/unit/test_assets_sound_prepare.py`: **15 passed, 0 failed, 0 skipped** (1.53 s; the earlier 66-test P1/P2/P7 run included the first 14), including 5.5 kHz content retention and explicit full-scale clipping rejection. Initial worker run had 12 passed. The final derived-ID test also distinguishes an explicitly changed normalization setting; default candidate IDs are unchanged.
- `py_compile` for `sound_prepare.py` and `prepare_speech_audio.py`: passed.
- `git diff --check` for the P7 files: passed.
- The real entry point ran with one CPU worker and BLAS/OMP/MKL thread counts set to 1. The initial detector found 307 candidates; the final encoding-safe set contains 296 prepared WAVs plus 11 explicit encoding rejections. A second invocation was intentionally checked and refused with `FileExistsError` for the existing manifest; this verifies no-clobber behavior.
- The parent regenerated the tool index and its test passed. Only this task's new tool entry is staged here; concurrent P4/P12 entries remain with those tasks.

3. 验收产物的路径，以及你亲自看过或听过的核对结果。

- Prepared set and manifest: `tmp/p7_prepared_audio_v3/prepared_manifest.json` (resolved under `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/`).
- Activity profiles: `tmp/p7_prepared_audio_v3/activity_profiles.json`.
- Non-speech inventory: `tmp/p7_prepared_audio_v3/nonverbal_source_inventory.json`.
- Ten-sample review table and JSON: `tmp/p7_prepared_audio_v3/listening_log_pending.md` and `tmp/p7_prepared_audio_v3/listening_samples_pending.json`.
- Stage-aware source provenance check (one M and one F row): `tmp/p7_prepared_audio_v1/provenance_stage_check.json`.
- Patch diagnostics moved under `tmp/p7_prepared_audio_v1/patch_diagnostics/`.
- The manifest retains all 613 speech registry rows: 307 detector candidates, of which **296 are prepared** and **11 are explicitly rejected because highpass output exceeds integer-PCM full scale**, plus 293 duration/activity rejections and 13 configured non-VCTK exclusions. Final prepared gender counts are **M=155, F=141**. All prepared rows have metadata and transcripts from original clip.json sidecars. The 307 detector candidate records remain inspectable; no candidate is silently removed from the denominator.
- Prepared duration is 1.92--4.98 s; source speech activity duration is 1.50--3.59 s; source audible span is 1.86--4.98 s. All 296 final prepared IDs are unique and all prepared durations are at most 5 s.
- I inspected the manifest, per-row measurements, profile JSON, inventory, and pending table. No human or model listening was performed, so no listening result is claimed.

4. 没做完的部分和原因，分清是题义不适用、接口未实现还是证据缺失。

- Evidence missing / unsampled: the ten required human listening records remain `pending_human`; reviewer, heard, consonant preservation, and notes are null.
- Evidence missing / calibration pending: non-speech thresholds remain explicit placeholders. Animal profiles use full-band energy with a 0.5 s placeholder minimum; device continuous profiles use full-band energy relative to a placeholder noise floor and require source activity to cover the query window; short prompts are repeatable and counted per event.
- Encoding failures: 11 detector candidates are retained with explicit full-scale overflow reasons; no clipped or normalized substitute is used.
- Interface not implemented: none within P7. The prepared manifest deliberately contains source activity only; listener audibility and wet-tail fields are absent and belong to P6.
- Not applicable by definition: none.
- The stage-aware check shows no same-object mismatch: for both one male and one female row, the registry provenance source SHA matches the prior generic prepared output (`sound_library_v1_prepared`), while the current source-library `clip.wav` has its own source SHA and P7 records that current source hash. The event URI has a third split-event SHA. These are different processing stages; the registry and source library were not changed.

5. 对 Claude 的接口有什么要求（字段名、函数签名、数据位置）。

- Use `prepare_speech_registry(registry, output_root, source_library_root=..., vctk_only=True)` to build a fresh derived set; the default VCTK policy is what yields the 307 strict machine candidates.
- Use `iter_prepared_speech_clips(manifest)` or `load_prepared_speech_manifest(path)` to consume only rows with `status=prepared`.
- Each prepared row exposes `prepared_audio_id`, `source_asset_id`, `prepared`, `source_pcm_path`, `source_metadata_path`, `source_sha256`, `gender`, `gender_source`, `transcript`, `transcript_source`, `source_offset_s`, `source_crop_start_sample`, `source_crop_end_sample_exclusive`, `filter_parameters`, `detector_parameters`, and `source_activity`.
- `source_activity` contains the source PCM audible span, union activity duration, interval count, and intervals. It does not assert listener-side audibility or wet-tail audibility.
- Pure clip APIs are `measure_speech_band(samples, rate)`, `prepare_speech_clip(samples, rate, ...)`, `bridge_speech_metadata(registry_row, source_library_root)`, and `activity_profile_for_class(sound_class)`.
- The ten-record table is selected by `listening_sample_records(manifest, count=10, seed=20260906)`; all human fields remain `pending_human`.
- The final prepared root is `tmp/p7_prepared_audio_v3`; no global sound registry was edited. The v1/v2 directories remain diagnostics superseded after bandwidth and PCM encoding corrections.

6. 任何与总前提冲突、需要 owner 拍板的地方，单独列出，不要自行决定。

- The default `vctk_only=True` reproduces the measured 600-row VCTK candidate scope. All 13 non-VCTK rows remain explicitly recorded; `--include-non-vctk` is a normal configuration option, not a new approval requirement. Unknown metadata is never invented or paired as known gender.
- No other owner decision was taken. Normalization is disabled for this derived speech-band set so the operation records filtering/cropping without adding an unrequested gain stage.


Parent integration review:

- Corrected a substantive first-version issue: the 300--3400 Hz bandpass now measures activity only. Delivered PCM receives the requested 80 Hz highpass and crop; high-frequency consonant content is retained. `filter_parameters` records output filtering; `activity_filter_parameters` records the detector band. Derived IDs change with this operation.
- Rebuilt all candidates without changing source PCM in fresh `tmp/p7_prepared_audio_v3` and directly read all 296 final WAVs, checking mono/16 kHz/sample count, <=5 s, finite values, metadata and output-filter scope. See `tmp/p7_prepared_audio_v3/parent_readback_check.json`. No original or v1 data was overwritten.
- The new entry point requires explicit registry/source/output/nonverbal paths; no private-server input path is hard-coded as an execution default. Reproduce with `tools/assets/prepare_speech_audio.py --registry /data/avengine_external/assets/sound_event_library_v1_20260903/sound_asset_registry_v1.json --source-library-root /data/avengine_external/assets/sound_library_v1 --output-root <fresh-output> --nonverbal-csv /data/datasets/omniaudio/tse_data/single_label_output.csv --prepared-set-id <new-set-id>`.
- No human or model listening record has been signed. P7 must not be marked fully accepted until its required ten real human records are available. Other implementation tasks continue.

Final encoding correction: the integer-PCM writer would clip 11 highpassed waveforms by up to a small positive dBFS margin. P7 now rejects these unrepresentable outputs before writing, with the actual peak in each reason; it applies no normalization or extra gain. See retained diagnostic `tmp/p7_prepared_audio_v2/full_scale_encoding_defect.json`. The final pool is 296, while the 307 detector-candidate denominator is retained. This is an explicit encoding failure, not a human acceptance decision or a claim that the source category is inapplicable.

The parent asked owner who will complete the ten real listening records. The report remains machine-complete/human-evidence-pending; independent P8 implementation is proceeding.
