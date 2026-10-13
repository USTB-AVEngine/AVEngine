# v13 telephone class merge

Owner decision, 2026-10-13 approximately 19:4x Asia/Singapore:
“电话声和电话铃可以合并”. The relay was approximately 19:5x.
Both times retain the precision supplied by the owner; no exact minute is inferred.

Run tools/qa/merge_sao_topup_classes.py with explicit --input, --cci-input,
--output and --pull-json-path arguments. Inputs are the immutable v12 pair.
The tool writes the v13 26-class pair, class_merge_map_v13.json,
cci_pull_list_v13.txt and DELIVERY_v13.json under the external output root.
The pull list contains only the new CCI JSON; the tool does not transfer it.

Only records whose top-level sound_class is telephone are changed:
sound_class and effective_qa21_answer_class become telephone_bell_ringing;
merged_from_sound_class and admission_gate_class retain telephone.
Earlier phone_ring merges remain unchanged. Prompts, IDs, paths, normalization,
activity, classifier targets/scores, voice checks, licenses and split restrictions
keep their original bytes. No model or admission step is rerun.

The lossless JSON edit preserves every untouched record's UTF-8 bytes and every
other existing value in changed records. It also preserves historical top-level
metadata and appends scope_change_6 with the owner decision and measured counts.
The 2,124 records retain their order and unique IDs, the merged class contains
the previous two classes combined, and the result has 26 classes.
Different-original-source counting follows the v12 origin-key preference.

The separate production-integration inventory is read-only. Future integration
must use {"telephone": "telephone_bell_ringing",
"phone_ring": "telephone_bell_ringing"} for answer/catalog selection while keeping
historical admission targets and the original AudioSet ontology intact.
Do not apply that map indiscriminately to recorded model evidence or STARSS class
names. Production registries, the integration worktree and other trials are not
modified by this preparation.
