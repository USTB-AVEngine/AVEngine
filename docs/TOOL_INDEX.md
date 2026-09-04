# 工具能力索引

> 由 `tools/build_tool_index.py` 从每个工具自己的 docstring 生成，
> 改完工具重新运行即可刷新（`--check` 已入单测回归，索引过期会红）。

自 2026-08-25 起，目录本身就是能力分组（阶段目录 m1…m7 已移除），
本表按目录列出每个工具做什么。
当前共 348 个工具脚本。

## 资产生成与装配（`tools/assets/`）

*从图像/网格到可运行资产：生成、修复、绑骨、动作、验收、打包、变体*

| 工具 | 做什么 |
|---|---|
| `tools/assets/add_missing_uv0.py` | Add deterministic zero UV0 accessors when a GLB safely omits them |
| `tools/assets/append_loop_closure.py` | Append one 15 Hz return-to-start sample to each GLB action channel |
| `tools/assets/assemble_variant_package.py` | Assemble a generic M2 animal research package from real QA evidence |
| `tools/assets/audit_candidate.py` | Write bounded automatic QA reports for an M2 research candidate |
| `tools/assets/audit_variant_candidate.py` | Run body-plan-neutral automatic M2 QA using explicit variant anchors |
| `tools/assets/audit_world_contacts.py` | Fit M2 root cadence and emit hash-bound world-contact artifacts |
| `tools/assets/bake_actions.py` | Bake the strict M2 Idle/Walking action artifact and hash-bound report |
| `tools/assets/bake_local_tr_actions.py` | Bake deterministic research-only local-translation action poses from one GLB |
| `tools/assets/bake_uniform_skin_scale.py` | Bake one positive uniform skin-ancestor scale into GLB payload data |
| `tools/assets/blender_level_generated_animal_support_plane.py` | Level a generated quadruped from two independent visible-foot authorities |
| `tools/assets/blender_normalize_generated_animal_heading.py` | Rigidly align a generated animal rig to a reviewed cardinal heading |
| `tools/assets/blender_retarget_quaternius_to_generated_quadruped.py` | Transfer Quaternius Walk/Idle to a generated, skinned quadruped rig |
| `tools/assets/build_appearance_variant_inputs.py` | Bind one realized appearance request into package spec and source lineage |
| `tools/assets/build_canary_request.py` | Build one hash-bound formal M2 canary capture request |
| `tools/assets/build_cross_species_appearance_lineage.py` | Build one diagnostic-only cross-species appearance lineage |
| `tools/assets/build_joint_mapping.py` | Emit the exact Habitat joint mapping bound to a rebase report |
| `tools/assets/build_research_review_request.py` | Build an exact 75-state M2 request for research-only human review |
| `tools/assets/build_sound_event_pool.py` | Build an avengine_sound_event_pool_v1 catalog from event_manifest.json |
| `tools/assets/build_sound_harvest_map.py` | 生成事件类到 FSD50K 标签的对照表草案。 |
| `tools/assets/capture_animal_variant_review.py` | Build and run one single-view animal-variant Habitat review capture |
| `tools/assets/capture_canary.py` | Run one formal 75-state M2 canary capture in Habitat |
| `tools/assets/capture_installed_research_review.py` | Run the current installed-prefix M2 Blender-room research review |
| `tools/assets/capture_research_review.py` | Run the 75-state single-view M2 research-review capture in Habitat |
| `tools/assets/check_prompt_token_budget.py` | Fail closed when a candidate profile's effective prompt cannot fit the model window |
| `tools/assets/compile_animal_package.py` | Compile a pinned Rocketbox Beagle M2 research-candidate package |
| `tools/assets/compose_topdown_review.py` | Derive synchronized RGB + navmesh/descriptor top-down QA review media |
| `tools/assets/derive_variant_contacts.py` | Derive generic package anchors and actor-space four-paw contacts |
| `tools/assets/force_matte_materials.py` | Force a complete opaque matte-dielectric GLB material policy |
| `tools/assets/gate_retopology.py` | Reject a mesh preparation that damaged the animal, before it is rigged |
| `tools/assets/gate_rigged_asset.py` | Accept or reject a rigged animal from how its surface tears during the walk |
| `tools/assets/generate_canonical_2d.py` | Generate one canonical 2D animal candidate from a clay pose guide |
| `tools/assets/generated_animal_support_plane.py` | Pure NumPy support-plane authority for generated quadrupeds |
| `tools/assets/generated_animal_support_plane_contract.py` | Stdlib-only validation contract for generated-animal support planes |
| `tools/assets/generated_quadruped_semantics.py` | Bone-name-independent semantic decomposition for generated quadruped rigs |
| `tools/assets/harvest_fsd50k_clips.py` | Fill the sound-effect classes from the FSD50K copy already on this machine |
| `tools/assets/harvest_vctk_speech.py` | Pull English speech with transcripts from the VCTK copy on this machine |
| `tools/assets/measure_deformation_stretch.py` | Measure how much of the surface a pose stretches |
| `tools/assets/measure_mesh_topology.py` | Compare mesh structure after welding, without a glTF round trip in between |
| `tools/assets/measure_static_resting_pose.py` | Measure the resting or mounting pose of a published rigid static asset |
| `tools/assets/measure_static_upright_correction.py` | Measure how far a rigid reconstruction is from standing straight |
| `tools/assets/measure_walk_deformation.py` | How badly a rigged animal's surface tears, over the whole walk cycle |
| `tools/assets/model_roots.py` | Where shared model weights live, resolved instead of hard-coded |
| `tools/assets/normalize_materials.py` | Normalize GLB PBR materials without modifying geometry or animation data |
| `tools/assets/plan_instance_variants.py` | Derive the instance-level variant plan for accepted source assets |
| `tools/assets/prepare_sound_library.py` | Convert the collected dry clips into the form the pipeline consumes |
| `tools/assets/preprocess_glb.py` | Select GLB actions and strip provably unweighted controller roots |
| `tools/assets/probe_habitat_skin_rest.py` | Generate and exercise a temporary Habitat skinned-AO rest-pose descriptor |
| `tools/assets/probe_heading_axis.py` |  |
| `tools/assets/promote_canary.py` | Promote one hash-closed M2 research candidate to a new canary package |
| `tools/assets/publish_animal_assets.py` | Publish accepted generated animals into the shared sound-source asset tree |
| `tools/assets/publish_static_source_assets.py` | Publish admitted static sound sources into the shared asset tree |
| `tools/assets/qc_sound_library.py` | Check every clip in the dry-sound library and say what is wrong, in words |
| `tools/assets/rebase_skin_root.py` | Create a Habitat-native, root-local GLB research candidate |
| `tools/assets/rebind_appearance_actions.py` | Reuse one validated M2 package action set on a compatible appearance rig |
| `tools/assets/rebind_appearance_visual.py` | Preserve a source M2 package actor frame on one appearance realization |
| `tools/assets/register_sound_event_assets.py` | Register split sound-event clips into avengine_m6_sound_asset_registry_v1 |
| `tools/assets/render_habitat_action_review.py` | Render hash-bound M2 Idle/Walk review media in Habitat |
| `tools/assets/render_habitat_local_tr_review.py` | Render a non-qualifying 75-frame Habitat local-TR v2 review |
| `tools/assets/render_turntable_review.py` | Orbit the camera once around a posed asset, with soft shadow-free lighting |
| `tools/assets/render_walk_review.py` | Render a walk cycle with soft, shadow-free lighting and the asset's own materials |
| `tools/assets/retime_actions.py` | Apply explicit action durations without changing sampled pose values |
| `tools/assets/retopologize_for_rigging.py` | Rebuild a generated mesh as a manifold surface, then reduce it for rigging |
| `tools/assets/rigger_loopback_bpy_server.py` | Loopback-only launcher for the vendored TokenRig bpy server |
| `tools/assets/run_generated_animal_chain.sh` | Take one generated animal from a raw reconstruction to a reviewable rigged |
| `tools/assets/run_pixal3d_mesh.py` | Run the local AVEngine Pixal3D image-to-3D stage on an RGBA candidate |
| `tools/assets/run_skintokens_rig.py` | Run local VAST-AI SkinTokens/TokenRig inference on one mesh |
| `tools/assets/segment_canonical_2d.py` | Cut the canonical 2D candidate out of its background with the local ISNet model |
| `tools/assets/skintokens_loopback_bpy_server.py` | Private Unix-socket Blender RPC endpoint for local SkinTokens inference |
| `tools/assets/spike_habitat_local_tr.py` | Prove one bounded local-translation-plus-rotation Habitat AO encoding |
| `tools/assets/split_sound_library_events.py` | Split a prepared sound library into one wav per sounding event |
| `tools/assets/wrap_uniform_scene_scale.py` | Wrap every root of one GLB scene in an explicit uniform-scale node |

## 房间（`tools/rooms/`）

*房间引入、制备、审计、资格金丝雀（Habitat 与 SPEAR/UE 两条腿都在这里）*

| 工具 | 做什么 |
|---|---|
| `tools/rooms/audit_real_surface_mesh.py` | Audit a glTF/GLB as evidence for the M1 real-surface geometry gate |
| `tools/rooms/audit_skokloster_glb.py` | Audit the exact Habitat Skokloster GLB for visual and acoustic staging |
| `tools/rooms/author_current_residential_visual_episode.py` | Author one current residential visual-only research episode |
| `tools/rooms/blender_build_exterior_proxy.py` | Blender-side builder for an inward-facing, textured exterior sphere |
| `tools/rooms/build_fixed_apartment_canary.py` | Build the fixed SPEAR Apartment S0--S5 M6.x review bundle |
| `tools/rooms/build_residential_source_episode.py` | Build AVEngine Timeline, Topdown and binaural audio for a residential room |
| `tools/rooms/emit_hm3d_room_manifest.py` | Emit an AVEngine room manifest for an HM3D scene directory |
| `tools/rooms/extract_interioragent_scene_metadata.py` | Extract a room polygon and navigation footprints from InteriorAgent USD |
| `tools/rooms/prepare_3d_front_toolbox_sample_proxy.py` | Build a clearly labelled 3D-FRONT Toolbox sample review proxy in USD |
| `tools/rooms/prepare_interioragent_kujiale_adapter.py` | Prepare an external InteriorAgent USD stage for UE's runtime USD importer |
| `tools/rooms/prepare_legacy_apartment.py` | Prepare the real-surface UE apartment export as an M1 Habitat room package |
| `tools/rooms/prepare_skokloster_interchange_glb.py` | Bake Skokloster's legacy source axes into a canonical glTF for UE import |
| `tools/rooms/prepare_spear_apartment_exterior.py` | Export UE's approaching_storm HDRI and build a visual-only Habitat GLB |
| `tools/rooms/rebuild_replicacad_obstacle_review.py` | Rebuild the retained ReplicaCAD review with live furniture obstacles |
| `tools/rooms/run_habitat_replicacad_lighting_canary.py` | Run the real ReplicaCAD Habitat capture with one shared lighting profile |
| `tools/rooms/run_room_qualification_attempt.py` | Run or verify the read-only M6 representative-room qualification attempt |
| `tools/rooms/run_spear_apartment_canary.py` | Render M6.x S0/S3/S4 through the native SPEAR Apartment map |
| `tools/rooms/run_spear_kujiale_canary.py` | Capture an external InteriorAgent/Kujiale room through SPEAR and UE |
| `tools/rooms/run_spear_mp3d_canary.py` | Render the retained 270-frame MP3D route through packaged SPEAR |
| `tools/rooms/run_spear_replicacad_canary.py` | Render the retained 270-frame ReplicaCAD route in an isolated SPEAR editor |
| `tools/rooms/run_spear_residential_episode.py` | Render one AVEngine residential human+Beagle episode through SPEAR/UE |

## 场景放置（`tools/scene/`）

*基于真实场景表面规划并核验实体放置*

| 工具 | 做什么 |
|---|---|
| `tools/scene/choose_listener_pose.py` | Choose listener poses for a route, so nothing downstream has to invent one |
| `tools/scene/plan_supported_placement.py` | Place speakers on surfaces that exist, and check the sightline against geometry |

## 声学（`tools/acoustics/`）

*声学场景包、材质、RIR 缓存与计划、声学核验*

| 工具 | 做什么 |
|---|---|
| `tools/acoustics/audit_jaeger_rir.py` | JAEGER SpatialSceneQA 公开包 RIR 混响审计脚本(可重跑版)。 |
| `tools/acoustics/audit_skokloster_cleanup_inventory.py` | Emit the exact face inventory for a Skokloster research cleanup |
| `tools/acoustics/build_asset_bound_rir_plan.py` | Bind concrete assets to generic root routes and plan on-demand RIR work |
| `tools/acoustics/compile_semantic_research_package.py` | Compile one room's semantic mesh into an M3/RLR research acoustic package |
| `tools/acoustics/derive_research_rlr_package.py` | Derive an RLR-loadable research package by removing QA-degenerate faces |
| `tools/acoustics/derive_skokloster_two_face_research_package.py` | Derive the bounded Skokloster package by removing exactly two QA faces |
| `tools/acoustics/extract_usd_acoustic_snapshot.py` | Expand a static USD room into one auditable M3 acoustic snapshot |
| `tools/acoustics/probe_room_front_back_pairs.py` | Measure whether one room supports front/back mirrored source pairs |
| `tools/acoustics/render_rir_cache.py` | Render a resumable native-RLR RIR cache from an M6.x job plan |
| `tools/acoustics/run_material_canary.py` | Run the hash-bound repeated M3 RLR material activation canary |
| `tools/acoustics/verify_material_canary.py` | Verify M3 canary schema, lineage, raw IRs and recomputed gates |
| `tools/acoustics/verify_package_frame_parity.py` | Cross-system frame parity: the same rays in Habitat and in the package |

## 相机与路径（`tools/routes/`）

*相机机位、可行域、路径库、轨迹选择、发声锚点*

| 工具 | 做什么 |
|---|---|
| `tools/routes/build_apartment_route_bank.py` | Precompute an apartment route bank from UE's own navigation system |
| `tools/routes/build_camera_pose_request.py` | Build one camera/listener-coherent M1 request at an arbitrary room pose |
| `tools/routes/build_four_motion_anchor_profile.py` | Build and benchmark the reusable emitter-anchor profile from a pilot |
| `tools/routes/build_hm3d_rir_plan.py` | Turn an HM3D floor bank and an accepted listener pose into a QA plan-dir |
| `tools/routes/compile_apartment_feasibility_bank.py` | Compile Apartment feasibility, a four-case trajectory bank, and Topdown QA |
| `tools/routes/compile_hm3d_dynamic_source_bank.py` | Ask whether a moving sound source can find legal routes in an HM3D scene |
| `tools/routes/compile_kujiale_feasibility_bank.py` | Compile a Kujiale room polygon into reusable source-center trajectories |
| `tools/routes/compile_mp3d_region_plan.py` | Build a bounded CPU MP3D .house region/camera/source-route plan |
| `tools/routes/filter_route_bank_by_ground.py` | Select one UE ground-height route domain from a multi-level route bank |
| `tools/routes/import_legacy_apartment_route.py` | Import the legacy 18 s apartment route into the M5.1 route manifest |
| `tools/routes/materialize_mp3d_actor_tracks.py` | Build CPU Habitat apply tracks for one planned MP3D region case |
| `tools/routes/materialize_mp3d_region_case.py` | Materialize one MP3D region route case into current AVEngine inputs |
| `tools/routes/plot_route_bank.py` | Render the apartment route bank as a top-down map: engine navigation vs the hand-mined corridors |
| `tools/routes/probe_camera_pose_native.py` | Render one lightweight native Habitat camera-pose probe |
| `tools/routes/select_asset_bound_trajectories.py` | Select asset-bound source-slot routes that pass the real center-point gate |
| `tools/routes/verify_route_legality.py` | Is the path itself legal? Occlusion is not the question here |

## episode 捕获（`tools/capture/`）

*演员级 episode 的视觉捕获与动作试点*

| 工具 | 做什么 |
|---|---|
| `tools/capture/capture_human_beagle_legacy.py` | Capture the committed 270-frame Rocketbox-human + Beagle legacy route |
| `tools/capture/capture_human_beagle_mp3d.py` | Run the real-navmesh 270-frame MP3D human + Beagle visual canary |
| `tools/capture/capture_human_beagle_replicacad.py` | Run the real ReplicaCAD apt_0 human + Beagle visual/placement review |
| `tools/capture/capture_mp3d_multi_actor.py` | Capture one explicit N-actor MP3D case through the installed Habitat runtime |
| `tools/capture/capture_two_human_mp3d.py` | Run the Habitat-native MP3D two-human production visual capture |
| `tools/capture/run_apartment_four_motion_pilot.py` | Run one shared Apartment capture for the four human/dog motion cases |

## 视觉回放（`tools/visual/`）

*在 AVEngine 原生视觉路径中回放、渲染与核验放置结果*

| 工具 | 做什么 |
|---|---|
| `tools/visual/render_moving_source_video.py` | Render the route the acoustic pass rendered, as frames that can carry that audio |
| `tools/visual/replay_placement_in_avengine.py` | Render the same speaker placement the acoustic chain used, in AVEngine's own runtime |

## 空间音频（`tools/audio/`）

*双耳/FOA 渲染与混音（src/avengine/spatial_audio，工具暂无）*

| 工具 | 做什么 |
|---|---|
| `tools/audio/build_hm3d_material_map.py` | Extend the acoustic material database to HM3D's category vocabulary |
| `tools/audio/calibrate_surface_materials.py` | Ask which material the room's dominant surfaces should point at |
| `tools/audio/compare_material_acoustics.py` | Does turning HM3D semantics into acoustic materials change the sound? |
| `tools/audio/fit_foa_axes.py` | Fit the map from SoundSpaces FOA channels to Habitat world axes |
| `tools/audio/hm3d_download.sh` | Download the HM3D val split into the shared dataset root. |
| `tools/audio/hm3d_download_pieces.sh` | Fetch HM3D pieces for one split. |
| `tools/audio/hm3d_semantic_download.sh` | Fetch the HM3D semantic annotations for the val split. |
| `tools/audio/insert_speakers_and_render_foa.py` | Put our published speakers into a Habitat scene and render RGB plus FOA |
| `tools/audio/make_source_orbit_bank.py` | Write a bank whose source circles the listener, for hearing the surround field |
| `tools/audio/measure_semantic_surface_area.py` | Rank HM3D semantic categories by surface area, which is what acoustics sees |
| `tools/audio/plan_from_navmesh.py` | Build a speaker placement from the scene's navmesh |
| `tools/audio/render_moving_source.py` | Render a moving sound source in one of the renderer's two output layouts |

## 出题与认证（`tools/qa/`）

*题型设计、出题、闸门核验、held-out 划分、评测与打分*

| 工具 | 做什么 |
|---|---|
| `tools/qa/adapt_strict_two_human_both_move_v1_preflight.py` | Bind the reviewed both-move geometry handoff to A's materializer contract |
| `tools/qa/assemble_qa_v3_room_pilot.py` | Assemble one quota-complete room-centric QA-v3 research pilot manifest |
| `tools/qa/audio_profiles.py` | Question-type audio profiles: one schedule per question type, not one for all |
| `tools/qa/audit_gatea_by_form.py` |  |
| `tools/qa/audit_gatea_semantics.py` |  |
| `tools/qa/audit_qa_v3_prescale_candidates.py` | Revalidate an existing room-pilot manifest against prescale QA-v3 rules |
| `tools/qa/audit_strict_two_human_camera_pan_motion_realism.py` | Audit camera-pan motion realism against authoritative 15 Hz camera poses |
| `tools/qa/audit_strict_two_human_room_expansion.py` | Audit two additional cooked SPEAR maps for the strict M/F/C room closure |
| `tools/qa/bind_native_paper_balance_episode.py` | Bind one full native SPEAR capture to the paper-balance QuestionSpec strata |
| `tools/qa/bind_native_pixel_fact_episode.py` | Bind one full native SPEAR pixel capture to Facts and QuestionSpecs |
| `tools/qa/bind_native_spear_episode.py` | Bind one compiled QA Fact table to retained native SPEAR/UE evidence |
| `tools/qa/build_batch_review_page.py` | Build a self-contained lazy-loading review page for one QA v2 batch |
| `tools/qa/build_full_episode_semantic_authority.py` | Build an approved full-Episode semantic authority without overwriting output |
| `tools/qa/build_full_episode_validation_batch.py` | Build one validation batch from an explicit request |
| `tools/qa/build_mcq_options.py` | MCQ option builder for the dual-source five-card pilot (work order 1.4) |
| `tools/qa/build_native_controlled_audio_program.py` | Build controlled dog/speech AudioProgram contracts for the A native canary |
| `tools/qa/build_native_full_occlusion_reappearance_episode.py` | Prepare one native, dynamic-rig full-occlusion/reappearance episode |
| `tools/qa/build_native_paper_balance_episode.py` | Plan one fail-closed native Episode that closes paper answer balance |
| `tools/qa/build_pixel_visibility_canaries.py` | Build five hermetic modal/target-only pixel-visibility canaries |
| `tools/qa/build_qa_v3_camera_clearance_table.py` | Per-scene camera clearance table: one actor-free depth cube ring per camera point |
| `tools/qa/build_qa_v3_human_calibration_pack.py` | Build a browser-ready full-AV human calibration pack from run02 media |
| `tools/qa/build_qa_v3_listening_copy.py` | Raise a calibration pack to a listenable level without touching the render |
| `tools/qa/build_qa_v3_n_actor_canary.py` | Build one scene-neutral four-actor/four-endpoint QA-v3 research canary |
| `tools/qa/build_qa_v3_programs.py` | Per-point audio-program generator for the v3 pilot (work order item 1.2) |
| `tools/qa/build_qa_v3_released_probe_items.py` | Build MCQ/Open shortcut-probe items from released run02-style media |
| `tools/qa/build_qa_v3_walkable_grid.py` | Build a scene's walkable-floor grid (see walkable_grid.py for what it is for) |
| `tools/qa/build_skokloster_strict_two_human_preflight.py` | Build a file-evidence-free CPU preflight for the Skokloster strict M/F Episode |
| `tools/qa/build_strict_two_human_camera_pan_v2_candidate.py` | Build one CPU-only camera-pan/both-static full75 geometry candidate |
| `tools/qa/build_strict_two_human_canary_preflight.py` | Validate and publish the CPU preflight for one strict two-human canary |
| `tools/qa/build_strict_two_human_canary_recipe.py` | Build one exact static two-human Apartment recipe and AudioProgram |
| `tools/qa/build_strict_two_human_debug_room_preflight.py` | Prepare fail-closed visual probes for two additional cooked SPEAR maps |
| `tools/qa/build_strict_two_human_distractor_moves_v2_candidate.py` | Build the CPU-only distractor-moves v2 geometry candidate |
| `tools/qa/build_strict_two_human_dynamic_canary_preflight.py` | Select four independent true-motion full75 canaries without launching a GPU |
| `tools/qa/build_strict_two_human_expansion_acoustic_batch.py` | Prepare and validate rows 2-8 of the strict two-human CPU acoustic batch |
| `tools/qa/build_strict_two_human_expansion_preflight.py` | Validate and publish the CPU-only strict two-human eight-row plan |
| `tools/qa/build_strict_two_human_full_episode_batch.py` | Build a fail-closed CPU plan for 100 independent strict two-human Episodes |
| `tools/qa/build_strict_two_human_ground_contact_diagnostic.py` | Build a fail-closed f0/f37/f74 live foot-floor diagnostic request |
| `tools/qa/build_strict_two_human_motion_realism_receipt.py` | Build a CPU-only, fail-closed motion-realism release receipt |
| `tools/qa/build_strict_two_human_mp3d_room_preflight.py` | Build a CPU-only MP3D strict-two-human room/capture/RIR preflight |
| `tools/qa/build_strict_two_human_native_rate_dynamic_candidates.py` | Build CPU-only native-rate full75 dynamic candidate preflights |
| `tools/qa/build_strict_two_human_row7_v2_acoustic.py` | Prepare and finalize the CPU-only strict two-human row7 v2 acoustics |
| `tools/qa/build_strict_two_human_row7_v2_preflight.py` | Validate and publish the CPU-only strict two-human row7 v2 overlay |
| `tools/qa/camera_clearance.py` | Camera clearance table: cube-ring depth geometry and the solver-side reader |
| `tools/qa/capture_qa_v3_timeline_pixel.py` | Capture native pixel visibility for one current QA-v3 visual timeline |
| `tools/qa/capture_skokloster_strict_two_human_episode.py` | Bind the native pixel runner to the isolated Skokloster package archive |
| `tools/qa/capture_spear_imported_glb_strict_two_human_episode.py` | Capture an imported-MP3D strict two-human SPEAR review Episode |
| `tools/qa/capture_spear_native_pixel_episode.py` | Capture a full native SPEAR RGB/depth/pixel-truth Episode |
| `tools/qa/capture_spear_native_pixel_room_batch.py` | Concrete one-process SPEAR adapter for strict full75 room batches |
| `tools/qa/certify_axis1_questions.py` | Fact-level axis-1 (route-swap) certification for mined simple questions |
| `tools/qa/compare_question_spec_fresh.py` | Fresh QuestionSpec re-evaluation versus retained bind-time records |
| `tools/qa/compile_apartment_fact_tables.py` | Compile per-episode QA fact tables for the asset-bound Apartment batch |
| `tools/qa/compile_question_protocol_coverage.py` | Compile or independently validate the 12-type native QuestionSpec protocol |
| `tools/qa/derive_native_occluder_evidence.py` | Derive static occluder identity from native modal/target-only pixels |
| `tools/qa/derive_twin_programs.py` | Derive sealed audio programs for Gate B twin points (qa-v3 pilot) |
| `tools/qa/design_qa_batch.py` | Design and author one constraint-driven QA v2 batch (reverse fitting) |
| `tools/qa/design_qa_v3_extended_profile.py` | Generate the QA-v3 profiles that require N actors, pixel truth, or segments |
| `tools/qa/design_qa_v3_pilot_batch.py` | Design one qa-v3 dual-source pilot batch (stage two assembler) |
| `tools/qa/design_qa_v3_scene_batch.py` | Integrated qa-v3 batch: generic scene solver + per-type audio + facts |
| `tools/qa/evaluate_question_specs.py` | Evaluate registry-bound QuestionSpecs and render a standalone review page |
| `tools/qa/filter_cross_time_points.py` | Cross-time sampling filter (pilot work order item 1.7) |
| `tools/qa/finalize_batch_visuals.py` | Retire raw rgb.npy arrays for a finished QA v2 batch (owner policy |
| `tools/qa/finalize_native_full_occlusion_reappearance_episode.py` | Compile Facts and finalize the native full-occlusion suite after audio render |
| `tools/qa/finalize_native_paper_balance_episode.py` | Finalize one paper-balance recipe after native RLR binaural rendering |
| `tools/qa/finalize_native_pixel_artifacts.py` | Finalize and verify a native SPEAR pixel-capture artifact inventory |
| `tools/qa/finalize_qa_v3_gateb_precheck.py` | Finalize representative Gate-B audio and native-pixel precert evidence |
| `tools/qa/finalize_qa_v3_room_pilot.py` | Validate and finalize representative runtime evidence for a room pilot |
| `tools/qa/finalize_strict_two_human_canary.py` | Finalize and fail-closed validate the strict two-human sparse canary |
| `tools/qa/finalize_strict_two_human_dynamic_full75_canary.py` | Fail-closed finalizer for strict two-human dynamic full75 canaries |
| `tools/qa/finalize_strict_two_human_full75_canary.py` | Fail-closed finalizer for one strict two-human 75-frame native canary |
| `tools/qa/finalize_strict_two_human_raw_episode.py` | CPU-only finalizer for one atomically published strict full75 raw spool |
| `tools/qa/floor_reference.py` | Per-room floor reference: the measured UE z of the walkable floor |
| `tools/qa/generate_qa_v2_questions.py` | QA v2 question generation for a constraint-driven batch |
| `tools/qa/generate_qa_v3_questions.py` | Generate qa-v3 pilot fact records + question candidates (cards ①⑦⑧⑨) |
| `tools/qa/join_qa_v3_extended_pixel.py` | Join native pixel truth to pixel-dependent QA-v3 candidates |
| `tools/qa/make_idle_then_walk_timeline.py` | Idle-then-walk timeline transform (pilot work order items 1.2/1.7 支撑件) |
| `tools/qa/materialize_qa_v3_dual_gateb.py` | Materialize renderable Gate-B twins for selected dual-source QA-v3 points |
| `tools/qa/materialize_strict_two_human_dynamic_canary.py` | Materialize one true-motion strict two-human full75 CPU closure |
| `tools/qa/measure_qa_v3_floor_z.py` | Measure a room's floor height in the engine and write its floor reference |
| `tools/qa/mine_simple_questions.py` | Mine simple (A-group) questions from compiled QA fact tables |
| `tools/qa/mine_temporal_questions.py` | Mine temporal (B-group) and numeric questions over intermittent fact tables |
| `tools/qa/pre_gpu_launch_ledger.py` | Fail-closed archival for prepared attempts that never reached a GPU launch |
| `tools/qa/preflight_camera_clearance_depth.py` | Camera-only depth preflight: is the view from a candidate camera pose clear? |
| `tools/qa/prepare_qa_v3_mcq.py` | Glue: facts (generator) -> split plan (1.3) -> MCQ items (1.4 input) |
| `tools/qa/probe_packaged_imported_glb_room.py` | Fail-closed NullRHI packaged readback for an imported-GLB room adapter |
| `tools/qa/probe_packaged_skokloster_room.py` | Fail-closed NullRHI packaged-object readback for Skokloster Castle |
| `tools/qa/probe_physical_features.py` | Physical-feature classifier probe (pilot work order item 1.6) |
| `tools/qa/probe_qa_v3_runtime_los_batch.py` | Probe QA-v3 candidate sightlines in one real packaged UE map |
| `tools/qa/probe_released_modality_shortcuts.py` | Probe text-, audio-, or video-only shortcuts from final released media |
| `tools/qa/probe_ue_capture_animation.py` | Probe a static-camera capture for scheduled-but-unrendered walk animation |
| `tools/qa/publish_strict_two_human_full75_canary_summary.py` | Publish a reviewable four-row summary for strict two-human full75 canaries |
| `tools/qa/publish_strict_two_human_review.py` | Publish a lightweight, server-linked review for the strict two-human gates |
| `tools/qa/qa_v3_actor_selection.py` | Resolve selected articulated or rigid source assets to their UE content |
| `tools/qa/qa_v3_arc.py` | 圆上的弧:起点加带符号扫角。有序的 [lo, hi] 表示不了它。 |
| `tools/qa/qa_v3_asset_policy.py` | Explicit per-request asset-pair policy for QA-v3 scene design |
| `tools/qa/qa_v3_azimuth.py` | The one place a camera-frame azimuth becomes a published azimuth |
| `tools/qa/qa_v3_pixel_thresholds.py` | Explicit placeholder pixel-answerability thresholds for QA-v3 base cards |
| `tools/qa/qa_v3_request.py` | Plan QA-v3 request budgets without hiding per-profile shortages |
| `tools/qa/recompile_native_pixel_truth.py` | Recompile retained metric-depth truth with the current lossless fields |
| `tools/qa/recompute_qa_v3_gateb_gold.py` | Recompute Gate-B gold for every selected QA-v3 pilot candidate |
| `tools/qa/refresh_strict_two_human_row8_ready.py` | Refresh the row8 sparse request against the current split visibility contract |
| `tools/qa/render_axis1_twin_audio.py` | Render route-swap twin binaural audio for axis-1 certified episodes |
| `tools/qa/render_intermittent_batch.py` | Render intermittent-window binaural mixtures for a declared episode subset |
| `tools/qa/report_qa_v3_card1_conditional_baseline.py` | Card1 realized conditional tables and best-response unimodal baselines |
| `tools/qa/route_synthesis.py` | Routes designed by the solver for the pose it has chosen |
| `tools/qa/run_qa_v3_audio_batch.py` | Sequential dynamic-audio runner for a qa-v3 design batch (stage two) |
| `tools/qa/run_qa_v3_capture_batch.py` | Sequential UE capture runner for a qa-v3 design batch (stage two) |
| `tools/qa/run_qa_v3_room_profile_scheduler.py` | Room-centric QA-v3 scene x profile scheduler |
| `tools/qa/run_scene_generalization_smoke.py` | Design-layer cross-scene smoke: one question-type config, several route domains |
| `tools/qa/run_strict_two_human_dynamic_full75_canary.py` | Launch one CPU-qualified dynamic full75 canary on physical GPU1 only |
| `tools/qa/run_strict_two_human_full75_canary.py` | Run one planned full75 canary only after the physical-GPU1 idle gate |
| `tools/qa/run_strict_two_human_full75_room_batch.py` | Fail-closed controller for one same-room strict full75 batch |
| `tools/qa/run_strict_two_human_ground_contact_diagnostic.py` | Prepare and launch one fail-closed f0/f37/f74 ground-contact diagnostic |
| `tools/qa/run_strict_two_human_mp3d_f15_probe.py` | Prepare and launch one MP3D strict-two-human diagnostic f15 probe |
| `tools/qa/run_strict_two_human_mp3d_f15_probe_v3.py` | Freeze the MP3D v2 failure and prepare the independent v3 f15 candidate |
| `tools/qa/run_strict_two_human_mp3d_f15_probe_v4.py` | Freeze the MP3D v3 failure and prepare the independent v4 f15 candidate |
| `tools/qa/run_strict_two_human_skokloster_f15_probe.py` | Prepare and run one fail-closed Skokloster strict-two-human f15 probe |
| `tools/qa/run_strict_two_human_skokloster_f15_probe_v2.py` | Freeze the Skokloster v1 environment failure and prepare f15 revision v2 |
| `tools/qa/scan_capture_listener_yaw.py` | Batch scan: does every capture's camera yaw match its audio listener? |
| `tools/qa/scene_sampler.py` | Scene-agnostic candidate search for qa-v3 question types |
| `tools/qa/score_open_answers.py` | Open-form answer scorer (pilot work order item 1.5) |
| `tools/qa/score_qa_v3_human_calibration.py` | Score QA-v3 human calibration responses without mixing binding errors |
| `tools/qa/search_mp3d_strict_two_human_nav_positions.py` | Search the real MP3D navmesh for a safer two-adult static probe pair |
| `tools/qa/search_skokloster_strict_listener.py` | Search one coupled Skokloster camera/listener for a strict two-adult probe |
| `tools/qa/select_qa_v3_card16_pixel_quota.py` | Select card16 candidates after native-pixel truth, stratified by gold state |
| `tools/qa/select_qa_v3_run02_dev.py` | Reproduce the QA-v3 run02-dev pixel-qualified 6-per-profile selection |
| `tools/qa/spear_imported_glb_room_adapter.py` | Runtime adapter for reload-verified GLB scenes imported into cooked SPEAR |
| `tools/qa/spear_room_batch_lifecycle.py` | SPEAR same-room Episode lifecycle gates staged for a two-Episode canary |
| `tools/qa/spear_skokloster_room_adapter.py` | Generic imported-GLB runtime contract specialized to Skokloster's one mesh |
| `tools/qa/spike_spear_native_pixel_visibility.py` | Capture one real SPEAR RGB/depth/modal/target-only visibility spike |
| `tools/qa/split_isolator.py` | Split isolator for pilot batches (work order item 1.3) |
| `tools/qa/strict_two_human_cpu_finalize_queue.py` | Low-priority one-worker CPU finalization queue for strict room batches |
| `tools/qa/strict_two_human_raw_spool.py` | Exact, crash-safe raw spool used by the shared-room full75 capture adapter |
| `tools/qa/transcribe_audio_review.py` | Transcribe declared review audio with an installed Whisper model |
| `tools/qa/upgrade_static_spear_suite_camera.py` | Bind legacy static-camera SPEAR plans to audited QA capture requests |
| `tools/qa/validate_stereo_channels.py` | Stereo-channel integrity validator (pilot work order item 1.1) |
| `tools/qa/validate_strict_two_human_camera_pan_motion_realism.py` | Validate or deterministically replay the camera-pan motion audit receipt |
| `tools/qa/validate_strict_two_human_motion_realism_receipt.py` | Validate a strict two-human CPU motion-realism receipt, optionally by replay |
| `tools/qa/validate_strict_two_human_native_rate_dynamic_candidates.py` | Validate a fail-closed native-rate full75 dynamic candidate pair |
| `tools/qa/validate_strict_two_human_publication_plan.py` | Fail-closed validation for the strict two-human publication plan |
| `tools/qa/validate_visibility_prediction.py` | Positive control for the visibility predictor: table prediction vs pixel truth |
| `tools/qa/verify_qa_v3_audio_batch.py` | Batch-level verification of qa-v3 pilot audio renders (post-render gate) |
| `tools/qa/verify_qa_v3_visual_batch.py` | Verify a materialized QA-v3 visual batch against its runtime readbacks |
| `tools/qa/visibility_prediction.py` | Predict actor visibility from the camera clearance table, before rendering |
| `tools/qa/walkable_grid.py` | Per-scene walkable-floor grid: where an actor may stand, and how far the |

## 数据集装配（`tools/dataset/`）

*训练/评测数据的规模化装配、重组、吞吐批与验证*

| 工具 | 做什么 |
|---|---|
| `tools/dataset/build_asset_bound_apartment_ue_bundle.py` | Materialize M7 source1/source2 routes as one reusable Apartment UE bundle |
| `tools/dataset/build_asset_bound_dataset_index.py` | Index 1,000 samples without copying visual, audio, or room media |
| `tools/dataset/build_asset_bound_visual_reviews.py` | Build Habitat-only internal visual QA reviews for the M7 throughput batch |
| `tools/dataset/build_cached_apartment_dataset_examples.py` | Build four cache-bound Apartment dataset examples for UE review |
| `tools/dataset/build_mp3d_room_evaluation_review.py` | Build one hash-bound MP3D room-evaluation listening review |
| `tools/dataset/build_room_evaluation_plan.py` | Select balanced generic source trajectories for one room evaluation |
| `tools/dataset/build_spear_apartment_review.py` | Bind one exact SPEAR Apartment RGB render to Habitat Topdown v3 and audio |
| `tools/dataset/compare_rir_cache_metrics.py` | Compare EDT/DRR/late-energy between two retained RIR caches on matched jobs |
| `tools/dataset/merge_spear_apartment_render_shards.py` | Merge independently rendered SPEAR Apartment shards without copying media |
| `tools/dataset/recombine_source_trajectory_bank.py` | Build many unique two-source episodes from one finite single-path pool |
| `tools/dataset/render_asset_bound_binaural_batch.py` | Assemble many binaural training items from one completed asset-bound cache |
| `tools/dataset/render_asset_bound_binaural_canary.py` | Render two real dry recordings through one completed asset-bound RIR cache |
| `tools/dataset/render_current_apartment_dynamic_audio.py` | Render motion-following binaural audio for a current UE research capture |
| `tools/dataset/render_room_evaluation_binaural.py` | Mix generic room-evaluation sound classes through a completed RIR cache |
| `tools/dataset/run_habitat_room_batch.py` | Batch Habitat-native RGB rendering for registry-selected rooms |
| `tools/dataset/verify_asset_bound_batch.py` | Verify the complete M7 asset-bound binaural throughput batch |

## 审阅（`tools/review/`）

*审阅页、交付片段、对比与评审证据*

| 工具 | 做什么 |
|---|---|
| `tools/review/build_current_mp3d_dynamic_review_clip.py` | Build the current MP3D dynamic-audio review clip from engine artifacts |
| `tools/review/build_legacy_delivery.py` | Build the final annotated 18-second M5.1 legacy comparison delivery |
| `tools/review/build_mp3d_delivery.py` | Build an 18-second Habitat-native human/Beagle annotated binaural review |
| `tools/review/build_review_index.py` | Build a small local review page for optional SPEAR/UE room renders |
| `tools/review/build_six_case_review.py` | Validate, plan, or build the immutable M6 six-case human-review package |
| `tools/review/machine_audition_hm3d_episode.py` | Machine-audit one rendered HM3D episode and write the verdict beside it |
| `tools/review/render_review_acoustics.py` | Render and retain variable-duration M5.1 binaural RIR evidence |

## 注册表与发布（`tools/registry/`）

*注册表发布与核验*

| 工具 | 做什么 |
|---|---|
| `tools/registry/publish_static_object_registry.py` | Publish or verify one fail-closed M6 static-object research registration |
| `tools/registry/reseal_examples.py` | Reseal and re-pin example evidence bindings after a legitimate content change |

## 发布（`tools/release/`）

*发布包组装*

| 工具 | 做什么 |
|---|---|
| `tools/release/build_manifest.py` | Prepare or verify the two-commit AVEngine cross-repository release manifest |

## Studio（`tools/studio/`）

*审阅与任务网页台*

| 工具 | 做什么 |
|---|---|
| `tools/studio/build_studio_scene_bundle.py` | Build a Studio scene bundle: preview mesh plus draft obstacle snapshot |
| `tools/studio/feed_hm3d_fleet.py` | Keep the studio queue fed with the next un-attempted HM3D houses |
| `tools/studio/make_paired_ablation.py` | Paired ablation variants for a rendered dynamic-audio bundle |
| `tools/studio/run_apartment_end_to_end.py` | Studio end-to-end Apartment chain: author timeline → UE capture → audio → clip |
| `tools/studio/run_hm3d_end_to_end.py` | One HM3D house, start to finish, in a single task |
| `tools/studio/run_hm3d_episode.py` | Render one HM3D moving-source episode: pose, FOA, first-person video, binaural |
| `tools/studio/run_kujiale_acoustic_package.py` | Compile a Kujiale USD room into an RLR-loadable research acoustic package |
| `tools/studio/run_mp3d_end_to_end.py` | Studio end-to-end MP3D chain: author route → capture → dynamic audio → clip |
| `tools/studio/run_studio_server.py` | Launch the AVEngine Studio backend server (loopback only) |
| `tools/studio/ue_export_apartment_gltf.py` | Headless UE editor export: apartment_0000 level -> textured glb |

## UE 工程（`tools/ue/`）

*UE 编辑器内脚本：导入、修复、建图、导出*

| 工具 | 做什么 |
|---|---|
| `tools/ue/assemble_package_stage.py` | Assemble a fresh UE package stage for the current Apartment visual route |
| `tools/ue/build_minimal_closure_report.py` | Build a minimal-closure report for the current Apartment visual stage |
| `tools/ue/create_spear_kujiale_map_editor.py` | Create a UE map containing one external USD stage |
| `tools/ue/export_apartment_gltf.py` | Export the legacy SPEAR apartment as real UE render-surface geometry |
| `tools/ue/export_asset_dependencies_editor.py` | Export real Unreal package dependencies for declared mounted content roots |
| `tools/ue/fix_spear_mp3d_materials_editor.py` | Repair and verify MP3D glTF color semantics inside an isolated UE project |
| `tools/ue/import_controlled_humans_editor.py` | Import one catalog-described controlled human with generic Unreal APIs |
| `tools/ue/import_spear_3d_front_sample_editor.py` | Import the local 3D-FRONT Toolbox sample proxy into a persistent UE map |
| `tools/ue/import_spear_replicacad_editor.py` | Import and assemble the prepared ReplicaCAD apt_0 scene inside UE 5.5 |
| `tools/ue/import_spear_skokloster_editor.py` | Import the prepared Skokloster GLB into one isolated SPEAR/UE content root |
| `tools/ue/import_static_source_editor.py` | Import declared rigid source GLBs into fresh Unreal content directories |
| `tools/ue/import_usd_stage_to_level_editor.py` | Import an external USD stage into ordinary Unreal assets and a saved level |
| `tools/ue/prepare_spear_mp3d_execution.py` | Bind a compiled Timeline-v2 visual plan to the imported MP3D UE scene |
| `tools/ue/prepare_spear_replicacad_scene.py` | Prepare the complete ReplicaCAD scene request for the optional UE backend |
| `tools/ue/probe_spear_replicacad_environment.py` | Probe whether ReplicaCAD can be imported/cooked without touching old SPEAR |
| `tools/ue/unreal_export_approaching_storm.py` | Unreal Editor-side exporter for the stock approaching_storm TextureCube |

## Blender 工程（`tools/blender/`）

*Blender 内运行的资产处理脚本*

| 工具 | 做什么 |
|---|---|
| `tools/blender/build_custom_room.py` | Build the controlled two-zone Blender room used by the M1 canary |
| `tools/blender/normalize_asset_skinned_glb.py` | Bake an armature object transform into a Habitat-oriented GLB candidate |
| `tools/blender/realize_animal_appearance.py` | Realize one hash-bound animal appearance request without changing its rig |
| `tools/blender/render_instance_diversity_check.py` | Render several finished assets in one frame at room distance |
| `tools/blender/verify_asset_rebased_glb.py` | Verify sampled deformation equivalence across the M2 GLB root rebase |
| `tools/blender/verify_motion_projection.py` | Measure sampled deformation drift from a source to a rotation-only GLB |

## 动作（`tools/motion/`）

*动作重定向与接触相位*

| 工具 | 做什么 |
|---|---|
| `tools/motion/audit_asset_retarget.py` | Audit one profile-bound retargeted M2 action with body-plan-neutral QA |
| `tools/motion/build_self_retarget_profile.py` | Build a hash-bound identity-skeleton retarget probe profile |
| `tools/motion/derive_contacts.py` | Derive hash-bound M2 four-paw contact phases from explicit inputs |
| `tools/motion/retarget_blender.py` | Retarget audited GLB motion onto an audited target GLB in Blender |
