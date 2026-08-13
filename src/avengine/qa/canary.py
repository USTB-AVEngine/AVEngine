"""QA Canary 生成器 — 构建五个可视化验收场景。

每个 canary 生成一个完整的 Episode 文档，包含合成语义分割数据、
可见性分析结果、声音/空间/运动事实，并自动验证答案可推导且事实可观察。

用法::

    from avengine.qa.canary import build_all_canaries, CANARY_BUILDERS

    canaries = build_all_canaries()
    for canary in canaries:
        print(canary["canary_id"], canary["qa_results"])
"""

from __future__ import annotations

import numpy as np

from avengine.m5.timeline import TICKS_PER_FRAME
from avengine.qa.answer_deriver import derive_answer
from avengine.qa.candidate_selector import ActorCandidate, AssetBinding, SoundCandidate
from avengine.qa.pixel_visibility import analyze_all_frames
from avengine.qa.question_spec import QuestionSpec

# ═══════════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════════

H, W = 64, 64
N_FRAMES = 5

# 语义 ID
TARGET_SEMANTIC_ID = 10
FURNITURE_SEMANTIC_ID = 50
OTHER_ACTOR_SEMANTIC_ID = 20

# 角色 ID
TARGET_ACTOR_ID = "dog_01"
OTHER_ACTOR_ID = "cat_01"

# 目标区域：中央 32×32
TARGET_SLICE = (slice(16, 48), slice(16, 48))
# 部分遮挡：覆盖目标上半
OCCLUDER_PARTIAL = (slice(16, 32), slice(0, 64))
# 完全遮挡
OCCLUDER_FULL = (slice(0, 64), slice(0, 64))


# ═══════════════════════════════════════════════════════════════════════════════
# 语义数组生成
# ═══════════════════════════════════════════════════════════════════════════════


def _make_target_only_semantic() -> np.ndarray:
    arr = np.zeros((H, W), dtype=np.int64)
    arr[TARGET_SLICE] = TARGET_SEMANTIC_ID
    return arr


def _make_normal_semantic(
    occluder_slices: list[tuple[slice, slice, int]] | None = None,
) -> np.ndarray:
    arr = np.zeros((H, W), dtype=np.int64)
    arr[TARGET_SLICE] = TARGET_SEMANTIC_ID
    if occluder_slices:
        for ys, xs, sem_id in occluder_slices:
            arr[ys, xs] = sem_id
    return arr


# ═══════════════════════════════════════════════════════════════════════════════
# Actor / Sound 构建
# ═══════════════════════════════════════════════════════════════════════════════


def _make_default_actors() -> list[dict]:
    return [
        {
            "actor_id": TARGET_ACTOR_ID,
            "entity_asset_id": "dog_asset_01",
            "identity": {"species_id": "狗", "breed_id": "比格犬"},
            "realized_visual_attributes": {
                "clothing": {"top_color": "棕色"},
                "size": "medium",
                "body_build": "average",
                "life_stage": "adult",
            },
            "semantic_id": TARGET_SEMANTIC_ID,
        },
    ]


def _make_dog_binding() -> AssetBinding:
    actor = ActorCandidate(
        actor_id=TARGET_ACTOR_ID,
        entity_asset_id="dog_asset_01",
        species_id="狗",
        semantic_id=TARGET_SEMANTIC_ID,
        attributes={"species_id": "狗", "breed_id": "比格犬"},
    )
    sound = SoundCandidate(
        sound_asset_id="bark_01",
        semantic_sound_class="dog_bark",
        transcript="",
        duration_samples=2 * TICKS_PER_FRAME,
        bound_to_actor=TARGET_ACTOR_ID,
    )
    return AssetBinding(
        actor=actor,
        sound=sound,
        attribute_values={"species_id": "狗", "breed_id": "比格犬"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Episode 文档构建
# ═══════════════════════════════════════════════════════════════════════════════


def _build_episode_doc(
    canary_id: str,
    normal_semantics: list[np.ndarray],
    target_only_semantics: list[np.ndarray],
    in_frustums: list[bool],
    *,
    actors: list[dict] | None = None,
    sound_facts: list[dict] | None = None,
    spatial_frames: list[dict] | None = None,
    motion_frames: list[dict] | None = None,
    events: list[dict] | None = None,
) -> dict:
    """从合成语义数组构建完整 Episode 文档。"""
    vis_records = analyze_all_frames(
        normal_semantics,
        target_only_semantics,
        target_semantic_id=TARGET_SEMANTIC_ID,
        actor_id=TARGET_ACTOR_ID,
        in_frustums=in_frustums,
        actor_semantic_map={OTHER_ACTOR_SEMANTIC_ID: OTHER_ACTOR_ID},
        furniture_semantic_map={FURNITURE_SEMANTIC_ID: ("table_01", "桌子")},
    )

    return {
        "schema": "avengine_qa_episode_v1",
        "episode_id": f"canary_{canary_id}",
        "assets_used": {
            "actors": actors or _make_default_actors(),
            "sounds": [],
        },
        "scene": {},
        "timeline": {},
        "facts": {
            "sound_facts": sound_facts or [
                {
                    "event_id": "evt_01",
                    "actor_id": TARGET_ACTOR_ID,
                    "sound_asset_id": "bark_01",
                    "start_tick": TICKS_PER_FRAME,
                    "end_tick": 3 * TICKS_PER_FRAME,
                    "start_frame": 1,
                    "end_frame": 3,
                    "transcript": "",
                },
            ],
            "spatial_facts": {"per_frame": spatial_frames or [
                {"frame_index": i, "actors": {
                    TARGET_ACTOR_ID: {"listener_relative": {"azimuth_deg": 0.0}},
                }} for i in range(N_FRAMES)
            ]},
            "motion_facts": {"per_frame": motion_frames or [
                {"frame_index": i, "actor_states": {TARGET_ACTOR_ID: "idle"}}
                for i in range(N_FRAMES)
            ]},
            "visibility_facts": {"per_frame": list(vis_records)},
            "events": events or [],
        },
        "qa_pairs": [],
        "sidecars": [],
        "provenance": {"canary_id": canary_id},
        "episode_content_sha256": "canary_placeholder",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 五个 Canary 构建函数
# ═══════════════════════════════════════════════════════════════════════════════


def build_canary_1_fully_visible() -> dict:
    """C1: 目标在所有帧完全可见。"""
    normal = [_make_normal_semantic() for _ in range(N_FRAMES)]
    target_only = [_make_target_only_semantic() for _ in range(N_FRAMES)]
    in_frustum = [True] * N_FRAMES
    return _build_episode_doc("c1_fully_visible", normal, target_only, in_frustum)


def build_canary_2_partial_occlusion() -> dict:
    """C2: 目标被家具部分遮挡。"""
    normal = [
        _make_normal_semantic(occluder_slices=[(*OCCLUDER_PARTIAL, FURNITURE_SEMANTIC_ID)])
        for _ in range(N_FRAMES)
    ]
    target_only = [_make_target_only_semantic() for _ in range(N_FRAMES)]
    in_frustum = [True] * N_FRAMES
    return _build_episode_doc("c2_partial_occlusion", normal, target_only, in_frustum)


def build_canary_3_fully_occluded() -> dict:
    """C3: 目标被另一角色完全遮挡。"""
    normal = [
        _make_normal_semantic(occluder_slices=[(*OCCLUDER_FULL, OTHER_ACTOR_SEMANTIC_ID)])
        for _ in range(N_FRAMES)
    ]
    target_only = [_make_target_only_semantic() for _ in range(N_FRAMES)]
    in_frustum = [True] * N_FRAMES
    actors = _make_default_actors() + [
        {
            "actor_id": OTHER_ACTOR_ID,
            "entity_asset_id": "cat_asset_01",
            "identity": {"species_id": "猫"},
            "realized_visual_attributes": {"clothing": {"top_color": "白色"}},
            "semantic_id": OTHER_ACTOR_SEMANTIC_ID,
        },
    ]
    return _build_episode_doc("c3_fully_occluded", normal, target_only, in_frustum,
                              actors=actors)


def build_canary_4_out_of_view_enter() -> dict:
    """C4: 帧 0-2 画外，帧 3 入画，帧 4 可见。"""
    normal = [_make_normal_semantic() for _ in range(N_FRAMES)]
    target_only = [_make_target_only_semantic() for _ in range(N_FRAMES)]
    in_frustum = [False, False, False, True, True]
    return _build_episode_doc(
        "c4_out_of_view_enter", normal, target_only, in_frustum,
        events=[{"event_type": "enter_frustum", "actor_id": TARGET_ACTOR_ID, "frame_index": 3}],
        spatial_frames=[
            {"frame_index": i, "actors": {
                TARGET_ACTOR_ID: {"listener_relative": {"azimuth_deg": -45.0}},
            }} for i in range(N_FRAMES)
        ],
    )


def build_canary_5_camera_motion_reappear() -> dict:
    """C5: 帧 0 可见 → 帧 1-3 遮挡 → 帧 4 重现。"""
    normal = [
        _make_normal_semantic(),                                                      # 帧 0: 无遮挡
        _make_normal_semantic(occluder_slices=[(*OCCLUDER_FULL, FURNITURE_SEMANTIC_ID)]),  # 帧 1-3: 家具全遮
        _make_normal_semantic(occluder_slices=[(*OCCLUDER_FULL, FURNITURE_SEMANTIC_ID)]),
        _make_normal_semantic(occluder_slices=[(*OCCLUDER_FULL, FURNITURE_SEMANTIC_ID)]),
        _make_normal_semantic(),                                                      # 帧 4: 无遮挡
    ]
    target_only = [_make_target_only_semantic() for _ in range(N_FRAMES)]
    in_frustum = [True] * N_FRAMES
    return _build_episode_doc(
        "c5_camera_motion_reappear", normal, target_only, in_frustum,
        events=[
            {"event_type": "occlusion_start", "actor_id": TARGET_ACTOR_ID, "frame_index": 1},
            {"event_type": "reappear", "actor_id": TARGET_ACTOR_ID, "frame_index": 4},
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Canary 验证
# ═══════════════════════════════════════════════════════════════════════════════


def verify_canary(
    canary_id: str,
    doc: dict,
    questions: list[dict],
) -> list[dict]:
    """对 canary 文档执行一组问题的答案推导验证。

    Args:
        canary_id: canary 标识。
        doc: 已构建的 Episode 文档。
        questions: 问题列表，每个问题含 ``spec`` (QuestionSpec) 和 ``expected_answer``。

    Returns:
        每个问题的验证结果列表，含 ``question_id``、``answer``、``passed`` 等字段。
    """
    binding = _make_dog_binding()
    results: list[dict] = []

    for q in questions:
        spec = q["spec"]
        answer, is_unique, is_observable = derive_answer(doc, spec, binding)
        passed = (
            answer == q.get("expected_answer", answer)
            and is_unique is True
            and is_observable is True
        )
        results.append({
            "canary_id": canary_id,
            "question_type": spec.question_type,
            "question_text": spec.template,
            "answer_text": answer,
            "expected_answer": q.get("expected_answer", ""),
            "answer_unique": is_unique,
            "fact_observable": is_observable,
            "passed": passed,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 预定义问题集
# ═══════════════════════════════════════════════════════════════════════════════


_SOUND_PRESENCE_SPEC = QuestionSpec(
    spec_id="canary_q1", question_type="sound_presence",
    template="比格犬是否发声？", answer_modality="sound_facts",
)
_SOUND_VIS_SPEC = QuestionSpec(
    spec_id="canary_q2", question_type="sound_visibility",
    template="比格犬叫时是否可见？", answer_modality="sound_visibility",
)
_OCCLUDER_SPEC = QuestionSpec(
    spec_id="canary_q3", question_type="occluder_identity",
    template="挡住狗的是什么？", answer_modality="occluder_identity",
)
_ENTER_DIR_SPEC = QuestionSpec(
    spec_id="canary_q4", question_type="enter_frustum_direction",
    template="狗从哪侧进入？", answer_modality="enter_frustum_direction",
)

# 每个 canary 的配套问题
_CANARY_QUESTIONS: dict[str, list[dict]] = {
    "c1_fully_visible": [
        {"spec": _SOUND_PRESENCE_SPEC, "expected_answer": "是"},
        {"spec": _SOUND_VIS_SPEC, "expected_answer": "清晰可见"},
    ],
    "c2_partial_occlusion": [
        {"spec": _OCCLUDER_SPEC, "expected_answer": "家具（桌子）"},
    ],
    "c3_fully_occluded": [
        {"spec": _SOUND_VIS_SPEC, "expected_answer": "完全遮挡"},
    ],
    "c4_out_of_view_enter": [
        {"spec": _ENTER_DIR_SPEC, "expected_answer": "左侧"},
    ],
    "c5_camera_motion_reappear": [
        {"spec": _SOUND_VIS_SPEC, "expected_answer": "完全遮挡"},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════


CANARY_BUILDERS: dict[str, callable] = {
    "c1_fully_visible": build_canary_1_fully_visible,
    "c2_partial_occlusion": build_canary_2_partial_occlusion,
    "c3_fully_occluded": build_canary_3_fully_occluded,
    "c4_out_of_view_enter": build_canary_4_out_of_view_enter,
    "c5_camera_motion_reappear": build_canary_5_camera_motion_reappear,
}


# 每个 canary 构建器需要的辅助参数（返回 None 表示不需要）
_CANARY_ARRAYS: dict[str, dict] = {
    "c1_fully_visible": {
        "normal": lambda: [_make_normal_semantic() for _ in range(N_FRAMES)],
        "target_only": lambda: [_make_target_only_semantic() for _ in range(N_FRAMES)],
        "in_frustums": [True] * N_FRAMES,
    },
    "c2_partial_occlusion": {
        "normal": lambda: [
            _make_normal_semantic(occluder_slices=[(*OCCLUDER_PARTIAL, FURNITURE_SEMANTIC_ID)])
            for _ in range(N_FRAMES)
        ],
        "target_only": lambda: [_make_target_only_semantic() for _ in range(N_FRAMES)],
        "in_frustums": [True] * N_FRAMES,
    },
    "c3_fully_occluded": {
        "normal": lambda: [
            _make_normal_semantic(occluder_slices=[(*OCCLUDER_FULL, OTHER_ACTOR_SEMANTIC_ID)])
            for _ in range(N_FRAMES)
        ],
        "target_only": lambda: [_make_target_only_semantic() for _ in range(N_FRAMES)],
        "in_frustums": [True] * N_FRAMES,
    },
    "c4_out_of_view_enter": {
        "normal": lambda: [_make_normal_semantic() for _ in range(N_FRAMES)],
        "target_only": lambda: [_make_target_only_semantic() for _ in range(N_FRAMES)],
        "in_frustums": [False, False, False, True, True],
    },
    "c5_camera_motion_reappear": {
        "normal": lambda: [
            _make_normal_semantic(),
            _make_normal_semantic(occluder_slices=[(*OCCLUDER_FULL, FURNITURE_SEMANTIC_ID)]),
            _make_normal_semantic(occluder_slices=[(*OCCLUDER_FULL, FURNITURE_SEMANTIC_ID)]),
            _make_normal_semantic(occluder_slices=[(*OCCLUDER_FULL, FURNITURE_SEMANTIC_ID)]),
            _make_normal_semantic(),
        ],
        "target_only": lambda: [_make_target_only_semantic() for _ in range(N_FRAMES)],
        "in_frustums": [True] * N_FRAMES,
    },
}


def build_all_canaries() -> list[dict]:
    """构建全部五个 canary 并运行验证。

    Returns:
        每个 canary 的完整信息字典，包含 ``canary_id``、``episode``、
        ``normal_semantics``、``target_only_semantics``、``qa_results``。
    """
    results: list[dict] = []
    for canary_id, builder in CANARY_BUILDERS.items():
        doc = builder()
        arrays = _CANARY_ARRAYS.get(canary_id, {})
        questions = _CANARY_QUESTIONS.get(canary_id, [])
        qa_results = verify_canary(canary_id, doc, questions)
        results.append({
            "canary_id": canary_id,
            "episode": doc,
            "normal_semantics": np.stack(arrays["normal"]()) if "normal" in arrays else None,
            "target_only_semantics": np.stack(arrays["target_only"]()) if "target_only" in arrays else None,
            "in_frustums": arrays.get("in_frustums", []),
            "qa_results": qa_results,
            "all_passed": all(r["passed"] for r in qa_results),
        })
    return results


__all__ = [
    "CANARY_BUILDERS",
    "build_all_canaries",
    "build_canary_1_fully_visible",
    "build_canary_2_partial_occlusion",
    "build_canary_3_fully_occluded",
    "build_canary_4_out_of_view_enter",
    "build_canary_5_camera_motion_reappear",
    "verify_canary",
]
