"""答案推导 — 从 Episode 事实数据中提取答案。

本模块从已构建的 Episode 文档中读取 ``sound_facts``、``visibility_facts``、
``motion_facts``、``spatial_facts`` 等事实数据，判断答案文本、唯一性和可观察性。

支持十种问题模态：::

    sound_facts           — 是否发声（"是"/"否"）
    sound_transcript      — 发声内容（transcript 文本）
    reverse_attribute     — 从发声内容反查外貌属性
    spatial_direction     — 发声者的空间方向（"左侧"/"右侧"/"正前方"）
    sound_order           — 两个角色的发声先后顺序
    overlap_sound         — 两个声音事件是否时间重叠
    sound_motion          — 发声时的运动状态
    enter_frustum_direction — 画外到入画的方向
    sound_visibility      — 发声时的像素遮挡状态
    occluder_identity     — 遮挡者识别
"""

from __future__ import annotations

from typing import Any

from avengine.m5.timeline import TICKS_PER_FRAME
from avengine.qa.candidate_selector import AssetBinding
from avengine.qa.question_spec import QuestionSpec


# ═══════════════════════════════════════════════════════════════════════════════
# 答案推导（主入口）
# ═══════════════════════════════════════════════════════════════════════════════


def derive_answer(
    doc: dict[str, Any],
    spec: QuestionSpec,
    binding: AssetBinding,
    time_window: tuple[int, int] | None = None,
) -> tuple[str, bool, bool]:
    """从 Episode 文档中推导答案。

    Args:
        doc: 已构建的 Episode 文档（``episode.build()`` 的返回值）。
        spec: 问题模板。
        binding: 资产绑定（角色 + 声音 + 属性值）。
        time_window: 时间窗口覆盖（优先于 spec.time_window）。

    Returns:
        ``(answer_text, is_unique, is_observable)`` 三元组。
    """
    effective_tw = time_window if time_window is not None else spec.time_window

    is_unique = check_answer_unique(doc, spec, binding)
    is_observable = check_fact_observable(doc, spec, binding, time_window=effective_tw)

    modality = spec.answer_modality

    # ── 原始三种模态 ────────────────────────────────────────────────────
    if modality == "sound_facts":
        answer = _derive_sound_answer(doc, binding, effective_tw)
    elif modality == "visibility_facts":
        answer = _derive_visibility_answer(doc, binding)
    elif modality == "motion_facts":
        answer = _derive_motion_answer(doc, binding)

    # ── 十类问题新增七种模态 ─────────────────────────────────────────────
    elif modality == "sound_transcript":
        answer = _derive_sound_transcript(doc, binding, effective_tw)
    elif modality == "reverse_attribute":
        answer = _derive_reverse_attribute(doc, spec, binding, effective_tw)
    elif modality == "spatial_direction":
        answer = _derive_spatial_direction(doc, binding, effective_tw)
    elif modality == "sound_order":
        answer = _derive_sound_order(doc, effective_tw)
    elif modality == "overlap_sound":
        answer = _derive_overlap_sound(doc, effective_tw)
    elif modality == "sound_motion":
        answer = _derive_sound_motion(doc, binding, effective_tw)
    elif modality == "enter_frustum_direction":
        answer = _derive_enter_frustum_direction(doc, binding)
    elif modality == "sound_visibility":
        answer = _derive_sound_visibility(doc, binding, effective_tw)
    elif modality == "occluder_identity":
        answer = _derive_occluder_identity(doc, binding, effective_tw)
    else:
        answer = "无法判定"

    return answer, is_unique, is_observable


# ═══════════════════════════════════════════════════════════════════════════════
# 唯一性检查
# ═══════════════════════════════════════════════════════════════════════════════


def check_answer_unique(
    doc: dict[str, Any],
    spec: QuestionSpec,
    binding: AssetBinding,
) -> bool:
    """检查场景中答案是否唯一。

    核心逻辑：如果场景中存在两个角色拥有相同的区分属性值
    （如两个 ``top_color=blue`` 的角色），则问题 "穿蓝衣的人是否发声"
    的答案不唯一。
    """
    actors: list[dict[str, Any]] = doc.get("assets_used", {}).get("actors", [])

    for attr_name, attr_value in binding.attribute_values.items():
        match_count = 0
        for actor in actors:
            actual_value = _extract_actor_attribute(actor, attr_name)
            if actual_value == attr_value:
                match_count += 1
        if match_count > 1:
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 可观察性检查（分发）
# ═══════════════════════════════════════════════════════════════════════════════


def check_fact_observable(
    doc: dict[str, Any],
    spec: QuestionSpec,
    binding: AssetBinding,
    time_window: tuple[int, int] | None = None,
) -> bool:
    """检查事实在 Episode 中是否可观察。"""
    actor_id = binding.actor.actor_id
    effective_tw = time_window if time_window is not None else spec.time_window
    modality = spec.answer_modality

    if modality == "sound_facts":
        return _sound_observable(doc, actor_id, effective_tw)
    elif modality == "visibility_facts":
        return _visibility_observable(doc, actor_id)
    elif modality == "motion_facts":
        return True
    elif modality == "sound_transcript":
        return _sound_observable(doc, actor_id, effective_tw)
    elif modality == "reverse_attribute":
        return _has_sound_facts(doc)
    elif modality == "spatial_direction":
        return _spatial_data_observable(doc, actor_id)
    elif modality == "sound_order":
        return _has_two_sound_actors(doc)
    elif modality == "overlap_sound":
        return _has_two_sound_actors(doc)
    elif modality == "sound_motion":
        return _motion_data_observable(doc, actor_id)
    elif modality == "enter_frustum_direction":
        return _has_pixel_data(doc) and _has_events(doc)
    elif modality == "sound_visibility":
        return _has_pixel_data(doc) and _has_visibility_frames(doc)
    elif modality == "occluder_identity":
        return _has_pixel_data(doc) and _has_visibility_frames(doc)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 推导函数：声音（已有 + 新增）
# ═══════════════════════════════════════════════════════════════════════════════


def _derive_sound_answer(
    doc: dict[str, Any],
    binding: AssetBinding,
    time_window: tuple[int, int] | None = None,
) -> str:
    """类型 1：外貌属性 → 是否发声。"""
    actor_id = binding.actor.actor_id
    sound_facts = _get_sound_facts(doc)

    has_sound = _actor_has_sound_in_window(actor_id, sound_facts, time_window)
    return "是" if has_sound else "否"


def _derive_sound_transcript(
    doc: dict[str, Any],
    binding: AssetBinding,
    time_window: tuple[int, int] | None = None,
) -> str:
    """类型 2：外貌属性 → 发声内容。"""
    actor_id = binding.actor.actor_id
    sound_facts = _get_sound_facts(doc)

    for fact in sound_facts:
        if fact.get("actor_id") != actor_id:
            continue
        if not _sound_overlaps_window(fact, time_window):
            continue
        transcript = fact.get("transcript", "")
        if transcript:
            return transcript

    return "（无内容）"


def _derive_reverse_attribute(
    doc: dict[str, Any],
    spec: QuestionSpec,
    binding: AssetBinding,
    time_window: tuple[int, int] | None = None,
) -> str:
    """类型 3：发声内容 → 外貌属性。

    从模板变量中提取目标属性名（非 transcript 变量），
    查找说了指定内容的角色，返回其对应属性值。
    """
    sound_facts = _get_sound_facts(doc)
    # 找到绑定中 transcript 的期望值
    target_transcript = binding.attribute_values.get("transcript", "")

    # 哪个角色说了这句话？
    speaking_actor_id: str | None = None
    for fact in sound_facts:
        if not _sound_overlaps_window(fact, time_window):
            continue
        if fact.get("transcript", "") == target_transcript:
            speaking_actor_id = fact.get("actor_id")
            break

    if speaking_actor_id is None:
        return "（无人说此话）"

    # 查找该角色的属性值
    actors: list[dict[str, Any]] = doc.get("assets_used", {}).get("actors", [])
    # 从 spec 变量中确定要返回的属性
    for var_name in spec.variable_names:
        if var_name in ("transcript", "time_window"):
            continue
        # 这是要查询的属性
        for actor in actors:
            if actor.get("actor_id") == speaking_actor_id:
                value = _extract_actor_attribute(actor, var_name)
                return value if value else "（未知）"

    return "（未知属性）"


# ═══════════════════════════════════════════════════════════════════════════════
# 推导函数：空间方向
# ═══════════════════════════════════════════════════════════════════════════════


def _derive_spatial_direction(
    doc: dict[str, Any],
    binding: AssetBinding,
    time_window: tuple[int, int] | None = None,
) -> str:
    """类型 4：发声者的空间方向。

    在声音事件中点帧读取 listener_relative.azimuth_deg。
    """
    actor_id = binding.actor.actor_id
    sound_facts = _get_sound_facts(doc)

    # 找到目标角色在时间窗口内的第一个声音事件
    target_frame: int | None = None
    for fact in sound_facts:
        if fact.get("actor_id") != actor_id:
            continue
        if not _sound_overlaps_window(fact, time_window):
            continue
        # 取声音事件中点帧
        start_f = fact.get("start_frame", 0)
        end_f = fact.get("end_frame", start_f)
        target_frame = (start_f + end_f) // 2
        break

    if target_frame is None:
        return "（未发声）"

    # 读取该帧的 spatial_facts
    azimuth = _get_azimuth_at_frame(doc, actor_id, target_frame)
    if azimuth is None:
        return "（无空间数据）"

    if azimuth < -30.0:
        return "左侧"
    elif azimuth > 30.0:
        return "右侧"
    else:
        return "正前方"


# ═══════════════════════════════════════════════════════════════════════════════
# 推导函数：双角色比较
# ═══════════════════════════════════════════════════════════════════════════════


def _derive_sound_order(
    doc: dict[str, Any],
    time_window: tuple[int, int] | None = None,
) -> str:
    """类型 5：两个角色谁先说话。

    比较场景中前两个有声音的角色，按 start_tick 排序。
    """
    sound_facts = _get_sound_facts(doc)
    actors = doc.get("assets_used", {}).get("actors", [])

    # 收集时间窗口内每个角色的最早发声 tick
    first_sound: dict[str, int] = {}
    for fact in sound_facts:
        if not _sound_overlaps_window(fact, time_window):
            continue
        aid = fact.get("actor_id", "")
        tick = fact.get("start_tick", 0)
        if aid not in first_sound or tick < first_sound[aid]:
            first_sound[aid] = tick

    if len(first_sound) < 2:
        return "（需要至少两个角色发声）"

    # 按最早发声排序
    sorted_actors = sorted(first_sound.items(), key=lambda item: item[1])
    # 查找角色名称
    first_name = _find_actor_label(actors, sorted_actors[0][0])
    second_name = _find_actor_label(actors, sorted_actors[1][0])

    return f"{first_name}先说话，{second_name}后说话"


def _derive_overlap_sound(
    doc: dict[str, Any],
    time_window: tuple[int, int] | None = None,
) -> str:
    """类型 6：两个角色是否同时发声。

    检查两个指定角色的声音事件是否有时间重叠。
    """
    sound_facts = _get_sound_facts(doc)
    actors = doc.get("assets_used", {}).get("actors", [])

    if len(actors) < 2:
        return "（需要两个角色）"

    # 取前两个角色的所有声音事件
    aid_a = actors[0].get("actor_id", "")
    aid_b = actors[1].get("actor_id", "")

    intervals_a = _collect_sound_intervals(sound_facts, aid_a, time_window)
    intervals_b = _collect_sound_intervals(sound_facts, aid_b, time_window)

    # 检查是否有任何重叠
    for sa, ea in intervals_a:
        for sb, eb in intervals_b:
            if not (ea < sb or sa > eb):
                return "是"

    return "否"


# ═══════════════════════════════════════════════════════════════════════════════
# 推导函数：复合模态
# ═══════════════════════════════════════════════════════════════════════════════


def _derive_sound_motion(
    doc: dict[str, Any],
    binding: AssetBinding,
    time_window: tuple[int, int] | None = None,
) -> str:
    """类型 7：发声时的运动状态。"""
    actor_id = binding.actor.actor_id
    sound_facts = _get_sound_facts(doc)

    # 找到声音事件的中点帧
    target_frame: int | None = None
    for fact in sound_facts:
        if fact.get("actor_id") != actor_id:
            continue
        if not _sound_overlaps_window(fact, time_window):
            continue
        start_f = fact.get("start_frame", 0)
        end_f = fact.get("end_frame", start_f)
        target_frame = (start_f + end_f) // 2
        break

    if target_frame is None:
        return "（未发声）"

    # 读取该帧的运动状态
    motion_state = _get_motion_at_frame(doc, actor_id, target_frame)
    if motion_state is None:
        return "（无运动数据）"

    # 中文映射
    motion_labels = {"idle": "静止", "walk": "走", "other": "其他动作"}
    return motion_labels.get(motion_state, motion_state)


def _derive_sound_visibility(
    doc: dict[str, Any],
    binding: AssetBinding,
    time_window: tuple[int, int] | None = None,
) -> str:
    """类型 9：发声时的像素遮挡状态。"""
    actor_id = binding.actor.actor_id
    sound_facts = _get_sound_facts(doc)

    # 找到声音事件中点帧
    target_frame: int | None = None
    for fact in sound_facts:
        if fact.get("actor_id") != actor_id:
            continue
        if not _sound_overlaps_window(fact, time_window):
            continue
        start_f = fact.get("start_frame", 0)
        end_f = fact.get("end_frame", start_f)
        target_frame = (start_f + end_f) // 2
        break

    if target_frame is None:
        return "（未发声）"

    vis_state = _get_visibility_at_frame(doc, actor_id, target_frame)
    if vis_state is None:
        return "（无可见性数据）"

    state_labels = {
        "visible_clear": "清晰可见",
        "visible_occluded": "部分遮挡",
        "fully_occluded": "完全遮挡",
        "out_of_view": "画外",
    }
    return state_labels.get(vis_state, vis_state)


def _derive_occluder_identity(
    doc: dict[str, Any],
    binding: AssetBinding,
    time_window: tuple[int, int] | None = None,
) -> str:
    """类型 10：遮挡者识别。"""
    actor_id = binding.actor.actor_id
    sound_facts = _get_sound_facts(doc)

    # 找到声音事件中点帧
    target_frame: int | None = None
    for fact in sound_facts:
        if fact.get("actor_id") != actor_id:
            continue
        if not _sound_overlaps_window(fact, time_window):
            continue
        start_f = fact.get("start_frame", 0)
        end_f = fact.get("end_frame", start_f)
        target_frame = (start_f + end_f) // 2
        break

    if target_frame is None:
        return "（未发声）"

    occluders = _get_occluders_at_frame(doc, actor_id, target_frame)
    if not occluders:
        return "无遮挡物"

    # 返回第一个（最多像素的）遮挡物
    first = occluders[0]
    otype = first.get("occluder_type", "unknown_static")
    if otype == "furniture":
        label = first.get("semantic_label", "家具")
        return f"家具（{label}）"
    elif otype == "actor":
        oid = first.get("actor_id", "未知")
        return f"另一只动物（{oid}）"
    else:
        return "未知物体"


def _derive_enter_frustum_direction(
    doc: dict[str, Any],
    binding: AssetBinding,
) -> str:
    """类型 8：画外到入画的方向。

    查找 enter_frustum 事件，获取进入帧的 spatial_facts 确定方向。
    """
    actor_id = binding.actor.actor_id
    events: list[dict[str, Any]] = doc.get("facts", {}).get("events", [])

    # 查找该角色的 enter_frustum 事件
    entry_frame: int | None = None
    for evt in events:
        if evt.get("event_type") == "enter_frustum" and evt.get("actor_id") == actor_id:
            entry_frame = evt.get("frame_index")
            break

    if entry_frame is None:
        return "（无入画事件）"

    azimuth = _get_azimuth_at_frame(doc, actor_id, entry_frame)
    if azimuth is None:
        return "（无空间数据）"

    if azimuth < -30.0:
        return "左侧"
    elif azimuth > 30.0:
        return "右侧"
    else:
        return "正前方"


# ═══════════════════════════════════════════════════════════════════════════════
# 推导函数：可见性 / 运动（已有）
# ═══════════════════════════════════════════════════════════════════════════════


def _derive_visibility_answer(
    doc: dict[str, Any],
    binding: AssetBinding,
) -> str:
    """从 visibility_facts 推导答案。"""
    actor_id = binding.actor.actor_id
    vis_frames = _get_visibility_frames(doc)

    total = 0
    visible = 0
    for frame in vis_frames:
        actor_vis = frame.get("actor_visibility", {}).get(actor_id)
        if actor_vis is None:
            continue
        total += 1
        state = actor_vis.get("visibility_state", "")
        if state in ("visible_clear", "visible_occluded"):
            visible += 1

    if total == 0:
        return "不可见"

    fraction = visible / total
    if fraction >= 0.9:
        return "基本无遮挡"
    elif fraction >= 0.05:
        return "部分遮挡"
    else:
        return "完全遮挡"


def _derive_motion_answer(
    doc: dict[str, Any],
    binding: AssetBinding,
) -> str:
    """从 motion_facts 推导答案。"""
    actor_id = binding.actor.actor_id
    motion_frames = _get_motion_frames(doc)

    for frame in motion_frames:
        states = frame.get("actor_states", {})
        if actor_id in states:
            return states[actor_id]

    return "未知"


# ═══════════════════════════════════════════════════════════════════════════════
# 可观察性检查（各模态具体实现）
# ═══════════════════════════════════════════════════════════════════════════════


def _sound_observable(
    doc: dict[str, Any],
    actor_id: str,
    time_window: tuple[int, int] | None = None,
) -> bool:
    """检查声音事件期间目标是否在画面中可见。"""
    vis_frames = _get_visibility_frames(doc)
    if not vis_frames:
        return True

    if time_window is not None:
        start_frame = time_window[0] // TICKS_PER_FRAME
        end_frame = time_window[1] // TICKS_PER_FRAME
    else:
        start_frame = 0
        end_frame = len(vis_frames) - 1

    relevant_frames = [
        f for f in vis_frames
        if start_frame <= f.get("frame_index", 0) <= end_frame
    ]
    if not relevant_frames:
        return True

    for frame in relevant_frames:
        actor_vis = frame.get("actor_visibility", {}).get(actor_id)
        if actor_vis is None:
            continue
        state = actor_vis.get("visibility_state", "")
        if state not in ("fully_occluded", "out_of_view"):
            return True

    return False


def _visibility_observable(doc: dict[str, Any], actor_id: str) -> bool:
    """可见性问题始终可观察。"""
    return True


def _spatial_data_observable(doc: dict[str, Any], actor_id: str) -> bool:
    """检查是否有空间数据。"""
    spatial_frames = _get_spatial_frames(doc)
    for frame in spatial_frames:
        actors = frame.get("actors", {})
        if actor_id in actors:
            return True
    return False


def _motion_data_observable(doc: dict[str, Any], actor_id: str) -> bool:
    """检查是否有运动数据。"""
    motion_frames = _get_motion_frames(doc)
    for frame in motion_frames:
        if actor_id in frame.get("actor_states", {}):
            return True
    return False


def _has_sound_facts(doc: dict[str, Any]) -> bool:
    """检查是否有声音事实数据。"""
    return len(_get_sound_facts(doc)) > 0


def _has_two_sound_actors(doc: dict[str, Any]) -> bool:
    """检查是否至少有两个角色有声音事件。"""
    sound_facts = _get_sound_facts(doc)
    actors_with_sound = {f.get("actor_id") for f in sound_facts if f.get("actor_id")}
    return len(actors_with_sound) >= 2


def _has_pixel_data(doc: dict[str, Any]) -> bool:
    """检查是否存在像素级可见性数据（必须有像素真值）。"""
    vis_frames = _get_visibility_frames(doc)
    if not vis_frames:
        return False
    for frame in vis_frames:
        for vis in frame.get("actor_visibility", {}).values():
            if vis.get("amodal_pixels", 0) > 0:
                return True
    return False


def _has_events(doc: dict[str, Any]) -> bool:
    """检查是否有事件数据。"""
    events = doc.get("facts", {}).get("events", [])
    return len(events) > 0


def _has_visibility_frames(doc: dict[str, Any]) -> bool:
    """检查是否有可见性帧数据。"""
    return len(_get_visibility_frames(doc)) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数：事实数据访问
# ═══════════════════════════════════════════════════════════════════════════════


def _get_sound_facts(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return doc.get("facts", {}).get("sound_facts", [])


def _get_spatial_frames(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return doc.get("facts", {}).get("spatial_facts", {}).get("per_frame", [])


def _get_motion_frames(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return doc.get("facts", {}).get("motion_facts", {}).get("per_frame", [])


def _get_visibility_frames(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return doc.get("facts", {}).get("visibility_facts", {}).get("per_frame", [])


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数：声音 / 帧数据查询
# ═══════════════════════════════════════════════════════════════════════════════


def _actor_has_sound_in_window(
    actor_id: str,
    sound_facts: list[dict[str, Any]],
    time_window: tuple[int, int] | None,
) -> bool:
    """检查角色在时间窗口内是否有声音事件。"""
    for fact in sound_facts:
        if fact.get("actor_id") != actor_id:
            continue
        if not _sound_overlaps_window(fact, time_window):
            continue
        return True
    return False


def _sound_overlaps_window(
    fact: dict[str, Any],
    time_window: tuple[int, int] | None,
) -> bool:
    """检查声音事实是否与时间窗口有交集。"""
    if time_window is None:
        return True
    start_tick = fact.get("start_tick", 0)
    end_tick = fact.get("end_tick", 0)
    tw_start, tw_end = time_window
    return not (end_tick < tw_start or start_tick > tw_end)


def _collect_sound_intervals(
    sound_facts: list[dict[str, Any]],
    actor_id: str,
    time_window: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    """收集角色在时间窗口内的所有声音区间 (start_tick, end_tick)。"""
    intervals: list[tuple[int, int]] = []
    for fact in sound_facts:
        if fact.get("actor_id") != actor_id:
            continue
        if not _sound_overlaps_window(fact, time_window):
            continue
        intervals.append((fact.get("start_tick", 0), fact.get("end_tick", 0)))
    return intervals


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数：逐帧数据查询
# ═══════════════════════════════════════════════════════════════════════════════


def _get_azimuth_at_frame(
    doc: dict[str, Any],
    actor_id: str,
    frame_index: int,
) -> float | None:
    """获取指定帧角色的 listener_relative.azimuth_deg。"""
    spatial_frames = _get_spatial_frames(doc)
    if frame_index < 0 or frame_index >= len(spatial_frames):
        return None
    frame = spatial_frames[frame_index]
    actor_spatial = frame.get("actors", {}).get(actor_id)
    if actor_spatial is None:
        return None
    return actor_spatial.get("listener_relative", {}).get("azimuth_deg")


def _get_motion_at_frame(
    doc: dict[str, Any],
    actor_id: str,
    frame_index: int,
) -> str | None:
    """获取指定帧角色的运动状态。"""
    motion_frames = _get_motion_frames(doc)
    if frame_index < 0 or frame_index >= len(motion_frames):
        return None
    frame = motion_frames[frame_index]
    return frame.get("actor_states", {}).get(actor_id)


def _get_visibility_at_frame(
    doc: dict[str, Any],
    actor_id: str,
    frame_index: int,
) -> str | None:
    """获取指定帧角色的可见性状态。"""
    vis_frames = _get_visibility_frames(doc)
    if frame_index < 0 or frame_index >= len(vis_frames):
        return None
    frame = vis_frames[frame_index]
    actor_vis = frame.get("actor_visibility", {}).get(actor_id)
    if actor_vis is None:
        return None
    return actor_vis.get("visibility_state")


def _get_occluders_at_frame(
    doc: dict[str, Any],
    actor_id: str,
    frame_index: int,
) -> list[dict[str, Any]]:
    """获取指定帧角色的遮挡物列表。"""
    vis_frames = _get_visibility_frames(doc)
    if frame_index < 0 or frame_index >= len(vis_frames):
        return []
    frame = vis_frames[frame_index]
    actor_vis = frame.get("actor_visibility", {}).get(actor_id)
    if actor_vis is None:
        return []
    return actor_vis.get("occluders", [])


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数：角色信息
# ═══════════════════════════════════════════════════════════════════════════════


def _find_actor_label(actors: list[dict[str, Any]], actor_id: str) -> str:
    """查找角色的可读标签。"""
    for actor in actors:
        if actor.get("actor_id") == actor_id:
            species = _extract_actor_attribute(actor, "species_id")
            top_color = _extract_actor_attribute(actor, "top_color")
            if top_color:
                return f"穿{top_color}衣的{species or '角色'}"
            return species or actor_id
    return actor_id


def _extract_actor_attribute(actor: dict[str, Any], attr_name: str) -> str | None:
    """从角色字典中提取指定属性值。"""
    if attr_name in ("species_id", "breed_id"):
        return actor.get("identity", {}).get(attr_name)

    attrs = actor.get("realized_visual_attributes", {})
    if attr_name == "top_color":
        return attrs.get("clothing", {}).get("top_color")
    if attr_name in ("size", "body_build", "life_stage"):
        return attrs.get(attr_name)

    return None


__all__ = [
    "check_answer_unique",
    "check_fact_observable",
    "derive_answer",
]
