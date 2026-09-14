"""可见状态轨迹必须按帧号的数值走，不能按字符串键的字典序走。

帧号是 JSON 对象的键，也就是字符串。``sorted(frames)`` 给的是 0, 1, 10, 100, …, 2, 20, …，
于是"相邻两帧"这件事整个错位：第 1 帧会跟第 10 帧凑成一对（凭空多出一次转场），第 9 帧和第 10 帧
却永远不相邻（真的转场看不见）。2026-09-14 在 v2 题库上实测，QA-11 的 144 道题里 14 道因此判反，
另有 503 个报出来的转场帧在数值序下根本不存在。

这里的用例故意用 12 帧——少于 10 帧时两种排序一样，钉不住任何东西。
"""

from __future__ import annotations

from pathlib import Path
import re

from avengine.qa.observation_predicates import (
    clear_after_partial_transitions,
    visibility_series,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "avengine"


def frames_dict(states):
    return {str(index): {"frame_index": index, "state": state} for index, state in enumerate(states)}


#: 第 1 帧部分遮挡、第 10 帧清晰可见，中间第 9 帧出画，所以数值序下没有任何
#: "部分遮挡紧接着清晰可见"的相邻对；字典序却会把第 1 帧和第 10 帧排到一起。
TRAP = frames_dict(
    ["visible_clear"]
    + ["visible_occluded"] * 8
    + ["out_of_view", "visible_clear", "visible_clear"]
)


def test_十帧以上时字典序和数值序确实不一样():
    assert len(TRAP) == 12
    assert sorted(TRAP) != [str(index) for index in sorted(map(int, TRAP))]
    assert sorted(TRAP)[:4] == ["0", "1", "10", "11"]


def test_摊开的轨迹按帧号数值排():
    series = visibility_series(TRAP)
    assert [record["frame_index"] for record in series] == list(range(12))


def test_字典序会凭空造出一次转场而数值序不会():
    lexicographic = [TRAP[key] for key in sorted(TRAP)]
    spurious = clear_after_partial_transitions(lexicographic)
    assert [(a["frame_index"], b["frame_index"]) for a, b in spurious] == [(1, 10)]
    # 数值序下第 1 帧后面是第 2 帧，也是部分遮挡，没有任何转场
    assert clear_after_partial_transitions(visibility_series(TRAP)) == []


def test_真的相邻转场在字典序下会被漏掉():
    track = frames_dict(["visible_occluded"] * 10 + ["visible_clear", "visible_clear"])
    numeric = clear_after_partial_transitions(visibility_series(track))
    assert [(a["frame_index"], b["frame_index"]) for a, b in numeric] == [(9, 10)]
    lexicographic = [track[key] for key in sorted(track)]
    assert (9, 10) not in [
        (a["frame_index"], b["frame_index"]) for a, b in clear_after_partial_transitions(lexicographic)
    ]


def test_不是字典就给空串不是报错():
    assert visibility_series(None) == []
    assert visibility_series([1, 2, 3]) == []


def test_认不出来的键排在最前面而不是让整串崩掉():
    series = visibility_series({"nope": {"frame_index": None}, "1": {"frame_index": 1}})
    assert [record.get("frame_index") for record in series] == [None, 1]


def test_要求是字典时把不是字典的那些行滤掉():
    series = visibility_series({"0": {"frame_index": 0}, "1": "broken"}, require_mapping=True)
    assert len(series) == 1


def test_源码里不再有按字符串键遍历帧字典的写法():
    """防复发：谁再写一次 sorted(frames) 去遍历帧字典，这条会红。"""

    pattern = re.compile(r"for\s+\w+\s+in\s+sorted\(\s*frames\s*\)")
    offenders = []
    for path in SOURCE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(SOURCE_ROOT)}:{number}: {line.strip()}")
    assert not offenders, (
        "这些地方在按字符串键的字典序遍历帧字典，改成 visibility_series(frames)：\n"
        + "\n".join(offenders)
    )
