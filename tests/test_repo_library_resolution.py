"""跑测试的时候用的必须是本检出的库,不是别的检出的。

2026-09-04:在 /data/jzy/tmp/wt-qa-v3-questionform 里跑 pytest,avengine 解析到的是
/data/jzy/code/AVEngine-lead-a/src/avengine——主检出那一份。我们同时开五个 worktree,
所以其中四个对 src/avengine 的改动本地测不到,而**测试是绿的**,因为它测的是别处。

那一天三个会话报出三个不同的 tests/unit 数字(38 failed / 0 / 0),花了两轮才归因。
接上本检出的 src 之后,38 个失败全部消失——它们的报错本来就是
"unexpected keyword argument"、"has no attribute" 这种库版本不匹配的形状。

这个文件钉两件:解析位置现在是对的,以及**守卫在解析错时真的会拦**——一道从不开火的
守卫跟没有守卫一样,而这个坑的表现形式恰恰是"看起来一切正常"。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import conftest as ROOT_CONFTEST  # noqa: E402


def test_avengine_comes_from_this_checkout():
    import avengine
    resolved = Path(avengine.__file__).resolve()
    src = (REPO / "src").resolve()
    assert resolved.is_relative_to(src), (
        f"avengine resolves to {resolved}, outside {src}. The suite would be "
        f"reporting on this checkout while testing another one's library."
    )


def test_this_checkouts_src_is_first_on_the_path():
    src = str((REPO / "src").resolve())
    assert src in sys.path
    earlier = [p for p in sys.path[:sys.path.index(src)] if "src" in p]
    assert not earlier, f"another src precedes this checkout's: {earlier}"


def test_the_guard_refuses_a_library_from_somewhere_else(monkeypatch):
    """反例:守卫必须真的会拦,否则它只是装饰。"""
    fake = SimpleNamespace(
        __file__="/data/jzy/code/AVEngine-lead-a/src/avengine/__init__.py")
    monkeypatch.setitem(sys.modules, "avengine", fake)
    with pytest.raises(RuntimeError, match="not to this checkout"):
        ROOT_CONFTEST.pytest_configure(None)


def test_the_guard_accepts_the_library_from_here(monkeypatch):
    fake = SimpleNamespace(
        __file__=str((REPO / "src" / "avengine" / "__init__.py")))
    monkeypatch.setitem(sys.modules, "avengine", fake)
    ROOT_CONFTEST.pytest_configure(None)   # 不该抛
