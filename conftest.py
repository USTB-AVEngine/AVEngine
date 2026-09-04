"""让测试用本检出的 src，并且在它不是本检出时整套失败。

2026-09-04 实测:在 /data/jzy/tmp/wt-qa-v3-questionform 里跑 pytest,``avengine``
解析到的是 /data/jzy/code/AVEngine-lead-a/src/avengine——**主检出的 src**,不是这个
worktree 的。我们同时跑五个 worktree,所以其中四个对 ``src/avengine`` 的改动本地测不到:
测试绿是因为它测的是主检出那一份。

这件事今天让三个会话报出三个互不相同的 tests/unit 数字(我 38 failed,另外两位各自
0 failed),而我们花了两轮才把它归因清楚。那 38 个的报错全是 "unexpected keyword
argument"、"has no attribute"、"does not bind canonical repository file"——库版本不匹配
的典型形状,不是代码坏了。

两件事一起做,缺一不可:

* 把本检出的 ``src`` 放到 ``sys.path`` 最前,所以导入拿到的是这里的代码;
* 断言导入回来的确实在这里,不符就整套失败。第二件才是关键——第一件只在没有别的
  东西抢先导入时有效,而"绿了但测的是别处"正是这个坑最难发现的地方。把它变成响的,
  比修好它更重要。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_SRC = _REPO / "src"

if _SRC.is_dir():
    _entry = str(_SRC)
    if sys.path and sys.path[0] == _entry:
        pass
    else:
        while _entry in sys.path:
            sys.path.remove(_entry)
        sys.path.insert(0, _entry)


def pytest_configure(config):
    """在收集任何测试之前确认 avengine 来自本检出。"""

    if not _SRC.is_dir():
        return
    try:
        import avengine
    except Exception:
        return
    resolved = Path(avengine.__file__).resolve()
    try:
        resolved.relative_to(_SRC.resolve())
    except ValueError:
        raise RuntimeError(
            f"avengine resolves to {resolved}, not to this checkout's "
            f"{_SRC}. The suite would be testing another checkout's library "
            f"while reporting on this one -- which is how three sessions got "
            f"three different tests/unit counts on 2026-09-04. Run with "
            f"PYTHONPATH={_SRC} , or install this checkout editable into the "
            f"environment you are using."
        ) from None
