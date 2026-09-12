"""Deterministic guard for the actual nested-lock launch-path deadlock."""
import ast
import inspect
import textwrap
from avengine.dataset.production_runner import ProductionRunner

def unsafe_calls(source):
    tree=ast.parse(textwrap.dedent(source));bad=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.With):
            continue
        if not any(isinstance(i.context_expr,ast.Attribute) and
                   i.context_expr.attr=="_accounting_lock" for i in node.items):
            continue
        for child in ast.walk(node):
            if not isinstance(child,ast.Call) or not isinstance(child.func,ast.Attribute):
                continue
            name=child.func.attr
            if name in {"_persist","_ensure_native_attempt","_file_failure"}:
                bad.append(name)
            if name=="_ensure_native_attempt_unlocked":
                explicit_false=any(k.arg=="persist" and isinstance(k.value,ast.Constant)
                                   and k.value.value is False for k in child.keywords)
                if not explicit_false:
                    bad.append(name)
    return bad

def test_launch_path_never_persists_while_accounting_lock_is_held():
    assert unsafe_calls(inspect.getsource(ProductionRunner.execute_work_item))==[]

def test_invariant_detects_the_actual_old_call_pattern():
    old="""def launch(self):
        with self._accounting_lock:
            self._ensure_native_attempt(scope,item,None)
            if exhausted:
                return self._file_failure(scope,item)
    """
    assert set(unsafe_calls(old))=={"_ensure_native_attempt","_file_failure"}
