from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from avengine.qa.backend_handlers import (
    DEFAULT_BACKEND_ID,
    HABITAT_NATIVE_BACKEND_ID,
    BackendHandler,
    BackendHandlerError,
    get_backend_handler,
    register_backend_handler,
    registered_backend_handlers,
    resolve_backend_id,
)


def test_backend_selection_uses_configured_backend_and_keeps_ue_default():
    assert resolve_backend_id(None) == DEFAULT_BACKEND_ID
    assert resolve_backend_id({"scene_id": "mp3d/room"}) == DEFAULT_BACKEND_ID
    assert get_backend_handler({"scene_id": "mp3d/room"}).backend_id == DEFAULT_BACKEND_ID
    assert get_backend_handler({"backend_id": "spear_unreal"}).backend_id == DEFAULT_BACKEND_ID
    assert get_backend_handler({"backend_id": HABITAT_NATIVE_BACKEND_ID}).backend_id == HABITAT_NATIVE_BACKEND_ID
    assert get_backend_handler({"backend": {"id": HABITAT_NATIVE_BACKEND_ID}}).backend_id == HABITAT_NATIVE_BACKEND_ID


def test_backend_registration_is_explicit_and_rejects_duplicates():
    from avengine.qa import backend_handlers as module

    handler = BackendHandler("fixture_backend", "fixture")
    try:
        assert register_backend_handler(handler).backend_id == "fixture_backend"
        try:
            register_backend_handler(handler)
        except BackendHandlerError:
            pass
        else:
            raise AssertionError("duplicate backend registration did not fail")
        assert "fixture_backend" in registered_backend_handlers()
    finally:
        module._REGISTRY.pop("fixture_backend", None)


def test_habitat_handler_exposes_lazy_capture_adapter_and_verifier():
    handler = get_backend_handler(HABITAT_NATIVE_BACKEND_ID)
    assert callable(handler.capture)
    assert callable(handler.audio_renderer)
    assert callable(handler.candidate_adapter)
    assert callable(handler.visual_verifier)
    ue = get_backend_handler(DEFAULT_BACKEND_ID)
    assert ue.capture is None
    assert ue.audio_renderer is None
    assert ue.candidate_adapter is None



def test_backend_selection_rejects_conflicting_explicit_fields():
    try:
        resolve_backend_id({
            "backend_id": DEFAULT_BACKEND_ID,
            "backend": HABITAT_NATIVE_BACKEND_ID,
        })
    except BackendHandlerError as error:
        assert "different handlers" in str(error)
    else:
        raise AssertionError("conflicting backend declarations did not fail")



def test_habitat_verifier_core_imports_with_only_installed_source(
    tmp_path: Path,
):
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from avengine.qa.backend_handlers import get_backend_handler;"
                "handler=get_backend_handler('habitat_native');"
                "assert handler.visual_verifier.__module__ == "
                "'avengine.qa.backend_handlers';"
                "from avengine.qa.mp3d_visual_verification import verify_batch;"
                "assert callable(verify_batch)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
