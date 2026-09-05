from __future__ import annotations

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
    assert callable(handler.candidate_adapter)
    assert callable(handler.visual_verifier)
    ue = get_backend_handler(DEFAULT_BACKEND_ID)
    assert ue.capture is None
    assert ue.candidate_adapter is None
