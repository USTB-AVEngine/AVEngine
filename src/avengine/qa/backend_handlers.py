"""Configuration driven visual backend handlers for QA candidates.

The QA design records describe *what* should be captured while a backend
handler describes *where* that capture is executed.  The handler is selected
from an explicit `backend_id` in the candidate/room configuration.  Older
records that do not carry that field retain the historical UE/SPEAR default.

The registry deliberately does not inspect `scene_id`.  A scene identifier
is an input to a selected backend, not a routing switch; using it here made a
same room change execution route when a scene path or alias changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
from typing import Any


DEFAULT_BACKEND_ID = "ue_spear"
HABITAT_NATIVE_BACKEND_ID = "habitat_native"
_BACKEND_ALIASES = {
    # The room runtime registry predates the QA scene spelling.
    "spear_unreal": DEFAULT_BACKEND_ID,
}


class BackendHandlerError(ValueError):
    """Raised when a backend configuration cannot be resolved."""


CandidateAdapter = Callable[..., Mapping[str, Any]]
AudioRenderer = Callable[..., Mapping[str, Any]]
VisualVerifier = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class BackendHandler:
    """One registered backend and its optional candidate adapters.

    `capture` and `visual_verifier` are intentionally optional.  Existing
    UE/SPEAR launchers remain the owners of their runtime execution, while the
    handler registry gives callers a stable place to discover route metadata.
    The MP3D adapter is loaded lazily so importing QA configuration never
    imports Habitat or a native runtime.
    """

    backend_id: str
    description: str
    capture: Callable[..., Any] | None = None
    audio_renderer: AudioRenderer | None = None
    candidate_adapter: CandidateAdapter | None = None
    visual_verifier: VisualVerifier | None = None

    def __post_init__(self) -> None:
        _validate_backend_id(self.backend_id)
        if not isinstance(self.description, str) or not self.description.strip():
            raise BackendHandlerError("backend handler description must be nonempty")


def _validate_backend_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise BackendHandlerError(
            "backend_id must be a nonempty string without surrounding whitespace "
            "or control characters"
        )
    return value


def _lazy_mp3d_candidate_adapter(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from avengine.qa.mp3d_candidate_adapter import adapt_mp3d_candidate

    return adapt_mp3d_candidate(*args, **kwargs)


def _lazy_mp3d_capture(*args: Any, **kwargs: Any) -> Any:
    from avengine.capture.mp3d_multi_actor import capture_mp3d_multi_actor

    return capture_mp3d_multi_actor(*args, **kwargs)


def _lazy_mp3d_audio_renderer(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from avengine.timeline.current_mp3d_dynamic_audio import (
        render_current_mp3d_dynamic_audio,
    )

    return render_current_mp3d_dynamic_audio(*args, **kwargs)


def _lazy_mp3d_visual_verifier(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from avengine.qa.mp3d_visual_verification import verify_batch

    return verify_batch(*args, **kwargs)


_REGISTRY: dict[str, BackendHandler] = {}


def register_backend_handler(
    handler: BackendHandler | None = None,
    *,
    backend_id: str | None = None,
    description: str | None = None,
    capture: Callable[..., Any] | None = None,
    audio_renderer: AudioRenderer | None = None,
    candidate_adapter: CandidateAdapter | None = None,
    visual_verifier: VisualVerifier | None = None,
    replace: bool = False,
) -> BackendHandler:
    """Register and return a backend handler.

    The object form is convenient for callers that already have a
    `BackendHandler`.  The keyword form also supports small test or plugin
    registrations without requiring a subclass.  Replacing an existing
    registration is explicit so a typo cannot silently change a route.
    """

    if handler is not None:
        if any(
            value is not None
            for value in (
                backend_id,
                description,
                capture,
                audio_renderer,
                candidate_adapter,
                visual_verifier,
            )
        ):
            raise BackendHandlerError(
                "handler object cannot be combined with handler fields"
            )
        selected = handler
    else:
        if backend_id is None or description is None:
            raise BackendHandlerError(
                "backend_id and description are required for a new handler"
            )
        selected = BackendHandler(
            backend_id=backend_id,
            description=description,
            capture=capture,
            audio_renderer=audio_renderer,
            candidate_adapter=candidate_adapter,
            visual_verifier=visual_verifier,
        )
    if selected.backend_id in _REGISTRY and not replace:
        raise BackendHandlerError(
            f"backend handler is already registered: {selected.backend_id!r}"
        )
    _REGISTRY[selected.backend_id] = selected
    return selected


def registered_backend_handlers() -> dict[str, BackendHandler]:
    """Return a shallow copy of the configured handler registry."""

    return dict(_REGISTRY)


def resolve_backend_id(
    config: Mapping[str, Any] | str | None = None,
    *,
    default: str = DEFAULT_BACKEND_ID,
) -> str:
    """Resolve a backend id from configuration without looking at scene ids.

    `backend` is accepted as a compatibility spelling for older candidate
    records.  If it is an object, its `backend_id` or `id` field is used.
    An absent backend keeps the historical `ue_spear` route.
    """

    _validate_backend_id(default)
    if config is None:
        return default
    if isinstance(config, str):
        return _validate_backend_id(config)
    if not isinstance(config, Mapping):
        raise BackendHandlerError("backend configuration must be an object or string")
    primary: Any = config.get("backend_id")
    compatibility: Any = config.get("backend")
    if isinstance(primary, Mapping):
        primary = primary.get("backend_id", primary.get("id"))
    if isinstance(compatibility, Mapping):
        compatibility = compatibility.get(
            "backend_id", compatibility.get("id")
        )
    if primary is not None and compatibility is not None:
        primary_id = _validate_backend_id(primary)
        primary_id = _BACKEND_ALIASES.get(primary_id, primary_id)
        compatibility_id = _validate_backend_id(compatibility)
        compatibility_id = _BACKEND_ALIASES.get(
            compatibility_id, compatibility_id
        )
        if primary_id != compatibility_id:
            raise BackendHandlerError(
                "backend_id and backend declare different handlers"
            )
        return primary_id
    value = primary if primary is not None else compatibility
    if value is None:
        return default
    backend_id = _validate_backend_id(value)
    return _BACKEND_ALIASES.get(backend_id, backend_id)


def get_backend_handler(
    config: Mapping[str, Any] | str | None = None,
    *,
    default: str = DEFAULT_BACKEND_ID,
) -> BackendHandler:
    """Return the registered handler selected by explicit configuration."""

    backend_id = resolve_backend_id(config, default=default)
    try:
        return _REGISTRY[backend_id]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise BackendHandlerError(
            f"unknown backend_id {backend_id!r}; registered handlers: {known}"
        ) from exc


# Short aliases keep integration code readable while retaining one registry
# implementation.  They are also useful to callers that use "select" rather
# than "get" terminology.
select_backend_handler = get_backend_handler
resolve_handler = get_backend_handler


register_backend_handler(
    BackendHandler(
        backend_id=DEFAULT_BACKEND_ID,
        description="Native UE/SPEAR visual execution",
    )
)
register_backend_handler(
    BackendHandler(
        backend_id=HABITAT_NATIVE_BACKEND_ID,
        description="Native Habitat-Sim MP3D visual execution",
        capture=_lazy_mp3d_capture,
        audio_renderer=_lazy_mp3d_audio_renderer,
        candidate_adapter=_lazy_mp3d_candidate_adapter,
        visual_verifier=_lazy_mp3d_visual_verifier,
    )
)


__all__ = [
    "BackendHandler",
    "BackendHandlerError",
    "DEFAULT_BACKEND_ID",
    "HABITAT_NATIVE_BACKEND_ID",
    "get_backend_handler",
    "register_backend_handler",
    "registered_backend_handlers",
    "resolve_backend_id",
    "resolve_handler",
    "select_backend_handler",
]
