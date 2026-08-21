"""AVEngine-local host/game SPEAR client namespace.

This selected adaptation is from spear-sim/spear at
251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7. It does not provide a global
spear package or a spear_ext compatibility alias. The optional native extension
uses the AVEngine-local module name avengine_spear_ext.
"""

from __future__ import annotations

_EXTENSION_MODULE = "avengine_spear_ext"

try:
    import avengine_spear_ext
except ModuleNotFoundError as error:
    if error.name != _EXTENSION_MODULE:
        raise
    avengine_spear_ext = None
    __can_import_avengine_spear_ext__ = False
else:
    __can_import_avengine_spear_ext__ = True

# Only host/game Python is staged here; upstream editor helpers are excluded.
__can_import_unreal__ = False


def require_native_extension():
    """Return the native RPC client or fail at the first operation that needs it."""

    if not __can_import_avengine_spear_ext__:
        raise RuntimeError(
            "AVEngine's SPEAR host/game client requires the optional "
            "'avengine_spear_ext' native extension when Instance is constructed. "
            "Build and install AVEngine's SPEAR extension before starting a "
            "SPEAR/UE instance."
        )
    return avengine_spear_ext


if __can_import_avengine_spear_ext__:
    DataBundle = avengine_spear_ext.DataBundle
    PackedArray = avengine_spear_ext.PackedArray

from .utils.config_utils import get_config
from .utils.func_utils import CallSyncEntryPointCaller, EditorEntryPointCaller, Future, PropertyValue, Service, Shared, from_script_result, to_array, to_arrays, to_data_bundle, to_data_bundle_dict, to_handle, to_handle_or_unreal_class, to_handle_or_unreal_object, to_handle_or_unreal_struct, to_json_string, to_json_strings, to_packed_array, to_packed_arrays, to_ptr, to_script_expr, to_script_struct_expr, to_shared, to_unreal_class, to_unreal_object, to_unreal_struct, try_to_dict, try_to_dicts
from .utils.log_utils import get_default_log_enabled, log, log_current_function, log_get_prefix, log_no_prefix, register_log_func, set_default_log_enabled, unregister_log_func
from .utils.system_utils import configure_system
from .services.async_loading_service import AsyncLoadingService
from .services.debug_service import DebugService
from .services.engine_globals_service import EngineGlobalsService
from .services.engine_service import EngineService
from .services.enhanced_input_service import EnhancedInputService
from .services.input_service import InputService
from .services.navigation_service import NavigationService
from .services.python_service import PythonService
from .services.rendering_service import EditorRenderingService, GameRenderingService, RenderingService
from .services.segmentation_service import SegmentationService
from .services.shared_memory_service import SharedMemoryService
from .services.sp_func_service import SpFuncService
from .services.unreal_service import UnrealService
from .services.world_registry_service import WorldRegistryService
from .unreal_object import UnrealClass, UnrealObject, UnrealStruct
from .utils import rendering_utils as rendering


def __getattr__(name: str):
    if name == "Instance":
        from .instance import Instance

        return Instance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AsyncLoadingService", "CallSyncEntryPointCaller", "DebugService",
    "EditorEntryPointCaller", "EditorRenderingService", "EngineGlobalsService",
    "EngineService", "EnhancedInputService", "Future", "GameRenderingService",
    "InputService", "Instance", "NavigationService", "PropertyValue",
    "PythonService", "RenderingService", "SegmentationService", "Service",
    "Shared", "SharedMemoryService", "SpFuncService", "UnrealClass",
    "UnrealObject", "UnrealService", "UnrealStruct", "WorldRegistryService",
    "avengine_spear_ext", "configure_system", "from_script_result", "get_config",
    "get_default_log_enabled", "log", "log_current_function", "log_get_prefix",
    "log_no_prefix", "register_log_func", "rendering", "require_native_extension",
    "set_default_log_enabled", "to_array", "to_arrays", "to_data_bundle",
    "to_data_bundle_dict", "to_handle", "to_handle_or_unreal_class",
    "to_handle_or_unreal_object", "to_handle_or_unreal_struct", "to_json_string",
    "to_json_strings", "to_packed_array", "to_packed_arrays", "to_ptr",
    "to_script_expr", "to_script_struct_expr", "to_shared", "to_unreal_class",
    "to_unreal_object", "to_unreal_struct", "try_to_dict", "try_to_dicts",
]
if __can_import_avengine_spear_ext__:
    __all__ += ["DataBundle", "PackedArray"]
