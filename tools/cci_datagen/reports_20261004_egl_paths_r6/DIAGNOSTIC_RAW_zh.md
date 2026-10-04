以下为现有实测日志原文摘录。完整原文以附带证据快照中的相同文件为准；P19 第六轮补库/新版 Mesa 未获 rank 时没有对应运行日志，不能伪造输出。

原路径 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_egl_paths_20261004_codex_r6/evidence/devbox/driver_package_assembly_v2/driver_extract_child.log`。文件字节数 1447，SHA256 `c89ea6a96d9a24b08c52e50e4a8231e88bd1d7f9d47a855ceade6909d8f59905`。
```text
Creating directory /mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_egl_paths_20261004_codex_r6/driver_runtime_verified/580.95.05/extract_quarantine/payload
Verifying archive integrity... OK
Uncompressing NVIDIA Accelerated Graphics Driver for Linux-x86_64 580.95.05............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
```

原路径 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_egl_paths_20261004_codex_r6/evidence/devbox/driver_package_assembly_v2/driver_tar_raw.log`。文件字节数 35，SHA256 `9536819cf0d8eb110e98d20e4a9b4a025acda7fc0a3c3f5e48d6cfbfb21cd570`。
```text
private extraction tar argv: xvf -
```

原路径 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_egl_diagnostic_20261004_codex_r5/output/scheduler_diagnostic/egl_diagnostic_r5/eval_job_a01_job_20261004T035951Z_b69cc9-t8wb6/inherited_devices.log`。文件字节数 635，SHA256 `dc2cc4baff2b69c9a4576f0f6afad6e4b0fe09006db8aee09a12485136a068da`。
```text
{"library": "/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/private_glvnd/libEGL.so.1", "assigned_gpu": "0", "MIG": false, "hardware_context_created": false, "client_extensions": "", "client_query_error": "0x3000", "device_count": 0, "enumeration_note": "client does not advertise EGL_EXT_device_enumeration", "devices": [], "default_display_handle": "0x0", "default_display_error": "0x300c", "default_display_initialized": false, "selected_leased_device": null, "device_initialize_note": "MIG graphics excluded or no unique CUDA-ID match; no other physical device initialized", "status": "metadata_complete"}
```

原路径 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_egl_diagnostic_20261004_codex_r5/output/scheduler_diagnostic/egl_diagnostic_r5/eval_job_a01_job_20261004T035951Z_b69cc9-t8wb6/EGL_PLATFORM_device_devices.log`。文件字节数 635，SHA256 `dc2cc4baff2b69c9a4576f0f6afad6e4b0fe09006db8aee09a12485136a068da`。
```text
{"library": "/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/private_glvnd/libEGL.so.1", "assigned_gpu": "0", "MIG": false, "hardware_context_created": false, "client_extensions": "", "client_query_error": "0x3000", "device_count": 0, "enumeration_note": "client does not advertise EGL_EXT_device_enumeration", "devices": [], "default_display_handle": "0x0", "default_display_error": "0x300c", "default_display_initialized": false, "selected_leased_device": null, "device_initialize_note": "MIG graphics excluded or no unique CUDA-ID match; no other physical device initialized", "status": "metadata_complete"}
```

原路径 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_egl_diagnostic_20261004_codex_r5/output/scheduler_diagnostic/egl_diagnostic_r5/eval_job_a01_job_20261004T035951Z_b69cc9-t8wb6/inherited_habitat.log`。文件字节数 3930，SHA256 `4701bc8f3fb7342a6609d6e560f0afdc1bcb219ca28f7c69f9bf05a70cdd1d3b`。
```text
{"status": "blocked", "returncode": 1, "stdout": "{\"runtime\": {\"import_workaround\": {\"workaround_id\": \"habitat_import_quaternion_first_v1\", \"required_import_order\": [\"quaternion\", \"habitat_sim\"], \"reason\": \"the pinned baseline can abort with an invalid free when habitat_sim is imported before numpy-quaternion\"}, \"quaternion_module\": {\"path\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/env/lib/python3.12/site-packages/quaternion/__init__.py\", \"version\": \"2024.0.13\"}, \"habitat_sim_module\": {\"path\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z/habitat_sim/__init__.py\", \"version\": \"0.3.3\"}, \"binding_api\": \"habitat_sim.RLRAcousticContext_v1\", \"installed_habitat_runtime\": {\"prefix\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z\", \"binding_path\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z/habitat_sim/_ext/habitat_sim_bindings.cpython-312-x86_64-linux-gnu.so\"}, \"external_rlr_sdk\": {\"root\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg\", \"header\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg/headers/RLRAudioPropagation.h\", \"library\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so\"}, \"runtime_mode\": \"current-installed\", \"runtime_identity\": {\"identity_schema\": \"avengine_current_installed_rlr_runtime_v1\", \"mode\": \"current-installed\", \"habitat_runtime_prefix\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z\", \"habitat_sim_module\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z/habitat_sim/__init__.py\", \"habitat_sim_binding\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z/habitat_sim/_ext/habitat_sim_bindings.cpython-312-x86_64-linux-gnu.so\", \"magnum_python_site\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages\", \"rlr_sdk_root\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg\", \"rlr_sdk_header\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg/headers/RLRAudioPropagation.h\", \"rlr_sdk_library\": \"/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex/native/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so\", \"rlr_adapter_enabled\": true, \"binding_api\": \"habitat_sim.RLRAcousticContext_v1\"}}, \"rlr_available\": true}\n", "stderr": "Platform::WindowlessEglApplication::tryCreateContext(): cannot get default EGL display: EGL_BAD_PARAMETER\nWindowlessContext: Unable to create windowless context\n", "wall_seconds": 7.7506013214588165, "mode": "egl_render", "libraries": {"EGL": null, "OpenGL": null, "GLdispatch": null, "cuda": "libcuda.so.1"}}
```

原路径 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_egl_paths_20261004_codex_r6/evidence/devbox/condaforge_mesa_v2/software_30s.log`。文件字节数 80725，SHA256 `797a08c5a497c405268ab416a0d0b6314fcf025e98f4a0c215d8d78270de1878`。
```text
AVEngine CPU EGL: software devices=1; hardware excluded=0
AVEngine CPU EGL: software devices=1; hardware excluded=0
MESA-EGL: warning: Not allowed to force software rendering when API explicitly selects a hardware device.
Renderer: llvmpipe (LLVM 23.1.2, 256 bits) by Mesa
OpenGL version: 4.6 (Core Profile) Mesa 26.2.4
```

原路径 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_egl_diagnostic_20261004_codex_r5/output/scheduler_diagnostic/egl_diagnostic_r5/eval_job_a01_job_20261004T035951Z_b69cc9-t8wb6/software_sample.log`。文件字节数 86773，SHA256 `b52daedaaf1b47cee31cd8d2052d87f76b51158d0fb76665ba566154a6febfd5`。
```text
AVEngine CPU EGL: software devices=1; hardware excluded=0
AVEngine CPU EGL: software devices=1; hardware excluded=0
Renderer: llvmpipe (LLVM 15.0.7, 256 bits) by Mesa
OpenGL version: 4.5 (Core Profile) Mesa 23.2.1-1ubuntu3.1~22.04.4
```

nvdiffrast CUDA：本轮按任务要求仅评估，没有安装、编译或运行；没有可提交的 CUDA 渲染实测原文。

原文输出与计算后的时间表分开保存。software_30s/RESULT.json 与 SAME_VERSION_48_COMPARE.json 包含全量计时、实际加载 DSO 路径以及原始数组哈希，不使用 MP4 编码哈希判定像素一致。
