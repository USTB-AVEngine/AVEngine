"""Read Vulkan device UUIDs in the current task's GPU device namespace."""
from __future__ import annotations


def enumerate_vulkan_devices():
    import ctypes as c

    class App(c.Structure):
        _fields_ = [('sType', c.c_int), ('pNext', c.c_void_p),
                    ('pApplicationName', c.c_char_p), ('applicationVersion', c.c_uint),
                    ('pEngineName', c.c_char_p), ('engineVersion', c.c_uint),
                    ('apiVersion', c.c_uint)]

    class Create(c.Structure):
        _fields_ = [('sType', c.c_int), ('pNext', c.c_void_p), ('flags', c.c_uint),
                    ('pApplicationInfo', c.POINTER(App)), ('enabledLayerCount', c.c_uint),
                    ('ppEnabledLayerNames', c.c_void_p), ('enabledExtensionCount', c.c_uint),
                    ('ppEnabledExtensionNames', c.c_void_p)]

    class DeviceID(c.Structure):
        _fields_ = [('sType', c.c_int), ('pNext', c.c_void_p),
                    ('deviceUUID', c.c_ubyte * 16), ('driverUUID', c.c_ubyte * 16),
                    ('deviceLUID', c.c_ubyte * 8), ('deviceNodeMask', c.c_uint),
                    ('deviceLUIDValid', c.c_uint)]

    class Properties(c.Structure):
        _fields_ = [('sType', c.c_int), ('pNext', c.c_void_p),
                    ('data', c.c_ubyte * 4096)]

    loader = c.CDLL('libvulkan.so.1')
    loader.vkCreateInstance.argtypes = [c.POINTER(Create), c.c_void_p, c.POINTER(c.c_void_p)]
    loader.vkEnumeratePhysicalDevices.argtypes = [
        c.c_void_p, c.POINTER(c.c_uint), c.POINTER(c.c_void_p)]
    loader.vkGetPhysicalDeviceProperties2.argtypes = [c.c_void_p, c.POINTER(Properties)]
    loader.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
    app = App(sType=0, pApplicationName=b'AVEngineGPUReadback',
              apiVersion=(1 << 22) | (1 << 12))
    create = Create(sType=1, pApplicationInfo=c.pointer(app))
    instance = c.c_void_p()
    if loader.vkCreateInstance(c.byref(create), None, c.byref(instance)) != 0:
        raise ValueError('cannot enumerate native Vulkan GPU UUIDs')
    try:
        count = c.c_uint()
        if loader.vkEnumeratePhysicalDevices(instance, c.byref(count), None) != 0:
            raise ValueError('cannot count native Vulkan GPUs')
        devices = (c.c_void_p * count.value)()
        if loader.vkEnumeratePhysicalDevices(instance, c.byref(count), devices) != 0:
            raise ValueError('cannot read native Vulkan GPUs')
        rows = []
        for index, device in enumerate(devices):
            identity = DeviceID(sType=1000071004)
            properties = Properties(sType=1000059001, pNext=c.addressof(identity))
            loader.vkGetPhysicalDeviceProperties2(device, c.byref(properties))
            raw = bytes(identity.deviceUUID).hex()
            uuid = 'GPU-' + '-'.join((raw[:8], raw[8:12], raw[12:16],
                                    raw[16:20], raw[20:]))
            rows.append(dict(vulkan_index=index, gpu_uuid=uuid,
                             device_name=bytes(properties.data)[20:276].split(b'\0', 1)[0].decode()))
        return rows
    finally:
        loader.vkDestroyInstance(instance, None)
