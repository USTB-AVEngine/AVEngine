import unittest
from private_nvidia_egl import exact_version,graphics_name
class PrivateGraphicsBoundary(unittest.TestCase):
 def test_open_kernel_version(self):
  self.assertEqual(exact_version('NVRM NVIDIA UNIX Open Kernel Module for x86_64  580.95.05 Release Build','580.95.05'),'580.95.05')
 def test_no_driver_mismatch(self):
  with self.assertRaises(ValueError):exact_version('NVIDIA Kernel Module 580.95.05','580.105.08')
 def test_no_ambiguous_version(self):
  with self.assertRaises(ValueError):exact_version('580.95.05 580.105.08','580.95.05')
 def test_graphics_only(self):
  self.assertTrue(graphics_name('libEGL_nvidia.so.580.95.05','580.95.05'))
  for x in ['libcuda.so.580.95.05','libnvidia-ml.so.580.95.05','nvidia-installer','libEGL_nvidia.so.580.105.08','libEGL_nvidia.so.0']:
   self.assertFalse(graphics_name(x,'580.95.05'))
