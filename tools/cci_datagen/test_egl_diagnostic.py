import unittest
from egl_device_diag import selected_device
class DiagnosticBoundary(unittest.TestCase):
 def test_only_exact_unique_leased_device(self):
  devices=[{'egl_index':0,'cuda_device':0},{'egl_index':1,'cuda_device':2}]
  self.assertEqual(selected_device(devices,'2')['egl_index'],1)
  self.assertIsNone(selected_device(devices,'3'))
 def test_MIG_cannot_borrow_physical_parent(self):
  self.assertIsNone(selected_device([{'egl_index':0,'cuda_device':0}],'MIG-devbox'))
 def test_ambiguous_mapping_fails_closed(self):
  self.assertIsNone(selected_device([{'cuda_device':0},{'cuda_device':0}],'0'))
