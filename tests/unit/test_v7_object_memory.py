"""Regression: CuMesh CUDA allocator failures must use the one reduced-resolution retry."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/assets'))
from v7_object_memory import is_cuda_oom

def test_cumesh_allocator_oom_is_retryable():
 assert is_cuda_oom(RuntimeError('[CuMesh] CUDA error: Error code: 2 Error text: out of memory'))

@pytest.mark.parametrize('error',[MemoryError('CPU out of memory'),RuntimeError('No module named flash_attn'),ValueError('invalid dimensions')])
def test_unrelated_errors_do_not_consume_gpu_oom_retry(error):
 assert not is_cuda_oom(error)
