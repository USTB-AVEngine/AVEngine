"""Recognize GPU OOM from PyTorch and CUDA extensions before retrying a smaller reconstruction."""
def is_cuda_oom(error):
 message=str(error).lower()
 return (type(error).__name__=='OutOfMemoryError' and type(error).__module__.startswith('torch')) or ('out of memory' in message and any(word in message for word in ['cuda','cumesh','cublas']))
