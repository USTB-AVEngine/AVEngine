"""Compare WAV payloads without libsndfile's PEAK creation timestamp.

This hashes a canonical view; source files are never edited. Whole-file hashes
remain the delivery commitments and must also be reported separately.
"""
import hashlib
from pathlib import Path
import struct

def canonical_wav_sha256(path):
    data=bytearray(Path(path).read_bytes())
    if data[:4]!=b'RIFF' or data[8:12]!=b'WAVE':raise ValueError('expected little-endian RIFF WAV')
    pos=12;timestamps=[]
    while pos+8<=len(data):
        tag,n=struct.unpack_from('<4sI',data,pos)
        if pos+8+n>len(data):raise ValueError('truncated WAV chunk')
        if tag==b'PEAK':
            if n<8:raise ValueError('invalid PEAK chunk')
            timestamps.append(struct.unpack_from('<I',data,pos+12)[0])
            data[pos+12:pos+16]=b'\x00'*4
        pos+=8+n+(n%2)
    return {'canonical_sha256':hashlib.sha256(data).hexdigest(),'peak_timestamps':timestamps,'normalization':'PEAK creation timestamp only; source unchanged'}
