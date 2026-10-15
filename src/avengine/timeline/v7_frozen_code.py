"""Verify a committed checkout or an immutable git-archive source snapshot."""
from pathlib import Path
import hashlib,json,subprocess
MARKER='.avengine_source_snapshot.json'

def source_commit(root):
    root=Path(root)
    if (root/MARKER).is_file():return json.loads((root/MARKER).read_text())['commit']
    return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()

def verify_source(root):
    root=Path(root);marker=root/MARKER
    if not marker.is_file():
        return not subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()
    info=json.loads(marker.read_text())
    if info.get('schema')!='avengine-committed-source-snapshot-v1' or not info.get('files'):return False
    for relative,digest in info['files'].items():
        path=root/relative
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():return False
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:return False
    return True
