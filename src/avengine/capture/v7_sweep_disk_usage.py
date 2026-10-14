"""Measure own sweep output despite concurrently removed private temporary files."""
from pathlib import Path
import os,subprocess


def output_disk_bytes(roots):
    roots=[Path(p).resolve()for p in roots]
    if not roots:raise ValueError('at least one owned output root required')
    env={**os.environ,'LC_ALL':'C'}
    result=subprocess.run(['du','-s','-B1',*[str(p)for p in roots]],text=True,capture_output=True,env=env)
    if result.returncode:
        errors=result.stderr.splitlines()
        if result.returncode!=1 or not errors:raise RuntimeError(result.stderr or 'du failed')
        # du still prints complete root totals when a private temporary disappears.
        # Other errors (permissions, I/O, a missing root) remain fatal.
        prefix="du: cannot access '";suffix="': No such file or directory"
        for line in errors:
            if not(line.startswith(prefix)and line.endswith(suffix)):raise RuntimeError(result.stderr)
            vanished=Path(line[len(prefix):-len(suffix)])
            if not any(vanished!=root and vanished.is_relative_to(root)for root in roots):raise RuntimeError(result.stderr)
    values={Path(line.split('\t',1)[1]).resolve():int(line.split('\t',1)[0])for line in result.stdout.splitlines()}
    if set(values)!=set(roots):raise RuntimeError('du did not return every owned root total')
    return sum(values.values())
