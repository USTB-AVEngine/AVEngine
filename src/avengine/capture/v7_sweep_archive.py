"""Lossless storage of own completed parent intermediates only."""
from pathlib import Path
import json,hashlib,os,gzip,shutil,datetime
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def compact_completed_parent(output, allowed_production_root):
 output=Path(output)
 if not output.resolve().is_relative_to(Path(allowed_production_root).resolve()):raise ValueError('outside own production')
 protected=set()
 for pointer in output.glob('parents/*/*_completed.json'):
  d=json.loads(pointer.read_text());base=pointer.parent/d['attempt']
  for r in d.get('required_files',[]):
   f=base/r
   if not f.is_file():raise ValueError('required artifact missing before compaction')
   protected.add(f)
 before={str(f.relative_to(output)):digest(f) for f in protected}
 compressed=[]
 names={'native_visible_visual_evidence.json','native_registered_visual_evidence.json','native_alpha_visual_evidence.json','native_visual_evidence.json'}
 for vd in output.glob('parents/*/visible_native_attempt*'):
  for f in vd.iterdir():
   if not f.is_file() or f.is_symlink() or f.name not in names or f in protected:continue
   sha=digest(f);target=f.with_name(f.name+'.gz')
   if target.exists():raise ValueError('existing compressed artifact; never overwrite')
   temporary=target.with_name(target.name+'.tmp.'+str(os.getpid()))
   with f.open('rb') as src,temporary.open('xb') as raw:
    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as dst:shutil.copyfileobj(src,dst,1024*1024)
   h=hashlib.sha256()
   with gzip.open(temporary,'rb') as src:
    for block in iter(lambda:src.read(1024*1024),b''):h.update(block)
   if h.hexdigest()!=sha:raise ValueError('lossless intermediate verification failed')
   os.replace(temporary,target);size=f.stat().st_size;f.unlink()
   compressed.append({'original_path':str(f),'gzip_path':str(target),'original_sha256':sha,'original_bytes':size,'compressed_bytes':target.stat().st_size,'not_required_by_any_completed_pointer':True})
 groups={};linked=[]
 for f in sorted(protected):
  sha=before[str(f.relative_to(output))];key=(f.stat().st_size,sha)
  if key not in groups:groups[key]=f;continue
  source=groups[key]
  if f.stat().st_ino==source.stat().st_ino:continue
  temporary=f.with_name(f.name+'.same_bytes.'+str(os.getpid()));os.link(source,temporary);os.replace(temporary,f);linked.append({'path':str(f),'same_bytes_as':str(source),'sha256':sha})
 after={str(f.relative_to(output)):digest(f) for f in protected}
 if before!=after:raise ValueError('required archive bytes changed')
 receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'only own completed parent; preserve every required file byte-for-byte','compressed_optional_native_intermediates':compressed,'identical_required_files_hardlinked':linked,'required_sha256_before_equals_after':True,'required_artifacts':len(protected),'restore_optional_receipt':'gzip -dc original.json.gz > original.json inside this own job, if needed for intermediate replay'}
 (output/'lossless_archive_compaction_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 return {'receipt':str(output/'lossless_archive_compaction_receipt.json'),'compressed_optional_files':len(compressed),'identical_required_files_hardlinked':len(linked),'required_bytes_unchanged':True}
