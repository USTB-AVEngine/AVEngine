"""Known bad transfers/geometry/PID identities must never authorize deletion."""
import copy,hashlib,io,json,os,tarfile
from pathlib import Path
import pytest
from tools.timeline import generate_v7_full_preproduction as producer
from tools.timeline import verify_v7_transferred_shard as receiver
from avengine.assets.mp3d_texture_stage import geometry_digest


def test_good_transfer_and_bad_archive_commit_counts_media():
    ready={'shard_id':3,'source_commit':'a'*40,'archive_SHA256':'b'*64,'archive_bytes':321,'AFS_destination':'/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_full_preproduction1500_20261005/shards/shard_003','requested_parents':50,'qualified_parents':31,'windows':100,'questions':1400}
    good={**ready,'status':'verified','every_file_SHA256_verified':True,'actual_media_readback':{'status':'pass'}}
    assert producer.receipt_matches(good,ready)
    for key,value in [('archive_SHA256','c'*64),('source_commit','d'*40),('windows',99),('questions',1399),('qualified_parents',30),('AFS_destination','/another/task'),('every_file_SHA256_verified',False),('actual_media_readback',{'status':'blocked'})]:
        bad=copy.deepcopy(good);bad[key]=value
        assert not producer.receipt_matches(bad,ready)


def test_whole_geometry_control_including_sparse_and_native_pose():
    doc={'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':8},{'buffer':0,'byteOffset':8,'byteLength':4}], 'accessors':[{'bufferView':0,'componentType':5126,'count':2,'type':'SCALAR','sparse':{'count':1,'indices':{'bufferView':1,'componentType':5121},'values':{'bufferView':0}}}], 'meshes':[{'primitives':[{'attributes':{'POSITION':0}}]}],'nodes':[{'mesh':0,'translation':[0,0,0]}],'scenes':[{'nodes':[0]}],'scene':0}
    payload=b'123456789abc';good=geometry_digest(doc,payload)
    assert geometry_digest(copy.deepcopy(doc),payload)==good
    for i in (0,8):
        bad=bytearray(payload);bad[i]^=1;assert geometry_digest(doc,bad)!=good
    bad=copy.deepcopy(doc);bad['nodes'][0]['translation']=[1,0,0];assert geometry_digest(bad,payload)!=good


def test_tar_traversal_link_and_unlisted_payload_rejected(tmp_path):
    def archive(name,kind,data=b'abc'):
        path=tmp_path/(name.replace('/','_')+'.tar.gz')
        with tarfile.open(path,'w:gz') as tf:
            m=tarfile.TarInfo(name);m.type=kind;m.size=len(data) if kind==tarfile.REGTYPE else 0
            if kind==tarfile.SYMTYPE:m.linkname='/outside/other_task'
            tf.addfile(m,io.BytesIO(data) if m.size else None)
        return path
    for name,kind in [('../escape',tarfile.REGTYPE),('/absolute',tarfile.REGTYPE),('symlink',tarfile.SYMTYPE)]:
        with pytest.raises(ValueError):receiver.extract_verified(archive(name,kind),tmp_path/('dest_'+str(len(name))))
    assert not (tmp_path.parent/'escape').exists()
    path=tmp_path/'unlisted.tar.gz'
    with tarfile.open(path,'w:gz') as tf:
        for name,data in [('FILE_SHA256.json',b'{}'),('unlisted',b'bad')]:
            m=tarfile.TarInfo(name);m.size=len(data);tf.addfile(m,io.BytesIO(data))
    with pytest.raises(ValueError,match='inventory'):receiver.extract_verified(path,tmp_path/'bad_inventory')


def test_exact_process_identity_never_accepts_unrelated_current_python(tmp_path):
    record={'pid':os.getpid(),'hostname':producer.socket.gethostname(),'boot_id':producer.boot_id(),'process_start':producer.process_start(os.getpid())}
    assert not producer.own_worker_live(record,tmp_path/'profile.json','a_scene')
    bad={**record,'process_start':'0'};assert not producer.own_worker_live(bad,tmp_path/'profile.json','a_scene')
    bad={**record,'boot_id':'wrong_boot'};assert not producer.own_worker_live(bad,tmp_path/'profile.json','a_scene')


def test_restart_extraction_retains_exact_inventory_and_rejects_mutation(tmp_path):
    content=b'opaque capture_by_path/absolute/data/jzy and unmodified answers'
    inventory={'evidence.json':hashlib.sha256(content).hexdigest()};path=tmp_path/'valid.tar.gz'
    with tarfile.open(path,'w:gz') as tf:
        for name,data in [('FILE_SHA256.json',json.dumps(inventory).encode()),('evidence.json',content)]:
            m=tarfile.TarInfo(name);m.size=len(data);tf.addfile(m,io.BytesIO(data))
    dest=tmp_path/'payload';assert receiver.extract_verified(path,dest)==inventory
    assert receiver.extract_verified(path,dest)==inventory
    assert (dest/'evidence.json').read_bytes()==content
    (dest/'evidence.json').write_bytes(b'wrong pose or gold')
    with pytest.raises(ValueError,match='differs'):receiver.extract_verified(path,dest)


def test_worker_scope_rejects_parent_symlink_escape(tmp_path,monkeypatch):
    root=tmp_path/'own';outside=tmp_path/'another_task';root.mkdir();outside.mkdir();(root/'shards').symlink_to(outside,target_is_directory=True)
    monkeypatch.setattr(producer,'ROOT',root)
    with pytest.raises(ValueError,match='escaped'):
        producer.storage_policy({},root/'shards'/'shard_000'/'work'/'parents'/'sid')
