"""Lossless-witness, direct-reader closure, tamper and version-binding regressions."""
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf
from avengine.qa.frozen_panel_retention import (encode_file, restore_file, hydrated_group,
                                                RetainedSoundFile, sha)
from avengine.qa.frozen_panel_versions import parent_hash, prepared_parent, validate_versions

pytestmark = pytest.mark.fast_unit


@pytest.mark.parametrize('array', [np.full((7, 3, 4), 1.25, '<f4'),
                                    np.arange(60, dtype='<i4').reshape(5, 3, 4),
                                    np.asfortranarray(np.arange(20).reshape(4, 5)),
                                    np.zeros((3, 0), '<f8')])
def test_exact_npy_bytes_without_original_input(tmp_path, array):
    (tmp_path/'retention').mkdir()
    source=tmp_path/'observed.npy';np.save(source,array)
    record=encode_file(tmp_path/'retention',source);expected=sha(source)
    source.unlink()  # Only this test's disposable fixture.
    restored=tmp_path/'restored.npy';restore_file(tmp_path,record,restored)
    assert sha(restored)==expected
    assert np.array_equal(np.load(restored),array)


def test_corrupt_constant_witness_is_rejected(tmp_path):
    (tmp_path/'retention').mkdir();source=tmp_path/'original.npy';np.save(source,np.ones((3,4),'<f4'))
    record=encode_file(tmp_path/'retention',source)
    record['sha256']='0'*64
    with pytest.raises(ValueError,match='differs from pre-prune'):
        restore_file(tmp_path,record,tmp_path/'restored.npy')


def test_varying_alpha_cannot_be_discarded(tmp_path):
    (tmp_path/'retention').mkdir();source=tmp_path/'rgb.npy'
    rgb=np.zeros((2,3,4,4),np.uint8);rgb[1,1,1,3]=255;np.save(source,rgb)
    with pytest.raises(ValueError,match='varying alpha'):
        encode_file(tmp_path/'retention',source,public_video='groups/random/video.mp4')


def test_lossless_rgba_restored_from_public_video_and_clock_checked(tmp_path):
    (tmp_path/'retention').mkdir();(tmp_path/'public').mkdir()
    rng=np.random.default_rng(73);rgb=rng.integers(0,256,(3,8,10,4),dtype=np.uint8);rgb[...,3]=255
    source=tmp_path/'rgb.npy';np.save(source,rgb)
    video=tmp_path/'public'/'video.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','10x8','-r','15','-i','pipe:0',
       '-frames:v','3','-c:v','libx264rgb','-crf','0','-threads','1',str(video)],input=rgb[...,:3].tobytes(),check=True)
    record=encode_file(tmp_path/'retention',source,public_video='video.mp4');source.unlink()
    restore_file(tmp_path,record,tmp_path/'restored.npy')
    assert np.array_equal(np.load(tmp_path/'restored.npy'),rgb)
    altered=copy.deepcopy(record);altered['shape'][0]=4
    with pytest.raises(ValueError,match='clock/decode'):
        restore_file(tmp_path,altered,tmp_path/'wrongclock.npy')


def test_direct_soundfile_dependency_reads_are_closed(tmp_path):
    root=tmp_path/'panel';(root/'retention').mkdir(parents=True)
    source=tmp_path/'external.wav';sf.write(source,np.arange(40,dtype='float32')/100,16000,subtype='FLOAT')
    expected,_=sf.read(source,dtype='float32');record=encode_file(root/'retention',source)
    manifest={'production_root':str(root/'production/g'), 'native_files':[], 'dependencies':[record]}
    source.unlink()  # The native verifier's string path now really does not exist.
    with hydrated_group(root,manifest) as resolver:
        proxy=RetainedSoundFile(sf,resolver);actual,rate=proxy.read(str(source),dtype='float32')
        assert rate==16000 and np.array_equal(actual,expected)
        assert proxy.reads[0]['retained']!=str(source)
        with pytest.raises(ValueError,match='unretained external'):
            proxy.read(str(tmp_path/'undeclared.wav'))
        with pytest.raises(ValueError,match='undeclared native parent'):
            resolver(root/'production/other/recipe.json')


def test_public_media_cannot_escape_retained_root(tmp_path):
    (tmp_path/'public').mkdir();(tmp_path/'outside').write_bytes(b'fixture')
    with pytest.raises(ValueError,match='escaped root'):
        restore_file(tmp_path,{'kind':'public_media','public_media':'../outside'},tmp_path/'restored')


def test_pending_formal_version_stops_before_opening_external_files():
    with pytest.raises(ValueError,match='three accepted'):
        validate_versions({'split':'test'})
    versions={k:{'status':'pending'} for k in ['generator','assets','render_environment']}
    with pytest.raises(ValueError,match='pending: generator'):
        validate_versions({'split':'test','production_versions':versions})


def test_prepared_seed_quota_and_pilot_exclusion(tmp_path):
    house='hm3d_val_fixture';d=tmp_path/house;d.mkdir()
    geometry={'camera_m':[0,1.55,0],'feet_m':[[1,0,0],[-1,0,0]],
              'basis':{'forward':[0,0,-1],'right':[1,0,0],'up':[0,1,0]}}
    row={'ordinal':0,'status':'cpu_preflight_pass','house':house,'room':'R0',
         'quota':{'voice':'same','side':'opposite'},'answers_generated':False,
         'geometry':geometry,'physical_parent_sha256':parent_hash(house,geometry),'scene_seed':456}
    path=d/'prepared.json';path.write_text(json.dumps({'groups':[row]}))
    config={'prepared_geometry_root':str(tmp_path),'groups_per_house':60,
            'houses':[{'house':house,'rooms':[{'room_label':'R0'}]}]}
    assert prepared_parent(config,0)['scene_seed']==456
    excluded=copy.deepcopy(config);excluded['excluded_physical_parent_sha256']=[row['physical_parent_sha256']]
    with pytest.raises(ValueError,match='reuses a pilot'):prepared_parent(excluded,0)
    row['quota']['side']='same_side';path.write_text(json.dumps({'groups':[row]}))
    with pytest.raises(ValueError,match='house/quota'):prepared_parent(config,0)


def pruning_fixture(tmp_path):
    import importlib.util
    tool=Path(__file__).resolve().parents[1]/'tools/qa/compact_frozen_panel.py'
    spec=importlib.util.spec_from_file_location('compact_prune_unit',tool);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'panel';g=root/'production/group';g.mkdir(parents=True)
    native=g/'audit.npy';native.write_bytes(b'disposable test fixture');st=native.stat()
    ledger={'source':str(native),'device':st.st_dev,'inode':st.st_ino,'bytes':st.st_size,'sha256':sha(native)}
    (root/'retention/groups').mkdir(parents=True)
    manifest={'group':'group','production_root':str(g),'certificate_sha256':'fixture', 'original_file_ledger':[ledger]}
    mp=root/'retention/groups/group.json';mp.write_text(json.dumps(manifest))
    (root/'retention/manifest.json').write_text(json.dumps({'groups':['groups/group.json']}))
    proof={'status':'pass','original_native_inputs_read':False,'external_dependency_reads_resolved_to_retained_files':True,
           'groups':[{'group':'group','certificate_sha256':'fixture','three_layers_verified':True}]}
    pp=root/'unit_proof.json';pp.write_text(json.dumps(proof))
    return module,root,native,manifest,mp,proof,pp


def test_prune_requires_closed_dependency_proof(tmp_path):
    module,root,native,manifest,mp,proof,pp=pruning_fixture(tmp_path)
    proof.pop('external_dependency_reads_resolved_to_retained_files');pp.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='passing retained-only'):module.prune(root,pp)
    assert native.read_bytes()==b'disposable test fixture'


def test_prune_refuses_changed_native_file_before_deletion(tmp_path):
    module,root,native,manifest,mp,proof,pp=pruning_fixture(tmp_path)
    native.write_bytes(b'an edited disposable fixture')
    with pytest.raises(ValueError,match='changed after audit'):module.prune(root,pp)
    assert native.exists()


def test_prune_refuses_parent_outside_certified_scope(tmp_path):
    module,root,native,manifest,mp,proof,pp=pruning_fixture(tmp_path)
    manifest['production_root']=str(tmp_path/'unrelated');mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='scope differs'):module.prune(root,pp)
    assert native.exists()


def test_accepted_version_identity_and_asset_tamper(tmp_path):
    repo=tmp_path/'generator';repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    entry=repo/'producer.py';entry.write_text('# disposable fixture entrypoint\n')
    subprocess.run(['git','-C',str(repo),'add','producer.py'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Unit Fixture','-c','user.email=fixture@example.invalid','commit','-q','-m','fixture'],check=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    evidence=tmp_path/'acceptance.json';evidence.write_text('{"status":"pass"}')
    registry=tmp_path/'registry.json';registry.write_text('{"assets":[]}')
    template=tmp_path/'template.json';actors=[{'asset_id':'fixture0','asset_revision':'r1'},{'asset_id':'fixture1','asset_revision':'r1'}]
    template.write_text(json.dumps({'visual_plan':{'actors':actors}}));runtime={'runtime_prefix':'disposable_fixture'}
    common={'status':'accepted','acceptance_report':str(evidence),'acceptance_report_sha256':sha(evidence),'files_sha256':{}}
    versions={'generator':{**common,'repo':str(repo),'revision':commit,'entrypoint':'producer.py','entrypoint_sha256':sha(entry),'stage_contract':'frozen_panel_stages_v1'},
        'assets':{**common,'revision':'fixed-assets-fixture','registry':str(registry),'registry_sha256':sha(registry),
                  'template_plan':str(template),'template_plan_sha256':sha(template),'actor_assets':actors},
        'render_environment':{**common,'revision':'runtime-fixture','machine':'CCI','runtime':runtime,'hostnames':['disposable-host']}}
    c={'split':'test','registry':str(registry),'template_plan':str(template),'runtime':runtime,'production_versions':versions}
    assert validate_versions(c,require_host=False)==str(entry)
    with pytest.raises(ValueError,match='host has not been accepted'):validate_versions(c)
    template.write_text('{"visual_plan":{"actors":[]}}')
    with pytest.raises(ValueError,match='asset input changed'):validate_versions(c,require_host=False)
    template.write_text(json.dumps({'visual_plan':{'actors':actors}}))
    entry.write_text('# changed disposable fixture entrypoint\n')
    with pytest.raises(ValueError,match='tracked modifications'):validate_versions(c,require_host=False)


def test_retained_configuration_binding_survives_assembly_but_refuses_changes(tmp_path):
    from avengine.qa.frozen_panel_retention import configuration_identity,require_configuration_binding
    config={'output':'/fixture/shard_0','split':'test','seed':321,'registry':'/fixture/registry'}
    manifest={'configuration_normalized_sha256':configuration_identity(config)}
    assembled={**config,'output':'/fixture/assembled'}
    require_configuration_binding(tmp_path,assembled,manifest)
    with pytest.raises(ValueError,match='frozen construction'):
        require_configuration_binding(tmp_path,{**assembled,'seed':322},manifest)


def test_legacy_configuration_bound_by_original_sealed_identity(tmp_path):
    from avengine.qa.frozen_panel_retention import require_configuration_binding
    (tmp_path/'inputs').mkdir();(tmp_path/'scoring_private').mkdir()
    c={'output':'/fixture/original','split':'pilot'};p=tmp_path/'inputs/config.json';p.write_text(json.dumps(c))
    identity={'explicit_inputs_sha256':{'/fixture/original/inputs/config.json':sha(p)}}
    (tmp_path/'scoring_private/build_code_identity.json').write_text(json.dumps(identity))
    require_configuration_binding(tmp_path,c,{})
    p.write_text(json.dumps({**c,'split':'test'}))
    with pytest.raises(ValueError,match='original scoring commitment'):
        require_configuration_binding(tmp_path,c,{})


def test_formal_launch_cannot_skip_new_version_canary():
    from avengine.qa.frozen_panel_versions import require_canary_acceptance
    with pytest.raises(ValueError,match='twenty-group canary'):
        require_canary_acceptance({'split':'test','production_versions':{'fixture':'accepted'}})
    require_canary_acceptance({'split':'pilot'})


def test_canary_for_other_versions_cannot_authorize_formal_launch():
    from avengine.qa.frozen_panel_versions import require_canary_acceptance
    with pytest.raises(ValueError,match='versions differ'):
        require_canary_acceptance({'split':'test','production_versions':{'fixture':'new'},
                                  'canary_acceptance':{'production_versions_sha256':'0'*64}})
