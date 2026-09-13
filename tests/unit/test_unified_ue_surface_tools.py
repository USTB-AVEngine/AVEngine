"""External UE package validation preserves provenance and byte checks."""
import importlib.util
import sys
from pathlib import Path
import pytest


def load_tool(name):
    path=Path(__file__).resolve().parents[2]/'tools/rooms'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_manifest(tmp_path, module):
    project=tmp_path/'Project'
    package=project/'Content/Room/Wall.uasset'
    package.parent.mkdir(parents=True)
    package.write_bytes(b'exported wall package')
    obj='/Game/Room/Wall.Wall'
    manifest={'selected_project_asset_package_count':1,
              'selected_project_asset_packages':[{'package_name':'/Game/Room/Wall',
                'repository_relative_path':'Content/Room/Wall.uasset','resolved_path':str(package),
                'byte_size':package.stat().st_size,'sha256':module.sha256_file(package),
                'git_tracked':False,'source_kind':'external_runtime_package','asset_object_paths':[obj]}],
              'actors':[{'static_mesh_components':[{'static_mesh_asset':obj,'material_assets':[]}]}]}
    return project, package, manifest


def test_external_package_is_read_back_and_detects_a_changed_mesh(tmp_path):
    module=load_tool('prepare_legacy_apartment')
    project, package, manifest=fixture_manifest(tmp_path,module)
    root=Path(__file__).resolve().parents[2]
    module.validate_selected_project_packages(manifest,root,project_dir=project)
    package.write_bytes(b'another package')
    with pytest.raises(ValueError,match='bytes changed'):
        module.validate_selected_project_packages(manifest,root,project_dir=project)


def test_external_manifest_cannot_omit_a_selected_material(tmp_path):
    module=load_tool('prepare_legacy_apartment')
    project, _, manifest=fixture_manifest(tmp_path,module)
    manifest['actors'][0]['static_mesh_components'][0]['material_assets']=['/Game/Room/Mat.Mat']
    with pytest.raises(ValueError,match='exactly closed'):
        module.validate_selected_project_packages(manifest,Path(__file__).resolve().parents[2],project_dir=project)


def test_external_package_cannot_claim_git_tracking(tmp_path):
    module=load_tool('prepare_legacy_apartment')
    project, _, manifest=fixture_manifest(tmp_path,module)
    manifest['selected_project_asset_packages'][0]['git_tracked']=True
    with pytest.raises(ValueError,match='tracking claim'):
        module.validate_selected_project_packages(manifest,Path(__file__).resolve().parents[2],project_dir=project)
