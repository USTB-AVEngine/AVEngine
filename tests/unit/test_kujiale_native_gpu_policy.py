import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from avengine.capture import kujiale_native_gpu_policy as gpu

pytestmark = pytest.mark.fast_unit


def policy(root):
    (root/gpu.POLICY_FILENAME).write_text(json.dumps({
        "schema":"kujiale_native_gpu_policy_v1", "allowed_gpu_indices":[2],
        "measured_peak_mib":3812, "headroom_mib":4096,
        "minimum_disk_free_bytes":50_000_000_000}))


def xml(free):
    return '<nvidia_smi_log>'+''.join(
        f'<gpu><uuid>card-{i}</uuid><fb_memory_usage><free>{n} MiB</free></fb_memory_usage></gpu>'
        for i,n in enumerate(free))+'</nvidia_smi_log>'


@pytest.mark.parametrize("card", [0,1,3])
def test_reserved_cards_rejected_before_native_import_or_child_launch(tmp_path, monkeypatch, card):
    policy(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("no subprocess or engine can start on a reserved card")
    monkeypatch.setattr(gpu.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="reserved"):
        gpu.native_gpu_preflight(tmp_path, card)
    from avengine.capture.kujiale_spear_runtime import launch_native
    with pytest.raises(ValueError, match="reserved"):
        launch_native({"output_root":str(tmp_path),"uproject":str(tmp_path/'private/project.uproject'),
                       "spear_ext_dir":str(tmp_path/'missing-sdk'),"graphics_adapter":card}, '/Game/Test')


def test_guard_rejects_old_gpu1_request_before_run_directory_or_process(tmp_path, monkeypatch):
    policy(tmp_path)
    from tools.rooms import v7_kujiale_ue_guard as guard
    request=tmp_path/'old.json';run=tmp_path/'must-not-start'
    request.write_text(json.dumps({"budget_root":str(tmp_path),"run_directory":str(run),"gpu_index":1}))
    monkeypatch.setattr('sys.argv', ['guard','--request',str(request)])
    with pytest.raises(ValueError,match="reserved"):
        guard.main()
    assert not run.exists()


def test_primary_gpu2_live_free_memory_is_used_not_other_empty_cards(tmp_path, monkeypatch):
    policy(tmp_path)
    monkeypatch.setattr(gpu.subprocess,"run",lambda *a,**kw:SimpleNamespace(stdout=xml([48520,48520,7000,48520])))
    with pytest.raises(RuntimeError,match="peak plus 4GiB"):
        gpu.native_gpu_preflight(tmp_path,2)


def test_gpu2_exact_measured_peak_plus_4g_and_disk_floor(tmp_path,monkeypatch):
    policy(tmp_path)
    monkeypatch.setattr(gpu.subprocess,"run",lambda *a,**kw:SimpleNamespace(stdout=xml([0,0,7908,0])))
    monkeypatch.setattr(gpu.shutil,"disk_usage",lambda p:SimpleNamespace(free=50_000_000_000))
    result=gpu.native_gpu_preflight(tmp_path,2)
    assert result['gpu_index']==2 and result['gpu_uuid']=='card-2' and result['required_free_mib']==7908
    monkeypatch.setattr(gpu.shutil,"disk_usage",lambda p:SimpleNamespace(free=49_999_999_999))
    with pytest.raises(RuntimeError,match="disk"):
        gpu.native_gpu_preflight(tmp_path,2)


def test_no_policy_preserves_other_campaigns_without_gpu_query(tmp_path,monkeypatch):
    monkeypatch.setattr(gpu.subprocess,"run",lambda *a,**kw:pytest.fail("unconfigured campaign behavior must be unchanged"))
    assert gpu.native_gpu_preflight(tmp_path,1) is None


def test_invalid_policy_cannot_reduce_required_headroom(tmp_path):
    policy(tmp_path);p=tmp_path/gpu.POLICY_FILENAME;x=json.loads(p.read_text());x['headroom_mib']=1024;p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match="at least 4GiB"):
        gpu.check_gpu_policy(tmp_path,2)


def test_isolated_visible_card_keeps_physical_uuid_reservation(tmp_path, monkeypatch):
    policy(tmp_path)
    path = tmp_path/gpu.POLICY_FILENAME
    value = json.loads(path.read_text())
    value["gpu_uuids"] = {"2": "card-2"}
    path.write_text(json.dumps(value))
    # A device namespace exposes only physical GPU 2 at visible ordinal zero.
    one_card = ('<nvidia_smi_log><gpu><uuid>card-2</uuid>'
                '<fb_memory_usage><free>7908 MiB</free></fb_memory_usage>'
                '</gpu></nvidia_smi_log>')
    monkeypatch.setattr(gpu.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(stdout=one_card))
    monkeypatch.setattr(gpu.shutil, "disk_usage",
                        lambda p: SimpleNamespace(free=50_000_000_000))
    result = gpu.native_gpu_preflight(tmp_path, 2)
    assert result["gpu_index"] == 2 and result["gpu_uuid"] == "card-2"
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout=one_card.replace("card-2", "card-0")))
    with pytest.raises(RuntimeError, match="UUID"):
        gpu.native_gpu_preflight(tmp_path, 2)
