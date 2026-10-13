import json
from pathlib import Path
import subprocess
import pytest
from avengine.capture.kujiale_pilot_environment import verify_gpu

pytestmark = pytest.mark.fast_unit


@pytest.fixture
def launch(tmp_path, monkeypatch):
    project = str(tmp_path / "private.uproject")
    temporary = tmp_path / "launch"
    temporary.mkdir()
    (temporary / "UnrealEditor.log").write_text(
        "LogVulkanRHI: Using device at index 1 of 3 as specfified by command line...\n")
    (tmp_path / "kujiale_native_gpu_policy.json").write_text(json.dumps({
        "allowed_gpu_indices": [1, 2, 3],
        "gpu_uuids": {"1": "GPU-one", "2": "GPU-two", "3": "GPU-three"}}))
    ue = dict(output_root=str(tmp_path), uproject=project,
              physical_gpu_index=2, graphics_adapter=1)
    receipt = dict(native_ue_pid=12345, temporary_directory=str(temporary),
                   vulkan_devices=[
                       dict(vulkan_index=0, gpu_uuid="GPU-one"),
                       dict(vulkan_index=1, gpu_uuid="GPU-two"),
                       dict(vulkan_index=2, gpu_uuid="GPU-three")])
    command = ["UnrealEditor", project, "-abslog=" + str(temporary / "UnrealEditor.log")]
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: (
        "\0".join(command).encode() if str(p) == "/proc/12345/cmdline"
        else original_read(p)))
    return ue, receipt, command


def install_nvml(monkeypatch, cards):
    data = "<nvidia_smi_log>" + "".join(
        "<gpu><uuid>" + uuid + "</uuid><processes>" + "".join(
            f"<process_info><pid>{pid}</pid><type>{kind}</type>"
            f"<used_memory>{memory} MiB</used_memory></process_info>"
            for pid, kind, memory in processes) + "</processes></gpu>"
        for uuid, processes in cards) + "</nvidia_smi_log>"
    def read(command, **kwargs):
        assert command == ["nvidia-smi", "-q", "-x"]
        return data
    monkeypatch.setattr(subprocess, "check_output", read)


def test_renderer_readback_accepts_and_reports_auxiliary_permitted_context(launch, monkeypatch):
    ue, receipt, _ = launch
    install_nvml(monkeypatch, [
        ("GPU-one", [(12345, "G", 6)]),
        ("GPU-two", [(12345, "C+G", 1925)]),
        ("GPU-three", [(67890, "C+G", 2100)])])
    result = verify_gpu(ue, receipt)
    assert result["expected_physical_gpu"] == 2
    assert [row["uuid"] for row in result["processes"]] == ["GPU-two"]
    assert [row["uuid"] for row in result["auxiliary_permitted_contexts"]] == ["GPU-one"]
    assert all(row["pid"] == 12345 for row in result["processes"])


def test_wrong_actual_renderer_uuid_rejected_even_when_expected_gpu_has_context(launch, monkeypatch):
    ue, receipt, _ = launch
    receipt["vulkan_devices"][1]["gpu_uuid"] = "GPU-three"
    install_nvml(monkeypatch, [("GPU-two", [(12345, "G", 6)])])
    with pytest.raises(ValueError, match="renderer is on an unauthorized"):
        verify_gpu(ue, receipt)


def test_other_same_project_pid_cannot_supply_own_renderer_readback(launch, monkeypatch):
    ue, receipt, _ = launch
    install_nvml(monkeypatch, [("GPU-two", [(67890, "C+G", 2000)])])
    with pytest.raises(ValueError, match="not found on its renderer GPU"):
        verify_gpu(ue, receipt)


@pytest.mark.parametrize("unexpected_uuid", ["GPU-zero", "GPU-unknown"])
def test_any_context_outside_task_gpu_scope_rejected(launch, monkeypatch, unexpected_uuid):
    ue, receipt, _ = launch
    install_nvml(monkeypatch, [
        ("GPU-two", [(12345, "C+G", 2000)]),
        (unexpected_uuid, [(12345, "G", 6)])])
    with pytest.raises(ValueError, match="outside the task scope"):
        verify_gpu(ue, receipt)


def test_known_uuid_not_in_authorized_subset_is_rejected(launch, monkeypatch):
    ue, receipt, _ = launch
    policy = Path(ue["output_root"]) / "kujiale_native_gpu_policy.json"
    data = json.loads(policy.read_text())
    data["allowed_gpu_indices"] = [2]
    policy.write_text(json.dumps(data))
    install_nvml(monkeypatch, [
        ("GPU-one", [(12345, "G", 6)]), ("GPU-two", [(12345, "C+G", 2000)])])
    with pytest.raises(ValueError, match="outside the task scope"):
        verify_gpu(ue, receipt)


def test_changed_project_command_does_not_match_launch(launch, monkeypatch):
    ue, receipt, command = launch
    command[1] = "another.uproject"
    with pytest.raises(ValueError, match="PID does not match"):
        verify_gpu(ue, receipt)


@pytest.mark.parametrize("selected", [0, 2, None])
def test_missing_or_different_ue_renderer_selection_rejected(launch, monkeypatch, selected):
    ue, receipt, _ = launch
    log = Path(receipt["temporary_directory"]) / "UnrealEditor.log"
    log.write_text("" if selected is None else
                   f"LogVulkanRHI: Using device at index {selected} of 3\n")
    with pytest.raises(ValueError, match="selection differs"):
        verify_gpu(ue, receipt)


def test_adapter_to_uuid_inventory_required(launch, monkeypatch):
    ue, receipt, _ = launch
    receipt["vulkan_devices"] = []
    with pytest.raises(ValueError, match="renderer is on an unauthorized"):
        verify_gpu(ue, receipt)
