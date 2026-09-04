"""Run local VAST-AI SkinTokens/TokenRig inference on one mesh.

The model and the Blender parser both come from AVEngine. The only external
inputs are explicit, host-local model roots resolved through
examples/assets/model_roots_v1.json. This entry point has no Gradio/training
path and rejects an upstream checkout argument.
"""

from __future__ import annotations

import argparse
import http.client
from copy import deepcopy
import os
from pathlib import Path
import pickle
import subprocess
import socket
import sys
import tempfile
import shutil
import time
from typing import Any, Sequence

REPOSITORY = Path(__file__).resolve().parents[2]
# Production inference is local-only: prevent a missing file from
# turning into an implicit Hugging Face/Datasets network lookup.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
SOURCE_ROOT = REPOSITORY / "src"
ASSET_ROOT = SOURCE_ROOT / "avengine" / "assets"
SERVER_SCRIPT = Path(__file__).with_name("skintokens_loopback_bpy_server.py")
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (ASSET_ROOT, SOURCE_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_roots import resolve as resolve_model_root  # noqa: E402
from skintokens.runtime import load_model, resolve_paths  # noqa: E402


MODEL_ROOT_NAMES = {
    "skintokens": "skintokens",
    "qwen": "qwen3_0_6b",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="one input mesh")
    parser.add_argument("--output", type=Path, required=True, help="fresh output GLB")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="local SkinTokens snapshot; defaults to model_roots_v1.json",
    )
    parser.add_argument(
        "--qwen-root",
        type=Path,
        default=None,
        help="local Qwen config snapshot; defaults to model_roots_v1.json",
    )
    parser.add_argument("--device", default="cuda", help="torch device, e.g. cuda or cpu")
    parser.add_argument("--blender", type=Path, default=None)
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--server-timeout", type=float, default=120.0)
    parser.add_argument("--class-name", default="articulation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2040)
    parser.add_argument("--use-skeleton", action="store_true")
    parser.add_argument("--use-transfer", action="store_true")
    parser.add_argument("--use-postprocess", action="store_true")
    parser.add_argument(
        "--rigger-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _resolve_root(
    override: Path | None,
    *,
    name: str,
    label: str,
) -> Path:
    try:
        value = resolve_model_root(name, override=override)
    except SystemExit as error:
        raise ValueError(str(error)) from error
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{label} directory is missing: {path}")
    return path


def _resolve_blender(configured: Path | None) -> Path:
    raw = str(configured) if configured is not None else os.environ.get(
        "AVENGINE_BLENDER", "blender"
    )
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    located = shutil.which(raw)
    if located:
        return Path(located).resolve()
    return candidate.resolve()


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float = 1200.0):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = Path(socket_path)

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(str(self.socket_path))
        self.sock = sock


def _request(socket_path: Path, endpoint: str, payload: Any) -> Any:
    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    connection = _UnixHTTPConnection(socket_path)
    try:
        connection.request(
            "POST",
            "/" + endpoint,
            body=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        response = connection.getresponse()
        encoded = response.read()
        if response.status >= 400:
            raise OSError(
                f"SkinTokens Blender RPC returned HTTP {response.status}: "
                f"{response.reason}"
            )
        value = pickle.loads(encoded)
    finally:
        connection.close()
    if isinstance(value, dict) and value.get("error") is not None:
        raise RuntimeError(value.get("traceback") or value["error"])
    return value


def _ping(socket_path: Path) -> bool:
    connection = _UnixHTTPConnection(socket_path, timeout=1.0)
    try:
        connection.request("GET", "/ping")
        response = connection.getresponse()
        return response.status == 200 and response.read() == b"pong"
    finally:
        connection.close()


class _BlenderRPC:
    def __init__(self, blender: Path, workdir: Path, timeout: float):
        self.blender = blender
        self.workdir = workdir
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self.log_handle = None
        self.log_path = workdir / "skintokens_bpy.log"
        self.session_dir = Path(
            tempfile.mkdtemp(prefix="avengine_skintokens_rpc_")
        ).resolve()
        os.chmod(self.session_dir, 0o700)
        self.socket_path = self.session_dir / "rpc.sock"

    def _remove_private_socket(self) -> None:
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        try:
            self.session_dir.rmdir()
        except OSError:
            pass

    def _stop(self) -> subprocess.Popen[str] | None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        self._remove_private_socket()
        return process

    def __enter__(self) -> "_BlenderRPC":
        if not self.blender.is_file():
            self._remove_private_socket()
            raise ValueError(f"Blender executable is missing: {self.blender}")
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        # The local server resolves its package from __file__; inherited
        # PYTHONPATH could otherwise reintroduce an upstream checkout.
        env.pop("PYTHONPATH", None)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.process = subprocess.Popen(
                [
                    str(self.blender),
                    "--background",
                    "--factory-startup",
                    "--python",
                    str(SERVER_SCRIPT),
                    "--",
                    "--socket",
                    str(self.socket_path),
                ],
                cwd=str(REPOSITORY),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + self.timeout
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Blender RPC exited with code {self.process.returncode}; "
                        f"see {self.log_path}"
                    )
                try:
                    if _ping(self.socket_path):
                        mode = self.socket_path.stat().st_mode & 0o777
                        if mode != 0o600:
                            raise PermissionError(
                                f"RPC socket is not mode 0600: {self.socket_path} "
                                f"({mode:04o})"
                            )
                        print(
                            f"SKINTOKENS_BPY_CONNECTED unix://{self.socket_path}"
                        )
                        return self
                except Exception as error:
                    last_error = error
                time.sleep(0.25)
            raise RuntimeError(
                f"timed out waiting for Blender RPC socket {self.socket_path}: "
                f"{last_error}; see {self.log_path}"
            )
        except Exception:
            self._stop()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        process_before_stop = self.process
        was_running = (
            process_before_stop is not None and process_before_stop.poll() is None
        )
        process = self._stop()
        # SIGTERM (-15) is the expected status for a server stopped by this
        # context manager. A process that had already died with an error still
        # fails the run.
        if (
            process is not None
            and process.returncode != 0
            and exc is None
            and not was_running
        ):
            raise RuntimeError(
                f"Blender RPC exited with code {process.returncode}; see {self.log_path}"
            )
        print(
            f"SKINTOKENS_BPY_EXITED code={process.returncode if process else None}"
        )

def _as_device(device: str):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"requested {device}, but CUDA is unavailable")
    return torch.device(device)


def _batch_from_processed(processed: dict[str, Any], device) -> dict[str, Any]:
    import numpy as np
    import torch

    batch: dict[str, Any] = {}
    for key, value in processed.items():
        if key == "non":
            if not isinstance(value, dict):
                raise ValueError("SkinTokens process_fn returned non-mapping metadata")
            for name, item in value.items():
                batch[name] = [item]
        elif key == "cat":
            raise ValueError("unexpected concatenated field in single-item batch")
        elif isinstance(value, np.ndarray):
            batch[key] = torch.from_numpy(value).to(device)
        elif isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        else:
            raise ValueError(f"unexpected processed field {key}: {type(value)}")
    return batch


def _generate_kwargs(args) -> dict[str, Any]:
    return {
        "max_new_tokens": int(args.max_new_tokens),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "temperature": float(args.temperature),
        "repetition_penalty": float(args.repetition_penalty),
        "num_return_sequences": 1,
        "num_beams": int(args.num_beams),
        "do_sample": True,
    }


def _postprocess(asset: Asset) -> None:
    from skintokens.data.vertex_group import voxel_skin

    voxel = asset.voxel(resolution=196)
    if asset.joints is None or asset.vertices is None or asset.faces is None:
        raise ValueError("SkinTokens output lacks geometry needed for voxel postprocess")
    asset.skin *= voxel_skin(
        grid=0,
        grid_coords=voxel.coords,
        joints=asset.joints,
        vertices=asset.vertices,
        faces=asset.faces,
        mode="square",
        voxel_size=voxel.voxel_size,
    )
    asset.normalize_skin()


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch
    from skintokens.model.spec import ModelInput
    from skintokens.rig_package.info.asset import Asset
    from skintokens.tokenizer.spec import TokenizeInput

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise ValueError(f"input mesh is missing: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if input_path.suffix.lower() not in {".glb", ".gltf", ".fbx", ".obj"}:
        raise ValueError(f"unsupported input extension: {input_path.suffix}")
    if args.use_transfer and output_path.suffix.lower() != ".glb":
        raise ValueError("--use-transfer requires a .glb output")

    model_root = _resolve_root(
        args.model_root,
        name=MODEL_ROOT_NAMES["skintokens"],
        label="SkinTokens model root",
    )
    qwen_root = _resolve_root(
        args.qwen_root,
        name=MODEL_ROOT_NAMES["qwen"],
        label="Qwen model root",
    )
    device = _as_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with _BlenderRPC(
        blender=_resolve_blender(args.blender),
        workdir=output_path.parent,
        timeout=float(args.server_timeout),
    ) as bpy_rpc:
        print(f"SKINTOKENS_LOAD_INPUT {input_path}")
        asset = _request(bpy_rpc.socket_path, "load", str(input_path))
        if not isinstance(asset, Asset):
            raise RuntimeError(f"Blender load returned {type(asset).__name__}, not Asset")
        asset.path = str(input_path)
        asset.cls = args.class_name
        has_skin = asset.skin is not None
        model, transform_config, tokenizer_config = load_model(
            resolve_paths(model_root, qwen_root),
            device=str(device),
            has_skin=has_skin,
        )
        from skintokens.data.transform import Transform
        from skintokens.tokenizer.parse import get_tokenizer

        transform = Transform.parse(**transform_config["predict_transform"])
        transform.apply(asset)
        tokenizer = get_tokenizer(**deepcopy(tokenizer_config))
        tokens = None
        if args.use_skeleton and asset.parents is not None and asset.joints is not None:
            tokens = tokenizer.tokenize(
                TokenizeInput(
                    joints=asset.joints,
                    parents=asset.parents.tolist(),
                    cls=asset.cls,
                    joint_names=asset.joint_names,
                )
            )
        processed = model._process_fn([ModelInput(asset=asset, tokens=tokens)])[0]
        batch = _batch_from_processed(processed, device)
        batch["generate_kwargs"] = _generate_kwargs(args)
        skeleton_tokens = [tokens] if tokens is not None else None
        print(
            f"SKINTOKENS_GENERATE device={device} input_vertices={asset.N} "
            f"sampled={asset.sampled_vertices.shape[0] if asset.sampled_vertices is not None else 0}"
        )
        with torch.inference_mode():
            results = model.predict_step(
                batch,
                skeleton_tokens=skeleton_tokens,
                make_asset=True,
            )["results"]
        if len(results) != 1 or results[0].asset is None:
            raise RuntimeError("SkinTokens did not return one rigged asset")
        predicted = results[0].asset
        if args.use_postprocess:
            _postprocess(predicted)
        if args.use_transfer:
            response = _request(
                bpy_rpc.socket_path,
                "transfer",
                {
                    "source_asset": predicted,
                    "target_path": str(input_path),
                    "export_path": str(output_path),
                    "group_per_vertex": 4,
                },
            )
        else:
            response = _request(
                bpy_rpc.socket_path,
                "export",
                {
                    "asset": predicted,
                    "filepath": str(output_path),
                    "group_per_vertex": 4,
                },
            )
        if response != "ok":
            raise RuntimeError(f"Blender export failed: {response!r}")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"SkinTokens output was not created: {output_path}")
    print(f"SKINTOKENS_RIG_OK {output_path}")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rigger_root is not None:
        print(
            "rig refused: --rigger-root is deprecated; SkinTokens source is "
            "AVEngine-local and the argument cannot select a checkout",
            file=sys.stderr,
        )
        return 2
    if args.port is not None:
        print(
            "rig refused: --port is deprecated; the RPC requires a private "
            "Unix socket and does not listen on TCP",
            file=sys.stderr,
        )
        return 2
    try:
        run(args)
    except (FileExistsError, OSError, ImportError, ValueError, RuntimeError) as error:
        print(f"rig refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
