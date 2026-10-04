"""Native CPU RLR and software pixels for the opt-in continuous-scene pilot.

Outputs are research diagnostics with procedural human proxies. No Habitat
visual Simulator, model, external real recording, or GPU is constructed here.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import math
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import time
import uuid

import numpy as np
from scipy import signal

from avengine.timeline.continuous_scene import (
    SAMPLE_RATE, FPS, camera_basis, position_at, raw_world_to_camera_sn3d,
    plan_rir_keyframes, partitioned_convolve, generate_v02_questions,
    interpolated_filter, angular_distance,
)
from avengine.acoustics.house_acceptance_metrics import foa_direction, rir_metrics


def write_json(path, value):
    path = Path(path)
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Unsupported metadata type: {type(value).__name__}")
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False)+"\n"
    # Reboot/kill before publication must leave only an uncommitted sibling,
    # never a truncated final JSON which would permanently block resumption.
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    from avengine.timeline.v7_storage_budget import check_write
    check_write(path, expected_bytes=len(payload.encode("utf-8")))
    with pending.open("x", encoding="utf-8") as f:
        f.write(payload)
        f.flush(); os.fsync(f.fileno())
    os.link(pending, path)
    pending.unlink()


class FormantSpeech:
    """Existing eSpeak NG synchronous formant library; no playback device."""
    def __init__(self):
        name = ctypes.util.find_library("espeak-ng")
        if not name:
            raise RuntimeError("existing CPU eSpeak NG library is required")
        self.lib = lib = ctypes.CDLL(name)
        lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.sample_rate = lib.espeak_Initialize(2, 0, None, 0x8000)
        if self.sample_rate <= 0:
            raise RuntimeError("existing eSpeak NG language data unavailable")
        self.parts = []
        cbtype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)
        @cbtype
        def callback(wav, count, events):
            if wav and count:
                self.parts.append(np.ctypeslib.as_array(wav, shape=(count,)).copy())
            return 0
        self.callback = callback
        lib.espeak_SetSynthCallback(callback)
        lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        lib.espeak_Synth.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int,
                                   ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        lib.espeak_Info.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        lib.espeak_Info.restype = ctypes.c_char_p
        data_path = ctypes.c_char_p()
        version = lib.espeak_Info(ctypes.byref(data_path)).decode()
        self.receipt = {"engine": "eSpeak NG formant", "library": name, "version": version,
                        "data_path": data_path.value.decode(), "native_sample_rate_hz": self.sample_rate,
                        "code_license": "GPL-3.0-or-later; system package copyright retained externally",
                        "upstream": "https://github.com/espeak-ng/espeak-ng",
                        "text_and_class_label_source": "author script and selected formant voice configuration",
                        "real_waveforms_or_STARSS_used": False,
                        "perceptual_voice_identity_or_gender_verified": False}

    def synthesize(self, text, voice):
        self.parts.clear()
        if self.lib.espeak_SetVoiceByName(voice.encode()) != 0:
            raise RuntimeError("requested formant voice unavailable")
        self.lib.espeak_SetParameter(1, 185, 0)  # words/minute
        self.lib.espeak_SetParameter(2, 75, 0)   # upstream volume, before common scene gain
        payload = text.encode()+b"\0"
        if self.lib.espeak_Synth(payload, len(payload), 0, 1, 0, 1, None, None) != 0:
            raise RuntimeError("formant synthesis failed")
        self.lib.espeak_Synchronize()
        if not self.parts:
            raise RuntimeError("formant synthesis returned no samples")
        a = np.concatenate(self.parts).astype(float)/32768
        divisor = math.gcd(self.sample_rate, SAMPLE_RATE)
        return signal.resample_poly(a, SAMPLE_RATE//divisor, self.sample_rate//divisor)


class PositionRIRCache:
    """Exact-pose primary keys, atomic per-RIR files and durable local progress.

    Configuration and the existing compiled package identity are fixed for a
    cache directory. No spatial quantization or implicit cross-material reuse.
    Independent workers may race; no-replace file publication keeps one result.
    """
    def __init__(self, root, renderer, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root/"configuration.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError("RIR cache configuration changed; use a new directory")
        else:
            try:
                write_json(path, identity)
            except FileExistsError:
                if json.loads(path.read_text()) != identity:
                    raise ValueError("RIR cache configuration changed; use a new directory")
        self.renderer = renderer
        self.db = sqlite3.connect(self.root/"positions.sqlite3", timeout=60)
        self.db.execute("CREATE TABLE IF NOT EXISTS rirs (id INTEGER PRIMARY KEY, pose TEXT UNIQUE NOT NULL, completed INTEGER NOT NULL DEFAULT 0)")
        self.db.commit()
        self.memory = {}
        self.stats = {"cache_hits": 0, "new_rirs": 0, "native_wall_seconds": 0., "native_cpu_seconds": 0.}

    def get(self, position):
        point = np.asarray(position, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("RIR source position must be finite xyz")
        key = json.dumps(point.tolist(), separators=(",", ":"))
        if key in self.memory:
            self.stats["cache_hits"] += 1
            return self.memory[key]
        self.db.execute("INSERT INTO rirs(pose) VALUES (?) ON CONFLICT(pose) DO NOTHING", (key,))
        self.db.commit()
        index, completed = self.db.execute("SELECT id,completed FROM rirs WHERE pose=?", (key,)).fetchone()
        path = self.root/f"rir_{index:06d}.npy"
        if path.exists():
            a = np.load(path, allow_pickle=False)
            if a.ndim != 2 or a.shape[0] != 4 or not np.isfinite(a).all() or not np.any(a):
                raise ValueError("existing cached RIR is invalid")
            self.stats["cache_hits"] += 1
        else:
            if completed:
                raise ValueError("completed RIR cache record lost its file")
            batch = self.renderer.render([point.tolist()])
            a = batch.samples[0]
            temp = self.root/f".rir_{index:06d}_{uuid.uuid4().hex}.partial"
            with temp.open("xb") as f:
                np.save(f, a, allow_pickle=False)
                f.flush(); os.fsync(f.fileno())
            try:
                os.link(temp, path)
            except FileExistsError:
                a = np.load(path, allow_pickle=False)
            self.stats["new_rirs"] += 1
            self.stats["native_wall_seconds"] += batch.wall_seconds
            self.stats["native_cpu_seconds"] += batch.process_cpu_seconds
        self.db.execute("UPDATE rirs SET completed=1 WHERE id=?", (index,)); self.db.commit()
        self.memory[key] = a
        return a


def make_scene(house, index, duration_s):
    p = house["room"]["placement"]
    a = np.asarray(p["source_1_m"], float)
    b = np.asarray(p["source_2_m"], float)
    far = b + (.48+.06*index)*(a-b)
    listener = np.asarray(p["camera_m"], float)
    forward = (a+b)/2-listener
    forward[1] = 0.; forward /= np.linalg.norm(forward)
    knots = []
    for base in range(0, int(duration_s), 10):
        for t, point in ((0, b), (2, b), (6, far), (8, far), (10, b)):
            if not knots or base+t > knots[-1]["time_s"]:
                knots.append({"time_s": float(base+t), "position_m": point.tolist()})
    sources = {
        "voice1": {"actor_id": 1, "class_id": 0, "voice": "en-us+f3", "mechanism": "human_live_formant",
                   "trajectory": [{"time_s": 0., "position_m": a.tolist()}, {"time_s": duration_s, "position_m": a.tolist()}]},
        "voice2": {"actor_id": 2, "class_id": 1, "voice": "en-us", "mechanism": "human_live_formant", "trajectory": knots},
        "clap1": {"actor_id": 1, "class_id": 2, "mechanism": "author_procedural_hand_clap",
                  "trajectory": [{"time_s": 0., "position_m": (a-[0., .4, 0.]).tolist()},
                                 {"time_s": duration_s, "position_m": (a-[0., .4, 0.]).tolist()}]},
    }
    events = []
    for base in range(0, int(duration_s), 10):
        schedules = {"voice1": [(0., 6.4), (7.2, 7.8), (8.4, 9.), (9.6, 9.9)],
                     "voice2": [(3.2, 5.), (5.3, 6.1), (6.6, 7.8), (8.2, 8.8), (9.2, 9.9)],
                     "clap1": [(6.3, 6.6), (8., 8.2)]}
        for endpoint, spans in schedules.items():
            for start, end in spans:
                lo, hi = round((base+start)*10), round(min(base+end, duration_s)*10)
                if hi > lo:
                    events.append({"event_id": f"activity{len(events):03d}", "endpoint_id": endpoint,
                                   "source_id": sources[endpoint]["actor_id"], "class_id": sources[endpoint]["class_id"],
                                   "start_tick": lo, "end_tick": hi})
    rng = np.random.default_rng(20261003+index)
    starts = [round(rng.uniform(4., 5.5)*10)/10, round(rng.uniform(10., 11.5)*10)/10,
              round(rng.uniform(duration_s-12., duration_s-10.)*10)/10]
    return {"scene_id": f"{house['house']}__s{index:02d}__{int(duration_s)}s",
            "house": house["house"], "split": house["split"], "seed": 20261003+index,
            "duration_s": duration_s, "sample_rate_hz": SAMPLE_RATE, "video_fps": FPS,
            "listener_m": listener.tolist(), "camera_forward": forward.tolist(),
            "sources": sources, "events": events, "window_starts_s": starts,
            "independent_parent_scene_id": f"{house['house']}__s{index:02d}",
            "paired_length_comparison": True, "visual_status": "CPU_procedural_proxy_research_only"}


def make_dry_buses(scene, tts):
    n = int(round(scene["duration_s"]*SAMPLE_RATE))
    buses = {k: np.zeros(n) for k in scene["sources"]}
    rng = np.random.default_rng(scene["seed"])
    phrases = ["The quiet morning brings a little sunshine", "We can read a story after lunch",
               "A small bird is resting by the window", "There is plenty of time to finish the work"]
    clips = {k: [tts.synthesize(text, v["voice"]) for text in phrases] for k, v in scene["sources"].items() if "voice" in v}
    for i, e in enumerate(scene["events"]):
        lo, hi = e["start_tick"]*1600, e["end_tick"]*1600
        length = hi-lo
        if e["endpoint_id"] == "clap1":
            t = np.arange(length)/SAMPLE_RATE
            clip = signal.sosfilt(signal.butter(2, [400., 6500.], btype="bandpass", fs=SAMPLE_RATE, output="sos"), rng.normal(0, 1, length))
            clip *= np.exp(-t/.025)*.035
        else:
            bank = clips[e["endpoint_id"]]
            clip = np.resize(bank[i % len(bank)], length)*.16
        fade = min(80, length//2)
        if fade:
            ramp = np.linspace(0, 1, fade)
            clip[:fade] *= ramp; clip[-fade:] *= ramp[::-1]
        buses[e["endpoint_id"]][lo:hi] += clip
    return buses, {**tts.receipt, "text": phrases, "voices": {k: v.get("voice") for k,v in scene["sources"].items()},
                   "speech_clip_policy": "deterministic trim/repeat of author text; robotic formant pilot, intelligibility not certified",
                   "clap": "seeded bandlimited noise with exponential decay; author procedural label"}


def validate_navigation(house, scene, habitat_module):
    pathfinder = habitat_module.PathFinder()
    if not pathfinder.load_nav_mesh(house["input"]["navmesh_source"]):
        raise ValueError("existing navigation mesh cannot be loaded read-only")
    checks = []
    track = scene["sources"]["voice2"]["trajectory"]
    for t in np.linspace(0, 10, 101):
        p = position_at(track, t)
        foot = p-[0., 1.3, 0.]
        snap = np.asarray(pathfinder.snap_point(foot), float)
        error = float(np.linalg.norm(snap-foot))
        checks.append({"time_s": float(t), "foot_to_navmesh_m": error})
    if any(not math.isfinite(c["foot_to_navmesh_m"]) or c["foot_to_navmesh_m"] > .15 for c in checks):
        raise ValueError("moving source route lacks navigation support")
    return {"read_only_navmesh": house["input"]["navmesh_source"], "sample_count": len(checks),
            "maximum_foot_to_navmesh_m": max(c["foot_to_navmesh_m"] for c in checks),
            "navmesh_generation_parameters_verified": False, "checks": checks}


class CPUProxyVideo:
    """Fixed-camera raw-scan BVH pixels, with occluded analytic human proxies."""
    def __init__(self, house, scene, inputs, output, width=320, height=180):
        from avengine.acoustics.house_acceptance_visual import load_raw_scan, NativeRawRayIntersector, camera_rays
        self.width, self.height = width, height
        self.listener = np.asarray(scene["listener_m"])
        self.rays, self.focal, self.forward, self.right, self.up = camera_rays(scene["camera_forward"], width, height)
        raw = Path(house["input"]["scene_directory"])/(house["input"]["scan_id"]+".glb")
        background_key = {"raw_asset":str(raw),"listener_m":scene["listener_m"],"camera_forward":scene["camera_forward"],
                          "width":width,"height":height,"hfov_deg":85.,"renderer_revision":1}
        background_path, config_path = output/"cpu_background.npz", output/"background_configuration.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config["key"] != background_key:
                raise ValueError("CPU background camera or raw asset changed")
            # Read prior completed fixed-name caches; new renders publish a
            # unique data file before the atomic completion pointer.
            background_path = output/config.get("data_file", "cpu_background.npz")
            with np.load(background_path,allow_pickle=False) as cache:
                self.depth = cache["depth"];self.background = cache["background"]
            if self.depth.shape != (width*height,) or self.background.shape != (width*height,3):
                raise ValueError("CPU background cache dimensions changed")
            self.receipt = {**config["receipt"],"background_cache_reused":True}
            return
        mesh, colors, modes = load_raw_scan(raw)
        result = {"acoustic_package_manifest": house["acoustic_package_manifest"],
                  "foa_checks": [{"listener_m": scene["listener_m"],
                                  "source_m": position_at(scene["sources"]["voice1"]["trajectory"], 0).tolist()}]}
        intersector = NativeRawRayIntersector(mesh, result, inputs, output)
        self.depth = np.full(len(self.rays), np.inf)
        self.background = np.full((len(self.rays), 3), [15, 18, 22], np.uint8)
        normals = mesh.face_normals
        for begin in range(0, len(self.rays), 512):
            end = min(len(self.rays), begin+512)
            hit, ri, tri = intersector.intersects_location(np.repeat(self.listener[None], end-begin, axis=0), self.rays[begin:end], False)
            if len(ri):
                ids = begin+ri
                self.depth[ids] = np.linalg.norm(hit-self.listener, axis=1)
                shade = .35+.65*np.abs(np.einsum("ij,ij->i", normals[tri], -self.rays[ids]))
                self.background[ids] = np.clip(colors[tri]*shade[:,None], 0, 255).astype(np.uint8)
        self.receipt = {"raw_visual_asset": str(raw), "color_modes": modes, "native_bvh": intersector.receipt,
                        "render_resolution": [width, height], "encoded_resolution": [1280,720],
                        "camera_static_background_reused": True, "human_geometry": "author analytic ellipsoid/sphere proxies",
                        "production_habitat_RGB": False, "compute_device": "CPU", "background_cache_reused":False}
        # An orphan from an interrupted attempt is retained and ignored.
        stamp = uuid.uuid4().hex
        pending = output/f".cpu_background_{stamp}.partial"
        background_path = output/f"cpu_background_{stamp}.npz"
        with pending.open("xb") as file:
            np.savez_compressed(file, depth=self.depth, background=self.background)
            file.flush();os.fsync(file.fileno())
        os.link(pending, background_path)
        write_json(config_path,{"key":background_key,"receipt":self.receipt,
                                "data_file":background_path.name})

    def frame(self, scene, time_s):
        pixels = self.background.copy(); depth = self.depth.copy()
        head_masks = {}
        actor_colors = {1: np.array([20,235,240], np.uint8), 2: np.array([245,110,35], np.uint8)}
        actors = sorted((v["actor_id"], position_at(v["trajectory"], time_s)) for v in scene["sources"].values() if "voice" in v)
        objects = []
        for actor, mouth in actors:
            color = actor_colors[actor]
            track = scene["sources"][f"voice{actor}"]["trajectory"]
            moving = np.linalg.norm(position_at(track,time_s+.05)-position_at(track,time_s)) > 1e-6
            phase = math.sin(2*math.pi*time_s*1.4)*.06 if moving else 0.
            # Body and head centers share the authoritative emission trajectory.
            objects += [(actor, "body", mouth+[0.,-.4,0.], [.22,.32,.14], (color*.52).astype(np.uint8)),
                        (actor, "leg", mouth+[.11,-.91,phase], [.075,.33,.075], (color*.35).astype(np.uint8)),
                        (actor, "leg", mouth+[-.11,-.91,-phase], [.075,.33,.075], (color*.35).astype(np.uint8)),
                        (actor, "arm", mouth+[.25,-.44,0.], [.07,.28,.07], (color*.7).astype(np.uint8)),
                        (actor, "arm", mouth+[-.25,-.44,0.], [.07,.28,.07], (color*.7).astype(np.uint8)),
                        (actor, "head", mouth, [.13,.15,.13], color)]
        # Last writer is still depth-tested; masks derive from final actual pixels.
        for actor, part, center, radii, color in objects:
            r = np.asarray(radii)
            origin = (self.listener-center)/r
            directions = self.rays/r
            aa = np.sum(directions**2, axis=1)
            bb = 2*(directions @ origin)
            cc = float(origin @ origin)-1
            discriminant = bb**2-4*aa*cc
            distance = (-bb-np.sqrt(np.maximum(discriminant,0)))/(2*aa)
            visible = (discriminant >= 0) & (distance > 0) & (distance < depth)
            depth[visible] = distance[visible]; pixels[visible] = color
        for actor, color in actor_colors.items():
            ids = np.flatnonzero(np.all(pixels == color, axis=1))
            head_masks[str(actor)] = {"visible_pixels": len(ids),
                                     "centroid_px": [float(np.mean(ids % self.width+.5)), float(np.mean(ids//self.width+.5))] if len(ids) else None,
                                     "color_rgb": color.tolist()}
        return pixels.reshape(self.height, self.width, 3), head_masks

    def encode(self, scene, path, diagnostics):
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-threads", "1",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}", "-r", str(FPS),
                   "-i", "pipe:0", "-vf", "scale=1280:720:flags=neighbor", "-c:v", "libx264", "-threads", "1",
                   "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        masks = []
        from PIL import Image
        try:
            for frame in range(int(scene["duration_s"]*FPS)):
                rgb, heads = self.frame(scene, frame/FPS)
                proc.stdin.write(rgb.tobytes())
                masks.append({"frame": frame, "time_s": frame/FPS, "heads": heads})
                if frame in diagnostics:
                    Image.fromarray(rgb).resize((1280,720), Image.Resampling.NEAREST).save(path.parent/f"frame_{frame:04d}.png")
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("CPU video encoding failed")
        except BaseException:
            if proc.poll() is None:
                proc.terminate(); proc.wait()
            raise
        return masks


def validate_interpolation(scene, cache, output, *, max_seconds=.25, max_distance_m=.5, max_angle_deg=None):
    """Independent 50 ms native RIR reference on the complete first traversal."""
    track = scene["sources"]["voice2"]["trajectory"]
    listener, forward = np.array(scene["listener_m"]), np.array(scene["camera_forward"])
    policies = {"quarter_second": {"max_seconds": .25, "max_distance_m": math.inf},
                "half_meter": {"max_seconds": math.inf, "max_distance_m": .5},
                "selected": {"max_seconds": max_seconds, "max_distance_m": max_distance_m, "max_angle_deg": max_angle_deg}}
    records = {}
    times = np.arange(0., 10.00001, .05)
    native = [cache.get(position_at(track, t)) for t in times]
    native_checks = [foa_direction(ir, listener, position_at(track,t), forward.copy())["check"] for t,ir in zip(times,native,strict=True)]
    for name, policy in policies.items():
        keys = plan_rir_keyframes(track, scene["duration_s"], listener, **policy)
        filters = [cache.get(k["position_m"]) for k in keys]
        samples = [k["sample"] for k in keys]
        comparisons = []
        for t, ref, refcheck in zip(times, native, native_checks, strict=True):
            blend = interpolated_filter(filters,samples,int(round(t*SAMPLE_RATE)))
            check = foa_direction(blend, listener, position_at(track,t), forward.copy())["check"]
            error = angular_distance(check["foa_world_direction"], refcheck["foa_world_direction"])
            distance = float(np.linalg.norm(position_at(track,t)-listener))
            refmetric, _ = rir_metrics(ref,SAMPLE_RATE,distance,4.)
            metric, _ = rir_metrics(blend,SAMPLE_RATE,distance,4.)
            comparisons.append({"time_s": float(t), "native_geometry_error_deg": refcheck["angle_deg"],
                                "interpolated_geometry_error_deg": check["angle_deg"], "native_vs_interpolated_foa_error_deg": error,
                                "drr_difference_db": metric["drr_db"]-refmetric["drr_db"],
                                "rt60_reference_s": refmetric["rt60_mid_s"], "rt60_interpolated_s": metric["rt60_mid_s"],
                                "interpolated_tail_truncated": metric["tail_truncated"]})
        records[name] = {"policy": {k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in policy.items()}, "keyframes": len(keys), "unique_positions": len({json.dumps(k['position_m']) for k in keys}),
                         "max_foa_error_deg": max(c["native_vs_interpolated_foa_error_deg"] for c in comparisons),
                         "max_geometry_error_deg": max(c["interpolated_geometry_error_deg"] for c in comparisons),
                         "pass_5deg": all(c["native_vs_interpolated_foa_error_deg"] <= 5 and c["interpolated_geometry_error_deg"] <= 5 for c in comparisons),
                         "checks": comparisons}
    # Native Monte Carlo repeat establishes waveform/late-energy variability.
    repeats = []
    position = position_at(track, 4.)
    for _ in range(3):
        batch = cache.renderer.render([position.tolist()]); metric, _ = rir_metrics(batch.samples[0],SAMPLE_RATE,float(np.linalg.norm(position-listener)),4.)
        repeats.append({"rt60_s": metric["rt60_mid_s"], "drr_db": metric["drr_db"],
                        "wall_seconds": batch.wall_seconds, "cpu_seconds": batch.process_cpu_seconds})
    result = {"reference_step_s": .05, "reference_samples": len(times), "source_time_crossfade": "linear_partition_of_unity",
              "policies": records, "native_repeats": repeats, "waveform_error_interpretation": "native late reverb is Monte Carlo; FOA direct window is the direction gate"}
    write_json(output/"interpolation_validation.json", result)
    return result


def measure_scene(scene, house, cache, tts, video, output, *, max_seconds=.25, max_distance_m=.5, max_angle_deg=None):
    """Produce one restartable scene and three related windows, with real timing."""
    from avengine.spatial_audio.audio import write_float32_wav
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    completed = output/"completed.json"
    policy = {"max_seconds":max_seconds,"max_distance_m":max_distance_m,"max_angle_deg":max_angle_deg,"crossfade":"linear_source_time_full_tail"}
    if completed.exists():
        receipt = json.loads(completed.read_text())
        if receipt["scene"] != scene or receipt.get("rir_update_policy") != policy:
            raise ValueError("completed scene plan changed")
        attempt = output/receipt["attempt"]
        if not all((attempt/p).is_file() for p in receipt["required_files"]):
            raise ValueError("completed scene lost an artifact")
        return receipt
    attempt = output/"attempts"/uuid.uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    before_cache = cache.stats.copy()
    stages = {}
    stamp = time.perf_counter()
    dry, dry_receipt = make_dry_buses(scene, tts)
    stages["dry_synthesis_wall_s"] = time.perf_counter()-stamp
    stamp = time.perf_counter()
    full_stems, rir_plan = {}, {}
    for source, spec in scene["sources"].items():
        track = spec["trajectory"]
        keys = plan_rir_keyframes(track, scene["duration_s"], np.array(scene["listener_m"]),
                                  max_seconds=max_seconds, max_distance_m=max_distance_m, max_angle_deg=max_angle_deg)
        # A truly static source/listener/material condition has exactly one filter.
        if all(np.array_equal(np.asarray(k["position_m"]),keys[0]["position_m"]) for k in keys):
            keys = keys[:1]
        filters = [cache.get(k["position_m"]) for k in keys]
        conv_start = time.perf_counter()
        full_stems[source] = partitioned_convolve(dry[source], filters, [k["sample"] for k in keys])
        rir_plan[source] = {"keyframes": keys, "unique_positions": len({json.dumps(k['position_m']) for k in keys}),
                            "convolution_wall_s": time.perf_counter()-conv_start,
                            "full_tail_samples": full_stems[source].shape[1]}
    stages["rir_and_convolution_wall_s"] = time.perf_counter()-stamp
    full_length = max(a.shape[1] for a in full_stems.values())
    mix = np.zeros((4,full_length))
    for source, a in full_stems.items():
        mix[:, :a.shape[1]] += a
        write_float32_wav(attempt/f"private_{source}_raw_full_tail.wav", a,SAMPLE_RATE,channel_axis=0)
    camera = raw_world_to_camera_sn3d(mix,scene["camera_forward"])
    write_float32_wav(attempt/"scene_foa_full_tail.wav",camera,SAMPLE_RATE,channel_axis=0)
    stamp = time.perf_counter()
    frame_ids = {round((start+.5)*FPS) for start in scene["window_starts_s"]}
    masks = video.encode(scene, attempt/"scene_video.mp4", frame_ids)
    write_json(attempt/"private_video_markers.json",masks)
    stages["video_wall_s"] = time.perf_counter()-stamp
    stamp = time.perf_counter()
    public, private, windows = [], [], []
    for i, window_start in enumerate(scene["window_starts_s"]):
        window_id = scene["scene_id"]+f"__w{i:02d}"
        folder = attempt/f"w{i:02d}"; folder.mkdir()
        lo, hi = round(window_start*SAMPLE_RATE), round((window_start+10)*SAMPLE_RATE)
        write_float32_wav(folder/"audio_foa.wav",camera[:,lo:hi],SAMPLE_RATE,channel_axis=0)
        write_float32_wav(folder/"audio_mono.wav",camera[:1,lo:hi],SAMPLE_RATE,channel_axis=0)
        # CPU frame-exact window extraction; no keyframe seek approximation.
        first = round(window_start*FPS)
        # 100 ms is not always a 15 fps frame; select the frame at/before window start.
        # This pilot constrains starts to 200 ms below in the worker plan.
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-n","-threads","1","-i",str(attempt/"scene_video.mp4"),
                        "-vf",f"select=between(n\\,{first}\\,{first+149}),setpts=N/({FPS}*TB)","-frames:v","150","-an",
                        "-c:v","libx264","-threads","1","-preset","veryfast","-crf","18","-pix_fmt","yuv420p",str(folder/"video.mp4")],check=True)
        media = {"video":"video.mp4","audio_foa":"audio_foa.wav","audio_mono":"audio_mono.wav",
                 "foa_format":"ACN_WYZX_SN3D_camera","video_visual_status":"CPU_proxy_research_only"}
        q,a, audit = generate_v02_questions(scene, window_start,window_id,media)
        write_json(folder/"public_requests.json",q);write_json(folder/"private_answers.json",a);write_json(folder/"private_protocol_audit.json",audit)
        public.extend(q); private.extend(a)
        windows.append({"window_id":window_id,"start_s":window_start,"path":str(folder.relative_to(output)),
                        "coverage":audit["coverage"],"question_count":len(q),"parent_scene_id":scene["independent_parent_scene_id"]})
    stages["windows_and_qa_wall_s"] = time.perf_counter()-stamp
    write_json(attempt/"scene_plan.json",scene); write_json(attempt/"dry_assets_receipt.json",dry_receipt)
    write_json(attempt/"rir_plan.json",rir_plan)
    child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = child_end.ru_utime+child_end.ru_stime-child_start.ru_utime-child_start.ru_stime
    receipt = {"scene":scene,"rir_update_policy":policy,"attempt":str(attempt.relative_to(output)),"wall_seconds":time.perf_counter()-start,
               "process_cpu_seconds":time.process_time()-cpu,"child_cpu_seconds":child_cpu,
               "total_cpu_seconds":time.process_time()-cpu+child_cpu,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               "stages":stages,"cache_delta":{k:cache.stats[k]-before_cache[k] for k in cache.stats},
               "windows":windows,"question_count":len(public),"compute_device":"CPU","gpu_renderer_created":False,
               "native_RLR_verified":True,"native_Habitat_RGB_verified":False,
               "required_files":[str(p.relative_to(attempt)) for p in attempt.rglob('*') if p.is_file() and not p.name.startswith('.') and p.suffix != '.partial']}
    temp=output/f".completed_{uuid.uuid4().hex}.partial";write_json(temp,receipt)
    os.link(temp,completed)
    return receipt
