/* AVEngine Studio 3D editor.
 *
 * Loads a scene bundle (clay mesh from the M3 acoustic package plus a draft
 * navmesh grid), lets the user drag authoring markers like in a DCC viewport,
 * shows millisecond draft validation and acoustic-cue previews, and submits
 * render tasks to the engine queue. All checks here are drafts: the native
 * gates inside the render chain remain the authority.
 */
"use strict";

const FRAME_COUNT = 75;
const state = {
  scene: null, camera: null, renderer: null, controls: null,
  bundle: null, roomId: null, floorY: 0,
  grid: null,               // {rows, cols, bits(Uint8Array cells), bounds, mpp}
  markers: {},              // label -> THREE.Mesh
  markerOrder: [],
  selectedMarker: null, dragging: false,
  trajectories: null,       // {source1: [[x,y,z]x75], source2: ...}
  listener: null,           // {position:[x,y,z], forward:[x,z] unit}
  frame: 0, playing: false,
  routeTaskId: null, lastSeed: null,
  raycaster: new THREE.Raycaster(), pointer: new THREE.Vector2(),
  floorPlane: null, actorDots: [],
};

/* ---------- boot ---------- */

init3d();
loadScenes();
setInterval(refreshTasks, 4000);
refreshTasks();

function init3d() {
  const canvas = document.getElementById("canvas3d");
  state.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  state.renderer.setPixelRatio(window.devicePixelRatio);
  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color(0x1e1f22);
  state.camera = new THREE.PerspectiveCamera(55, 1, 0.05, 500);
  state.camera.position.set(6, 7, 6);
  state.controls = new THREE.OrbitControls(state.camera, canvas);
  state.controls.enableDamping = true;

  state.scene.add(new THREE.HemisphereLight(0xf4f2ec, 0x35363b, 1.05));
  const sun = new THREE.DirectionalLight(0xffffff, 0.9);
  sun.position.set(8, 14, 6);
  state.scene.add(sun);

  window.addEventListener("resize", resize);
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  resize();
  requestAnimationFrame(tick);
}

function resize() {
  const box = document.getElementById("viewport").getBoundingClientRect();
  state.renderer.setSize(box.width, box.height, false);
  state.camera.aspect = box.width / box.height;
  state.camera.updateProjectionMatrix();
}

function tick(timeMs) {
  requestAnimationFrame(tick);
  state.controls.update();
  if (state.playing) {
    setFrame(Math.floor((timeMs / 1000) * 15) % FRAME_COUNT);
  }
  updateActorDots();
  state.renderer.render(state.scene, state.camera);
}

/* ---------- scene bundles ---------- */

async function loadScenes() {
  const payload = await (await fetch("/api/scenes")).json();
  const select = document.getElementById("sceneSelect");
  select.innerHTML = "";
  for (const scene of payload.scenes) {
    const option = document.createElement("option");
    option.value = scene.room_id;
    option.textContent = `${scene.display_name} (${scene.triangle_count} tris)`;
    select.appendChild(option);
  }
  select.onchange = () => loadBundle(select.value);
  if (payload.scenes.length) loadBundle(payload.scenes[0].room_id);
  else select.innerHTML = "<option>无场景包：先运行 build_studio_scene_bundle</option>";
}

async function fetchBuffer(roomId, name, attempt = 0) {
  try {
    const response = await fetch(`/api/scenes/${roomId}/files/${name}`);
    if (!response.ok) throw new Error(`fetch ${name}: ${response.status}`);
    return await response.arrayBuffer();
  } catch (error) {
    if (attempt >= 2) throw error;
    await new Promise((resolve) => setTimeout(resolve, 800));
    return fetchBuffer(roomId, name, attempt + 1);
  }
}

async function loadBundle(roomId) {
  try {
    await loadBundleInner(roomId);
  } catch (error) {
    document.getElementById("sceneHint").textContent = `场景加载失败：${error.message}（自动重试…）`;
    await new Promise((resolve) => setTimeout(resolve, 1200));
    try {
      await loadBundleInner(roomId);
    } catch (retryError) {
      document.getElementById("sceneHint").textContent = `场景加载失败：${retryError.message}`;
    }
  }
}

async function loadBundleInner(roomId) {
  state.roomId = roomId;
  const bundle = await (await fetch(`/api/scenes/${roomId}/bundle.json`)).json();
  state.bundle = bundle;
  document.getElementById("sceneHint").textContent =
    `${bundle.room_id} · 声学网格黏土视图 · ` +
    (bundle.obstacle_map ? "含草稿 navmesh 栅格" : "无栅格（提交时由引擎权威校验）");

  clearRoom();
  const [positions, indices, materialIds] = await Promise.all([
    fetchBuffer(roomId, "mesh_positions.bin"),
    fetchBuffer(roomId, "mesh_indices.bin"),
    fetchBuffer(roomId, "mesh_material_ids.bin"),
  ]);
  buildRoomMesh(new Float32Array(positions), new Uint32Array(indices),
                new Uint32Array(materialIds), bundle.mesh.materials);
  buildGrid(bundle);
  applyRoofClip();
  frameCameraOnBounds(bundle.mesh.bounds_m);
  setupAuthoring(bundle);
  runDraftValidation();
}

function applyRoofClip() {
  const room = state.scene.getObjectByName("room");
  if (!room) return;
  const enabled = document.getElementById("roofClip").checked;
  state.renderer.localClippingEnabled = true;
  room.material.clippingPlanes = enabled
    ? [new THREE.Plane(new THREE.Vector3(0, -1, 0), state.floorY + 2.0)]
    : [];
  room.material.needsUpdate = true;
}

function clearRoom() {
  for (const name of ["room", "navgrid", "markers", "paths", "walkable-overlay"]) {
    const old = state.scene.getObjectByName(name);
    if (old) state.scene.remove(old);
  }
  state.markers = {}; state.markerOrder = []; state.trajectories = null;
  state.actorDots = [];
}

function buildRoomMesh(positions, indices, materialIds, materials) {
  const colorById = new Map(materials.map((m) => [m.id, new THREE.Color(m.color)]));
  const fallback = new THREE.Color("#8d99ae");
  const triCount = materialIds.length;
  if (triCount > 1500000) {
    // very large mesh: keep it indexed with a uniform clay color to hold
    // GPU memory down (per-face colors need de-indexing)
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
      color: 0x9aa3b2, flatShading: false, roughness: 0.95, metalness: 0.0,
      side: THREE.DoubleSide,
    }));
    mesh.name = "room";
    state.scene.add(mesh);
    return;
  }
  const flat = new Float32Array(triCount * 9);
  const colors = new Float32Array(triCount * 9);
  for (let t = 0; t < triCount; t++) {
    const color = colorById.get(materialIds[t]) || fallback;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = indices[t * 3 + corner];
      const out = t * 9 + corner * 3;
      flat[out] = positions[vertex * 3];
      flat[out + 1] = positions[vertex * 3 + 1];
      flat[out + 2] = positions[vertex * 3 + 2];
      colors[out] = color.r; colors[out + 1] = color.g; colors[out + 2] = color.b;
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(flat, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true, flatShading: true, roughness: 0.95, metalness: 0.0,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "room";
  state.scene.add(mesh);
}

function buildGrid(bundle) {
  state.grid = null;
  const om = bundle.obstacle_map;
  const bounds = om ? om.bounds_m : bundle.mesh.bounds_m;
  state.floorY = om ? om.floor_height_m : bounds[0][1];
  // invisible floor plane for drag raycasts
  const size = Math.max(bounds[1][0] - bounds[0][0], bounds[1][2] - bounds[0][2]) * 2;
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(size, size),
    new THREE.MeshBasicMaterial({ visible: false }),
  );
  plane.rotation.x = -Math.PI / 2;
  plane.position.set((bounds[0][0] + bounds[1][0]) / 2, state.floorY,
                     (bounds[0][2] + bounds[1][2]) / 2);
  plane.name = "navgrid";
  state.scene.add(plane);
  state.floorPlane = plane;
  if (!om) return;
  const raw = atob(om.navmesh_grid_packbits_b64);
  const packed = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) packed[i] = raw.charCodeAt(i);
  const [rows, cols] = om.grid_shape;
  const bits = new Uint8Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) {
    bits[i] = (packed[i >> 3] >> (7 - (i & 7))) & 1;
  }
  state.grid = { rows, cols, bits, bounds: om.bounds_m, mpp: om.meters_per_pixel };
  drawWalkableOverlay(om);
}

function drawWalkableOverlay(om) {
  // Translucent green carpet over every draft-walkable cell.
  const grid = state.grid;
  const data = new Uint8Array(grid.rows * grid.cols * 4);
  for (let i = 0; i < grid.rows * grid.cols; i++) {
    const walkable = grid.bits[i];
    data[i * 4] = 40; data[i * 4 + 1] = walkable ? 220 : 0;
    data[i * 4 + 2] = 90; data[i * 4 + 3] = walkable ? 95 : 0;
  }
  const texture = new THREE.DataTexture(data, grid.cols, grid.rows, THREE.RGBAFormat);
  texture.needsUpdate = true;
  texture.magFilter = THREE.NearestFilter;
  const width = grid.cols * grid.mpp;
  const depth = grid.rows * grid.mpp;
  const overlay = new THREE.Mesh(
    new THREE.PlaneGeometry(width, depth),
    new THREE.MeshBasicMaterial({
      map: texture, transparent: true, depthWrite: false, depthTest: false,
      side: THREE.DoubleSide, opacity: 0.85,
    }),
  );
  overlay.renderOrder = 2; // draw through the floor slab regardless of heights
  overlay.rotation.x = -Math.PI / 2;
  // PlaneGeometry maps texture v from bottom; after the -90° X rotation the
  // plane's local +Y ends up along world -Z, so texture rows follow +Z when
  // we flip v — verified visually against the room mesh.
  overlay.scale.y = -1;
  overlay.position.set(
    om.bounds_m[0][0] + width / 2,
    state.floorY + 0.03,
    om.bounds_m[0][2] + depth / 2,
  );
  overlay.name = "walkable-overlay";
  const old = state.scene.getObjectByName("walkable-overlay");
  if (old) state.scene.remove(old);
  state.scene.add(overlay);
}

function gridWalkable(x, z) {
  const grid = state.grid;
  if (!grid) return null;
  const row = Math.floor((z - grid.bounds[0][2]) / grid.mpp);
  const col = Math.floor((x - grid.bounds[0][0]) / grid.mpp);
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      const r = row + dr, c = col + dc;
      if (r >= 0 && r < grid.rows && c >= 0 && c < grid.cols &&
          grid.bits[r * grid.cols + c]) return true;
    }
  }
  return false;
}

function frameCameraOnBounds(bounds) {
  const center = new THREE.Vector3(
    (bounds[0][0] + bounds[1][0]) / 2,
    (bounds[0][1] + bounds[1][1]) / 2,
    (bounds[0][2] + bounds[1][2]) / 2,
  );
  const span = Math.max(bounds[1][0] - bounds[0][0], bounds[1][2] - bounds[0][2]);
  state.controls.target.copy(center);
  state.camera.position.set(center.x + span * 0.8, center.y + span * 0.9, center.z + span * 0.8);
}

/* ---------- authoring markers ---------- */

const MARKER_STYLES = {
  camera:       { color: 0xb0b0b0, label: "相机（锁定）", locked: true },
  human_start:  { color: 0x4f83ff, label: "人类 起点" },
  human_end:    { color: 0x9fc0ff, label: "人类 终点" },
  beagle_start: { color: 0xff9440, label: "比格犬 起点" },
  beagle_end:   { color: 0xffc79a, label: "比格犬 终点" },
};

function ueCmToWorld(ue) { return [ue[0] / 100, ue[2] / 100, ue[1] / 100]; }
function worldToUeCm(p)  { return [p[0] * 100, p[2] * 100, p[1] * 100]; }

function setupAuthoring(bundle) {
  const mode = (bundle.authoring || {}).mode;
  document.getElementById("apartmentPanel").style.display =
    mode === "explicit_points" ? "block" : "none";
  document.getElementById("mp3dPanel").style.display =
    mode === "seed_route" ? "block" : "none";
  const group = new THREE.Group();
  group.name = "markers";
  state.scene.add(group);

  if (mode === "explicit_points") {
    const authoring = bundle.authoring;
    for (const key of Object.keys(MARKER_STYLES)) {
      const ue = authoring.defaults_ue_cm[key];
      if (!ue) continue;
      const world = ueCmToWorld(ue);
      addMarker(group, key, world, MARKER_STYLES[key]);
    }
    state.listener = {
      position: ueCmToWorld(authoring.defaults_ue_cm.camera),
      // UE camera yaw → habitat yaw (see avengine.m7.apartment_dynamic_audio)
      yawDeg: -90 - authoring.camera_yaw_deg,
    };
    rebuildApartmentTrajectories();
    renderMarkerList();
    document.getElementById("btnApartmentValidate").onclick = () =>
      submitTask("apartment_author", apartmentOverrides(), "作者化校验");
    document.getElementById("btnApartmentRender").onclick = () =>
      submitTask("apartment_end_to_end", apartmentOverrides(), "公寓完整渲染");
  } else if (mode === "seed_route") {
    document.getElementById("btnRoutePreview").onclick = submitRoutePreview;
    document.getElementById("btnMp3dRender").onclick = submitMp3dRender;
  }
}

function addMarker(group, key, world, style) {
  const isCamera = key === "camera";
  const geometry = isCamera
    ? new THREE.ConeGeometry(0.14, 0.34, 4)
    : new THREE.SphereGeometry(0.12, 20, 14);
  const material = new THREE.MeshStandardMaterial({
    color: style.color, emissive: style.color, emissiveIntensity: 0.35,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(world[0], world[1] + (isCamera ? 0 : 0.12), world[2]);
  mesh.userData = { key, locked: !!style.locked, baseColor: style.color };
  group.add(mesh);
  state.markers[key] = mesh;
  state.markerOrder.push(key);
}

function markerWorld(key) {
  const mesh = state.markers[key];
  return [mesh.position.x, state.floorY, mesh.position.z];
}

function apartmentOverrides() {
  const authoring = state.bundle.authoring;
  const overrides = { camera_yaw_deg: authoring.camera_yaw_deg };
  overrides.camera_position_ue_cm = authoring.defaults_ue_cm.camera;
  for (const [key, param] of [
    ["human_start", "human_start_ue_cm"], ["human_end", "human_end_ue_cm"],
    ["beagle_start", "beagle_start_ue_cm"], ["beagle_end", "beagle_end_ue_cm"],
  ]) {
    const world = markerWorld(key);
    const defaults = authoring.defaults_ue_cm[key];
    const ue = worldToUeCm(world);
    ue[2] = defaults[2]; // keep the authored UE height; only X/Y move on drag
    overrides[param] = [Math.round(ue[0] * 10) / 10, Math.round(ue[1] * 10) / 10, defaults[2]];
  }
  return overrides;
}

function rebuildApartmentTrajectories() {
  const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
  const t1 = [], t2 = [];
  const hs = markerWorld("human_start"), he = markerWorld("human_end");
  const bs = markerWorld("beagle_start"), be = markerWorld("beagle_end");
  for (let f = 0; f < FRAME_COUNT; f++) {
    const t = f / (FRAME_COUNT - 1);
    t1.push(lerp(hs, he, t));
    t2.push(lerp(bs, be, t));
  }
  state.trajectories = { source1: t1, source2: t2 };
  drawPaths();
  updateCues();
}

function drawPaths() {
  const old = state.scene.getObjectByName("paths");
  if (old) state.scene.remove(old);
  if (!state.trajectories) return;
  const group = new THREE.Group();
  group.name = "paths";
  const colors = { source1: 0x4f83ff, source2: 0xff9440 };
  state.actorDots = [];
  for (const [key, points] of Object.entries(state.trajectories)) {
    const vertices = points.map((p) => new THREE.Vector3(p[0], p[1] + 0.06, p[2]));
    const geometry = new THREE.BufferGeometry().setFromPoints(vertices);
    group.add(new THREE.Line(geometry,
      new THREE.LineBasicMaterial({ color: colors[key] || 0xffffff })));
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.09, 14, 10),
      new THREE.MeshBasicMaterial({ color: colors[key] || 0xffffff }),
    );
    dot.userData.trajectory = key;
    group.add(dot);
    state.actorDots.push(dot);
  }
  if (state.listener) {
    const p = state.listener.position;
    const listenerMark = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.13),
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true }),
    );
    listenerMark.position.set(p[0], p[1], p[2]);
    group.add(listenerMark);
  }
  state.scene.add(group);
}

function updateActorDots() {
  if (!state.trajectories) return;
  for (const dot of state.actorDots) {
    const points = state.trajectories[dot.userData.trajectory];
    if (!points) continue;
    const p = points[Math.min(state.frame, points.length - 1)];
    dot.position.set(p[0], p[1] + 0.1, p[2]);
  }
}

function renderMarkerList() {
  const container = document.getElementById("markerList");
  container.innerHTML = "";
  for (const key of state.markerOrder) {
    const style = MARKER_STYLES[key];
    const row = document.createElement("div");
    const world = markerWorld(key);
    row.innerHTML = `<span>${style.label}</span>` +
      `<span style="color:#${style.color.toString(16).padStart(6, "0")}">` +
      `${world[0].toFixed(2)}, ${world[2].toFixed(2)}</span>`;
    row.onclick = () => selectMarker(key);
    if (state.selectedMarker === key) row.classList.add("selected");
    container.appendChild(row);
  }
}

function selectMarker(key) {
  state.selectedMarker = key;
  renderMarkerList();
}

/* ---------- pointer dragging ---------- */

function pointerRay(event) {
  const rect = state.renderer.domElement.getBoundingClientRect();
  state.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  state.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  state.raycaster.setFromCamera(state.pointer, state.camera);
}

function onPointerDown(event) {
  if (!state.bundle) return;
  pointerRay(event);
  const markers = Object.values(state.markers).filter((m) => !m.userData.locked);
  const hits = state.raycaster.intersectObjects(markers, false);
  if (hits.length) {
    state.selectedMarker = hits[0].object.userData.key;
    state.dragging = true;
    state.controls.enabled = false;
    renderMarkerList();
  }
}

function onPointerMove(event) {
  if (!state.dragging || !state.selectedMarker || !state.floorPlane) return;
  pointerRay(event);
  const hits = state.raycaster.intersectObject(state.floorPlane, false);
  if (!hits.length) return;
  const point = hits[0].point;
  const mesh = state.markers[state.selectedMarker];
  mesh.position.set(point.x, state.floorY + 0.12, point.z);
  const walkable = gridWalkable(point.x, point.z);
  mesh.material.color.set(walkable === false ? 0xe5534b : mesh.userData.baseColor);
  if (state.bundle.authoring?.mode === "explicit_points") rebuildApartmentTrajectories();
  renderMarkerList();
}

function onPointerUp() {
  if (!state.dragging) return;
  state.dragging = false;
  state.controls.enabled = true;
  runDraftValidation();
}

/* ---------- draft validation ---------- */

async function runDraftValidation() {
  const container = document.getElementById("validation");
  if (!state.bundle) return;
  const points = state.markerOrder
    .filter((key) => !state.markers[key].userData.locked)
    .map((key) => ({ label: key, position_m: markerWorld(key) }));
  if (!points.length) { container.textContent = "此场景通过任务提交时校验。"; return; }
  const response = await fetch("/api/validate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ room_id: state.roomId, points }),
  });
  const result = await response.json();
  if (!response.ok) { container.textContent = result.error; return; }
  if (result.all_ok === null) { container.textContent = result.claim; return; }
  container.innerHTML = result.points.map((p) =>
    `<div><span class="badge ${p.ok ? "ok" : "bad"}">${p.ok ? "可放置" : "不可放置"}</span> ` +
    `${MARKER_STYLES[p.label]?.label || p.label}` +
    (p.reason ? ` <span style="color:var(--dim)">${p.reason}</span>` : "") + "</div>"
  ).join("") + `<div class="hint">${result.claim}</div>`;
  for (const p of result.points) {
    const mesh = state.markers[p.label];
    if (mesh) mesh.material.color.set(p.ok ? mesh.userData.baseColor : 0xe5534b);
  }
}

/* ---------- acoustic cue preview (draft heuristics) ---------- */

function updateCues() {
  const canvas = document.getElementById("azimuthStrip");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#141518";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const warnings = [];
  if (!state.trajectories || !state.listener) {
    document.getElementById("warnings").innerHTML = "";
    return;
  }
  const yaw = (state.listener.yawDeg ?? 0) * Math.PI / 180;
  // habitat frame: forward = -Z rotated by yaw around +Y
  const forward = [-Math.sin(yaw), -Math.cos(yaw)];
  const right = [-forward[1], forward[0]];
  const listener = state.listener.position;
  const keys = Object.keys(state.trajectories);
  const azimuths = {};
  for (const key of keys) {
    azimuths[key] = state.trajectories[key].map((p) => {
      const vx = p[0] - listener[0], vz = p[2] - listener[2];
      return Math.atan2(vx * right[0] + vz * right[1],
                        vx * forward[0] + vz * forward[1]) * 180 / Math.PI;
    });
  }
  const laneHeight = canvas.height / keys.length;
  keys.forEach((key, lane) => {
    const values = azimuths[key];
    for (let f = 0; f < FRAME_COUNT; f++) {
      const az = Math.max(-90, Math.min(90, values[f]));
      const intensity = Math.abs(az) / 90;
      const channel = Math.round(120 + 135 * intensity);
      ctx.fillStyle = az < 0
        ? `rgb(60,100,${channel + 60 > 255 ? 255 : channel + 60})`
        : `rgb(${channel},70,60)`;
      const x = (f / FRAME_COUNT) * canvas.width;
      ctx.fillRect(x, lane * laneHeight + 2, canvas.width / FRAME_COUNT + 1, laneHeight - 4);
      if (state.grid && lineBlocked(listener, state.trajectories[key][f])) {
        ctx.fillStyle = "rgba(0,0,0,.55)";
        ctx.fillRect(x, lane * laneHeight + 2, 1.5, laneHeight - 4);
      }
    }
    const sweep = Math.max(...values) - Math.min(...values);
    if (sweep < 10) warnings.push(`⚠ ${key} 全程方位变化仅 ${sweep.toFixed(1)}°，左右线索弱`);
  });
  if (keys.length === 2) {
    const separation = azimuths[keys[0]].map((a, i) => Math.abs(a - azimuths[keys[1]][i]));
    const narrow = separation.filter((s) => s < 20).length;
    if (narrow > FRAME_COUNT / 2) {
      warnings.push(`⚠ 两源夹角 <20° 的帧占 ${(narrow / FRAME_COUNT * 100).toFixed(0)}%，空间可分性差`);
    }
  }
  document.getElementById("warnings").innerHTML = warnings.length
    ? warnings.map((w) => `<div><span class="badge warn">${w}</span></div>`).join("")
    : '<div><span class="badge ok">未触发防废案警示</span></div>';
  updateQuestionPreview(azimuths, listener);
}

function updateQuestionPreview(azimuths, listener) {
  // Draft heuristics only: which QA families this layout could support.
  // The engine QuestionSpec evaluation on real fact tables stays authoritative.
  const container = document.getElementById("questionPreview");
  if (!container) return;
  const keys = Object.keys(state.trajectories);
  const distances = {};
  for (const key of keys) {
    distances[key] = state.trajectories[key].map((p) => Math.hypot(
      p[0] - listener[0], p[2] - listener[2]));
  }
  const rows = [];
  const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
  for (const key of keys) {
    const az = azimuths[key];
    const med = median(az);
    const stableSign = az.filter((a) => Math.sign(a) === Math.sign(med)).length / az.length;
    const lateral = Math.abs(med) >= 15 && stableSign >= 0.8;
    rows.push([`左右判断 · ${key}`, lateral,
      lateral ? `中位方位 ${med.toFixed(0)}°` : `中位方位仅 ${med.toFixed(0)}° 或不稳定 → 拒答落点`]);
    const dd = distances[key][FRAME_COUNT - 1] - distances[key][0];
    const radial = Math.abs(dd) >= 0.5;
    rows.push([`趋近/远离 · ${key}`, radial,
      radial ? `${dd > 0 ? "远离" : "趋近"} ${Math.abs(dd).toFixed(1)}m` : `径向位移仅 ${Math.abs(dd).toFixed(1)}m → 拒答落点`]);
  }
  if (keys.length === 2) {
    const ratio = median(distances[keys[0]]) / median(distances[keys[1]]);
    const closer = ratio >= 1.3 || ratio <= 1 / 1.3;
    rows.push(["谁更近", closer,
      closer ? `距离比 ${ratio.toFixed(2)}` : `距离比 ${ratio.toFixed(2)} 太接近 → 拒答落点`]);
    const diff = azimuths[keys[0]].map((a, i) => a - azimuths[keys[1]][i]);
    const crossing = diff.some((d) => d > 5) && diff.some((d) => d < -5);
    rows.push(["方位交叉", crossing, crossing ? "两源相对方位发生交叉" : "全程无交叉"]);
    rows.push(["先后发声次序", true, "轮流发声 program 恒可产出"]);
  }
  container.innerHTML = rows.map(([name, ok, note]) =>
    `<div><span class="badge ${ok ? "ok" : "bad"}">${ok ? "可产出" : "不可产出"}</span> ` +
    `${name} <span style="color:var(--dim)">${note}</span></div>`).join("");
}

function lineBlocked(a, b) {
  // 2D DDA over the draft navmesh grid: any unwalkable cell blocks the ray.
  const steps = 48;
  for (let s = 1; s < steps; s++) {
    const t = s / steps;
    const x = a[0] + (b[0] - a[0]) * t;
    const z = a[2] + (b[2] - a[2]) * t;
    if (gridWalkable(x, z) === false) return true;
  }
  return false;
}

/* ---------- timeline ---------- */

document.getElementById("frameBar").oninput = (event) => setFrame(+event.target.value);
document.getElementById("btnPlay").onclick = () => {
  state.playing = !state.playing;
  document.getElementById("btnPlay").textContent = state.playing ? "暂停 ⏸" : "播放 ▶";
};

function setFrame(frame) {
  state.frame = frame;
  document.getElementById("frameBar").value = frame;
  document.getElementById("frameLabel").textContent = `帧 ${frame} / ${FRAME_COUNT - 1}`;
}

/* ---------- MP3D route preview ---------- */

async function submitRoutePreview() {
  const seed = +document.getElementById("seedInput").value;
  const cameraSelection = document.getElementById("cameraSelect").value;
  state.lastSeed = seed;
  const task = await submitTask("mp3d_route_author",
    { seed, camera_selection: cameraSelection }, "路线授权");
  if (!task) return;
  state.routeTaskId = task.task_id;
  pollRoutePreview(task.task_id);
}

async function pollRoutePreview(taskId) {
  const detail = await (await fetch(`/api/tasks/${taskId}`)).json();
  const status = detail.task.status;
  if (status === "queued" || status === "running") {
    setTimeout(() => pollRoutePreview(taskId), 3000);
    return;
  }
  if (status !== "pass") return;
  const explain = await fetchArtifactJson(taskId,
    "route/two_beagle_route_explanation.json");
  const m1 = await fetchArtifactJson(taskId, "route/research_m1_request.json");
  const trajectories = extractTrajectories(explain);
  if (trajectories) {
    state.trajectories = trajectories;
    state.listener = extractListener(m1) || state.listener;
    drawPaths();
    updateCues();
  }
  document.getElementById("btnMp3dRender").disabled = false;
}

async function fetchArtifactJson(taskId, relative) {
  const response = await fetch(
    `/api/tasks/${taskId}/artifact?path=${encodeURIComponent(relative)}`);
  return response.ok ? response.json() : null;
}

function extractTrajectories(node, found = {}) {
  // best-effort: any key holding a [>=25][3] finite numeric array is a path
  if (!node || typeof node !== "object") return null;
  for (const [key, value] of Object.entries(node)) {
    if (Array.isArray(value) && value.length >= 25 &&
        Array.isArray(value[0]) && value[0].length === 3 &&
        value.every((p) => Array.isArray(p) && p.length === 3 &&
                     p.every((n) => typeof n === "number" && isFinite(n)))) {
      found[key] = value;
    } else if (typeof value === "object") {
      extractTrajectories(value, found);
    }
  }
  const actorKeys = Object.keys(found).filter((k) => /actor/i.test(k));
  if (actorKeys.length >= 2) {
    return Object.fromEntries(actorKeys.map((k) => [k, found[k]]));
  }
  return Object.keys(found).length ? found : null;
}

function extractListener(node) {
  if (!node || typeof node !== "object") return null;
  // prefer the primary camera rig's world_from_rig transform (M1 request)
  const rig = node.primary_camera_rig?.world_from_rig;
  let position = null, quaternion = null;
  if (rig && Array.isArray(rig.translation_m)) {
    position = rig.translation_m;
    quaternion = rig.rotation_xyzw || null;
  } else {
    (function walk(current) {
      if (!current || typeof current !== "object") return;
      for (const [key, value] of Object.entries(current)) {
        if (!position && /position_m$|translation_m$/.test(key) &&
            Array.isArray(value) && value.length === 3) {
          position = value;
        }
        if (!quaternion && /orientation|rotation/.test(key) &&
            Array.isArray(value) && value.length === 4) {
          quaternion = value;
        }
        if (typeof value === "object") walk(value);
      }
    })(node);
  }
  if (!position) return null;
  let yawDeg = 0;
  if (quaternion) {
    const q = new THREE.Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(q);
    yawDeg = Math.atan2(-dir.x, -dir.z) * 180 / Math.PI;
  }
  return { position, yawDeg };
}

async function submitMp3dRender() {
  const cameraSelection = document.getElementById("cameraSelect").value;
  await submitTask("mp3d_end_to_end",
    { seed: state.lastSeed ?? +document.getElementById("seedInput").value,
      camera_selection: cameraSelection },
    "MP3D 完整渲染");
}

/* ---------- tasks ---------- */

async function submitTask(template, overrides, label) {
  const response = await fetch("/api/tasks", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ template, overrides }),
  });
  const body = await response.json();
  if (!response.ok) { alert(`${label} 提交失败：${body.error}`); return null; }
  refreshTasks();
  return body.task;
}

async function refreshTasks() {
  const payload = await (await fetch("/api/tasks")).json().catch(() => null);
  if (!payload) return;
  const container = document.getElementById("tasks");
  container.innerHTML = "";
  for (const task of payload.tasks.slice(0, 8)) {
    const div = document.createElement("div");
    div.className = "task";
    const clip = (task.artifacts || []).find(
      (a) => a.endsWith(".mp4") && !a.endsWith(".base.mp4"));
    div.innerHTML =
      `<span class="badge ${task.status}">${task.status}</span> ` +
      `${task.template}<br><span style="color:var(--dim)">${task.task_id}</span>` +
      (clip ? ` · <a href="#" data-task="${task.task_id}" data-clip="${clip}">▶ 播放成片</a>` : "");
    const link = div.querySelector("a[data-clip]");
    if (link) link.onclick = (event) => {
      event.preventDefault();
      const player = document.getElementById("player");
      player.src = `/api/tasks/${task.task_id}/artifact?path=${encodeURIComponent(clip)}`;
      player.style.display = "block";
      player.play();
    };
    container.appendChild(div);
  }
}
