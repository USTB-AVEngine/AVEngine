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
  state.viewSize = { width: box.width, height: box.height };
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
  const renderer = state.renderer;
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, state.viewSize.width, state.viewSize.height);
  renderer.render(state.scene, state.camera);
  renderMinimap();
}

function renderMinimap() {
  if (!state.miniCamera || !state.viewSize) return;
  const renderer = state.renderer;
  const width = 260, height = 180, pad = 11;
  const x = state.viewSize.width - width - pad;
  const y = state.viewSize.height - height - pad; // WebGL origin = bottom-left
  if (x < 40 || y < 40) return;
  // planning aids stay visible on the minimap even in FPV
  const toggled = [];
  for (const name of ["walkable-overlay", "camera-frustum"]) {
    const object = state.scene.getObjectByName(name);
    if (object && !object.visible) { object.visible = true; toggled.push(object); }
  }
  if (state.viewDot) {
    state.viewDot.visible = true;
    state.viewDot.position.set(
      state.camera.position.x, state.floorY + 0.2, state.camera.position.z);
  }
  renderer.setScissorTest(true);
  renderer.setScissor(x, y, width, height);
  renderer.setViewport(x, y, width, height);
  renderer.clearDepth();
  renderer.render(state.scene, state.miniCamera);
  renderer.setScissorTest(false);
  for (const object of toggled) object.visible = false;
  if (state.viewDot) state.viewDot.visible = false;
}

function buildMinimapCamera(bounds) {
  const cx = (bounds[0][0] + bounds[1][0]) / 2;
  const cz = (bounds[0][2] + bounds[1][2]) / 2;
  let halfW = ((bounds[1][0] - bounds[0][0]) / 2) * 1.08;
  let halfD = ((bounds[1][2] - bounds[0][2]) / 2) * 1.08;
  const aspect = 260 / 180;
  if (halfW / halfD < aspect) halfW = halfD * aspect;
  else halfD = halfW / aspect;
  const camera = new THREE.OrthographicCamera(-halfW, halfW, halfD, -halfD, 0.1, 200);
  camera.position.set(cx, bounds[1][1] + 20, cz);
  camera.up.set(0, 0, -1);
  camera.lookAt(cx, 0, cz);
  state.miniCamera = camera;
  if (!state.viewDot) {
    state.viewDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 10, 8),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    state.viewDot.visible = false;
    state.scene.add(state.viewDot);
  }
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
  if (payload.scenes.length) {
    // first paint fast: start with the lightest scene
    const lightest = [...payload.scenes].sort(
      (a, b) => (a.triangle_count || 0) - (b.triangle_count || 0))[0];
    select.value = lightest.room_id;
    loadBundle(lightest.room_id);
  } else {
    select.innerHTML = "<option>无场景包：先运行 build_studio_scene_bundle</option>";
  }
}

const loadProgress = { loaded: 0, total: 0 };

function showLoading(text) {
  const box = document.getElementById("loading");
  box.style.display = "block";
  document.getElementById("loadingText").textContent = text;
}

function setLoadingProgress() {
  if (!loadProgress.total) return;
  const percent = Math.min(100, (loadProgress.loaded / loadProgress.total) * 100);
  document.getElementById("loadingBar").style.width = percent.toFixed(1) + "%";
  document.getElementById("loadingText").textContent =
    `下载场景网格 ${(loadProgress.loaded / 1048576).toFixed(1)} / ` +
    `${(loadProgress.total / 1048576).toFixed(1)} MB`;
}

function hideLoading() {
  document.getElementById("loading").style.display = "none";
}

async function fetchBuffer(roomId, name, attempt = 0) {
  try {
    const response = await fetch(`/api/scenes/${roomId}/files/${name}`);
    if (!response.ok) throw new Error(`fetch ${name}: ${response.status}`);
    const total = Number(response.headers.get("Content-Length") || 0);
    loadProgress.total += total;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      loadProgress.loaded += value.length;
      setLoadingProgress();
    }
    const buffer = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
    return buffer.buffer;
  } catch (error) {
    if (attempt >= 2) throw error;
    await new Promise((resolve) => setTimeout(resolve, 800));
    return fetchBuffer(roomId, name, attempt + 1);
  }
}

async function loadBundle(roomId) {
  try {
    await loadBundleInner(roomId);
    hideLoading();
  } catch (error) {
    showLoading(`场景加载失败：${error.message}，自动重试…`);
    await new Promise((resolve) => setTimeout(resolve, 1200));
    try {
      await loadBundleInner(roomId);
      hideLoading();
    } catch (retryError) {
      showLoading(`场景加载失败：${retryError.message}（刷新页面重试）`);
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
  loadProgress.loaded = 0;
  loadProgress.total = 0;
  showLoading("下载场景网格…");
  const [positions, indices, materialIds] = await Promise.all([
    fetchBuffer(roomId, "mesh_positions.bin"),
    fetchBuffer(roomId, "mesh_indices.bin"),
    fetchBuffer(roomId, "mesh_material_ids.bin"),
  ]);
  showLoading("构建几何与法线…（大场景需数秒）");
  await new Promise((resolve) => requestAnimationFrame(() =>
    requestAnimationFrame(resolve)));
  buildRoomMesh(new Float32Array(positions), new Uint32Array(indices),
                new Uint32Array(materialIds), bundle.mesh.materials);
  buildGrid(bundle);
  buildMinimapCamera(bundle.mesh.bounds_m);
  applyRoofClip();
  frameCameraOnBounds(bundle.mesh.bounds_m);

  document.getElementById("texturedRow").style.display =
    bundle.textured_mesh ? "flex" : "none";
  // real textures by default whenever the room has them
  document.getElementById("texturedView").checked = !!bundle.textured_mesh;
  const refFrame = document.getElementById("refFrame");
  if (bundle.reference_frame) {
    refFrame.src = `/api/scenes/${roomId}/files/reference_frame.png`;
    refFrame.style.width = "240px";
    refFrame.style.display = "block";
  } else {
    refFrame.style.display = "none";
  }

  await loadActorModels(bundle);
  setupAuthoring(bundle);
  runDraftValidation();
  if (bundle.textured_mesh) await applyTexturedView();
}

const actorTemplateCache = new Map();

async function loadActorModels(bundle) {
  state.actorTemplates = {};
  const models = bundle.actor_models || {};
  const hints = bundle.authoring?.actor_display || {};
  const loader = new THREE.GLTFLoader();
  for (const name of Object.keys(models)) {
    try {
      if (!actorTemplateCache.has(name)) {
        showLoading(`下载角色模型 ${name}…`);
        const gltf = await new Promise((resolve, reject) => loader.load(
          `/api/scenes/${state.roomId}/files/actor_${name}.glb`, resolve,
          (event) => {
            if (event.total) {
              document.getElementById("loadingText").textContent =
                `下载角色模型 ${name} ${(event.loaded / 1048576).toFixed(1)} / ` +
                `${(event.total / 1048576).toFixed(1)} MB`;
            }
          },
          reject));
        actorTemplateCache.set(name, gltf.scene);
      }
      const template = actorTemplateCache.get(name).clone(true);
      const hint = hints[name] || {};
      if (hint.rotate_x_deg) template.rotation.x = hint.rotate_x_deg * Math.PI / 180;
      // skinned meshes render through bone transforms, which makes both the
      // bounding box and any root scaling unreliable — replace them with
      // static meshes in bind pose (fine for a placement stand-in)
      const skinnedNodes = [];
      template.traverse((node) => { if (node.isSkinnedMesh) skinnedNodes.push(node); });
      for (const node of skinnedNodes) {
        const static_ = new THREE.Mesh(node.geometry, node.material);
        static_.position.copy(node.position);
        static_.quaternion.copy(node.quaternion);
        static_.scale.copy(node.scale);
        node.parent.add(static_);
        node.parent.remove(node);
      }
      template.updateMatrixWorld(true);
      let box = new THREE.Box3().setFromObject(template);
      const size = new THREE.Vector3();
      box.getSize(size);
      // pick the unit scale that puts the standing height in a plausible
      // actor range — UE assets are cm, some habitat assets carry bone-scale
      // compensated tiny geometry
      let scale = hint.scale || 1.0;
      if (!hint.scale) {
        for (const candidate of [0.01, 0.1, 1.0, 10.0, 100.0]) {
          if (size.y * candidate >= 0.2 && size.y * candidate <= 2.5) {
            scale = candidate;
            break;
          }
        }
      }
      template.scale.setScalar(scale);
      template.updateMatrixWorld(true);
      box = new THREE.Box3().setFromObject(template);
      // stand the model on the floor: shift so its bbox bottom sits at y=0
      const wrapper = new THREE.Group();
      wrapper.add(template);
      template.position.y -= box.min.y;
      state.actorTemplates[name] = wrapper;
    } catch (error) {
      console.warn("actor model failed:", name, error);
    }
  }
}

async function applyTexturedView() {
  const on = document.getElementById("texturedView").checked;
  const clay = state.scene.getObjectByName("room");
  let textured = state.scene.getObjectByName("room-textured");
  if (on && !textured) {
    try {
      showLoading("下载贴图网格…（数据集真实外观）");
      const loader = new THREE.GLTFLoader();
      const gltf = await new Promise((resolve, reject) => loader.load(
        `/api/scenes/${state.roomId}/files/textured.glb`, resolve,
        (event) => {
          if (event.total) {
            document.getElementById("loadingText").textContent =
              `下载贴图网格 ${(event.loaded / 1048576).toFixed(1)} / ` +
              `${(event.total / 1048576).toFixed(1)} MB`;
          }
        },
        reject));
      textured = gltf.scene;
      textured.name = "room-textured";
      alignTexturedToClay(textured);
      state.scene.add(textured);
      await loadComposition();
      hideLoading();
    } catch (error) {
      showLoading(`贴图网格加载失败：${error.message || error}`);
      document.getElementById("texturedView").checked = false;
      setTimeout(hideLoading, 4000);
      return;
    }
  }
  if (textured) textured.visible = on;
  const objects = state.scene.getObjectByName("room-objects");
  if (objects) objects.visible = on;
  if (clay) clay.visible = !on;
  applyRoofClip();
}

async function loadComposition() {
  // full scene composition: dataset object glbs placed per scene_instance
  if (!state.bundle.composition) return;
  if (state.scene.getObjectByName("room-objects")) return;
  const response = await fetch(`/api/scenes/${state.roomId}/files/composition.json`);
  if (!response.ok) return;
  const composition = await response.json();
  const group = new THREE.Group();
  group.name = "room-objects";
  state.scene.add(group);
  const loader = new THREE.GLTFLoader();
  const cache = new Map();
  const loadGlb = (relative) => {
    if (!cache.has(relative)) {
      cache.set(relative, new Promise((resolve, reject) => loader.load(
        `/api/scenes/${state.roomId}/dataset/${relative}`, resolve, undefined, reject)));
    }
    return cache.get(relative);
  };
  let placed = 0;
  for (const record of composition.objects) {
    try {
      const gltf = await loadGlb(record.glb);
      const instance = gltf.scene.clone(true);
      instance.position.set(...record.translation);
      const [w, x, y, z] = record.rotation_wxyz;
      instance.quaternion.set(x, y, z, w);
      if (record.scale) instance.scale.set(...record.scale);
      group.add(instance);
    } catch (error) {
      console.warn("composition object failed:", record.glb, error);
    }
    placed += 1;
    document.getElementById("loadingText").textContent =
      `摆放物件 ${placed} / ${composition.objects.length}`;
  }
}

function alignTexturedToClay(textured) {
  // dataset glbs may be z-up; pick the orientation whose bounds best match
  // the acoustic clay mesh
  const clayBounds = state.bundle.mesh.bounds_m;
  const claySize = [
    clayBounds[1][0] - clayBounds[0][0],
    clayBounds[1][1] - clayBounds[0][1],
    clayBounds[1][2] - clayBounds[0][2],
  ];
  const measure = () => {
    const box = new THREE.Box3().setFromObject(textured);
    const size = new THREE.Vector3();
    box.getSize(size);
    return Math.abs(size.x - claySize[0]) + Math.abs(size.y - claySize[1]) +
           Math.abs(size.z - claySize[2]);
  };
  const identityError = measure();
  textured.rotation.x = -Math.PI / 2;
  textured.updateMatrixWorld(true);
  const rotatedError = measure();
  if (identityError <= rotatedError) {
    textured.rotation.x = 0;
    textured.updateMatrixWorld(true);
  }
  // snap centers so the textured shell sits on the clay footprint
  const box = new THREE.Box3().setFromObject(textured);
  const center = new THREE.Vector3();
  box.getCenter(center);
  textured.position.x += (clayBounds[0][0] + clayBounds[1][0]) / 2 - center.x;
  textured.position.z += (clayBounds[0][2] + clayBounds[1][2]) / 2 - center.z;
  textured.position.y += clayBounds[0][1] - box.min.y;
}

function applyRoofClip() {
  const enabled = document.getElementById("roofClip").checked;
  const planes = enabled
    ? [new THREE.Plane(new THREE.Vector3(0, -1, 0), state.floorY + 2.0)]
    : [];
  state.renderer.localClippingEnabled = true;
  for (const name of ["room", "room-textured", "room-objects"]) {
    const object = state.scene.getObjectByName(name);
    if (!object) continue;
    object.traverse((node) => {
      if (node.material) {
        node.material.clippingPlanes = planes;
        node.material.needsUpdate = true;
      }
    });
  }
}

/* ---------- view presets ---------- */

function renderCameraPose() {
  // The authoring/render camera: apartment = the locked M1 pose marker;
  // MP3D = the listener extracted from the authored route's M1 request.
  if (!state.listener) return null;
  const yaw = (state.listener.yawDeg ?? 0) * Math.PI / 180;
  return {
    position: state.listener.position,
    forward: [-Math.sin(yaw), 0, -Math.cos(yaw)],
    hfovDeg: state.bundle?.authoring?.hfov_degrees ?? 105.0,
  };
}

function setViewMode(mode) {
  const hint = document.getElementById("viewHint");
  const bounds = state.bundle ? state.bundle.mesh.bounds_m : null;
  if (!bounds) return;
  // the X-ray walkable overlay and the frustum helper are planning aids;
  // they only add noise inside the render camera's own view
  for (const name of ["walkable-overlay", "camera-frustum"]) {
    const object = state.scene.getObjectByName(name);
    if (object) object.visible = mode !== "fpv";
  }
  const center = [
    (bounds[0][0] + bounds[1][0]) / 2,
    (bounds[0][1] + bounds[1][1]) / 2,
    (bounds[0][2] + bounds[1][2]) / 2,
  ];
  const span = Math.max(bounds[1][0] - bounds[0][0], bounds[1][2] - bounds[0][2]);
  const roofClip = document.getElementById("roofClip");
  if (mode === "top") {
    if (!roofClip.checked) { roofClip.checked = true; applyRoofClip(); }
    state.camera.fov = 55;
    state.camera.position.set(center[0], state.floorY + span * 1.15, center[2] + 0.01);
    state.controls.target.set(center[0], state.floorY, center[2]);
    hint.textContent = "俯视图：滚轮缩放，右键平移。";
  } else if (mode === "fpv") {
    const pose = renderCameraPose();
    if (!pose) {
      hint.textContent = "第一视角需要相机位姿：公寓自带；MP3D 先生成路线预览。";
      return;
    }
    if (roofClip.checked) { roofClip.checked = false; applyRoofClip(); }
    const aspect = state.camera.aspect;
    const hfov = pose.hfovDeg * Math.PI / 180;
    state.camera.fov = 2 * Math.atan(Math.tan(hfov / 2) / aspect) * 180 / Math.PI;
    state.camera.position.set(...pose.position);
    state.controls.target.set(
      pose.position[0] + pose.forward[0] * 2.5,
      pose.position[1],
      pose.position[2] + pose.forward[2] * 2.5,
    );
    hint.textContent = "相机第一视角：渲染相机所见（视场与渲染 hfov 对齐）。";
  } else {
    if (!roofClip.checked) { roofClip.checked = true; applyRoofClip(); }
    state.camera.fov = 55;
    frameCameraOnBounds(bounds);
    hint.textContent = "";
  }
  state.camera.updateProjectionMatrix();
  state.controls.update();
}

function drawCameraFrustum() {
  const old = state.scene.getObjectByName("camera-frustum");
  if (old) state.scene.remove(old);
  const pose = renderCameraPose();
  if (!pose) return;
  const aspect = 1280 / 720;
  const hfov = pose.hfovDeg * Math.PI / 180;
  const vfovDeg = 2 * Math.atan(Math.tan(hfov / 2) / aspect) * 180 / Math.PI;
  const preview = new THREE.PerspectiveCamera(vfovDeg, aspect, 0.15, 3.0);
  preview.position.set(...pose.position);
  preview.lookAt(
    pose.position[0] + pose.forward[0],
    pose.position[1],
    pose.position[2] + pose.forward[2],
  );
  preview.updateMatrixWorld();
  const helper = new THREE.CameraHelper(preview);
  helper.name = "camera-frustum";
  state.scene.add(helper);
}

function clearRoom() {
  for (const name of ["room", "room-textured", "room-objects", "navgrid", "markers", "paths", "walkable-overlay", "camera-frustum"]) {
    const old = state.scene.getObjectByName(name);
    if (old) state.scene.remove(old);
  }
  state.markers = {}; state.markerOrder = []; state.trajectories = null;
  state.actorDots = []; state.listener = null;
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
    drawCameraFrustum();
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
  const preset = bundle.authoring?.default_listener;
  if (preset && !state.listener) {
    // pre-authored render-camera pose: FPV works before any route is authored
    state.listener = { position: preset.position_m, yawDeg: preset.yaw_deg };
    drawCameraFrustum();
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
    const modelName = state.bundle?.authoring?.actor_model_by_source?.[key];
    let dot;
    if (modelName && state.actorTemplates?.[modelName]) {
      dot = state.actorTemplates[modelName].clone(true);
      dot.userData.isModel = true;
    } else {
      dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.09, 14, 10),
        new THREE.MeshBasicMaterial({ color: colors[key] || 0xffffff }),
      );
    }
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
    const index = Math.min(state.frame, points.length - 1);
    const p = points[index];
    if (dot.userData.isModel) {
      dot.position.set(p[0], state.floorY, p[2]);
      const ahead = points[Math.min(index + 1, points.length - 1)];
      const dx = ahead[0] - p[0], dz = ahead[2] - p[2];
      if (Math.abs(dx) + Math.abs(dz) > 1e-6) {
        dot.rotation.y = Math.atan2(dx, dz);
      }
    } else {
      dot.position.set(p[0], p[1] + 0.1, p[2]);
    }
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
    drawCameraFrustum();
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
