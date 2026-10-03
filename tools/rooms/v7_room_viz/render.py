"""Native Habitat RGB / textured cutaways, adapted from demo_render at 0465ecd."""
from __future__ import annotations
import math
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import shape, box
from .common import read_json, write_json, project, now
from .textures import prepare_texture_scene


class Renderer:
    def __init__(self, source, runtime):
        self.rt = runtime
        self.hs = hs = runtime.habitat_sim
        cfg = hs.SimulatorConfiguration()
        cfg.scene_id = str(source)
        cfg.gpu_device_id = 0  # The supervisor sets CUDA_VISIBLE_DEVICES explicitly.
        cfg.enable_physics = False
        cfg.load_semantic_mesh = False
        cfg.physics_config_file = str(runtime.physics_config_path)
        cfg.requires_textures = True
        from habitat_sim.gfx import NO_LIGHT_KEY
        cfg.scene_light_setup = NO_LIGHT_KEY
        top = hs.CameraSensorSpec()
        top.uuid = 'top'
        top.sensor_type = hs.SensorType.COLOR
        top.sensor_subtype = hs.SensorSubType.ORTHOGRAPHIC
        top.resolution = runtime.magnum.Vector2i([1536, 1536])
        top.position = runtime.magnum.Vector3(0, 0, 0)
        top.orientation = runtime.magnum.Vector3(-math.pi / 2, 0, 0)
        top.near, top.far = .05, 2.3
        specs = [top]
        for name, kind in [('rgb', hs.SensorType.COLOR), ('depth', hs.SensorType.DEPTH)]:
            sensor = hs.CameraSensorSpec()
            sensor.uuid = name
            sensor.sensor_type = kind
            sensor.sensor_subtype = hs.SensorSubType.PINHOLE
            sensor.resolution = runtime.magnum.Vector2i([720, 1280])
            sensor.position = runtime.magnum.Vector3(0, 0, 0)
            sensor.hfov = 85.0
            sensor.near, sensor.far = .05, 1000
            specs.append(sensor)
        agent = hs.agent.AgentConfiguration()
        agent.sensor_specifications = specs
        self.sim = hs.Simulator(hs.Configuration(cfg, [agent]))
        self.agent = self.sim.get_agent(0)
        self.source = str(source)

    def pose(self, eye, yaw=0, pitch=0):
        state = self.hs.AgentState()
        state.position = np.asarray(eye, dtype=np.float32)
        state.rotation = (self.rt.quaternion.from_rotation_vector([0, yaw, 0]) *
                          self.rt.quaternion.from_rotation_vector([pitch, 0, 0]))
        self.agent.set_state(state, reset_sensors=True)

    def grab(self, key):
        obs = self.sim.get_sensor_observations()
        camera = self.sim._sensors[key]._sensor_object.render_camera
        view = np.asarray(camera.camera_matrix, dtype=float)
        projection = np.asarray(camera.projection_matrix, dtype=float)
        image = Image.fromarray(np.asarray(obs[key])[:, :, :3].astype(np.uint8))
        return image, view, projection, obs.get('depth')

    def topdown(self, geometry, floor_y):
        x0, z0, x1, z1 = geometry.bounds
        span = max(2.0, max(x1 - x0, z1 - z0) * 1.12 + .25)
        eye = [(x0 + x1) / 2, floor_y + 1.65, (z0 + z1) / 2]
        self.pose(eye)
        camera = self.sim._sensors['top']._sensor_object.render_camera
        initial = np.asarray(camera.projection_matrix, dtype=float)
        self.sim._sensors['top']._sensor_object.zoom((2 / abs(initial[0, 0])) / span)
        image, view, projection, _ = self.grab('top')
        if not np.allclose(view @ np.r_[eye, 1], [0, 0, 0, 1], atol=2e-4):
            raise ValueError('native view does not map the camera origin')
        if not np.isclose(2 / abs(projection[0, 0]), span, rtol=2e-5):
            raise ValueError('native orthographic span calibration failed')
        return image, dict(camera_m=eye, view=view.tolist(), projection=projection.tolist(),
                           span_m=span, floor_y_m=floor_y, width=image.width, height=image.height,
                           cutaway_above_floor_m=1.60, cutaway_below_floor_m=.65,
                           orientation='+X right, +Z down; native Habitat matrices')

    def interior(self, check):
        eye = np.asarray(check['listener_m'], dtype=float)
        direction = np.asarray(check.get('camera_forward', np.asarray(check['source_m']) - eye))
        yaw = math.atan2(-direction[0], -direction[2])
        self.pose(eye, yaw)
        image, view, projection, depth = self.grab('rgb')
        hfov = math.degrees(2 * math.atan(1 / abs(projection[0, 0])))
        if abs(hfov - 85) > .01:
            raise ValueError('native RGB horizontal field of view differs from 85 degrees')
        if not np.allclose(view @ np.r_[eye, 1], [0, 0, 0, 1], atol=2e-4):
            raise ValueError('native RGB camera differs from the acceptance listener')
        return image, depth, dict(camera_m=eye.tolist(), view=view.tolist(), projection=projection.tolist(),
                                 horizontal_fov_deg=hfov, width=image.width, height=image.height,
                                 probe=check.get('probe'), room_label=check['room_label'],
                                 camera_role=check.get('camera_role', 'measured_acceptance_listener'))

    def close(self):
        self.sim.close()


def font(size):
    for name in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def pieces(geometry):
    if geometry.geom_type == 'Polygon':
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from pieces(part)


def same(a, b):
    return np.allclose(a, b, atol=1e-5)


def label(draw, at, text, color, used, image_size, size=23):
    f = font(size)
    b = draw.textbbox((0, 0), text, font=f)
    w, h = b[2] - b[0] + 14, b[3] - b[1] + 12
    width, height = image_size
    for dx, dy in [(14, -h - 8), (14, 8), (-w - 12, 8), (-w - 12, -h - 8),
                   (14, 36), (-w - 12, 36), (14, -h - 36)]:
        x, y = max(6, min(width - w - 6, at[0] + dx)), max(6, min(height - h - 6, at[1] + dy))
        rect = (x, y, x + w, y + h)
        if all(rect[2] <= r[0] or rect[0] >= r[2] or rect[3] <= r[1] or rect[1] >= r[3] for r in used):
            break
    used.append(rect)
    draw.rounded_rectangle(rect, radius=4, fill=(255, 255, 255, 239), outline=color, width=2)
    draw.text((x + 7, y + 4 - b[1]), text, font=f, fill=(22, 40, 52, 255))
    return rect


def mark_top(image, calibration, job, level):
    image = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(overlay)
    used, markers = [], []
    fy = level['floor_y_m']
    view, proj = calibration['view'], calibration['projection']
    def px(point):
        return project([point[0], fy, point[2]], view, proj, *image.size)
    def coords(geom):
        return [(p[0], p[1]) for p in [px([x, fy, z]) for x, z in geom.exterior.coords] if p]
    checked = set(level['acoustic_room_labels'])
    for outline in level['outlines']:
        geom = shape(outline['geometry'])
        for part in pieces(geom):
            vertices = coords(part)
            if len(vertices) >= 3:
                draw.line(vertices, fill=(215, 226, 231, 140), width=2)
    rooms = [r for r in job['house']['acoustic_rooms'] if r['room_label'] in checked]
    for room in rooms:
        geom = shape(room['floor_polygon'])
        for part in pieces(geom):
            vertices = coords(part)
            if len(vertices) >= 3:
                draw.polygon(vertices, fill=(22, 182, 158, 25))
                draw.line(vertices, fill=(18, 208, 183, 255), width=5)
        main = max(pieces(geom), key=lambda p: p.area)
        p = main.representative_point()
        at = px([p.x, fy, p.y])
        label(draw, at, room['room_label'], (15, 137, 114, 255), used, image.size, 27)
    for room in rooms:
        name = room['room_label']
        rirs = [r for r in job['result'].get('rirs', []) if r.get('room_label') == name]
        witness = room['placement']
        listeners = [witness['camera_m']]
        sources = [witness['source_1_m'], witness['source_2_m']]
        for rir in rirs:
            if not any(same(rir['listener_m'], p) for p in listeners):
                listeners.append(rir['listener_m'])
            if not any(same(rir['source_m'], p) for p in sources):
                sources.append(rir['source_m'])
        for i, source in enumerate(sources):
            p = px(source)
            if p is None:
                continue
            x, y = p[:2]
            rad = 10 if i < 2 else 6
            draw.ellipse((x-rad, y-rad, x+rad, y+rad), fill=(250, 167, 46, 255),
                         outline=(255, 255, 255, 255), width=2)
            if i < 2:
                label(draw, p, 'S'+str(i+1), (213, 126, 18, 255), used, image.size, 23)
            markers.append(dict(kind='source', room_label=name, index=i+1,
                                position_m=source, pixel=list(p[:2]), witness=i < 2))
        for i, listener in enumerate(listeners):
            p = px(listener)
            if p is None:
                continue
            x, y = p[:2]
            rad = 13 if i == 0 else 9
            draw.polygon([(x, y-rad), (x-rad, y+rad), (x+rad, y+rad)],
                         fill=(34, 193, 218, 255), outline=(255, 255, 255, 255), width=2)
            if i == 0:
                label(draw, p, 'L', (10, 131, 153, 255), used, image.size, 23)
            markers.append(dict(kind='listener', room_label=name, index=i+1,
                                position_m=listener, pixel=list(p[:2])))
    holes = [r for r in job['boundary'].get('escaped_ray_boundary_candidates', []) if r['room_label'] in checked]
    for hole in holes:
        position = hole['candidate_boundary_midpoint_m']
        p = px(position)
        if p is None:
            continue
        x, y = p[:2]
        draw.ellipse((x-5, y-5, x+5, y+5), fill=(246, 53, 80, 235),
                     outline=(255, 255, 255, 250), width=1)
        markers.append(dict(kind='leak_candidate', room_label=hole['room_label'],
                            position_m=position, pixel=list(p[:2])))
    # A calibrated scale bar remains readable after image compression.
    length = min(5, max(1, int(calibration['span_m'] / 6)))
    bar = image.width * length / calibration['span_m']
    draw.rounded_rectangle((24, image.height-75, 74+bar, image.height-20), radius=7, fill=(255,255,255,235))
    draw.line([(43,image.height-38),(43+bar,image.height-38)],fill=(30,45,58,255),width=5)
    draw.text((43,image.height-68),str(length)+' m',font=font(22),fill=(30,45,58,255))
    return Image.alpha_composite(image, overlay).convert('RGB'), markers, len(holes)


def mark_interior(image, depth, calibration, job, check):
    draw = ImageDraw.Draw(image)
    view, proj = calibration['view'], calibration['projection']
    room = next(r for r in job['house']['acoustic_rooms'] if r['room_label'] == check['room_label'])
    sources = [room['placement']['source_1_m'], room['placement']['source_2_m']]
    markers, used = [], []
    for i, point in enumerate(sources):
        p = project(point, view, proj, *image.size)
        inside = p is not None and 0 <= p[0] < image.width and 0 <= p[1] < image.height and -1 <= p[2] <= 1
        record = dict(source_index=i+1, position_m=point, inside_frustum=bool(inside))
        if inside:
            x, y = p[:2]
            camera_point = np.asarray(view) @ np.r_[point, 1]
            axial = -camera_point[2]
            observed = float(depth[min(image.height-1,max(0,int(y))),min(image.width-1,max(0,int(x)))])
            visible = observed <= 0 or observed >= axial - .15
            record.update(pixel=[x,y], axial_distance_m=float(axial), native_depth_m=observed,
                          center_depth_visible=bool(visible))
            color = (255, 177, 45) if visible else (255, 115, 70)
            if visible:
                draw.ellipse((x-14,y-14,x+14,y+14),outline=(255,255,255),width=6)
                draw.ellipse((x-12,y-12,x+12,y+12),outline=color,width=4)
                draw.line([(x-20,y),(x+20,y)],fill=color,width=2)
                draw.line([(x,y-20),(x,y+20)],fill=color,width=2)
            else:
                for start in range(0,360,45):
                    draw.arc((x-14,y-14,x+14,y+14),start,start+23,fill=color,width=4)
            label(draw,(x,y),'S'+str(i+1)+('' if visible else ' (occluded)'),color,used,image.size,24)
        markers.append(record)
    return image, markers


def view_checks(job):
    checks = [dict(c) for c in job['visual'].get('checks', [])[:3]]
    for check in checks:
        check.setdefault('room_label', check['probe'].split('_', 1)[0])
    if not checks:
        checks = [dict(c) for c in job['result'].get('foa_checks', [])[:3]]
    if not checks:
        for room in job['house']['acoustic_rooms'][:3]:
            w = room['placement']
            checks.append(dict(room_label=room['room_label'], listener_m=w['camera_m'],
                               source_m=w['source_1_m'], probe=room['room_label']+'_planned',
                               camera_role='planned_listener_acoustics_not_completed'))
    if len(checks) < 3:
        original = list(checks)
        for c in original:
            if len(checks) >= 3:
                break
            second = dict(c)
            room = next(r for r in job['house']['acoustic_rooms'] if r['room_label'] == c['room_label'])
            second['source_m'] = room['placement']['source_2_m']
            second.pop('camera_forward', None)
            second['probe'] = room['room_label']+'_source2_display_heading'
            checks.append(second)
    return checks[:3]



def top_view_scope(job, level):
    """Frame every measured position and boundary candidate, even outside room bounds."""
    x0, z0, x1, z1 = shape(level['geometry']).bounds
    rooms = set(level['acoustic_room_labels'])
    points = [h['candidate_boundary_midpoint_m'] for h in
              job['boundary'].get('escaped_ray_boundary_candidates', []) if h['room_label'] in rooms]
    for room in job['house']['acoustic_rooms']:
        if room['room_label'] in rooms:
            points.extend(room['placement'][key] for key in ['camera_m','source_1_m','source_2_m'])
    for rir in job['result'].get('rirs', []):
        if rir.get('room_label') in rooms:
            points.extend([rir['listener_m'],rir['source_m']])
    if points:
        positions=np.asarray(points)
        x0=min(x0,float(positions[:,0].min()))
        z0=min(z0,float(positions[:,2].min()))
        x1=max(x1,float(positions[:,0].max()))
        z1=max(z1,float(positions[:,2].max()))
    return box(x0,z0,x1,z1)


def render_house(job_path, output):
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
    from tools.rooms.room_selection.runtime import habitat_runtime_options
    output = Path(output)
    job = read_json(job_path)
    house = job['house']['house']
    started = time.monotonic()
    scene, texture = prepare_texture_scene(job['source'], output)
    runtime = prepare_installed_habitat_runtime(**habitat_runtime_options())
    renderer = Renderer(scene, runtime)
    asset_dir = output/'page'/'assets'/house
    raw_dir = output/'cache'/'native_frames'/house
    asset_dir.mkdir(parents=True,exist_ok=True)
    raw_dir.mkdir(parents=True,exist_ok=True)
    receipt = dict(house=house,source_glb=job['source'],runtime_glb=str(scene),texture=texture,
                   input_paths=job['input_paths'],native_backend='Habitat-Sim',topdowns=[],interiors=[])
    try:
        for level in job['levels']:
            name = 'level_'+str(level['level_id'])
            image, calibration = renderer.topdown(top_view_scope(job, level), level['floor_y_m'])
            raw = raw_dir/(name+'.png')
            image.save(raw)
            annotated, markers, holes = mark_top(image,calibration,job,level)
            target = asset_dir/(name+'.webp')
            annotated.save(target,format='WEBP',quality=90,method=6)
            receipt['topdowns'].append(dict(path=str(target),raw_frame=str(raw),calibration=calibration,
                markers=markers,hole_candidate_count=holes,room_labels=level['room_labels'],
                acoustic_room_labels=level['acoustic_room_labels'],kind=level['kind'],level_id=level['level_id']))
        for index, check in enumerate(view_checks(job),1):
            image, depth, calibration = renderer.interior(check)
            raw = raw_dir/('interior_'+str(index)+'.png')
            image.save(raw)
            image, markers = mark_interior(image,depth,calibration,job,check)
            target = asset_dir/('interior_'+str(index)+'.jpg')
            image.save(target,quality=87,subsampling=0,optimize=True)
            receipt['interiors'].append(dict(path=str(target),raw_frame=str(raw),calibration=calibration,
                markers=markers,annotation='projected emitter positions; hollow dashed ring indicates center occlusion',
                mean_rgb=float(np.asarray(image).mean())))
        receipt.update(status='complete',at_sgt=now(),seconds=time.monotonic()-started)
        write_json(output/'rendered_houses'/(house+'.json'),receipt)
        print('NATIVE_COMPLETE',house,len(receipt['topdowns']),len(receipt['interiors']),flush=True)
    finally:
        renderer.close()
