"""Orthographic CPU triangle rasterization of original USD, with a depth buffer."""
from __future__ import annotations
import numpy as np
from numba import njit
from PIL import Image
from shapely.geometry import shape
import shapely
from tools.rooms.room_split_kujiale.adapter import dump


@njit(cache=False)
def raster_tri(image, depth, tri, colour, cx, cz, span):
    h, w = depth.shape
    x = (tri[:, 0] - cx) / span * w + w / 2
    z = (tri[:, 2] - cz) / span * h + h / 2
    den = (z[1]-z[2])*(x[0]-x[2]) + (x[2]-x[1])*(z[0]-z[2])
    if abs(den) < 1e-10: return
    lo_x = max(0, int(np.floor(x.min()))); hi_x = min(w-1, int(np.ceil(x.max())))
    lo_z = max(0, int(np.floor(z.min()))); hi_z = min(h-1, int(np.ceil(z.max())))
    for iz in range(lo_z, hi_z+1):
        for ix in range(lo_x, hi_x+1):
            px, pz = ix+.5, iz+.5
            a = ((z[1]-z[2])*(px-x[2]) + (x[2]-x[1])*(pz-z[2])) / den
            b = ((z[2]-z[0])*(px-x[2]) + (x[0]-x[2])*(pz-z[2])) / den
            c = 1-a-b
            if min(a, b, c) < -1e-7: continue
            y = a*tri[0,1] + b*tri[1,1] + c*tri[2,1]
            if y > depth[iz, ix]:
                depth[iz, ix] = y
                image[iz, ix] = colour


@njit(cache=False)
def raster_all(verts, faces, colours, walls, ceilings, floor_y, cx, cz, span, size):
    image = np.full((size, size, 3), 241, np.uint8)
    depth = np.full((size, size), -np.inf, np.float32)
    rendered = 0; hidden = 0
    for i in range(len(faces)):
        tri = verts[faces[i]]
        if ceilings[i] or tri[:,1].max() < floor_y-.15 or (not walls[i] and tri[:,1].min() > floor_y+2.2):
            hidden += 1; continue
        rgb = np.empty(3, np.uint8)
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0]); nn = np.linalg.norm(normal)
        # Fixed neutral diffuse light for review; geometry and authored colours remain visible.
        shade = .78 + .22*abs(normal[1])/nn if nn > 0 else 1.
        for j in range(3):
            v = min(1., max(0., colours[i,j]*shade))
            v = 12.92*v if v <= .0031308 else 1.055*v**(1/2.4)-.055
            rgb[j] = int(min(255., max(0., v*255)))
        if walls[i] or tri[:,1].max() <= floor_y+2.2:
            raster_tri(image, depth, tri, rgb, cx, cz, span); rendered += 1; continue
        # Sutherland-Hodgman clipping against the review ceiling plane.
        poly = np.empty((4,3), np.float32); n = 0; upper = floor_y+2.2
        for j in range(3):
            a, b = tri[j], tri[(j+1)%3]
            if a[1] <= upper: poly[n] = a; n += 1
            if (a[1] <= upper) != (b[1] <= upper):
                t = (upper-a[1])/(b[1]-a[1]); poly[n] = a + t*(b-a); n += 1
        for j in range(1,n-1):
            tt = np.empty((3,3), np.float32);tt[0]=poly[0];tt[1]=poly[j];tt[2]=poly[j+1]
            raster_tri(image, depth, tt, rgb, cx, cz, span)
        rendered += 1
    return image, depth, rendered, hidden


def render_house(house, scene, directory, size=1600):
    from pathlib import Path
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    union = shapely.union_all([shape(r['cad_polygon_xz_m']) for r in house['rooms']])
    x0, z0, x1, z1 = union.bounds; cx, cz = (x0+x1)/2, (z0+z1)/2
    span = max(x1-x0, z1-z0)+2.
    records=[]
    for y in sorted({round(r['floor_y_m'],3) for r in house['rooms']}):
        rgb, depth, rendered, hidden = raster_all(scene.vertices, scene.faces, scene.colours, scene.wall, scene.ceiling, y, cx, cz, span, size)
        path = directory / (house['house'] + '__Y%+.3f.jpg' % y)
        Image.fromarray(rgb).save(path, quality=86)
        # world (x,y,z,1) -> NDC: screen top is smaller z; pixel convention matches owner redraw.
        projection = [[2/span,0,0,-2*cx/span],[0,0,-2/span,2*cz/span],[0,1,0,0],[0,0,0,1]]
        meta=dict(path=str(path), projection=projection, view=np.eye(4).tolist(), floor_y_m=y,
                  span_m=span, size_px=[size,size], house=house['house'],
                  renderer='Numba CPU orthographic original-USD triangle rasterizer; no Habitat or UE renderer',
                  source_stage=house['source_stage'], source_to_canonical=house['source_to_canonical'],
                  material_colour=scene.receipt['material_colour'],
                  review_clip='hide authored ceilings; nonwall upper plane floor+2.2m; wall meshes intact; floor-0.15m lower selection',
                  rendered_triangles=rendered, review_hidden_triangles=hidden,
                  geometry_pixels=int(np.isfinite(depth).sum()), source_modified=False,
                  production_render_or_acoustics_modified=False)
        dump(path.with_suffix('.json'), meta); records.append(meta)
    return records
