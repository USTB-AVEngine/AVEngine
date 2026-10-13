"""Anatomical-region surface QA for unchanged cat meshes.

Counts are diagnostic only. Contact fragments are clipped to the opposing
triangle footprint; unrelated distant corners contribute no area or depth.
Complete-region containment requires outward exits before nearest surface
distance is compared with the approved binding pose. The source is unchanged.
"""
import numpy as np
import pathlib,runpy
_contact=runpy.run_path(str(pathlib.Path(__file__).with_name('cat_contact_patches.py')))
clipped_internal_depths=_contact['clipped_internal_depths']
clipped_penetrating_patches=_contact['clipped_penetrating_patches']
from mathutils import Vector
from mathutils.bvhtree import BVHTree

REGIONS = ("torso", "head", "tail", "FL", "FR", "HL", "HR")
AREA_LIMIT = 25e-6
DEPTH_LIMIT = .002

def anatomical_regions(parts, document):
    vertices = []
    for part in parts:
        score = np.zeros((len(part["positions"]), len(REGIONS)))
        for column, bone in enumerate(part["skin_joints"]):
            name = document["nodes"][bone].get("name", "")
            region = (2 if name.startswith("Tail") else
                      1 if name in ("Head", "Neck") else
                      next((3+i for i, paw in enumerate(REGIONS[3:])
                            if name.endswith("_"+paw)), 0))
            score[:, region] += np.sum(part["weights"] *
                                      (part["joints"] == column), axis=1)
        vertices.append(np.argmax(score, axis=1))
    return np.concatenate(vertices)

def internal_plane_depth(vertices, faces, pairs):
    if not pairs:
        return 0.
    ids = np.asarray(list(pairs), int)
    aa, bb = vertices[faces[ids[:, 0]]], vertices[faces[ids[:, 1]]]
    return float(clipped_internal_depths(aa,bb).max(initial=0))

class RegionSurfaceChecks:
    def __init__(self, vertices, faces, vertex_regions, crossings,
                 new_disjoint_contact_pairs):
        self.rest = np.asarray(vertices, float)
        self.faces = np.asarray(faces, int)
        self.vregions = np.asarray(vertex_regions, int)
        votes = np.stack([np.sum(self.vregions[self.faces] == r, axis=1)
                          for r in range(len(REGIONS))], axis=1)
        self.fregions = np.argmax(votes, axis=1)
        self.crossings = crossings
        self.new_contacts = new_disjoint_contact_pairs
        self.baseline = crossings(self.rest, self.faces)
        triangles = self.rest[self.faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0],
                           triangles[:, 2]-triangles[:, 0])
        norms = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(norms[:, None], 1e-30)
        centers = triangles.mean(1)
        full_surface = self.tree(self.rest, self.faces)
        self.rest_normals = normals.copy()
        distance = float(np.linalg.norm(self.rest.max(0)-self.rest.min(0))*2+1)
        self.visible = np.zeros(len(self.faces), bool)
        self.orientation = np.ones(len(self.faces))
        # Global exposure excludes internal layers at anatomical cut edges.
        # Oblique exterior rays retain faces occluded only along their normal
        # by a separate limb, without treating internal region cuts as visible.
        for index, (center, normal) in enumerate(zip(centers, normals)):
            if norms[index] < 1e-20:
                continue
            tangent = np.cross(normal, [1., 0., 0.])
            if np.linalg.norm(tangent) < .1:
                tangent = np.cross(normal, [0., 1., 0.])
            tangent /= np.linalg.norm(tangent)
            other = np.cross(normal, tangent)
            directions = [(normal, 1.), (-normal, -1.)]
            directions.extend((sign*normal+8*axis, sign)
                              for sign in (1., -1.)
                              for axis in (tangent, -tangent, other, -other))
            for direction, sign in directions:
                direction = direction/np.linalg.norm(direction)
                hit, _, _, _ = full_surface.ray_cast(
                    Vector(center+direction*distance), Vector(-direction),
                    distance+.00001)
                if hit is not None and np.linalg.norm(np.asarray(hit)-center) <= .00001:
                    self.visible[index] = True
                    self.orientation[index] = sign
                    break
        self.outer_faces = [np.flatnonzero(self.visible & (self.fregions == r))
                            for r in range(len(REGIONS))]
        self.outer_vertices = [np.intersect1d(
            np.unique(self.faces[ids]), np.flatnonzero(self.vregions == r))
            for r, ids in enumerate(self.outer_faces)]
        self.rest_trees = [self.region_tree(self.rest, r)
                           for r in range(len(REGIONS))]
        self.closed_region = []
        _, welded = np.unique(np.round(self.rest/1e-7).astype(np.int64),
                              axis=0, return_inverse=True)
        for face_ids in self.outer_faces:
            triangles = welded[self.faces[face_ids]]
            edges = np.sort(np.concatenate([triangles[:, [0, 1]],
                                             triangles[:, [1, 2]],
                                             triangles[:, [2, 0]]]), axis=1)
            _, counts = np.unique(edges, axis=0, return_counts=True)
            self.closed_region.append(bool(len(counts) and np.all(counts >= 2)))
        self.rest_region_bounds = []
        for ids in self.outer_faces:
            points = self.rest[np.unique(self.faces[ids])]
            self.rest_region_bounds.append((points.min(0), points.max(0)) if len(points) else None)
        self.closed_baseline_depths = {}
        self.reference_ray_info=self.ray_info(np.eye(3))
        self.baseline_visible_faces = {}
        for i, j in self.baseline:
            a, b = int(self.fregions[i]), int(self.fregions[j])
            if a != b and self.visible[i] and self.visible[j]:
                key = tuple(sorted((a, b)))
                self.baseline_visible_faces.setdefault(key, set()).update((i, j))

    @staticmethod
    def tree(vertices, faces):
        if not len(faces):
            return None
        indices, inverse = np.unique(faces, return_inverse=True)
        return BVHTree.FromPolygons([Vector(x) for x in vertices[indices]],
                                   inverse.reshape(-1, 3).tolist(),
                                   all_triangles=True, epsilon=1e-8)

    def region_tree(self, vertices, region):
        return self.tree(vertices, self.faces[self.outer_faces[region]])

    @staticmethod
    def ray_info(basis):
        return [(Vector(d),Vector(d)*1e-7)
                for d in np.concatenate([basis.T,-basis.T])]

    def inside(self,point,tree,region,rays):
        exits=0;base=Vector(point)
        for index,(direction,offset) in enumerate(rays):
            hit,normal,face_index,_=tree.ray_cast(base+offset,direction)
            if hit is not None:
                sign=self.orientation[self.outer_faces[region][face_index]]
                if normal.dot(direction)*sign>.1:exits+=1
            if exits>=4:return True
            if exits+5-index<4:return False
        return False

    def depths(self,vertices,region,against,tree):
        ids=self.outer_vertices[region];result=np.zeros(len(ids))
        if tree is None or not len(ids):return result
        triangles=self.faces[self.outer_faces[against]]
        outer=vertices[np.unique(triangles)]
        low,high=outer.min(0)-.00005,outer.max(0)+.00005
        eligible=np.flatnonzero(np.all((vertices[ids]>=low)&(vertices[ids]<=high),axis=1))
        for k in eligible:
            point=vertices[ids[k]]
            if not self.inside(point,tree,against,self.posed_ray_info[against]):continue
            nearest,normal,face_index,distance=tree.find_nearest(Vector(point))
            if nearest is None:continue
            key=(region,against,int(ids[k]))
            if key not in self.closed_baseline_depths:
                baseline=0.;reference=self.rest[ids[k]];bounds=self.rest_region_bounds[against]
                if bounds is not None and np.all(reference>=bounds[0]) and np.all(reference<=bounds[1]):
                    reference_tree=self.rest_trees[against]
                    if reference_tree is not None and self.inside(reference,reference_tree,against,self.reference_ray_info):
                        ref_point,_,_,ref_distance=reference_tree.find_nearest(Vector(reference))
                        if ref_point is not None:baseline=float(ref_distance)
                self.closed_baseline_depths[key]=baseline
            result[k]=max(0.,float(distance)-self.closed_baseline_depths[key])
        return result

    def inspect(self, vertices, pairs=None):
        vertices = np.asarray(vertices, float)
        pairs = self.crossings(vertices, self.faces) if pairs is None else pairs
        novel = pairs-self.baseline
        physical = self.new_contacts(novel, self.rest, self.faces)
        same = {(i, j) for i, j in physical
                if self.fregions[i] == self.fregions[j]}
        same_depth = internal_plane_depth(vertices, self.faces, same)
        visible_contacts = {tuple(sorted((int(self.fregions[i]), int(self.fregions[j]))))
                            for i,j in pairs if self.fregions[i] != self.fregions[j]
                            and self.visible[i] and self.visible[j]}
        patches={}
        for i,j in physical:
            a,b=int(self.fregions[i]),int(self.fregions[j])
            if a!=b and self.visible[i] and self.visible[j]:
                key=tuple(sorted((a,b)))
                patches.setdefault(key,[]).append((i,j))
        triangles=vertices[self.faces]
        patch_measurements={}
        for key,pairs_for_key in patches.items():
            pair_ids=np.asarray(pairs_for_key,int)
            aa,bb=triangles[pair_ids[:,0]],triangles[pair_ids[:,1]]
            first,_=clipped_penetrating_patches(aa,bb,self.orientation[pair_ids[:,1]])
            second,_=clipped_penetrating_patches(bb,aa,self.orientation[pair_ids[:,0]])
            patch_measurements[key]=(float((first+second).sum()*.5),
                                     float(clipped_internal_depths(aa,bb).max(initial=0)))
        self.posed_ray_info=[]
        for region,ids in enumerate(self.outer_vertices):
            rotation=np.eye(3)
            if len(ids)>=3:
                original=self.rest[ids]-self.rest[ids].mean(0)
                posed=vertices[ids]-vertices[ids].mean(0)
                u,_,vt=np.linalg.svd(original.T@posed)
                rotation=vt.T@u.T
                if np.linalg.det(rotation)<0:
                    vt[-1]*=-1;rotation=vt.T@u.T
            self.posed_ray_info.append(self.ray_info(rotation))
        contact_keys={tuple(sorted((int(self.fregions[i]),int(self.fregions[j])))) for i,j in pairs if self.fregions[i]!=self.fregions[j]}
        bounds=[]
        for ids in self.outer_faces:
            points=vertices[np.unique(self.faces[ids])]
            bounds.append((points.min(0),points.max(0)) if len(points) else None)
        region_rows = []
        total_area = 0.
        max_depth = 0.
        posed_trees = [self.region_tree(vertices, r)
                       for r in range(len(REGIONS))]
        # All overlapping region bounds are checked, even when one complete
        # region is contained and no boundary triangles intersect this frame.
        for a in range(len(REGIONS)):
            for b in range(a+1, len(REGIONS)):
                key = (a, b)
                area,depth=patch_measurements.get(key,(0.,0.))
                contained=False
                if bounds[a] is not None and bounds[b] is not None:
                    contained=any(np.all(bounds[x][0]>=bounds[y][0]-1e-8) and
                                  np.all(bounds[x][1]<=bounds[y][1]+1e-8)
                                  for x,y in ((a,b),(b,a)))
                # A disjoint pair with neither boundary contacts nor complete
                # containment has no new penetration. Complete containment is
                # still checked even when its boundary has no intersections.
                if key not in contact_keys and not contained:continue
                for region, against in ((a, b), (b, a)):
                    now = self.depths(vertices, region, against, posed_trees[against])
                    if not np.all(np.isfinite(now)):
                        raise ValueError('nonfinite_surface_penetration_depth')
                    depth = max(depth, float(now.max(initial=0)))
                total_area += area
                max_depth = max(max_depth, depth)
                if area or depth:
                    region_rows.append({"regions": [REGIONS[a], REGIONS[b]],
                                        "new_visible_area_m2": area,
                                        "max_new_depth_m": depth})
        failures = []
        if total_area > AREA_LIMIT:
            failures.append("new_interregion_visible_area")
        if max_depth > DEPTH_LIMIT:
            failures.append("new_interregion_penetration_depth")
        if same_depth > DEPTH_LIMIT:
            failures.append("new_intraregion_penetration_depth")
        return {
            "self_intersection_pairs": len(pairs),
            "baseline_self_intersection_pairs": len(self.baseline),
            "self_intersection_count_excess": len(pairs)-len(self.baseline),
            "new_self_intersection_pairs": len(novel),
            "counts_participate_in_acceptance": False,
            "new_interregion_visible_area_m2": total_area,
            "max_new_interregion_depth_m": max_depth,
            "max_new_intraregion_depth_m": same_depth,
            "region_contacts": region_rows,
            "surface_failures": failures,
        }

    def provenance(self):
        return {
            "rule": "owner_20261014_023x_visible_penetration",
            "measurement_revision": "contact_footprints_v3",
            "regions": list(REGIONS),
            "region_assignment": "fixed binding-pose summed skin-weight family; face majority",
            "visible_triangle_count": int(self.visible.sum()),
            "closed_regions": [REGIONS[i] for i,closed in enumerate(self.closed_region) if closed],
            "counts_participate_in_acceptance": False,
            "new_interregion_area_limit_m2": AREA_LIMIT,
            "new_penetration_depth_limit_m": DEPTH_LIMIT,
            "exposure_method": "global binding surface: two normal rays and eight oblique rays; anatomical cut surfaces do not become exterior; 10 micrometre ray endpoint accuracy",
            "area_method": "half-sum of actual penetrating triangle fragments clipped to the opposing triangular footprint; pair sum conservatively bounds overlapping fragments; binding-pose physical contacts excluded",
            "depth_method": "relative complete-region nearest-surface distance only after four of six outward exit rays confirm containment; axes follow each material region rigid fit; subtract actual binding containment distance; intersecting pairs use separating face depths clipped to their mutual footprints; pair counts do not gate",
            "baseline_contact_gap_m": .0001,
            "mesh_repaired": False,
        }
