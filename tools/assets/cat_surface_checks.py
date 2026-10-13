"""Anatomical-region surface QA for unchanged cat meshes.

Counts are diagnostic only. Areas are a conservative upper bound: half the
sum of unique visible contacting triangle areas, without repairing the mesh.
Depth is relative to the approved binding pose. Exterior depth uses the nearest
outward surface of the other region, checking the complete material region,
including vertices beyond the intersection boundary. Internal layer depth uses
the smaller positive/negative extent across each crossing triangle plane.
"""
import numpy as np
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
    result = np.full(len(ids), np.inf)
    for source, target in ((aa, bb), (bb, aa)):
        normal = np.cross(target[:, 1]-target[:, 0],
                          target[:, 2]-target[:, 0])
        length = np.linalg.norm(normal, axis=1)
        normal /= np.maximum(length[:, None], 1e-30)
        signed = np.sum((source-target[:, 0, None])*normal[:, None], axis=2)
        extent = np.minimum(np.maximum(signed.max(1), 0),
                            np.maximum(-signed.min(1), 0))
        result = np.minimum(result, extent)
    return float(result.max(initial=0))

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
        region_surfaces = [self.tree(self.rest, self.faces[self.fregions == r])
                           for r in range(len(REGIONS))]
        distance = float(np.linalg.norm(self.rest.max(0)-self.rest.min(0))*2+1)
        self.visible = np.zeros(len(self.faces), bool)
        self.orientation = np.ones(len(self.faces))
        # The rest-pose geometric face is exposed if a ray from infinity along
        # either face-normal direction reaches that face first within its own region.
        # Other limbs may occlude an exterior face from a particular view; that
        # occlusion does not turn it into an internal mesh layer. No edits.
        for index, (center, normal) in enumerate(zip(centers, normals)):
            if norms[index] < 1e-20:
                continue
            for sign in (1., -1.):
                hit, _, _, _ = region_surfaces[int(self.fregions[index])].ray_cast(Vector(center+normal*sign*distance),
                                             Vector(-normal*sign),
                                             distance+.00005)
                if hit is not None and np.linalg.norm(np.asarray(hit)-center) <= .00005:
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
        self.rest_depth = {}
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

    def depths(self, vertices, region, against, tree):
        ids = self.outer_vertices[region]
        result = np.zeros(len(ids))
        if tree is None or not len(ids):
            return result
        triangles = self.faces[self.outer_faces[against]]
        outer = vertices[np.unique(triangles)]
        low, high = outer.min(0)-.00005, outer.max(0)+.00005
        eligible = np.flatnonzero(np.all((vertices[ids] >= low) &
                                        (vertices[ids] <= high), axis=1))
        for k in eligible:
            point = vertices[ids[k]]
            nearest, normal, face_index, _ = tree.find_nearest(Vector(point))
            if nearest is None:
                continue
            outward = np.asarray(normal)*self.orientation[self.outer_faces[against][face_index]]
            signed = float(np.dot(point-np.asarray(nearest), outward))
            result[k] = max(0., -signed)
        return result

    def inspect(self, vertices, pairs=None):
        vertices = np.asarray(vertices, float)
        pairs = self.crossings(vertices, self.faces) if pairs is None else pairs
        novel = pairs-self.baseline
        same = {(i, j) for i, j in novel
                if self.fregions[i] == self.fregions[j]}
        same_depth = internal_plane_depth(vertices, self.faces, same)
        physical = self.new_contacts(novel, self.rest, self.faces)
        patches = {}
        for i, j in physical:
            a, b = int(self.fregions[i]), int(self.fregions[j])
            if a != b and self.visible[i] and self.visible[j]:
                key = tuple(sorted((a, b)))
                patches.setdefault(key, set()).update((i, j))
        triangles = vertices[self.faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                        triangles[:, 2]-triangles[:, 0]), axis=1)*.5
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
                active = patches.get(key, set())-self.baseline_visible_faces.get(key, set())
                area = float(areas[list(active)].sum()*.5) if active else 0.
                depth = 0.
                for region, against in ((a, b), (b, a)):
                    cache_key = (region, against)
                    if cache_key not in self.rest_depth:
                        self.rest_depth[cache_key] = self.depths(
                            self.rest, region, against, self.rest_trees[against])
                    now = self.depths(vertices, region, against, posed_trees[against])
                    if not np.all(np.isfinite(now)) or not np.all(np.isfinite(self.rest_depth[cache_key])):
                        raise ValueError('nonfinite_surface_penetration_depth')
                    depth = max(depth, float(np.maximum(
                        now-self.rest_depth[cache_key], 0).max(initial=0)))
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
            "regions": list(REGIONS),
            "region_assignment": "fixed binding-pose summed skin-weight family; face majority",
            "visible_triangle_count": int(self.visible.sum()),
            "counts_participate_in_acceptance": False,
            "new_interregion_area_limit_m2": AREA_LIMIT,
            "new_penetration_depth_limit_m": DEPTH_LIMIT,
            "area_method": "conservative half-sum of unique newly contacting exposed triangle areas across all region pairs per frame; binding-pose physical contacts excluded",
            "depth_method": "complete exposed material region nearest outward opposing-region surface; subtract each material vertex binding-pose depth; same-region new crossings use the minimum separating translation across either face plane",
            "baseline_contact_gap_m": .0001,
            "mesh_repaired": False,
        }
