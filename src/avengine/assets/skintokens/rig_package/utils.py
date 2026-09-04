from numpy import ndarray
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except ImportError:  # Blender's bundled Python does not ship scipy.
    cKDTree = None  # type: ignore

try:
    from mathutils import kdtree as _blender_kdtree  # type: ignore
except ImportError:  # regular model Python has no Blender mathutils.
    _blender_kdtree = None


def assert_ndarray(arr, name: str="arr", shape: Optional[Tuple[int, ...]]=None, dtype=None):
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray or None, got {type(arr)}")
    if shape is not None:
        # shape may contain None as wildcard
        if len(shape) != arr.ndim:
            raise ValueError(f"{name}: expected shape length {len(shape)} but array ndim is {arr.ndim}")
        for i, (exp, actual) in enumerate(zip(shape, arr.shape)):
            if exp > 0 and exp != actual:
                raise ValueError(f"{name} shape mismatch at axis {i}: expected {exp}, got {actual}")
    if dtype is not None:
        if not np.issubdtype(arr.dtype, dtype):
            raise ValueError(f"{name} dtype must be {dtype}, got {arr.dtype}")


def assert_list(arr, name: str="arr", dtype=None):
    if not isinstance(arr, list):
        raise ValueError(f"found type {type(arr)}, expect a list")
    if dtype is not None:
        for x in arr:
            if not isinstance(x, dtype):
                raise ValueError(f"found type {type(x)} in {name}, expect all to be {dtype}")


NEAREST_NEIGHBOR_TEMP_BYTES = 8 * 1024 * 1024
NEAREST_NEIGHBOR_QUERY_TILE = 256


def nearest_neighbors(
    queries: ndarray,
    references: ndarray,
    k: int = 1,
) -> Tuple[ndarray, ndarray]:
    """Return nearest reference distances and indices without a scipy dependency.

    The fallback visits both dimensions in bounded tiles. It never allocates a
    queries-by-references-by-3 array; its largest distance tile is capped at
    NEAREST_NEIGHBOR_TEMP_BYTES (apart from caller-owned inputs and the small
    k-neighbor accumulator).
    """
    queries = np.asarray(queries)
    references = np.asarray(references)
    if queries.ndim != 2 or references.ndim != 2 or queries.shape[1] != references.shape[1]:
        raise ValueError(
            f"nearest-neighbor arrays must be (N,D)/(M,D), got "
            f"{queries.shape} and {references.shape}"
        )
    if references.shape[0] == 0:
        raise ValueError("nearest-neighbor reference array is empty")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    k_eff = min(int(k), references.shape[0])
    if cKDTree is not None:
        distances, indices = cKDTree(references).query(queries, k=k_eff)
        return np.asarray(distances), np.asarray(indices)
    if _blender_kdtree is not None:
        tree = _blender_kdtree.KDTree(references.shape[0])
        for index, reference in enumerate(references):
            tree.insert(tuple(float(value) for value in reference), index)
        tree.balance()
        if k_eff == 1:
            pairs = [tree.find(tuple(float(value) for value in query)) for query in queries]
            distances = np.asarray([pair[2] for pair in pairs], dtype=np.float64)
            indices = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
            return distances, indices
        pairs = [
            tree.find_n(tuple(float(value) for value in query), k_eff)
            for query in queries
        ]
        distances = np.asarray([[pair[2] for pair in row] for row in pairs], dtype=np.float64)
        indices = np.asarray([[pair[1] for pair in row] for row in pairs], dtype=np.int64)
        return distances, indices

    q_tile = max(1, min(NEAREST_NEIGHBOR_QUERY_TILE, queries.shape[0]))
    dimensions = max(1, queries.shape[1])
    ref_tile = max(
        1,
        min(
            4096,
            NEAREST_NEIGHBOR_TEMP_BYTES // (8 * q_tile * dimensions),
        ),
    )
    distances = np.empty((queries.shape[0], k_eff), dtype=np.float64)
    indices = np.empty((queries.shape[0], k_eff), dtype=np.int64)
    for q_start in range(0, queries.shape[0], q_tile):
        q_stop = min(q_start + q_tile, queries.shape[0])
        q_block = queries[q_start:q_stop]
        best_dist2 = np.full((q_stop - q_start, k_eff), np.inf, dtype=np.float64)
        best_indices = np.full((q_stop - q_start, k_eff), -1, dtype=np.int64)
        for r_start in range(0, references.shape[0], ref_tile):
            r_stop = min(r_start + ref_tile, references.shape[0])
            delta = q_block[:, None, :] - references[r_start:r_stop][None, :, :]
            candidate_dist2 = np.einsum("qrd,qrd->qr", delta, delta)
            candidate_count = min(k_eff, r_stop - r_start)
            if candidate_count < r_stop - r_start:
                candidate_indices = np.argpartition(
                    candidate_dist2, kth=candidate_count - 1, axis=1
                )[:, :candidate_count]
                candidate_dist2 = np.take_along_axis(
                    candidate_dist2, candidate_indices, axis=1
                )
            else:
                candidate_indices = np.broadcast_to(
                    np.arange(r_stop - r_start, dtype=np.int64),
                    candidate_dist2.shape,
                )
            candidate_indices = candidate_indices + r_start
            combined_dist2 = np.concatenate((best_dist2, candidate_dist2), axis=1)
            combined_indices = np.concatenate((best_indices, candidate_indices), axis=1)
            chosen = np.argpartition(
                combined_dist2, kth=k_eff - 1, axis=1
            )[:, :k_eff]
            best_dist2 = np.take_along_axis(combined_dist2, chosen, axis=1)
            best_indices = np.take_along_axis(combined_indices, chosen, axis=1)
        order = np.argsort(best_dist2, axis=1)
        best_dist2 = np.take_along_axis(best_dist2, order, axis=1)
        best_indices = np.take_along_axis(best_indices, order, axis=1)
        distances[q_start:q_stop] = np.sqrt(best_dist2)
        indices[q_start:q_stop] = best_indices
    if int(k) == 1:
        return distances[:, 0], indices[:, 0]
    return distances, indices


def compute_mesh_normals(vertices: ndarray, faces: ndarray) -> Tuple[ndarray, ndarray]:
    """Compute face and vertex normals with numpy for Blender-side parsing."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    edge0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_normals = np.cross(edge0, edge1)
    face_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(face_lengths, 1e-12)
    vertex_normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals)
    vertex_lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(vertex_lengths, 1e-12)
    return vertex_normals.astype(np.float32), face_normals.astype(np.float32)


def linear_blend_skinning(
    vertices: ndarray,
    matrix_local: ndarray,
    matrix: ndarray,
    skin: ndarray,
    pad: int=1,
    value: float=1.0,
) -> ndarray:
    """
    Args:
        vertices: (N, 4-pad)
        matrix_local: (J, 4, 4)
        matrix: (J, 4, 4)
        skin: (N, J)
        pad: 0 or 1
        value: value to pad
    Returns:
        (N, 3) vertices using LBS algorithm: Skinning with dual quaternions, Kavan, 2007
    """
    J = matrix_local.shape[0]
    N = vertices.shape[0]
    assert_ndarray(vertices, name='vertices', shape=(N, 3))
    assert_ndarray(matrix_local, name="matrix_local", shape=(J, 4, 4))
    assert_ndarray(matrix, name="matrix", shape=(J, 4, 4))
    assert_ndarray(skin, name="skin", shape=(N, J))
    assert vertices.shape[-1] + pad == 4
    padded = np.pad(vertices, ((0, 0), (0, pad)), 'constant', constant_values=(0, value)).T
    trans = matrix @ np.linalg.inv(matrix_local)
    weighted_per_bone_matrix = []
    mask = (skin > 0).T
    for i in range(J):
        offset = np.zeros((4, N), dtype=np.float32)
        offset[:, mask[i]] = (trans[i] @ padded[:, mask[i]]) * skin.T[i, mask[i]]
        weighted_per_bone_matrix.append(offset)
    weighted_per_bone_matrix = np.stack(weighted_per_bone_matrix)
    g = np.sum(weighted_per_bone_matrix, axis=0)
    final = g[:3, :] / (np.sum(skin, axis=1) + 1e-8)
    return final.T


def axis_angle_to_matrix(axis_angle: ndarray) -> ndarray:
    """Turn one or more axis-angle vectors into homogeneous matrices."""
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    if axis_angle.ndim != 2 or axis_angle.shape[1] != 3:
        raise ValueError(f"axis_angle must have shape (N, 3), got {axis_angle.shape}")
    if cKDTree is not None:
        try:
            from scipy.spatial.transform import Rotation  # type: ignore
            rotation = Rotation.from_rotvec(axis_angle).as_matrix()
        except ImportError:
            rotation = None
    else:
        rotation = None
    if rotation is None:
        theta = np.linalg.norm(axis_angle, axis=1, keepdims=True)
        safe_theta = np.maximum(theta, 1e-12)
        unit = axis_angle / safe_theta
        x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
        zero = (theta[:, 0] <= 1e-12)
        x = np.where(zero, 0.0, x)
        y = np.where(zero, 0.0, y)
        z = np.where(zero, 0.0, z)
        c = np.cos(theta[:, 0])
        s = np.sin(theta[:, 0])
        one_c = 1.0 - c
        rotation = np.empty((axis_angle.shape[0], 3, 3), dtype=np.float64)
        rotation[:, 0, 0] = c + x * x * one_c
        rotation[:, 0, 1] = x * y * one_c - z * s
        rotation[:, 0, 2] = x * z * one_c + y * s
        rotation[:, 1, 0] = y * x * one_c + z * s
        rotation[:, 1, 1] = c + y * y * one_c
        rotation[:, 1, 2] = y * z * one_c - x * s
        rotation[:, 2, 0] = z * x * one_c - y * s
        rotation[:, 2, 1] = z * y * one_c + x * s
        rotation[:, 2, 2] = c + z * z * one_c
    res = np.zeros((axis_angle.shape[0], 4, 4), dtype=np.float64)
    res[:, :3, :3] = rotation
    res[:, -1, -1] = 1.0
    return res.astype(np.float32)


def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    mask: Optional[ndarray]=None,
) -> Tuple[ndarray, ndarray, ndarray]:
    '''
    Randomly pick samples proportional to face area.

    See sample_surface: https://github.com/mikedh/trimesh/blob/main/trimesh/sample.py

    Args:
        mask: (num_faces,), only sample points on the faces where value is True.
    Return:
        vertex_samples, original_face_index, random_lengths
    '''
    original_face_indices = np.arange(len(faces))
    if mask is not None:
        assert_ndarray(arr=mask, name="mask", shape=(faces.shape[0],))
        original_face_indices = original_face_indices[mask]
        faces = faces[mask]
    if faces.shape[0] == 0:
        raise ValueError("cannot sample a mesh with no selected faces")
    offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_weight = np.linalg.norm(np.cross(offset_0, offset_1, axis=-1), axis=-1)
    weight_cum = np.cumsum(face_weight, axis=0)
    if weight_cum[-1] <= 1e-12:
        raise ValueError("cannot sample a degenerate mesh")
    face_pick = np.random.rand(num_samples) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)
    original_face_index = original_face_indices[face_index]
    tri_origins = vertices[faces[:, 0]][face_index]
    tri_vectors = vertices[faces[:, 1:]][face_index] - tri_origins[:, None, :]
    random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    return vertex_samples, original_face_index, random_lengths


def sample_barycentric(
    vertex_group: ndarray,
    faces: ndarray,
    face_index: ndarray,
    random_lengths: ndarray,
) -> ndarray:
    v_origins = vertex_group[faces[face_index, 0]]
    v_vectors = vertex_group[faces[face_index, 1:]] - v_origins[:, np.newaxis, :]
    sample_vector = (v_vectors * random_lengths).sum(axis=1)
    v_samples = sample_vector + v_origins
    return v_samples


def sample_vertex_groups(
    vertices: ndarray,
    faces: ndarray,
    num_samples: int,
    num_vertex_samples: Optional[int]=None,
    vertex_normals: Optional[ndarray]=None,
    face_normals: Optional[ndarray]=None,
    vertex_groups: Optional[ndarray]=None,
    face_mask: Optional[ndarray]=None,
    shuffle: bool=True,
    same: bool=False,
) -> Tuple[ndarray, ndarray|None, ndarray|None]:
    """
    Choose num_samples samples on the mesh and get their positions and normals.
    If vertex_group is provided, get its weights using barycentric sampling.
    """
    if num_vertex_samples is None:
        num_vertex_samples = 0
    if num_vertex_samples > num_samples:
        raise ValueError(f"num_vertex_samples cannot be larger than num_samples, found: {num_vertex_samples} > {num_samples}")
    def get_mask_perm(mask: Optional[ndarray]):
        if mask is None:
            vertex_mask = np.arange(vertices.shape[0])
        else:
            vertex_mask = np.unique(mask)
        perm = np.random.permutation(vertex_mask.shape[0])
        return vertex_mask[perm[:num_vertex_samples]]
    if vertex_groups is not None:
        if vertex_groups.ndim == 1:
            assert_ndarray(arr=vertex_groups, name="vertex_groups", shape=(vertices.shape[0],))
            vertex_groups = vertex_groups[:, None]
        else:
            assert_ndarray(arr=vertex_groups, name="vertex_groups", shape=(vertices.shape[0], -1))
        if face_mask is not None:
            if face_mask.ndim == 1:
                assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0],))
            else:
                assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0], vertex_groups.shape[1]))
        list_sampled_vertices = []
        list_sampled_normals = []
        list_sampled_vertex_groups = []
        perm = None
        _mask = None
        same = same and (face_mask is None or (face_mask is not None and face_mask.ndim != 2))
        for i in range(vertex_groups.shape[1]):
            if face_mask is not None:
                if face_mask.ndim == 1:
                    perm = get_mask_perm(faces[face_mask])
                    _mask = face_mask
                else:
                    perm = get_mask_perm(faces[face_mask[:, i]])
                    _mask = face_mask[:, i]
            else:
                perm = get_mask_perm(None)
                _mask = None
            _num_samples = num_samples - len(perm)
            face_vertices, face_index, random_lengths = sample_surface(
                num_samples=_num_samples,
                vertices=vertices,
                faces=faces,
                mask=_mask,
            )
            list_sampled_vertices.append(np.concatenate([vertices[perm], face_vertices], axis=0))
            if vertex_normals is not None and face_normals is not None:
                list_sampled_normals.append(np.concatenate([vertex_normals[perm], face_normals[face_index]], axis=0))
            if same:
                g = sample_barycentric(vertex_groups, faces, face_index, random_lengths)
                list_sampled_vertex_groups.append(np.concatenate([vertex_groups[perm], g], axis=0))
                break
            g = sample_barycentric(vertex_groups[:, i:i+1], faces, face_index, random_lengths)[:, 0]
            list_sampled_vertex_groups.append(np.concatenate([vertex_groups[:, i][perm], g], axis=0))
        sampled_vertices = np.stack(list_sampled_vertices, axis=1)
        sampled_normals = np.stack(list_sampled_normals, axis=1) if list_sampled_normals else None
        sampled_vertex_groups = list_sampled_vertex_groups[0] if same else np.stack(list_sampled_vertex_groups, axis=1)
    else:
        if face_mask is not None:
            assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0],))
            perm = get_mask_perm(faces[face_mask])
        else:
            perm = get_mask_perm(None)
        num_samples -= len(perm)
        n_vertex = vertices[perm]
        face_vertices, face_index, random_lengths = sample_surface(num_samples, vertices, faces, face_mask)
        sampled_vertices = np.concatenate([n_vertex, face_vertices], axis=0)
        sampled_normals = (
            np.concatenate([vertex_normals[perm], face_normals[face_index]], axis=0)
            if vertex_normals is not None and face_normals is not None else None
        )
        sampled_vertex_groups = None
    return sampled_vertices, sampled_normals, sampled_vertex_groups
