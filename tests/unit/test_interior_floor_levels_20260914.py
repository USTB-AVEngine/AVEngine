"""A navigable level is a floor only when the room has geometry overhead.

Recast walks a render surface wherever an agent fits, so a house exports its
roof as navigable alongside its rooms. These fixtures build the two cases that
have to come out differently - a single-storey box whose roof is walkable, and
a genuine two-storey box - from the same criterion, with no rule that names
either of them.
"""
import numpy as np
import pytest

from avengine.qa.answerability import MeshHandle
from avengine.rooms import conditioned_sampler as cs
from avengine.rooms import floor_levels


def slab(height, *, span=(0.0, 8.0), hole_fraction=0.0, cells=8):
    """A horizontal surface at ``height`` as ``cells`` x ``cells`` quads.

    ``hole_fraction`` drops that share of the quads, which is what a scan or an
    exported mesh looks like from below.
    """
    low, high = span
    step = (high - low) / cells
    vertices, triangles = [], []
    dropped = 0
    total = cells * cells
    for row in range(cells):
        for column in range(cells):
            index = row * cells + column
            if hole_fraction and dropped < round(hole_fraction * total) and index % 3 == 0:
                dropped += 1
                continue
            x0, z0 = low + column * step, low + row * step
            x1, z1 = x0 + step, z0 + step
            base = len(vertices)
            vertices.extend([[x0, height, z0], [x1, height, z0],
                             [x1, height, z1], [x0, height, z1]])
            triangles.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    return vertices, triangles


def mesh_of(*slabs):
    vertices, triangles = [], []
    for part_vertices, part_triangles in slabs:
        offset = len(vertices)
        vertices.extend(part_vertices)
        triangles.extend([[a + offset, b + offset, c + offset] for a, b, c in part_triangles])
    return MeshHandle(np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64))


class BoxSpace:
    """Navigation whose points sit on the given levels, as a raster space does."""

    def __init__(self, levels, span=(0.5, 7.5), step=1.0, resolution_m=None):
        self.metadata = {'authority': 'fixture_box_navmesh',
                         'floor_height_m': float(min(height for height, _ in levels))}
        if resolution_m is not None:
            # Declaring a cell size is what lets a raster space state an area.
            self.metadata['resolution_m'] = float(resolution_m)
        points = []
        for height, extent in levels:
            low, high = extent
            for x in np.arange(low, high + 1e-9, step):
                for z in np.arange(low, high + 1e-9, step):
                    points.append([float(x), float(height), float(z)])
        self._points = np.asarray(points, dtype=float)
        self._span = span

    def points(self, region=None):
        points = self._points
        if region is not None:
            bounds = np.asarray(region, dtype=float)
            points = points[np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)]
        return points

    def bounds(self):
        return np.array([[0.0, self._points[:, 1].min() - 1.0, 0.0],
                         [8.0, self._points[:, 1].max() + 1.0, 8.0]], dtype=float)

    def sample_navigable(self, rng, region=None):
        points = self.points(region)
        if not len(points):
            raise ValueError('requested region has no navigable cells')
        return points[int(rng.integers(len(points)))].copy()

    def is_navigable(self, point):
        point = np.asarray(point, dtype=float)
        return bool(np.any(np.all(np.abs(self._points - point) < 1e-6, axis=1)))

    def route_bank(self):
        return None


def single_storey():
    """One room, a ceiling at 2.8 m, and a walkable roof at 3.2 m."""
    space = BoxSpace([(0.0, (0.5, 7.5)), (3.2, (0.5, 7.5))])
    mesh = mesh_of(slab(0.0), slab(2.8), slab(3.2))
    return space, mesh


def two_storey():
    """Two rooms stacked, a ceiling at 5.8 m, and a walkable roof at 6.2 m."""
    space = BoxSpace([(0.0, (0.5, 7.5)), (3.0, (0.5, 7.5)), (6.2, (0.5, 7.5))])
    mesh = mesh_of(slab(0.0), slab(3.0), slab(5.8), slab(6.2))
    return space, mesh


def verdicts(record):
    return {round(row['height_m'], 3): row['verdict'] for row in record['levels']}


def test_roof_of_a_single_storey_room_is_not_a_floor():
    space, mesh = single_storey()
    record = floor_levels.interior_floor_levels(space, mesh)
    assert record['status'] == 'measured'
    assert verdicts(record) == {0.0: 'interior', 3.2: 'open_to_sky'}
    assert [round(value, 3) for value in record['legal_heights_m']] == [0.0]
    roof = next(row for row in record['levels'] if round(row['height_m'], 3) == 3.2)
    assert roof['covered_share'] == 0.0
    assert 'without meeting geometry' in roof['reason']


def test_both_storeys_of_a_two_storey_room_stay_floors():
    space, mesh = two_storey()
    record = floor_levels.interior_floor_levels(space, mesh)
    assert record['status'] == 'measured'
    assert verdicts(record) == {0.0: 'interior', 3.0: 'interior', 6.2: 'open_to_sky'}
    assert [round(value, 3) for value in record['legal_heights_m']] == [0.0, 3.0]


def test_a_ceiling_with_holes_still_reads_as_interior():
    space = BoxSpace([(0.0, (0.5, 7.5)), (3.2, (0.5, 7.5))])
    mesh = mesh_of(slab(0.0), slab(2.8, hole_fraction=0.25), slab(3.2, hole_fraction=0.25))
    record = floor_levels.interior_floor_levels(space, mesh)
    ground = next(row for row in record['levels'] if round(row['height_m'], 3) == 0.0)
    assert ground['verdict'] == 'interior'
    assert ground['covered_share'] < 1.0
    assert verdicts(record)[3.2] == 'open_to_sky'


def test_without_a_mesh_the_declaration_is_returned_and_the_record_says_so():
    space, _mesh = single_storey()
    room = {'room_package': {'floor_heights_m': [0.0, 3.2]}}
    record = cs.planning_floor_decision(room, space, None)
    assert record['status'] == 'unmeasured'
    assert 'no static mesh' in record['reason']
    assert record['legal_heights_m'] == [0.0, 3.2]
    assert cs.declared_floor_heights_m(room, space, None) == [0.0, 3.2]
    assert {row['verdict'] for row in record['levels']} == {'unmeasured'}
    drawn = {round(cs.lock_same_floor_region(space, np.random.default_rng(seed), None, room)[1], 3)
             for seed in range(24)}
    assert drawn == {0.0, 3.2}


def test_a_room_that_declares_nothing_gets_its_floor_from_geometry():
    space, mesh = single_storey()
    room = {'room_package': {}}
    assert cs._room_declared_floor_heights_m(room, space) == []
    drawn = {round(cs.lock_same_floor_region(space, np.random.default_rng(seed), None, room, mesh)[1], 3)
             for seed in range(24)}
    assert drawn == {0.0}
    without_mesh = {round(cs.lock_same_floor_region(space, np.random.default_rng(seed), None, room)[1], 3)
                    for seed in range(24)}
    assert without_mesh == {0.0, 3.2}


def test_dropping_the_roof_does_not_promote_a_platform_into_a_floor():
    """The share that decides a floor is a share of the whole room.

    Half the navigation of this fixture is roof and a twentieth of it is a
    raised platform. Measured against the levels left after the roof goes, the
    platform looks like a tenth of the room and would draw plans; measured
    against the room, it stays the platform it is.
    """
    space = BoxSpace([(0.0, (0.5, 7.5)), (3.2, (0.5, 7.5))])
    platform = np.array([[float(x), 0.55, float(z)]
                         for x in np.arange(2.0, 4.1, 1.0) for z in np.arange(2.0, 3.1, 1.0)])
    space._points = np.concatenate([space._points, platform])
    mesh = mesh_of(slab(0.0), slab(0.55, span=(2.0, 4.0), cells=2), slab(2.8), slab(3.2))
    record = floor_levels.interior_floor_levels(space, mesh)
    assert verdicts(record) == {0.0: 'interior', 0.55: 'interior', 3.2: 'open_to_sky'}
    legal = [row['height_m'] for row in record['levels'] if row['verdict'] == 'interior']
    weights = cs._floor_navigable_weights(space, legal, decision=record)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == 0.0
    room = {'room_package': {}}
    drawn = {round(cs.lock_same_floor_region(space, np.random.default_rng(seed), None, room, mesh)[1], 3)
             for seed in range(24)}
    assert drawn == {0.0}


def test_the_decision_is_measured_once_per_space_and_mesh():
    space, mesh = single_storey()
    first = floor_levels.floor_level_decision(space, mesh)
    assert floor_levels.floor_level_decision(space, mesh) is first
    other = mesh_of(slab(0.0), slab(2.8))
    assert floor_levels.floor_level_decision(space, other) is not first


def test_a_navigation_package_may_declare_its_interior_floors():
    from avengine.capture import qa_plan_adapters

    assert qa_plan_adapters._declared_planning_floors({}) == {}
    assert qa_plan_adapters._declared_planning_floors(
        {'walkable_space': {'planning_floors_m': [0.0, 3.0]}}
    ) == {'planning_floors_m': [0.0, 3.0]}
    with pytest.raises(ValueError):
        qa_plan_adapters._declared_planning_floors(
            {'walkable_space': {'planning_floors_m': [float('nan')]}})
    space, mesh = single_storey()
    space.metadata['planning_floors_m'] = [0.0]
    assert cs._room_declared_floor_heights_m({'room_package': {}}, space) == [0.0]
    assert cs.declared_floor_heights_m({'room_package': {}}, space, mesh) == [0.0]


def test_levels_are_taken_as_dense_bands_rather_than_chained_by_a_staircase():
    ramp = np.array([[0.0, height, 0.0] for height in np.arange(0.0, 3.01, 0.15)])
    ground = np.array([[float(x), 0.0, float(z)]
                       for x in range(6) for z in range(6)])
    upper = np.array([[float(x), 3.0, float(z)]
                      for x in range(6) for z in range(6)])
    levels = floor_levels.navigable_level_clusters(
        np.concatenate([ground, ramp, upper]), tolerance_m=0.3)
    heights = [round(height, 2) for height, _ in levels]
    assert heights[0] == pytest.approx(0.0, abs=0.05)
    assert heights[-1] == pytest.approx(3.0, abs=0.05)
    assert len(levels) >= 4


class FakePathFinder:
    """The part of a Recast pathfinder the builder's floor probe uses."""

    def __init__(self, points):
        self._points = np.asarray(points, dtype=float)
        self._cursor = 0

    def seed(self, value):
        self._cursor = int(value) % len(self._points)

    def get_random_navigable_point(self):
        point = self._points[self._cursor % len(self._points)]
        self._cursor += 1
        return point


def test_a_built_navigation_package_declares_the_levels_inside_the_building(tmp_path):
    from avengine.rooms import navigation_preparation

    space, mesh = single_storey()
    vertices = tmp_path / 'vertices.npy'
    triangles = tmp_path / 'triangles.npy'
    np.save(vertices, mesh.vertices)
    np.save(triangles, mesh.triangles)
    finder = FakePathFinder(space.points())
    floors = navigation_preparation._interior_floors_of(finder, vertices, triangles)
    assert floors['status'] == 'measured'
    assert [round(value, 3) for value in floors['legal_heights_m']] == [0.0]
    assert verdicts(floors)[3.2] == 'open_to_sky'
    assert navigation_preparation._interior_floors_of(finder, None, None)['status'] == 'unmeasured'


# ------------------------------------------------------- how big a level has to be

def storeyed(upper_extent, *, resolution_m=1.0):
    """A ground floor, an upper level of the given extent, and a walkable roof."""
    space = BoxSpace([(0.0, (0.5, 7.5)), (3.0, upper_extent), (6.2, (0.5, 7.5))],
                     resolution_m=resolution_m)
    mesh = mesh_of(slab(0.0), slab(2.8), slab(3.0, span=(upper_extent[0] - 0.5, upper_extent[1] + 0.5),
                                              cells=2),
                   slab(5.8), slab(6.2))
    return space, mesh


def test_an_indoor_level_too_small_to_hold_an_episode_is_not_a_floor():
    space, mesh = storeyed((3.5, 4.5))
    record = floor_levels.interior_floor_levels(space, mesh)
    assert record['navigable_area_source'] == 'navigable_cells_times_declared_cell_area'
    by_height = {round(row['height_m'], 3): row for row in record['levels']}
    assert by_height[0.0]['area_m2'] == pytest.approx(64.0)
    assert by_height[0.0]['verdict'] == 'interior'
    assert by_height[3.0]['area_m2'] == pytest.approx(4.0)
    assert by_height[3.0]['verdict'] == 'too_small'
    assert 'less navigable surface' in by_height[3.0]['reason']
    # The area rule does not rescue a roof, and does not have to judge it either.
    assert by_height[6.2]['verdict'] == 'open_to_sky'
    assert [round(value, 3) for value in record['legal_heights_m']] == [0.0]


def test_a_second_storey_above_the_minimum_area_stays_a_floor():
    space, mesh = storeyed((0.5, 7.5))
    record = floor_levels.interior_floor_levels(space, mesh)
    by_height = {round(row['height_m'], 3): row for row in record['levels']}
    assert by_height[3.0]['area_m2'] == pytest.approx(64.0)
    assert by_height[3.0]['verdict'] == 'interior'
    assert [round(value, 3) for value in record['legal_heights_m']] == [0.0, 3.0]


def test_a_space_that_cannot_measure_area_keeps_every_indoor_level():
    space, mesh = storeyed((3.5, 4.5), resolution_m=None)
    record = floor_levels.interior_floor_levels(space, mesh)
    assert record['navigable_area_source'] == 'unavailable'
    assert all(row['area_m2'] is None for row in record['levels'])
    by_height = {round(row['height_m'], 3): row for row in record['levels']}
    assert by_height[3.0]['verdict'] == 'interior'
    assert 'cannot measure level area' in by_height[3.0]['reason']
    assert [round(value, 3) for value in record['legal_heights_m']] == [0.0, 3.0]


class NavmeshLikeSpace(BoxSpace):
    """A space that cannot list its cells but knows its total navigable area."""

    class _PathFinder:
        def __init__(self, navigable_area):
            self.navigable_area = float(navigable_area)

    def __init__(self, levels, navigable_area, **kwargs):
        super().__init__(levels, **kwargs)
        self.pathfinder = self._PathFinder(navigable_area)
        self.metadata.pop('resolution_m', None)

    # A native navmesh answers point queries; it cannot enumerate its cells.
    points = None

    def sample_navigable(self, rng, region=None):
        return self._points[int(rng.integers(len(self._points)))].copy()


def test_a_native_navmesh_measures_level_area_from_its_own_total():
    space = NavmeshLikeSpace([(0.0, (0.5, 7.5)), (3.0, (3.5, 4.5)), (6.2, (0.5, 7.5))],
                             navigable_area=132.0)
    mesh = mesh_of(slab(0.0), slab(2.8), slab(3.0, span=(3.0, 5.0), cells=2), slab(5.8), slab(6.2))
    record = floor_levels.interior_floor_levels(space, mesh)
    assert record['navigable_area_source'] == 'navmesh_navigable_area_times_pool_share'
    assert record['navigable_pool_source'] == 'fixed_seed_sample_navigable'
    by_height = {round(row['height_m'], 3): row for row in record['levels']}
    # 64 + 4 + 64 navigable cells share 132 m2, so the upper level is about 4 m2.
    assert by_height[3.0]['area_m2'] == pytest.approx(4.0, abs=2.0)
    assert by_height[3.0]['verdict'] == 'too_small'
    assert by_height[0.0]['area_m2'] == pytest.approx(64.0, abs=8.0)
    assert [round(value, 3) for value in record['legal_heights_m']] == [0.0]


def test_a_room_whose_every_indoor_level_is_small_keeps_them_all():
    space = BoxSpace([(0.0, (3.5, 4.5)), (6.2, (0.5, 7.5))], resolution_m=1.0)
    mesh = mesh_of(slab(0.0, span=(3.0, 5.0), cells=2), slab(2.8), slab(6.2))
    record = floor_levels.interior_floor_levels(space, mesh)
    assert record['status'] == 'no_level_above_minimum_area'
    assert 'rather than leaving the room with no floor' in record['reason']
    assert [round(value, 3) for value in record['legal_heights_m']] == [0.0]
    by_height = {round(row['height_m'], 3): row for row in record['levels']}
    assert by_height[0.0]['verdict'] == 'too_small'


def test_the_plan_record_carries_the_measured_area_of_every_level():
    space, mesh = storeyed((3.5, 4.5))
    summary = floor_levels.decision_summary(floor_levels.interior_floor_levels(space, mesh))
    assert summary['navigable_area_source'] == 'navigable_cells_times_declared_cell_area'
    assert [row['area_m2'] for row in summary['levels']] == [
        pytest.approx(64.0), pytest.approx(4.0), pytest.approx(64.0)]
    assert summary['criterion']['minimum_navigable_area_m2'] == floor_levels.MIN_FLOOR_NAVIGABLE_AREA_M2
