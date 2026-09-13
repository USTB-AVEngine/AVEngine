"""Render-surface declarations distinguish authority from array storage."""
import pytest
from avengine.rooms.conditioned_visibility import geometry_authority_for_package

@pytest.mark.parametrize("declaration,expected", [(True,"visual_mesh"),(False,"acoustic_proxy_mesh"),(None,"acoustic_proxy_mesh"),("true","acoustic_proxy_mesh")])
def test_render_surface_requires_an_explicit_boolean(declaration,expected):
    package={"static_geometry":{"source":"acoustic_package_arrays","render_surface":declaration}}
    assert geometry_authority_for_package(package)==expected

def test_missing_geometry_is_unknown():
    assert geometry_authority_for_package({})=="unknown"
