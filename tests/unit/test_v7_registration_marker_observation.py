import numpy as np
import pytest
from avengine.capture.v7_visible_native import attachment_template
pytestmark=pytest.mark.fast_unit


def test_registration_marker_density_does_not_reject_real_sole_contact():
    vertices=np.array([[x,y,z]for x in (-.2,.2)for z in (-.3,.3)for y in (0.,.05)])
    landmarks={'foot_'+str(i):[x,.025,z]for i,(x,z)in enumerate([(-.2,-.3),(-.2,.3),(.2,-.3),(.2,.3)])}
    anchor,probes,evidence=attachment_template(vertices,'floor','animal',landmarks)
    projected=evidence['registered_foot_original_surface_projection']
    assert len(projected)==4
    assert all(p['distance_m']>.015 and p['registration_marker_projection_warning']for p in projected)
    assert all(p['original_surface_point_m'][1]==0 for p in projected)
    assert all(p['marker_distance_admission_effect'].startswith('record_only')for p in projected)
