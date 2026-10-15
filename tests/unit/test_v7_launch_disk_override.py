import datetime
import pytest
from avengine.timeline.v7_storage_budget import launch_minimum_free_bytes
pytestmark=pytest.mark.fast_unit
POLICY={'minimum_free_bytes':60_000_000_000,'expires_utc':'2026-10-15T04:30:00+00:00','after_expiry_minimum_free_bytes':100_000_000_000}

def test_default_launch_headroom_remains_explicit():
    assert launch_minimum_free_bytes()==100_000_000_000

def test_temporary_sixty_gb_applies_before_singapore_1230():
    assert launch_minimum_free_bytes(POLICY,datetime.datetime(2026,10,15,4,29,59,tzinfo=datetime.timezone.utc))==60_000_000_000

def test_expired_override_restores_one_hundred_gb():
    assert launch_minimum_free_bytes(POLICY,datetime.datetime(2026,10,15,4,30,tzinfo=datetime.timezone.utc))==100_000_000_000
