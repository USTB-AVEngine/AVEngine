"""The retired stage names stay as CLI aliases of the capability commands."""
from avengine.cli import build_parser


def test_stage_aliases_parse_like_the_capability_names() -> None:
    parser = build_parser()
    for alias, primary in [
        (["m1", "verify", "evidence.json"], ["rooms", "verify", "evidence.json"]),
        (["m3", "validate-package", "pkg"], ["acoustics", "validate-package", "pkg"]),
        (["m4", "validate-request", "req"], ["spatial-audio", "validate-request", "req"]),
    ]:
        a = parser.parse_args(alias)
        b = parser.parse_args(primary)
        assert a.handler is b.handler
