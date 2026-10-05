import unittest

from tools.rooms.room_screening.screening_policy import ScreeningFacts, screen_room


class ScreeningPolicyTests(unittest.TestCase):
    def test_area_minimum_uses_semantic_main_area(self):
        result = screen_room(ScreeningFacts(area_m2=9.99))
        self.assertEqual(result.suggestion, "discard_candidate")
        self.assertIn("area_below_10m2", result.reason_codes)

    def test_probe_upper_bound_holds_threshold_sensitive_scope(self):
        result = screen_room(
            ScreeningFacts(area_m2=8, scope_verified=False, unassigned_ground_upper_m2=4)
        )
        self.assertEqual(result.suggestion, "scope_uncertain_hold")
        self.assertEqual(result.area_m2, 8)  # Candidate upper bound is not added.

    def test_probe_upper_bound_below_minimum_does_not_prevent_small_area_filter(self):
        result = screen_room(
            ScreeningFacts(area_m2=8, scope_verified=False, unassigned_ground_upper_m2=1)
        )
        self.assertEqual(result.suggestion, "discard_candidate")

    def test_scene_type_requires_no_room_body(self):
        with_body = screen_room(
            ScreeningFacts(area_m2=14, stair_dominated=True, room_body_present=True)
        )
        without_body = screen_room(
            ScreeningFacts(area_m2=14, stair_dominated=True, room_body_present=False)
        )
        self.assertEqual(with_body.suggestion, "usable_candidate")
        self.assertEqual(without_body.suggestion, "discard_candidate")

    def test_shape_checks_are_joint_not_single_metric(self):
        narrow_only = screen_room(
            ScreeningFacts(area_m2=16, elongation_long_short=5.2,
                           largest_component_median_width_m=0.7)
        )
        fragmented = screen_room(
            ScreeningFacts(area_m2=16, component_count_ge_min_area=3,
                           largest_component_fraction=0.4)
        )
        self.assertEqual(narrow_only.suggestion, "discard_candidate")
        self.assertEqual(fragmented.suggestion, "discard_candidate")

    def test_large_area_is_review_or_crop_candidate_not_auto_split(self):
        result = screen_room(ScreeningFacts(area_m2=50))
        self.assertEqual(result.suggestion, "crop_candidate")
        self.assertIn("area_at_least_50m2_review_boundary", result.reason_codes)

    def test_missing_area_is_not_zero(self):
        result = screen_room(ScreeningFacts(area_m2=None, area_assessable=False))
        self.assertEqual(result.suggestion, "unassessable_hold")


if __name__ == "__main__":
    unittest.main()
