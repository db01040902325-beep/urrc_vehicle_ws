#!/usr/bin/env python3
"""Run without ROS/Gazebo: python3 scripts/test_steering_guard.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/urrc_lidar_racer'))
from urrc_lidar_racer.steering_guard import SteeringReversalGuard


class SteeringGuardTests(unittest.TestCase):
    def test_straight_and_same_turn(self):
        guard = SteeringReversalGuard()
        self.assertEqual([guard.filter(x) for x in (0, 5, 10, 15)], [0, 5, 10, 15])

    def test_isolated_opposite_is_suppressed(self):
        guard = SteeringReversalGuard()
        self.assertEqual([guard.filter(x) for x in (12, -12, 12)], [12, 0, 12])

    def test_real_reversal_after_three_frames(self):
        guard = SteeringReversalGuard()
        self.assertEqual([guard.filter(x) for x in (12, -12, -13, -14, -15)],
                         [12, 0, 0, -14, -15])

    def test_small_noise_does_not_change_turn_direction(self):
        guard = SteeringReversalGuard()
        self.assertEqual([guard.filter(x) for x in (10, -2, -10, 10)],
                         [10, -2, 0, 10])

    def test_reset_allows_hairpin_direction(self):
        guard = SteeringReversalGuard()
        guard.filter(10)
        guard.reset()
        self.assertEqual(guard.filter(-20), -20)


if __name__ == '__main__':
    unittest.main()
