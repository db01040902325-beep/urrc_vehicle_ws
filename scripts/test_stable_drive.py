#!/usr/bin/env python3
"""Offline safety checks for stable_drive geometry; ROS 2 is not required."""
import ast
import math
import statistics
import unittest
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'src/urrc_lidar_racer/urrc_lidar_racer/stable_drive.py'
tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
names = {'clamp', 'median_wall', 'corridor', 'choose_command'}
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert {f.name for f in functions} == names, 'Controller pure helpers not found'
namespace = {'math': math, 'statistics': statistics}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
choose_command = namespace['choose_command']


def walls(near_center=0.0, far_center=0.0, width=2.4):
    pts = []
    for x in [1.0, 1.15, 1.3, 1.45, 1.6, 1.75,
              3.25, 3.4, 3.55, 3.7, 3.85, 4.0]:
        center = near_center if x < 2.4 else far_center
        pts.extend([(x, center + width / 2), (x, center - width / 2)])
    return pts


class StableDriveTests(unittest.TestCase):
    def test_straight_corridor_does_not_force_center(self):
        speed, steer, reason = choose_command(walls(0.15, 0.15), 0, 0, 0.05)
        self.assertEqual(reason, 'DRIVING')
        self.assertAlmostEqual(steer, 0.0, places=5)
        self.assertGreater(speed, 0.0)

    def test_missing_wall_blocks_motion(self):
        speed, steer, reason = choose_command(
            [(1.0, 1.2), (3.4, 1.2)], 0, 0, 0.05)
        self.assertEqual((speed, steer, reason), (0.0, 0.0, 'PATH_UNVERIFIED'))

    def test_corner_steer_sign(self):
        _, left, reason = choose_command(walls(0.0, 0.6), 0, 0.4, 0.05)
        self.assertEqual(reason, 'DRIVING')
        self.assertGreater(left, 0.0)
        _, right, reason = choose_command(walls(0.0, -0.6), 0, 0.4, 0.05)
        self.assertEqual(reason, 'DRIVING')
        self.assertLess(right, 0.0)

    def test_front_obstacle_stops(self):
        p = walls() + [(0.8, 0.0)]
        self.assertEqual(choose_command(p, 0, 1, 0.05)[2], 'FRONT_BLOCKED')

    def test_high_heading_is_limited(self):
        speed, steer, reason = choose_command(walls(0.0, 0.72), 0, 0, 0.05)
        self.assertEqual(reason, 'DRIVING')
        self.assertLessEqual(abs(math.degrees(steer)), 1.25 + 1e-8)
        self.assertLessEqual(speed, 2.2)

    def test_wall_too_close_stops(self):
        speed, steer, reason = choose_command(walls(1.03, 1.03), 0, 0, 0.05)
        self.assertEqual((speed, steer, reason), (0.0, 0.0, 'SIDE_WALL_TOO_CLOSE'))


if __name__ == '__main__':
    unittest.main()
