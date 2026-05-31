#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clear tvc_traj_player RViz displays (one-shot helper, no extra ROS package)."""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


def _empty_path(frame_id: str, stamp) -> Path:
    msg = Path()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    return msg


def _empty_pose(frame_id: str, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    return msg


def _delete_all_markers(frame_id: str, stamp) -> MarkerArray:
    delete_markers = MarkerArray()
    delete_all = Marker()
    delete_all.header.stamp = stamp
    delete_all.header.frame_id = frame_id
    delete_all.action = Marker.DELETEALL
    delete_markers.markers.append(delete_all)
    return delete_markers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'frame_id', nargs='?', default='world', help='Header frame_id (default: world)'
    )
    parser.add_argument(
        '--mode',
        choices=('executed', 'all'),
        default='executed',
        help='executed: clear executed_path + current_setpoint; '
             'all: also clear planned_path + waypoint_markers',
    )
    parser.add_argument(
        '--hold', type=float, default=1.5,
        help='Seconds to republish clears (mode all)',
    )
    args = parser.parse_args(argv[1:] if argv else [])

    rclpy.init(args=argv)
    node = Node('tvc_clear_traj_viz')
    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    pubs = [
        (node.create_publisher(Path, '/tvc_traj_player/executed_path', qos), 'executed_path'),
        (
            node.create_publisher(
                PoseStamped, '/tvc_traj_player/current_setpoint', qos
            ),
            'current_setpoint',
        ),
    ]
    if args.mode == 'all':
        pubs.insert(
            0,
            (node.create_publisher(Path, '/tvc_traj_player/planned_path', qos), 'planned_path'),
        )
        pubs.append(
            (
                node.create_publisher(
                    MarkerArray, '/tvc_traj_player/waypoint_markers', qos
                ),
                'waypoint_markers',
            )
        )

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if any(pub.get_subscription_count() > 0 for pub, _ in pubs):
            break

    hold_until = time.monotonic() + max(0.3, float(args.hold))
    while time.monotonic() < hold_until:
        stamp = node.get_clock().now().to_msg()
        empty_path = _empty_path(args.frame_id, stamp)
        empty_pose = _empty_pose(args.frame_id, stamp)
        delete_markers = _delete_all_markers(args.frame_id, stamp)
        for pub, kind in pubs:
            if kind == 'current_setpoint':
                pub.publish(empty_pose)
            elif kind == 'waypoint_markers':
                pub.publish(delete_markers)
            else:
                pub.publish(empty_path)
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.15)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
