#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, Image


class BirdEyeCalibrator(Node):
    def __init__(self):
        super().__init__('bird_eye_calibrator')

        self.declare_parameter(
            'image_topic',
            '/camera/color/image_raw/compressed',
        )
        self.declare_parameter('img_width', 640)
        self.declare_parameter('img_height', 480)
        self.declare_parameter('dst_margin_ratio', 0.25)

        self.image_topic = self.get_parameter('image_topic').value
        self.img_width = int(self.get_parameter('img_width').value)
        self.img_height = int(self.get_parameter('img_height').value)
        self.dst_margin_ratio = float(
            self.get_parameter('dst_margin_ratio').value
        )

        self.cv_bridge = CvBridge()
        self.frame = None
        self.src_points = []
        self.dst_points = self.make_default_dst_points()
        self.warp_mat = None
        self.printed_current_points = False

        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_cb,
            qos_profile_sensor_data,
        )
        self.bird_eye_pub = self.create_publisher(
            Image,
            '/bird_eye_calib/image',
            10,
        )

        self.source_window = 'bird_eye_source'
        self.result_window = 'bird_eye_result'
        cv.namedWindow(self.source_window, cv.WINDOW_NORMAL)
        cv.namedWindow(self.result_window, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self.source_window, self.mouse_cb)

        self.timer = self.create_timer(1.0 / 30.0, self.process)
        self.get_logger().info(
            'Click 4 points on source image: LT, RT, LB, RB. '
            'Keys: u=undo, r=reset, p=print, q=quit.'
        )

    def make_default_dst_points(self):
        margin = self.img_width * self.dst_margin_ratio
        return np.float32([
            [margin, 0.0],
            [self.img_width - margin, 0.0],
            [margin, self.img_height - 1.0],
            [self.img_width - margin, self.img_height - 1.0],
        ])

    def image_cb(self, image_msg):
        try:
            frame = self.cv_bridge.compressed_imgmsg_to_cv2(
                image_msg,
                desired_encoding='bgr8',
            )
        except Exception as exc:
            self.get_logger().warning(f'Failed to decode camera image: {exc}')
            return

        if frame.shape[1] != self.img_width or frame.shape[0] != self.img_height:
            frame = cv.resize(frame, (self.img_width, self.img_height))

        self.frame = frame

    def mouse_cb(self, event, x, y, flags, param):
        del flags
        del param

        if event != cv.EVENT_LBUTTONDOWN:
            return

        if len(self.src_points) >= 4:
            self.src_points = []

        self.src_points.append([float(x), float(y)])
        self.printed_current_points = False
        self.update_warp_matrix()

    def update_warp_matrix(self):
        if len(self.src_points) != 4:
            self.warp_mat = None
            return

        src = np.float32(self.src_points)
        self.warp_mat = cv.getPerspectiveTransform(src, self.dst_points)
        self.print_current_points()

    def print_current_points(self):
        if len(self.src_points) != 4 or self.warp_mat is None:
            return

        src = np.float32(self.src_points)
        self.get_logger().info('Current bird eye view points:')
        self.get_logger().info(f'src_points = {src.tolist()}')
        self.get_logger().info(f'dst_points = {self.dst_points.tolist()}')
        self.get_logger().info(f'warp_mat = {self.warp_mat.tolist()}')
        self.get_logger().info(
            'pre_lane_follow.py style:\n'
            f'self.src_points = np.float32({src.tolist()})\n'
            f'self.dst_points = np.float32({self.dst_points.tolist()})'
        )
        self.printed_current_points = True

    def draw_source_view(self):
        if self.frame is None:
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        view = self.frame.copy()
        labels = ['LT', 'RT', 'LB', 'RB']

        for idx, point in enumerate(self.src_points):
            x, y = int(point[0]), int(point[1])
            cv.circle(view, (x, y), 6, (0, 255, 255), -1)
            cv.putText(
                view,
                labels[idx],
                (x + 8, y - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv.LINE_AA,
            )

        if len(self.src_points) == 4:
            pts = np.int32(self.src_points)
            cv.polylines(view, [pts], True, (0, 255, 0), 2)

        status = f'points: {len(self.src_points)}/4'
        cv.putText(
            view,
            status,
            (20, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            view,
            'u undo | r reset | p print | q quit',
            (20, self.img_height - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        return view

    def make_bird_eye_view(self):
        if self.frame is None:
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        if self.warp_mat is None:
            return self.frame.copy()

        return cv.warpPerspective(
            self.frame,
            self.warp_mat,
            (self.img_width, self.img_height),
        )

    def handle_key(self, key):
        if key in (ord('q'), 27):
            rclpy.shutdown()
        elif key == ord('r'):
            self.src_points = []
            self.warp_mat = None
            self.printed_current_points = False
            self.get_logger().info('Reset source points.')
        elif key == ord('u'):
            if self.src_points:
                removed = self.src_points.pop()
                self.update_warp_matrix()
                self.get_logger().info(f'Undo point: {removed}')
        elif key == ord('p'):
            if len(self.src_points) == 4:
                self.print_current_points()
            else:
                self.get_logger().info(
                    f'Need 4 points before printing. '
                    f'Current: {len(self.src_points)}/4'
                )

    def process(self):
        source_view = self.draw_source_view()
        bird_eye_view = self.make_bird_eye_view()

        if self.warp_mat is not None:
            self.bird_eye_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(bird_eye_view, encoding='bgr8')
            )

        cv.imshow(self.source_window, source_view)
        cv.imshow(self.result_window, bird_eye_view)

        key = cv.waitKey(1) & 0xFF
        if key != 255:
            self.handle_key(key)


def main(args=None):
    rclpy.init(args=args)
    node = BirdEyeCalibrator()

    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        cv.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
