#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import deque

import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64


class LaneFollow(Node):
    def __init__(self):
        super().__init__('lane_follow')

        self.cv_bridge = CvBridge()

        self.declare_parameter('img_width', 640)
        self.declare_parameter('img_height', 480)
        self.declare_parameter('white_lower', [0, 0, 180])
        self.declare_parameter('white_upper', [180, 20, 255])
        self.declare_parameter('debug_view', True)
        self.declare_parameter('process_hz', 30.0)
        self.declare_parameter('steer_k', 0.005)
        self.declare_parameter('yaw_k', 1.0)
        self.declare_parameter('max_steer', 0.409)
        self.declare_parameter('steer_smoothing_alpha', 0.35)
        self.declare_parameter('steer_slowdown_ratio', 0.35)
        self.declare_parameter('min_smooth_speed', 0.45)
        self.declare_parameter('lane_width_px', 250.0)
        self.declare_parameter('min_lane_overlap_px', 50.0)
        self.declare_parameter('min_lane_pixels', 30)
        self.declare_parameter('use_history_lane_fallback', True)
        self.declare_parameter('lane_history_size', 10)
        self.declare_parameter('min_lane_history_samples', 3)
        self.declare_parameter('history_compare_y_ratio', 0.85)

        self.img_width = int(self.get_parameter('img_width').value)
        self.img_height = int(self.get_parameter('img_height').value)
        self.white_lower = np.array(
            self.get_parameter('white_lower').value,
            dtype=np.uint8,
        )
        self.white_upper = np.array(
            self.get_parameter('white_upper').value,
            dtype=np.uint8,
        )
        self.debug_view = bool(self.get_parameter('debug_view').value)
        process_hz = float(self.get_parameter('process_hz').value)
        self.steer_k = float(self.get_parameter('steer_k').value)
        self.yaw_k = float(self.get_parameter('yaw_k').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.steer_smoothing_alpha = float(
            self.get_parameter('steer_smoothing_alpha').value
        )
        self.steer_slowdown_ratio = float(
            self.get_parameter('steer_slowdown_ratio').value
        )
        self.min_smooth_speed = float(
            self.get_parameter('min_smooth_speed').value
        )
        self.lane_width_px = float(self.get_parameter('lane_width_px').value)
        self.min_lane_overlap_px = float(
            self.get_parameter('min_lane_overlap_px').value
        )
        self.min_lane_pixels = int(self.get_parameter('min_lane_pixels').value)
        self.use_history_lane_fallback = bool(
            self.get_parameter('use_history_lane_fallback').value
        )
        self.lane_history_size = int(
            self.get_parameter('lane_history_size').value
        )
        self.min_lane_history_samples = int(
            self.get_parameter('min_lane_history_samples').value
        )
        self.history_compare_y_ratio = float(
            self.get_parameter('history_compare_y_ratio').value
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_cb,
            qos_profile_sensor_data,
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10,
        )
        self.roi_img_pub = self.create_publisher(Image, '/roi_img', 10)

        self.binary_img_pub = self.create_publisher(Image, '/binary_img', 10)

        self.debug_publisher1 = self.create_publisher(
            Image,
            '/debugging_image1',
            10,
        )
        self.debug_publisher2 = self.create_publisher(
            Image,
            '/debugging_image2',
            10,
        )
        self.src_points = np.float32([[119.0, 233.0], [492.0, 215.0], [13.0, 304.0], [622.0, 326.0]])
        self.dst_points = np.float32([[160.0, 0.0], [480.0, 0.0], [160.0, 479.0], [480.0, 479.0]])

        self.warp_mat = cv.getPerspectiveTransform(
            self.src_points,
            self.dst_points,
        )
        self.inv_warp_mat = cv.getPerspectiveTransform(
            self.dst_points,
            self.src_points,
        )

        self.bgr = None
        self.warp_img0 = None
        self.warp_img = None
        self.white_img = None
        self.filtered_img = None
        self.gaussian_sigma = 1
        self.gear = 3
        self.yaw = 0.0
        self.error = 0.0
        self.steer = 0.0
        self.angular_velocity = 0.0
        self.prev_steer = None
        self.cmd_speed = 0.0
        self.prev_lfit = None
        self.prev_rfit = None
        self.left_fit_history = deque(maxlen=self.lane_history_size)
        self.right_fit_history = deque(maxlen=self.lane_history_size)
        self.last_lane_status = 'none'

        period = 1.0 / process_hz if process_hz > 0.0 else 1.0 / 30.0
        self.timer = self.create_timer(period, self.process)
        self.get_logger().info('ROS2 lane_follow node initialized')

    def image_cb(self, image_msg):
        try:
            bgr = self.cv_bridge.compressed_imgmsg_to_cv2(
                image_msg,
                desired_encoding='bgr8',
            )
        except Exception as exc:
            self.get_logger().warning(f'Failed to decode camera image: {exc}')
            return

        if bgr.shape[1] != self.img_width or bgr.shape[0] != self.img_height:
            bgr = cv.resize(bgr, (self.img_width, self.img_height))

        self.bgr = bgr


    def warpping(self, img):
        h, w = img.shape[:2]
        return cv.warpPerspective(img, self.warp_mat, (w, h))

    def gaussian_filter(self, img):
        return cv.GaussianBlur(img, (0, 0), self.gaussian_sigma)

    def white_color_filter_hsv(self, img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        white_hsv = cv.inRange(hsv, self.white_lower, self.white_upper)
        return cv.bitwise_and(img, img, mask=white_hsv)

    def binary_filter(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
        return binary

    def detect_original_bottom_lane_side(self):
        if self.bgr is None:
            return None

        white_img = self.white_color_filter_hsv(self.bgr)
        gray = cv.cvtColor(white_img, cv.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        bottom_roi = gray[int(h * 0.85):h, :]
        midpoint = w // 2

        left_count = cv.countNonZero(bottom_roi[:, :midpoint])
        right_count = cv.countNonZero(bottom_roi[:, midpoint:])

        if left_count == right_count:
            return None
        if left_count > right_count:
            return 'left'
        return 'right'

    def update_lane_history(self, lfit, rfit):
        self.left_fit_history.append(np.array(lfit, dtype=float))
        self.right_fit_history.append(np.array(rfit, dtype=float))

    def detect_lane_side_by_history(self, fit, img_h):
        if not self.use_history_lane_fallback:
            return None
        if len(self.left_fit_history) < self.min_lane_history_samples:
            return None
        if len(self.right_fit_history) < self.min_lane_history_samples:
            return None

        avg_lfit = np.mean(np.array(self.left_fit_history), axis=0)
        avg_rfit = np.mean(np.array(self.right_fit_history), axis=0)
        compare_y = (img_h - 1) * self.history_compare_y_ratio

        detected_x = fit[0] * compare_y + fit[1]
        avg_left_x = avg_lfit[0] * compare_y + avg_lfit[1]
        avg_right_x = avg_rfit[0] * compare_y + avg_rfit[1]

        left_dist = abs(detected_x - avg_left_x)
        right_dist = abs(detected_x - avg_right_x)

        if left_dist == right_dist:
            return None
        if left_dist < right_dist:
            return 'left'
        return 'right'

    def roi_set(self, img):
        h = img.shape[0]
        w = img.shape[1]
        return img[int(h*0.8):h, :]

    def cal_steering(self, yaw, error, gear=3):
        gear = self.gear

        if gear == 3:
            base_speed = 0.35
        elif gear == 2:
            base_speed = 0.55
        elif gear == 1:
            base_speed = 0.25
        else:
            base_speed = 0.35
        base_speed = 0.99
        wheelbase = 0.23

        # Stanley 제어기로 조향각 delta 계산
        steering_angle = (
            self.yaw_k * yaw
            + np.arctan2(
                self.steer_k * error,
                max(abs(base_speed), 0.01),
            )
        )

        # LIMO의 등가 중심 조향각 제한
        raw_steering_angle = float(
            np.clip(
                steering_angle,
                -self.max_steer,
                self.max_steer,
            )
        )

        if self.prev_steer is None:
            steering_delta = 0.0
            steering_angle = raw_steering_angle
        else:
            steering_delta = raw_steering_angle - self.prev_steer
            alpha = float(np.clip(self.steer_smoothing_alpha, 0.0, 1.0))
            steering_angle = self.prev_steer + alpha * steering_delta
            steering_angle = float(
                np.clip(
                    steering_angle,
                    -self.max_steer,
                    self.max_steer,
                )
            )

        steer_change_ratio = min(
            abs(steering_delta) / max(abs(self.max_steer), 0.01),
            1.0,
        )
        speed_scale = 1.0 - self.steer_slowdown_ratio * steer_change_ratio
        base_speed = max(base_speed * speed_scale, self.min_smooth_speed)

        # 조향각을 차체 각속도로 변환
        angular_velocity = (
            base_speed * np.tan(steering_angle) / wheelbase
        )

        self.steer = steering_angle
        self.prev_steer = steering_angle
        self.cmd_speed = float(base_speed)
        self.angular_velocity = float(angular_velocity)

        msg = Twist()
        msg.linear.x = float(base_speed)
        msg.angular.z = self.angular_velocity

        self.cmd_vel_pub.publish(msg)

    def sliding_window(self, img, n_windows=10, margin=12, minpix=5):
        y = img.shape[0]
        histogram = np.sum(img[y // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        leftx_current = int(np.argmax(histogram[:midpoint]))
        rightx_current = int(np.argmax(histogram[midpoint:]) + midpoint)

        window_height = int(y / n_windows)
        nz = img.nonzero()

        left_lane_inds = []
        right_lane_inds = []

        out_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        for window in range(n_windows):
            win_yl = y - (window + 1) * window_height
            win_yh = y - window * window_height

            win_xll = leftx_current - margin
            win_xlh = leftx_current + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            cv.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh),
                         (0, 255, 0), 2)
            cv.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh),
                         (0, 255, 0), 2)

            good_left_inds = (
                (nz[0] >= win_yl) & (nz[0] < win_yh)
                & (nz[1] >= win_xll) & (nz[1] < win_xlh)
            ).nonzero()[0]
            good_right_inds = (
                (nz[0] >= win_yl) & (nz[0] < win_yh)
                & (nz[1] >= win_xrl) & (nz[1] < win_xrh)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nz[1][good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nz[1][good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        left_detected = len(left_lane_inds) >= self.min_lane_pixels
        right_detected = len(right_lane_inds) >= self.min_lane_pixels

        lfit = None
        rfit = None

        if left_detected:
            lfit = np.polyfit(
                nz[0][left_lane_inds],
                nz[1][left_lane_inds],
                1,
            )
        if right_detected:
            rfit = np.polyfit(
                nz[0][right_lane_inds],
                nz[1][right_lane_inds],
                1,
            )

        # NOTE: prev_lfit / prev_rfit는 "양쪽 차선이 실제로 검출된 경우"에만
        # 갱신한다. left_only / right_only / default 분기에서 만들어진 가상
        # 차선을 prev_*에 저장하면, 다음 프레임에서 양쪽 다 안 보일 때
        # (previous 분기) 왜곡된 추정값이 그대로 재사용되어 오차가 누적된다.
        if left_detected and right_detected:
            self.last_lane_status = 'both'
            y_bottom = y - 1
            left_x = lfit[0] * y_bottom + lfit[1]
            right_x = rfit[0] * y_bottom + rfit[1]
            lane_width = abs(right_x - left_x)

            if lane_width < self.min_lane_overlap_px:
                bottom_y = int(y * 0.7)
                left_bottom_count = np.count_nonzero(
                    nz[0][left_lane_inds] >= bottom_y
                )
                right_bottom_count = np.count_nonzero(
                    nz[0][right_lane_inds] >= bottom_y
                )

                if left_bottom_count >= right_bottom_count:
                    self.last_lane_status = 'overlap_left_only'
                    rfit = np.array([lfit[0], lfit[1] + self.lane_width_px])
                else:
                    self.last_lane_status = 'overlap_right_only'
                    lfit = np.array([rfit[0], rfit[1] - self.lane_width_px])
            else:
                self.prev_lfit = lfit.copy()
                self.prev_rfit = rfit.copy()
                self.update_lane_history(lfit, rfit)

        elif left_detected:
            history_side = self.detect_lane_side_by_history(lfit, y)
            original_side = (
                history_side
                if history_side is not None
                else self.detect_original_bottom_lane_side()
            )

            # 오른쪽 차선이 보이지 않으면 왼쪽 차선과 같은 기울기를
            # 유지하면서 lane_width_px만큼 평행 이동해 가상 오른쪽
            # 차선을 만든다.
            if original_side == 'right':
                if history_side == 'right':
                    self.last_lane_status = 'history_right_only'
                else:
                    self.last_lane_status = 'origin_right_only'
                rfit = lfit.copy()
                lfit = np.array([rfit[0], rfit[1] - self.lane_width_px])
            else:
                if history_side == 'left':
                    self.last_lane_status = 'history_left_only'
                elif original_side == 'left':
                    self.last_lane_status = 'origin_left_only'
                else:
                    self.last_lane_status = 'left_only'
                rfit = np.array([lfit[0], lfit[1] + self.lane_width_px])

        elif right_detected:
            history_side = self.detect_lane_side_by_history(rfit, y)
            original_side = (
                history_side
                if history_side is not None
                else self.detect_original_bottom_lane_side()
            )

            # 왼쪽 차선이 보이지 않으면 오른쪽 차선과 같은 기울기를
            # 유지하면서 lane_width_px만큼 평행 이동해 가상 왼쪽 차선을
            # 만든다.
            if original_side == 'left':
                if history_side == 'left':
                    self.last_lane_status = 'history_left_only'
                else:
                    self.last_lane_status = 'origin_left_only'
                lfit = rfit.copy()
                rfit = np.array([lfit[0], lfit[1] + self.lane_width_px])
            else:
                if history_side == 'right':
                    self.last_lane_status = 'history_right_only'
                elif original_side == 'right':
                    self.last_lane_status = 'origin_right_only'
                else:
                    self.last_lane_status = 'right_only'
                lfit = np.array([rfit[0], rfit[1] - self.lane_width_px])

        elif self.prev_lfit is not None and self.prev_rfit is not None:
            self.last_lane_status = 'previous'
            lfit = self.prev_lfit.copy()
            rfit = self.prev_rfit.copy()
        else:
            self.last_lane_status = 'default'
            lane_center = img.shape[1] / 2.0
            half_lane = self.lane_width_px / 2.0
            lfit = np.array([0.0, lane_center - half_lane])
            rfit = np.array([0.0, lane_center + half_lane])

        out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]

        y_top = 0
        y_bottom = y - 1
        left_top = int(np.clip(lfit[0] * y_top + lfit[1], 0, img.shape[1] - 1))
        left_bottom = int(np.clip(
            lfit[0] * y_bottom + lfit[1],
            0,
            img.shape[1] - 1,
        ))
        right_top = int(np.clip(
            rfit[0] * y_top + rfit[1],
            0,
            img.shape[1] - 1,
        ))
        right_bottom = int(np.clip(
            rfit[0] * y_bottom + rfit[1],
            0,
            img.shape[1] - 1,
        ))
        cv.line(
            out_img,
            (left_top, y_top),
            (left_bottom, y_bottom),
            (255, 255, 0),
            3,
        )
        cv.line(
            out_img,
            (right_top, y_top),
            (right_bottom, y_bottom),
            (0, 255, 255),
            3,
        )
        self.debug_publisher1.publish(
            self.cv_bridge.cv2_to_imgmsg(out_img, encoding='bgr8')
        )

        if self.debug_view:
            #cv.imshow('viewer', out_img)
            pass

        return lfit, rfit

    def cal_center_line(self, lfit, rfit):
        cfit = (lfit + rfit) / 2.0

        if self.filtered_img is not None:
            h, w = self.filtered_img.shape[:2]
        else:
            h, w = 160, self.img_width

        y_eval = h * 0.9
        a, b = cfit
        x_center = a * y_eval + b
        yaw = np.arctan(a)

        img_center_x = w / 2.0
        error = -x_center + img_center_x

        return yaw, error

    def draw_lane(self, image, warp_roi, warp_img0, inv_mat, left_fit, right_fit):
        if warp_img0 is not None:
            base_warp = warp_img0
        else:
            base_warp = warp_roi

        full_h, _ = base_warp.shape[:2]
        roi_h, _ = warp_roi.shape[:2]
        roi_offset_y = full_h - roi_h

        ploty = np.linspace(0, roi_h - 1, roi_h)
        left_fitx = left_fit[0] * ploty + left_fit[1]
        right_fitx = right_fit[0] * ploty + right_fit[1]
        ploty_full = ploty + roi_offset_y

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty_full]))])
        pts_right = np.array([
            np.flipud(np.transpose(np.vstack([right_fitx, ploty_full])))
        ])
        pts = np.hstack((pts_left, pts_right))

        color_warp = np.zeros_like(base_warp).astype(np.uint8)
        cv.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
        cv.polylines(color_warp, np.int32(pts_left), False, (255, 255, 0), 5)
        cv.polylines(color_warp, np.int32(pts_right), False, (0, 255, 255), 5)

        newwarp = cv.warpPerspective(
            color_warp,
            inv_mat,
            (image.shape[1], image.shape[0]),
        )
        result = cv.addWeighted(image, 1, newwarp, 0.3, 0)

        steer_deg = math.degrees(self.steer)
        text1 = (
            f'yaw: {self.yaw:.3f} rad / steer: {steer_deg:.1f} deg '
            f'/ ang_z: {self.angular_velocity:.2f}'
        )
        text2 = f'err: {self.error:.1f} px / v: {self.cmd_speed:.2f}'
        text3 = f'lane: {self.last_lane_status}'
        cv.putText(result, text1, (30, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(result, text2, (30, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(result, text3, (30, 145),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv.LINE_AA)

        if self.debug_view:
            #cv.imshow('debug_warp', color_warp)
            pass

        return result

    def process(self):
 

        if self.bgr is None:
            return

        self.warp_img0 = self.warpping(self.bgr)
        self.warp_img = self.roi_set(self.warp_img0)
        g_filtered = self.gaussian_filter(self.warp_img)

        self.roi_img_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(g_filtered, encoding='bgr8')
        )

        self.white_img = self.white_color_filter_hsv(g_filtered)
        self.filtered_img = self.binary_filter(self.white_img)
        self.binary_img_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(self.filtered_img, encoding='mono8')
        )

        lfit, rfit = self.sliding_window(self.filtered_img)
        self.yaw, self.error = self.cal_center_line(lfit, rfit)
        self.cal_steering(yaw=self.yaw, error=self.error)

        debug2_img = self.draw_lane(
            self.bgr,
            self.warp_img,
            self.warp_img0,
            self.inv_warp_mat,
            lfit,
            rfit,
        )
        self.debug_publisher2.publish(
            self.cv_bridge.cv2_to_imgmsg(debug2_img, encoding='bgr8')
        )

        if self.debug_view:
            #cv.imshow('debug_lane', debug2_img)
            #cv.imshow('roi_warp', self.warp_img)
            #cv.imshow('binary_img', self.filtered_img)
            #cv.waitKey(1)
            pass


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollow()

    try:
        node.get_logger().info(
        'mission start!!! / Lane Following is always working...'
         )
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
