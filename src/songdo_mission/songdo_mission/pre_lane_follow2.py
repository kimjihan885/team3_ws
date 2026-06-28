

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


class LaneFollow(Node):
    def __init__(self, node_name='pre_lane_follow2', start_timer=True):
        super().__init__(node_name)

        self.cv_bridge = CvBridge()

        self.declare_parameter('img_width', 640)
        self.declare_parameter('img_height', 480)
        self.declare_parameter('white_lower', [0, 0, 180])
        self.declare_parameter('white_upper', [180, 20, 255])
        self.declare_parameter('debug_view', True)
        self.declare_parameter('process_hz', 30.0)
        self.declare_parameter('steer_k', 0.002)
        self.declare_parameter('yaw_k', 1.0)
        self.declare_parameter('max_steer', 0.6)
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
        self.declare_parameter('single_lane_track_alpha', 0.80)
        self.declare_parameter('side_match_gate_ratio', 0.60)
        self.declare_parameter('side_match_margin_ratio', 0.12)
        self.declare_parameter('lane_width_update_alpha', 0.10)
        self.declare_parameter('side_switch_frames', 3)
        self.declare_parameter('side_hold_gate_ratio', 0.85)

        # 참고 코드 기준 ROI 시작점
        self.declare_parameter('cutting_idx', 300)

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
        self.process_hz = float(self.get_parameter('process_hz').value)
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
        self.single_lane_track_alpha = float(
            self.get_parameter('single_lane_track_alpha').value
        )
        self.side_match_gate_ratio = float(
            self.get_parameter('side_match_gate_ratio').value
        )
        self.side_match_margin_ratio = float(
            self.get_parameter('side_match_margin_ratio').value
        )
        self.lane_width_update_alpha = float(
            self.get_parameter('lane_width_update_alpha').value
        )
        self.side_switch_frames = max(
            1,
            int(self.get_parameter('side_switch_frames').value),
        )
        self.side_hold_gate_ratio = float(
            self.get_parameter('side_hold_gate_ratio').value
        )

        self.cutting_idx = int(self.get_parameter('cutting_idx').value)

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

        self.roi_img_pub = self.create_publisher(
            Image,
            '/roi_img',
            10,
        )

        self.binary_img_pub = self.create_publisher(
            Image,
            '/binary_img',
            10,
        )

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

        # ==========================================================
        # BEV / Bird-Eye-View 좌표 수정 부분
        # 참고로 방금 준 코드의 SRC_POINTS / DST_POINTS 그대로 반영함
        # ==========================================================
        self.src_points = np.float32([
            [215.0, 250.0],   # 좌상
            [435.0, 250.0],   # 우상
            [0.0,   445.0],   # 좌하
            [636.0, 445.0],   # 우하
        ])

        self.dst_points = np.float32([
            [240.0, 230.0],   # 좌상
            [400.0, 230.0],   # 우상
            [240.0, 470.0],   # 좌하
            [400.0, 470.0],   # 우하
        ])

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
        self.last_observed_side = None
        self.pending_single_side = None
        self.pending_single_count = 0
        self.single_left_score = float('inf')
        self.single_right_score = float('inf')

        self.timer = None
        if start_timer:
            self.start_process_timer()

        self.get_logger().info(f'ROS2 {self.get_name()} node initialized')

    def start_process_timer(self):
        if self.timer is not None:
            return

        period = (
            1.0 / self.process_hz
            if self.process_hz > 0.0
            else 1.0 / 30.0
        )
        self.timer = self.create_timer(period, self.process)

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

    def update_lane_history(self, lfit, rfit):
        self.left_fit_history.append(np.array(lfit, dtype=float))
        self.right_fit_history.append(np.array(rfit, dtype=float))

    @staticmethod
    def fit_x(fit, y_values):
        y_values = np.asarray(y_values, dtype=float)
        return fit[0] * y_values + fit[1]

    def fit_distance(self, fit_a, fit_b, img_h):
        y_values = (img_h - 1) * np.array([0.15, 0.35, 0.55, 0.75, 0.95])
        distance = np.abs(
            self.fit_x(fit_a, y_values)
            - self.fit_x(fit_b, y_values)
        )
        return float(np.median(distance))

    def classify_single_lane(self, fit, img_h):
        if self.prev_lfit is None or self.prev_rfit is None:
            self.single_left_score = float('inf')
            self.single_right_score = float('inf')
            return None, self.single_left_score, self.single_right_score

        left_score = self.fit_distance(fit, self.prev_lfit, img_h)
        right_score = self.fit_distance(fit, self.prev_rfit, img_h)

        slope_weight = 0.20 * img_h
        left_score += slope_weight * abs(fit[0] - self.prev_lfit[0])
        right_score += slope_weight * abs(fit[0] - self.prev_rfit[0])

        self.single_left_score = float(left_score)
        self.single_right_score = float(right_score)

        lane_width = max(abs(self.lane_width_px), 1.0)
        match_gate = max(35.0, self.side_match_gate_ratio * lane_width)
        hold_gate = max(match_gate, self.side_hold_gate_ratio * lane_width)
        decision_margin = max(12.0, self.side_match_margin_ratio * lane_width)

        best_score = min(left_score, right_score)
        score_gap = abs(left_score - right_score)

        if best_score > match_gate or score_gap < decision_margin:
            raw_side = None
        elif left_score < right_score:
            raw_side = 'left'
        else:
            raw_side = 'right'

        if self.last_observed_side is None:
            if raw_side is not None:
                self.last_observed_side = raw_side
                self.pending_single_side = None
                self.pending_single_count = 0
            return raw_side, left_score, right_score

        current_side = self.last_observed_side
        current_score = left_score if current_side == 'left' else right_score

        if raw_side is None:
            self.pending_single_side = None
            self.pending_single_count = 0
            if current_score <= hold_gate:
                return current_side, left_score, right_score
            return None, left_score, right_score

        if raw_side == current_side:
            self.pending_single_side = None
            self.pending_single_count = 0
            return current_side, left_score, right_score

        if self.pending_single_side == raw_side:
            self.pending_single_count += 1
        else:
            self.pending_single_side = raw_side
            self.pending_single_count = 1

        if self.pending_single_count >= self.side_switch_frames:
            self.last_observed_side = raw_side
            self.pending_single_side = None
            self.pending_single_count = 0
            return raw_side, left_score, right_score

        if current_score <= hold_gate:
            return current_side, left_score, right_score

        return None, left_score, right_score

    def update_single_lane_track(self, observed_fit, side):
        alpha = float(np.clip(self.single_lane_track_alpha, 0.0, 1.0))
        observed_fit = np.asarray(observed_fit, dtype=float)

        if side == 'left':
            if self.prev_lfit is None:
                tracked = observed_fit.copy()
            else:
                tracked = (
                    alpha * observed_fit
                    + (1.0 - alpha) * self.prev_lfit
                )
            lfit = tracked
            rfit = np.array([tracked[0], tracked[1] + self.lane_width_px])
        else:
            if self.prev_rfit is None:
                tracked = observed_fit.copy()
            else:
                tracked = (
                    alpha * observed_fit
                    + (1.0 - alpha) * self.prev_rfit
                )
            rfit = tracked
            lfit = np.array([tracked[0], tracked[1] - self.lane_width_px])

        self.prev_lfit = lfit.copy()
        self.prev_rfit = rfit.copy()
        self.last_observed_side = side

        return lfit, rfit

    def update_both_lane_track(self, lfit, rfit, img_h, img_w):
        y_values = (img_h - 1) * np.array([0.25, 0.50, 0.75, 0.95])
        left_x = self.fit_x(lfit, y_values)
        right_x = self.fit_x(rfit, y_values)
        widths = right_x - left_x
        measured_width = float(np.median(widths))

        if self.min_lane_overlap_px <= measured_width <= img_w * 0.85:
            alpha = float(np.clip(self.lane_width_update_alpha, 0.0, 1.0))
            self.lane_width_px = (
                (1.0 - alpha) * self.lane_width_px
                + alpha * measured_width
            )

        self.prev_lfit = np.asarray(lfit, dtype=float).copy()
        self.prev_rfit = np.asarray(rfit, dtype=float).copy()
        self.update_lane_history(lfit, rfit)

        self.last_observed_side = None
        self.pending_single_side = None
        self.pending_single_count = 0

    def roi_set(self, img):
        h = img.shape[0]
        w = img.shape[1]
        return img[int(h*0.8):h, :]

    def cal_steering(self, yaw, error, gear=3):
        gear = self.gear

        base_speed = 1.0
        wheelbase = 0.23

        steering_angle = (
            self.yaw_k * yaw
            + np.arctan2(
                self.steer_k * error,
                max(abs(base_speed), 0.01),
            )
        )

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

            cv.rectangle(
                out_img,
                (win_xll, win_yl),
                (win_xlh, win_yh),
                (0, 255, 0),
                2,
            )
            cv.rectangle(
                out_img,
                (win_xrl, win_yl),
                (win_xrh, win_yh),
                (0, 255, 0),
                2,
            )

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

        candidates = []

        if left_detected:
            candidates.append((lfit, len(left_lane_inds), 'window_left'))

        if right_detected:
            candidates.append((rfit, len(right_lane_inds), 'window_right'))

        if len(candidates) == 2:
            duplicate_distance = self.fit_distance(
                candidates[0][0],
                candidates[1][0],
                y,
            )

            if duplicate_distance < self.min_lane_overlap_px:
                candidates = [max(candidates, key=lambda item: item[1])]

        if len(candidates) == 2:
            fit_a = np.asarray(candidates[0][0], dtype=float)
            fit_b = np.asarray(candidates[1][0], dtype=float)
            compare_y = (y - 1) * 0.80

            if self.fit_x(fit_a, [compare_y])[0] <= self.fit_x(fit_b, [compare_y])[0]:
                lfit, rfit = fit_a, fit_b
            else:
                lfit, rfit = fit_b, fit_a

            lane_width = self.fit_distance(lfit, rfit, y)

            if lane_width >= self.min_lane_overlap_px:
                self.last_lane_status = 'both'
                self.update_both_lane_track(lfit, rfit, y, img.shape[1])
            else:
                candidate = max(candidates, key=lambda item: item[1])[0]

                side, left_score, right_score = self.classify_single_lane(
                    candidate,
                    y,
                )

                if side is None:
                    self.last_lane_status = 'single_ambiguous'

                    if self.prev_lfit is not None and self.prev_rfit is not None:
                        lfit = self.prev_lfit.copy()
                        rfit = self.prev_rfit.copy()
                    else:
                        lane_center = img.shape[1] / 2.0
                        half_lane = self.lane_width_px / 2.0
                        lfit = np.array([0.0, lane_center - half_lane])
                        rfit = np.array([0.0, lane_center + half_lane])
                else:
                    lfit, rfit = self.update_single_lane_track(candidate, side)
                    self.last_lane_status = f'tracked_{side}_only'

        elif len(candidates) == 1:
            candidate = np.asarray(candidates[0][0], dtype=float)

            side, left_score, right_score = self.classify_single_lane(
                candidate,
                y,
            )

            if side is None:
                self.last_lane_status = 'single_ambiguous'

                if self.prev_lfit is not None and self.prev_rfit is not None:
                    lfit = self.prev_lfit.copy()
                    rfit = self.prev_rfit.copy()
                else:
                    lane_center = img.shape[1] / 2.0
                    half_lane = self.lane_width_px / 2.0
                    lfit = np.array([0.0, lane_center - half_lane])
                    rfit = np.array([0.0, lane_center + half_lane])
            else:
                lfit, rfit = self.update_single_lane_track(candidate, side)
                self.last_lane_status = f'tracked_{side}_only'

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
            pass

        return lfit, rfit

    def cal_center_line(self, lfit, rfit):
        cfit = (lfit + rfit) / 2.0

        if self.filtered_img is not None:
            h, w = self.filtered_img.shape[:2]
        else:
            h, w = 180, self.img_width

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

        pts_left = np.array([
            np.transpose(np.vstack([left_fitx, ploty_full]))
        ])

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
        text4 = f'ROI start: {self.cutting_idx}'

        if np.isfinite(self.single_left_score) and np.isfinite(self.single_right_score):
            text5 = (
                f'match L:{self.single_left_score:.1f} '
                f'R:{self.single_right_score:.1f}'
            )
        else:
            text5 = 'match L:- R:-'

        cv.putText(
            result,
            text1,
            (30, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

        cv.putText(
            result,
            text2,
            (30, 110),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )

        cv.putText(
            result,
            text3,
            (30, 145),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv.LINE_AA,
        )

        cv.putText(
            result,
            text4,
            (30, 180),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        cv.putText(
            result,
            text5,
            (30, 215),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        if self.debug_view:
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
