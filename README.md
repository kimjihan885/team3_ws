# team3_ws

ROS 2 workspace for the Songdo mission robot code.

The current focus is `pre_lane_follow`, a camera-based lane following node for
detecting white lane markings, estimating a drivable line, and publishing
`/cmd_vel`.

## Package

```text
src/songdo_mission/
├── songdo_mission/
│   ├── pre_lane_follow.py       # Main lane-following node
│   └── bird_eye_calibrator.py   # Bird-eye calibration helper
├── package.xml
└── setup.py
```

## pre_lane_follow Structure

`pre_lane_follow.py` is organized as a simple vision-to-control pipeline:

1. Subscribe to the compressed camera image.
2. Resize the frame to the configured image size.
3. Warp the image into a bird-eye view.
4. Crop the lower ROI used for lane tracking.
5. Filter white lane markings in HSV.
6. Convert the filtered image to a binary mask.
7. Detect lane pixels with a sliding-window search.
8. Fit left/right lane lines.
9. Handle one-lane or overlapping-lane cases with fallback heuristics.
10. Compute yaw and lateral error from the estimated center line.
11. Convert the steering result into a `Twist` command.
12. Publish debug images for ROI, binary mask, sliding windows, and final overlay.

The node is still being tuned, so the heuristics in `pre_lane_follow.py` are
expected to change as camera calibration and driving behavior improve.

## Main Topics

Input:

- `/camera/color/image_raw/compressed`

Output:

- `/cmd_vel`
- `/roi_img`
- `/binary_img`
- `/debugging_image1`
- `/debugging_image2`

## Build and Run

```bash
cd ~/team3_ws
colcon build --symlink-install
source install/setup.bash
ros2 run songdo_mission pre_lane_follow
```

## Bird-eye Calibration

Use the calibrator when the camera angle or perspective points need adjustment:

```bash
ros2 run songdo_mission bird_eye_calibrator
```

Click four source points in this order:

```text
LT, RT, LB, RB
```

Useful keys:

- `u`: undo
- `r`: reset
- `p`: print points
- `q` or `Esc`: quit

Copy the printed `src_points` and `dst_points` into `pre_lane_follow.py`.

## Git Notes

The repository tracks source code and lightweight project files. Local runtime
artifacts such as ROS build outputs, logs, bags, extracted images, and Python
cache files are ignored.
