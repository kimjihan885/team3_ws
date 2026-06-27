# Songdo Mission ROS 2 Workspace

ROS 2 Python workspace for camera-based lane following and bird-eye calibration.
The main package, `songdo_mission`, detects white lane markings from a compressed
RGB camera stream, estimates a lane/center line in a warped ROI, and publishes
velocity commands for a mobile robot.

## Repository Layout

```text
team3_ws/
├── src/songdo_mission/
│   ├── songdo_mission/
│   │   ├── pre_lane_follow.py       # Lane detection and steering node
│   │   └── bird_eye_calibrator.py   # Interactive perspective calibration tool
│   ├── package.xml
│   └── setup.py
├── raw_image/image_extractor.py     # Utility script for extracting bag images
├── best.pt                          # Local model artifact kept with the project
└── .gitignore
```

Generated ROS workspace outputs and local datasets are intentionally ignored:
`build/`, `install/`, `log/`, `my_bag/`, `raw_image/output_images/`, and Python
cache directories.

## Nodes

### `pre_lane_follow`

Lane-following node with:

- Bird-eye perspective transform
- HSV white lane filtering
- Binary thresholding
- Sliding-window line fitting
- One-lane fallback using fixed lane-width heuristics
- Stanley-style steering-angle calculation converted to `cmd_vel.angular.z`
- Debug overlays for ROI windows, fitted lines, lane status, error, steering, and angular velocity

Run:

```bash
ros2 run songdo_mission pre_lane_follow
```

Subscribed topics:

| Topic | Type | Purpose |
| --- | --- | --- |
| `/camera/color/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Input camera image |
| `/mission_num` | `std_msgs/msg/Float64` | Mission state input |

Published topics:

| Topic | Type | Purpose |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Robot velocity command |
| `/mission_num` | `std_msgs/msg/Float64` | Mission state output |
| `/roi_img` | `sensor_msgs/msg/Image` | Warped ROI image |
| `/binary_img` | `sensor_msgs/msg/Image` | Binary lane mask |
| `/debugging_image1` | `sensor_msgs/msg/Image` | Sliding-window debug view |
| `/debugging_image2` | `sensor_msgs/msg/Image` | Final lane overlay debug view |

Key parameters:

| Parameter | Default | Description |
| --- | ---: | --- |
| `img_width` | `640` | Camera image width |
| `img_height` | `480` | Camera image height |
| `white_lower` | `[0, 0, 200]` | HSV lower bound for white filtering |
| `white_upper` | `[180, 40, 255]` | HSV upper bound for white filtering |
| `process_hz` | `30.0` | Processing loop frequency |
| `steer_k` | `0.005` | Stanley cross-track gain |
| `yaw_k` | `1.0` | Heading gain |
| `max_steer` | `0.9` | Maximum angular command parameter |
| `lane_width_px` | `250.0` | Fixed lane-width heuristic in warped pixels |
| `min_lane_overlap_px` | `50.0` | Overlap threshold for duplicate lane fits |
| `min_lane_pixels` | `30` | Minimum pixels required for a lane detection |

### `bird_eye_calibrator`

Interactive calibration helper for selecting four source points and printing
`pre_lane_follow.py`-ready `src_points` and `dst_points`.

Run:

```bash
ros2 run songdo_mission bird_eye_calibrator
```

Controls:

| Input | Action |
| --- | --- |
| Left mouse click | Add points in `LT`, `RT`, `LB`, `RB` order |
| `u` | Undo last point |
| `r` | Reset points |
| `p` | Print current calibration points |
| `q` or `Esc` | Quit |

## Build

From the workspace root:

```bash
cd ~/team3_ws
colcon build --symlink-install
source install/setup.bash
```

Run tests:

```bash
colcon test
colcon test-result --verbose
```

## Typical Workflow

1. Calibrate the bird-eye transform:

   ```bash
   ros2 run songdo_mission bird_eye_calibrator
   ```

2. Copy the printed `src_points` / `dst_points` into `pre_lane_follow.py`.

3. Rebuild and source:

   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```

4. Start lane following:

   ```bash
   ros2 run songdo_mission pre_lane_follow
   ```

5. Inspect debug topics in `rqt_image_view`:

   ```bash
   rqt_image_view
   ```

Recommended views:

- `/debugging_image1` for sliding-window and fitted-line diagnostics
- `/debugging_image2` for final overlay and steering text
- `/binary_img` for white-threshold tuning

## Notes

- The node assumes a white lane/line target and uses HSV thresholding before
  fitting lines.
- `lane_width_px` is a fixed heuristic. It is not updated at runtime.
- One-lane cases are resolved with a combination of bird-eye fitting and original
  image bottom-region white-pixel checks.
- Large local bags and extracted image sequences are not tracked in git.
