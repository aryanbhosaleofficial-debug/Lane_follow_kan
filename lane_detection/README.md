# Classical Lane Detection — White Lines on a Black Floor

This page describes the preserved **legacy raw-image compatibility path**. M03's
BEV-based advanced detector, structured observations, CLI and tests are documented
in [advanced/README.md](advanced/README.md). The two paths use different input and
coordinate contracts; legacy IPM JSON is not an M02 metric profile.
Current M03 repairs and reproducible evidence are in
[M03_FINAL_REPORT.md](M03_FINAL_REPORT.md).

A lightweight, CPU-based lane detector using OpenCV Canny edge detection, a region of interest (ROI), the probabilistic Hough transform, and least-squares line fitting.

The intended platform is a small forward-facing camera on a 1:10 autonomous vehicle.

## 1. What the program does

The program processes each camera frame through this pipeline:

1. Capture a frame from the camera.
2. Convert the frame to grayscale.
3. Apply Gaussian blur to reduce image noise.
4. Run Canny edge detection.
5. Mask the image to a trapezoidal road region of interest.
6. Detect line segments with `cv2.HoughLinesP`.
7. Group segments into left and right lane boundaries.
8. Fit one mathematical line to each boundary.
9. Evaluate both lines at a lookahead row and calculate the lane midpoint.
10. Calculate lateral error relative to the image center.
11. Display the annotated camera feed and processing FPS.

**Important:** This is a vision prototype. It displays lane geometry and an error signal; it does not send steering or throttle commands to the car.

## 2. Files

Keep these two files in the same project folder:

```text
lane_detection/
├── lane_detection.py
└── README.md
```

## 3. Requirements

- Python 3.9 or newer recommended
- A USB camera or the vehicle's camera exposed as a camera device
- Windows, Linux, or Raspberry Pi OS
- OpenCV and NumPy

Install the dependencies:

```bash
python -m pip install opencv-python numpy
```

If `python` does not work on Windows, try:

```powershell
py -m pip install opencv-python numpy
```

No PyTorch, TensorFlow, CUDA, or neural-network weights are required.

## 4. Run the program

Open a terminal in the folder containing `lane_detection.py`.

Default run (camera index 0):

```bash
python lane_detection.py
```

Windows alternative:

```powershell
py lane_detection.py
```

If the camera is index 1:

```bash
python lane_detection.py --camera 1
```

The program opens:
- **Lane Detection:** camera frame with ROI, fitted boundaries, midpoint, error, and FPS.
- **Canny ROI:** edge image restricted to the road region (unless `--no-debug` is used).

Press **Q** or **ESC** while a video window is focused to quit.

## 5. Command-line options

| Option | Default | Meaning |
|---|---:|---|
| `--camera` | `0` | Camera device index |
| `--width` | `640` | Requested camera width |
| `--height` | `480` | Requested camera height |
| `--canny-low` | `50` | Canny lower threshold |
| `--canny-high` | `150` | Canny upper threshold |
| `--hough-threshold` | `25` | Hough accumulator threshold |
| `--min-line-length` | `25` | Minimum Hough segment length in pixels |
| `--max-line-gap` | `60` | Maximum gap between segments that Hough may join |
| `--roi-top` | `0.55` | Top edge of ROI as a fraction of frame height |
| `--lookahead` | `0.75` | Image row used to measure lane center, as a fraction of height |
| `--no-debug` | off | Hide the Canny ROI window |

Example:

```bash
python lane_detection.py --camera 0 --width 640 --height 480 --canny-low 40 --canny-high 130 --roi-top 0.60 --lookahead 0.78
```

## 6. Understanding the mathematics

### 6.1 Line representation

Each lane boundary is approximated by:

\[
x = ay + b
\]

where:
- \(x\) is the horizontal image coordinate,
- \(y\) is the vertical image coordinate,
- \(a\) is the fitted slope in this coordinate convention,
- \(b\) is the horizontal intercept.

The code uses `numpy.polyfit` to fit this line to the endpoints of the Hough segments.

### 6.2 Lane midpoint

At the selected lookahead row \(y_p\):

\[
x_L = a_L y_p + b_L
\]

\[
x_R = a_R y_p + b_R
\]

The target midpoint is:

\[
x_M = \frac{x_L+x_R}{2}
\]

### 6.3 Lateral error

For an image of width \(W\), the image center is \(W/2\).

\[
e = x_M - \frac{W}{2}
\]

Interpretation:
- `e > 0`: the detected lane center is to the right of the image center.
- `e < 0`: the detected lane center is to the left of the image center.
- `e ≈ 0`: the detected lane center is approximately centered.

The program also displays normalized error:

\[
e_n = \frac{e}{W/2}
\]

This makes the error less dependent on image resolution. It is a dimensionless image-space error, not a physical distance in metres.

### 6.4 Temporal smoothing

The program smooths the displayed error using an exponential moving average:

\[
e_t^{smooth} = \alpha e_t + (1-\alpha)e_{t-1}^{smooth}
\]

The current value of \(\alpha\) is `0.35`. Increase it for a faster response; decrease it for smoother output. Smoothing is reset when valid two-sided lane geometry is lost.

## 7. How to tune it for the physical track

Tune with the car stationary first. Put the camera in its final mounting position and point it at the actual track.

### A. ROI

The cyan trapezoid shows the region being analyzed.

- If the top of the trapezoid includes too much background, increase `--roi-top` (for example, `0.60`).
- If the ROI cuts off the lane markings, decrease it (for example, `0.48`).
- If the lane is outside the left/right edges, adjust the polygon coordinates in `make_roi_mask()`.

### B. Canny thresholds

The white lane markings should create clear, continuous edges in the Canny ROI window.

- If there are too many noisy edges, try increasing the thresholds.
- If the lane markings disappear, try lowering them.
- Start with `50` and `150`; tune using the actual lighting and camera exposure.

Example:

```bash
python lane_detection.py --canny-low 40 --canny-high 120
```

### C. Hough parameters

- Increase `--hough-threshold` to reject weaker detections.
- Decrease it if lane segments are not being detected.
- Increase `--min-line-length` to reject short fragments.
- Increase `--max-line-gap` if a painted/taped line appears broken in the camera image.

Change one parameter at a time and observe the result.

### D. Lookahead

`--lookahead 0.75` evaluates the lane center at 75% of the image height.

- A larger value looks closer to the vehicle.
- A smaller value looks farther ahead.

The best value depends on camera mounting angle, field of view, vehicle speed, and track geometry.

## 8. Output and interpretation

When both boundaries are detected, the display shows:

- Blue/orange line: fitted left boundary.
- Red line: fitted right boundary.
- Green dot: estimated lane midpoint at the lookahead row.
- Yellow line: image center to estimated lane midpoint.
- `Error`: smoothed horizontal error in pixels.
- `Normalized`: smoothed error divided by half the image width.
- `FPS`: approximate processing frames per second.

When either boundary is missing, the program displays a warning and does **not** report a valid centerline. It deliberately does not invent the missing boundary from an old frame.

## 9. Known limitations

This first version makes simplifying assumptions:

1. The camera looks forward and is approximately centered.
2. The road is represented by two visible lane boundaries.
3. Each boundary can be approximated by a straight line over the selected ROI.
4. Left/right grouping by image half is suitable for the camera's orientation.
5. The floor and lighting provide enough contrast for Canny to see the white markings.

Potential failure cases include:
- Sharp curves where a straight-line model is inadequate.
- Strong shadows, reflections, glare, or overexposure.
- Intersections, branches, or extra white markings.
- One boundary being blocked or outside the camera view.
- Camera vibration or a changing camera angle.

If the track has significant curves, the next step is to fit a second-degree polynomial or use a sliding-window approach. If the camera sees multiple unrelated white lines, add geometric constraints and temporal tracking.

## 10. Safety and vehicle integration

Do not connect the displayed error directly to motor control without testing and calibration.

Before vehicle integration:

1. Verify left/right lane detection across the full test track.
2. Verify the sign of the error with the vehicle stationary: move the vehicle/camera left and right relative to the lane and confirm the error changes as expected.
3. Calibrate the steering servo's neutral point, direction, and safe limits.
4. Use a low test speed and a physical emergency stop.
5. If either boundary is lost, the controller should command a safe stop or enter a separately designed recovery state.
6. Add speed limits, steering limits, and command timeouts in the vehicle-control layer.

The image-space error is not a steering angle. A steering controller must be calibrated for the vehicle geometry, camera placement, steering mechanism, and speed.

## 11. Troubleshooting

**Camera will not open**
- Close apps that may already be using it.
- Try `--camera 1` or another camera index.
- Check Windows camera permissions.
- Confirm that the camera appears in Device Manager.

**No lane lines appear**
- Look at the Canny ROI window.
- Confirm the white markings are visible in the raw camera feed.
- Adjust ROI and Canny thresholds.
- Reduce the Hough threshold or minimum line length.

**Only one boundary appears**
- Check that both lane lines are inside the ROI.
- Check whether the camera is centered and aimed correctly.
- Adjust ROI polygon coordinates.
- Check for glare, shadows, or obstruction.

**Lines jump around**
- Improve lighting and camera mounting stability.
- Tune Canny/Hough parameters.
- Adjust the smoothing alpha in the code.
- Consider adding temporal tracking after the baseline detector works.

---

**Recommended first test:** Run the script with the vehicle stationary, positioned in the middle of the track. Confirm that both fitted boundaries and the green midpoint follow the actual lane before attempting any autonomous movement.
