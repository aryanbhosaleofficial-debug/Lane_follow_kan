# Classical Lane Perception for an Autonomous Car

A lightweight, classical-computer-vision perception system for a small autonomous vehicle operating on a track with **white lane markings on a dark floor**.

The project starts with OpenCV-based edge and line detection, then progressively adds bird's-eye-view geometry, temporal lane tracking, waypoint generation, and intersection perception. The first major release targets reliable lane perception and local waypoint output. Motion compensation and route-level navigation are planned as later milestones.

> **Project status:** M01 acquisition and M02 calibration/BEV have regression-tested implementations. M03 now adds per-frame straight, curved, dashed and broken boundary detection over M02 BEV frames, with conservative ambiguity and calibration handling. See [M03 usage](lane_detection/advanced/README.md) and [final corrective validation](lane_detection/M03_FINAL_REPORT.md). Physical-track and Raspberry Pi validation remain outstanding. This is not a complete autonomous-driving or vehicle-control system.

## Roadmap

| Version | Milestone | Main outcome | Status |
|---|---|---|---|
| `v0.1.0` | Classical lane-detection baseline | Canny, ROI, Hough segments, left/right line fitting, midpoint and lateral error | Baseline implemented; needs regression tests |
| `v0.2.0` | Inverse Perspective Mapping (IPM) | Transform camera view into a bird's-eye view (BEV) using a homography | M02 implemented; physical calibration validation depends on deployment |
| `v0.3.0` | Temporal lane tracking | Stabilize lane observations across frames and bridge short gaps in broken markings | Planned |
| `v0.4.0` | Waypoint generation | Convert the detected lane centreline into an ordered sequence of local path points | Planned |
| `v0.5.0` | Intersection perception | Detect junction geometry and report possible path branches without choosing a route | Planned |
| `v0.6.0` | ROS 2 integration | Connect the tested perception/waypoint output to the vehicle architecture; access timestamped IMU data | Deferred until local work is ready |
| `v1.0.0` | Lane perception + waypoint milestone | Demonstrate stable lane perception and waypoint output, with a defined ROS 2 interface and documented tests | Future target |
| `v1.1.0+` | Motion compensation | Use vehicle motion estimates, including IMU and odometry where available, to update lane/path estimates | Post-v1.0 |
| `v1.2.0+` | Navigation | Add route/mission input, path selection at intersections, and route-level decision-making | Post-v1.0 |
| Later | Full vehicle validation | Validate perception, navigation and control together on the physical track | Future goal |

The roadmap describes targets, not features that are already implemented. Version 1.0 is deliberately limited to **perception and waypoint generation**. Route planning and navigation decisions are a separate post-v1.0 milestone.

**Current priority:** complete and validate the perception pipeline locally before requesting ROS 2 bags, lab recordings, or vehicle-side integration. For now, use your own webcam, local images, prerecorded videos, and synthetic test frames where useful. ROS 2 integration is deferred until the local stages are ready; this also means IMU-based motion compensation is not a dependency for the first local prototypes.

---

## 1. Problem statement

The vehicle follows a track with white markings on a dark floor. The initial OpenCV detector works reasonably well on straight, clearly visible lane boundaries, but has important limitations:

1. **Broken lane markings:** individual frames may not contain enough visible marking to produce stable detections.
2. **Intersections:** the assumption of exactly two lane boundaries does not hold when the track branches or markings form a junction.
3. **Perspective distortion:** the forward-facing camera makes ground-plane geometry difficult to reason about directly in image pixels.
4. **Temporal instability:** detections can jump between frames due to gaps, noise, lighting, and camera movement.

The project addresses these problems incrementally using geometry and classical computer vision rather than relying on a large neural network.

## 2. Design principles

- **CPU-friendly:** prefer lightweight OpenCV and NumPy operations.
- **Geometry first:** use measurable geometric relationships wherever possible.
- **Modular:** separate image acquisition, preprocessing, lane extraction, perspective mapping, tracking, and navigation logic.
- **Reproducible:** test changes against the same local images and video clips.
- **Confidence-aware:** distinguish a current observation from a prediction based on old observations.
- **Fail safely:** missing or uncertain lane information must not be treated as a valid steering command.
- **Documented and versioned:** record meaningful changes in Git tags and `CHANGELOG.md`.

## 3. Vehicle architecture and software responsibilities

### Confirmed vehicle components

The current vehicle architecture, as understood by this project, contains:
- **Raspberry Pi 5:** high-level computer running the ROS 2 architecture and, eventually, our perception software.
- **Camera:** provides the forward-facing road/track image.
- **IMU:** provides vehicle motion measurements; our software expects to access these through ROS 2 integration.
- **Raspberry Pi Pico and motor controller:** part of the low-level actuation system.
- **Steering servo and drive motors:** physically steer and propel the vehicle.

Pratik is currently working on the PID/control side and getting the car running. Our project focuses on perception and path/waypoint generation. We should coordinate the software interface with the control side, but we should not duplicate or silently replace the PID controller.

The exact ROS 2 topics, message types, coordinate frames, and Raspberry Pi 5-to-Pico communication mechanism still need to be confirmed during integration. This README does not assume those details.

### Responsibility boundaries

| Component | Responsibility | Owner / status |
|---|---|---|
| Lane perception | Detect lane markings and estimate lane geometry | This project |
| Waypoint generation | Turn the estimated lane centreline into ordered local path points | This project |
| Intersection perception | Identify junction geometry and possible outgoing paths | This project, planned |
| Navigation | Decide which path/route the vehicle should follow | Post-v1.0 feature |
| PID / vehicle control | Convert a target path or tracking error into steering/motor commands | Pratik is currently handling this |
| Low-level actuation | Drive the steering servo and motors through the Pico/motor-controller setup | Existing vehicle architecture |

### Target data flow

```text
Camera
  |
  v
Raspberry Pi 5
  |
  +--> Image preprocessing + IPM/BEV
  |         |
  |         v
  |    Lane geometry + temporal tracking
  |         |
  |         v
  |    Centreline / local waypoint generation
  |         |
  |         v
  |    ROS 2 perception output (integration phase)
  |         |
  |         +----> Local waypoints (v1.0 output)
  |         |          |
  |         |          +----> Controller / PID (Pratik's work)
  |         |
  |         +----> Navigation (post-v1.0)
  |                    |
  |                    v
  |              Selected path / target
  |                    |
  +<---- IMU data through ROS 2 (integration phase)
                       |
                       v
              Controller / PID
                       |
                       v
              Pico + motor controller
  |
  +--> Steering servo
  +--> Drive motors
```

This is a responsibility-level diagram, not a final ROS 2 node graph. The precise message contract and topic names will be decided with the vehicle team during integration.

### Waypoints are not the same as navigation

- **Lane perception** answers: where are the visible lane boundaries and centreline?
- **Waypoint generation** answers: which ordered points describe the locally drivable centreline?
- **Navigation** answers: which route or branch should the vehicle take to reach its goal?
- **PID/control** answers: what actuator commands are needed to follow the selected target?

For the first milestone, waypoint generation will produce a local path from the detected lane geometry. At an intersection, perception may report multiple candidate branches; it will not choose left, straight, or right until the navigation milestone is implemented.

### Planned perception output

The exact ROS 2 message is not fixed yet. Conceptually, the perception module should eventually provide:
- Timestamp of the camera observation.
- Ordered waypoint coordinates in a documented coordinate frame.
- Lane/path confidence or validity status.
- Optional lane width and detected boundary information for debugging.
- Intersection/path-candidate information when available.

Before integration, we must agree with the controller side on coordinate frame, units, waypoint ordering, lookahead convention, update rate, and what to do when the output is invalid. Do not assume pixel coordinates can be passed directly to the PID controller.

## 4. Development phases

### Phase 1 — Classical lane-detection baseline (`v0.1.0`)

**Purpose:** establish a simple, working reference implementation.

```text
Frame
  -> Grayscale
  -> Gaussian blur
  -> Canny edge detection
  -> Region of interest (ROI)
  -> Probabilistic Hough line transform
  -> Left/right segment grouping
  -> Line fitting
  -> Lane midpoint and lateral error
```

Each boundary is initially approximated as:

\[
x = ay + b
\]

At a selected lookahead row \(y_p\), evaluate the left and right boundaries and calculate their midpoint:

\[
x_M = \frac{x_L + x_R}{2}
\]

For image width \(W\), image-space lateral error is:

\[
e = x_M - \frac{W}{2}
\]

**Known limitations:**
- Best suited to straight or gently curving visible boundaries.
- Can struggle when markings are broken or temporarily absent.
- Can produce ambiguous results at intersections.
- Pixel coordinates are affected by perspective and camera placement.

### Phase 2 — Inverse Perspective Mapping (`v0.2.0`)

**Purpose:** transform the forward-facing camera image into a top-down or bird's-eye view of the ground plane.

IPM uses a planar projective transformation (homography). Four corresponding points in the camera image and four points in the destination BEV image define the transformation.

```text
Forward camera image             Bird's-eye view

      \       /                  |           |
       \     /                   |           |
        \   /                    |           |
         \ /                     |           |
          V                      |           |
```

OpenCV functions:
- `cv2.getPerspectiveTransform()` calculates a homography from four point pairs.
- `cv2.warpPerspective()` applies the transformation.
- `cv2.undistort()` corrects lens distortion when camera intrinsics and distortion coefficients are available.

**Initial calibration approach:**

1. Capture a representative image using the intended camera and mounting position.
2. Select four points that lie on the track's ground plane.
3. Map those points to a rectangular destination region.
4. Inspect the BEV result and adjust the mapping.
5. Measure physical lane width and use it to validate the transformation.
6. Store calibration points and output dimensions in configuration, not as unexplained constants scattered through code.

The first prototype may use manually selected points. A later calibrated version should account for camera intrinsics, lens distortion, camera height, pitch, and known ground-plane measurements.

**Limitation:** a single homography models a plane. Objects above the ground plane will be distorted in BEV. IPM is for track markings, not general 3D reconstruction.

**Phase 2 acceptance criteria:**
- The selected track region maps into a usable top-down view.
- Both visible lane boundaries remain detectable in BEV.
- Apparent lane width is reasonably consistent over the useful region.
- Calibration parameters are saved and documented.
- The transform works on multiple local test images or video frames.

### Phase 3 — Temporal lane tracking (`v0.3.0`)

**Purpose:** reduce flicker and handle short gaps in broken lane markings.

The tracker will maintain a short history of lane observations, initially targeting approximately two seconds of data.

Store compact, timestamped observations rather than simply retaining and overlaying old images. An observation may include:
- Timestamp.
- Detected lane points or fitted boundary parameters.
- Estimated lane width or centerline.
- Observation confidence.
- Whether the value came from current image evidence or prediction.

Use a time-based buffer rather than assuming a fixed frame rate. For example, two seconds at 20 FPS is about 40 observations, but the implementation must tolerate a changing frame rate.

When current markings disappear briefly:
1. Check recent lane history.
2. Estimate whether the previous lane model is still plausible.
3. Predict only for a limited period.
4. Decrease confidence as observations become stale.
5. Mark the lane as lost when confidence falls below a defined threshold.

Historical observations must not be treated as current measurements without accounting for vehicle movement. Motion compensation is deferred until after the v1.0 milestone, when timestamped IMU (and odometry, if available) data can be accessed through ROS 2.

### Phase 4 — Waypoint generation (`v0.4.0`)

**Purpose:** convert the estimated lane centreline into an ordered sequence of local path points that a downstream controller can follow.

The initial implementation will generate waypoints in a consistent BEV/ground-plane coordinate system rather than raw image pixels. Each waypoint should have a documented unit and coordinate convention. The first version can use a fixed lookahead region and evenly spaced points along the estimated centreline.

Questions to settle during implementation:
- Are coordinates expressed in metres, centimetres, or normalized BEV units?
- Which direction is forward, and which direction is positive lateral displacement?
- How far ahead should waypoints extend?
- How should waypoint spacing adapt to curvature and confidence?
- What should be output when one or both lane boundaries are missing?

**Acceptance criteria:**
- Waypoints follow the centreline on straight and curved local test footage.
- Waypoint order is consistent from near to far (or another explicitly documented convention).
- Units and coordinate frame are documented.
- Low-confidence or invalid estimates are clearly marked rather than emitted as trustworthy points.

This phase produces a local path. It does not perform route planning or decide which branch to take at an intersection.

### Phase 5 — Intersection perception (`v0.5.0`)

**Purpose:** stop assuming every road segment consists of exactly two continuous lane boundaries, and identify possible outgoing path branches.

At an intersection, perception may observe multiple branches, widening or merging markings, stop lines, crossings, or temporary absence of ordinary lane boundaries.

Separate two responsibilities:

- **Perception:** identify visible geometry and possible paths.
- **Navigation:** decide which path the vehicle should take.

The intended state machine:

```text
LANE_FOLLOWING
      |
      v
APPROACHING_INTERSECTION
      |
      v
INTERSECTION
      |
      v
TURNING / PATH_FOLLOWING
      |
      v
EXITING_INTERSECTION
      |
      v
LANE_FOLLOWING

Uncertain or unsafe perception -> LOST / SAFE_STOP
```

For this milestone, perception reports available path candidates and their geometry. It does not choose `LEFT`, `STRAIGHT`, or `RIGHT`; that decision belongs to the post-v1.0 navigation component.

Initial intersection cues may include:
- Sudden changes in detected marking width.
- Multiple line orientations or connected components in BEV.
- Branch-like geometry.
- Loss of the normal two-boundary model.

The initial safe strategy should favour slowing or stopping when the path is ambiguous rather than guessing.

### Phase 6 — ROS 2 integration (`v0.6.0`)

**Purpose:** connect the locally tested perception and waypoint-generation pipeline to the vehicle's existing ROS 2 architecture.

This phase is deliberately deferred until the local perception and waypoint stages are stable. ROS 2 integration is also when our software expects to gain access to the vehicle IMU stream. We will not block the initial local development on IMU data or ROS 2 bags.

Integration tasks:
- Confirm the camera topic, message type, resolution, frame rate, and timestamp behaviour.
- Confirm the IMU topic, message type, units, orientation convention, and timestamp source.
- Confirm whether wheel odometry is available and how it is represented.
- Agree on coordinate frames and transforms.
- Define the perception output/waypoint message and invalid-output behaviour.
- Agree with the PID/controller side on waypoint or tracking-error inputs, units, sign conventions, update rate, and timeout behaviour.
- Verify the communication path between the Raspberry Pi 5, Pico, and motor-controller system without assuming a particular transport in advance.

The perception node publishes observations and path information. The existing controller/PID remains a separate component. No unvalidated vision prototype should send direct steering or throttle commands.

### Phase 7 — Version 1.0 milestone: perception and waypoints (`v1.0.0`)

**Purpose:** deliver a tested lane-perception system that can produce a stable local waypoint sequence and expose it through the agreed vehicle interface.

Version 1.0 is not the complete navigation system. Its scope is:
- Robust lane geometry on representative straight and curved track sections.
- Short-gap handling for broken markings.
- Local waypoint generation with documented coordinates and confidence.
- Basic intersection geometry detection and reporting of candidate paths, where feasible.
- ROS 2 integration and a clearly defined output contract.
- Documented local and controlled on-vehicle tests.

### Post-v1.0 — Motion compensation and navigation (`v1.1.0+`)

Once the v1.0 perception/waypoint interface is stable, extend the system in two separate directions.

**Motion compensation:** use timestamped IMU and, if available, wheel-odometry measurements to account for vehicle movement between camera observations. Yaw-rate integration can provide an approximate heading change:

\[
\Delta\psi = \int_{t_0}^{t_1}\omega_z(t)\,dt
\]

Gyroscope drift, sensor alignment, timestamps, and coordinate frames must be handled. IMU data alone does not identify lane position. Evaluate sensor fusion only when there is a measurable need.

**Navigation:** add a route/mission input and a decision layer that chooses among path candidates at intersections. This requires deciding how the route is represented (for example, a sequence of left/straight/right decisions, a track graph, or another route representation). Navigation will select a path; perception will continue to report what the camera sees; the PID/controller will continue to follow the selected target.

### Later — Full vehicle validation

Validate perception, waypoint generation, navigation, and control together on the physical track. Cover straight sections, curves, broken markings, intersections, different speeds, temporary lane loss, camera vibration, lighting changes, and safe-stop behaviour. A release should be based on repeatable tests and documented limitations, not merely on a successful demonstration.

## 5. Local development and testing strategy

Until the pre-ROS milestones are complete, all development happens locally.

### Input sources

The vision pipeline should support:
- Webcam input.
- Still images.
- Prerecorded video files.
- Network streams when a suitable stream endpoint is available.

### Suggested local dataset layout

```text
datasets/
├── README.md
├── images/
│   ├── straight/
│   ├── curves/
│   ├── broken_markings/
│   └── intersections/
└── videos/
    ├── straight/
    ├── curves/
    ├── slow/
    ├── fast/
    └── intersections/
```

Do not commit large videos or raw recordings to Git unless the repository explicitly intends to store them. Keep dataset acquisition instructions and metadata in `datasets/README.md`; use an external shared location for large files when needed.

### Testing principles

For each change:
1. Keep a small set of representative images and video clips.
2. Run the same inputs before and after the change.
3. Inspect intermediate images as well as final lane geometry.
4. Record failures, not just successful examples.
5. Measure processing speed as well as visual quality.
6. Avoid tuning against only one convenient frame.

Useful measurements:
- Processing FPS and frame latency.
- Percentage of frames with valid left/right boundaries.
- Lane-center error on manually selected test frames.
- Duration of successful tracking through marking gaps.
- False lane detections at intersections.
- Recovery behaviour after lane loss.

## 6. Repository structure

The codebase will be separated by responsibility as features are implemented.

```text
lane_detection/
├── README.md                  # This project plan and architecture
├── CHANGELOG.md               # Version history
├── requirements.txt           # Python dependencies
│
├── config/
│   ├── camera.yaml            # Camera settings/calibration
│   ├── ipm.yaml               # BEV source/destination points
│   └── lane_detection.yaml    # Vision parameters
│
├── vision/
│   ├── camera.py
│   ├── preprocessing.py
│   ├── edge_detection.py
│   ├── lane_extraction.py
│   ├── perspective.py
│   └── visualization.py
│
├── tracking/
│   ├── lane_tracker.py
│   ├── temporal_buffer.py
│   └── motion_compensation.py
│
├── navigation/
│   ├── intersection_detector.py
│   ├── path_selector.py
│   └── state_machine.py
│
├── ros2/                      # Introduced during ROS 2 integration
│   ├── lane_detection_node.py
│   └── lane_controller_adapter.py
│
├── scripts/
│   ├── run_camera.py
│   └── run_video.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_perspective.py
│   ├── test_lane_geometry.py
│   └── test_temporal_tracking.py
│
└── datasets/
    └── README.md
```

This is the intended organization. Do not create empty modules just to match the tree; introduce files when their responsibilities are implemented.

## 7. Versioning and contribution workflow

Use semantic versioning: `MAJOR.MINOR.PATCH`.

- **MAJOR:** incompatible architecture or interface changes.
- **MINOR:** new functionality or a completed development milestone.
- **PATCH:** backwards-compatible fixes and small improvements.

Example release sequence:

```text
v0.1.0  Initial Canny + Hough baseline
v0.2.0  IPM / bird's-eye transformation
v0.3.0  Temporal lane tracking
v0.4.0  Local waypoint generation
v0.5.0  Intersection geometry and path candidates
v0.6.0  ROS 2 perception/waypoint interface + IMU access
v1.0.0  Validated lane perception and waypoint output
v1.1.0+ IMU/odometry motion compensation
v1.2.0+ Route planning and navigation
```

### Branching

Use feature branches for substantial changes:

```bash
git checkout -b feature/ipm
```

Commit focused changes:

```bash
git add .
git commit -m "Add initial inverse perspective mapping"
```

Merge only after testing:

```bash
git checkout main
git merge feature/ipm
git tag v0.2.0
```

### Changelog

Maintain `CHANGELOG.md` with:
- Version and date.
- Added features.
- Changed behaviour.
- Fixed bugs.
- Known limitations.
- Tests performed.

A planned feature must not be listed as completed until it is implemented and tested.

## 8. Safety boundaries

This repository develops perception, local waypoint generation, and later navigation. It does not own the PID controller or low-level actuation, and it does not, by itself, establish that the vehicle is safe to operate autonomously.

Before physical integration:
- Verify the camera-to-vehicle coordinate convention.
- Verify lane-error sign and units.
- Calibrate steering neutral and limits in the vehicle-control system.
- Use a low initial test speed and a physical emergency stop.
- Apply a safe stop or separately validated recovery behaviour when lane confidence is lost.
- Keep command timeouts and actuator safety checks in the control layer.

**The perception system must report uncertainty honestly. A missing lane is not a valid lane estimate.**

## 9. Immediate next steps

The next development cycle is local and focused:

1. Preserve the current working Canny + Hough implementation as the `v0.1.0` baseline.
2. Add `CHANGELOG.md` and keep the project plan in this README.
3. Collect a small local set of representative images and videos.
4. Implement an IPM prototype with manually selected source and destination points.
5. Validate the BEV output on multiple frames and measure apparent lane-width consistency.
6. Add temporal tracking, then generate local waypoints from the tracked centreline.
7. Add intersection geometry/path-candidate detection without implementing route decisions yet.
8. Integrate with ROS 2 when the local pipeline is ready; obtain IMU data at that stage.
9. Keep route planning/navigation as a post-v1.0 milestone.

## Project principle

Build the simplest measurable solution first. Keep each stage independently testable, preserve working versions, and only add complexity when a real test demonstrates the need for it.
