# Stereo Pipeline Specification: Home Dining Monitoring System

## Overview

A stereo-camera-based system for monitoring dining behavior in home experiment settings. The system captures 3D body pose data of subjects during meals, identifies individuals, and annotates dining zone occupancy.

**Core principle**: Reliable video capture is the foundation. All analysis is post-processing.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    REAL-TIME (Phase 1)                       │
│                                                             │
│  Fixed Stereo Camera (always on after calibration)          │
│       │                                                     │
│       ▼                                                     │
│  Smart Recorder                                             │
│  - Continuous frame reading                                 │
│  - Lightweight person detection (every Nth frame)           │
│  - State machine: IDLE → RECORDING → COOLDOWN → IDLE       │
│  - Output: complete video segments per visit                │
│                                                             │
│  Output: visit_YYYYMMDD_HHMMSS/port_1.mp4, port_2.mp4      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  POST-PROCESSING (Phase 2)                  │
│                                                             │
│  Step A: Person Identification                              │
│  Step B: 3D Reconstruction (caliscope)                      │
│  Step C: Dining Zone Occupancy Analysis                     │
│  Step D: Multi-Camera Alignment (food camera)               │
│  Step E: Activity Summary & Export                          │
│                                                             │
│  Each step independent, re-runnable, non-destructive        │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Smart Recorder (Real-Time)

### Purpose
Automatically record complete video segments when a person is present in the stereo camera's field of view. Must be simple, robust, and never miss a visit.

### Person Detection
- **Method**: Lightweight detector on ONE stereo view (port_1), downscaled
- **Options** (in order of recommendation):
  1. **YOLOv8-nano** — fast, accurate person detection, ~30fps on CPU
  2. **OpenCV HOG person detector** — no extra dependencies, slightly less accurate
  3. **MediaPipe pose detection** — already in caliscope env, heavier
- **Frequency**: Run detection every 5 frames (at 60fps camera = 12 checks/sec)
- **Resolution**: Downscale to 640x480 for detection; record at full 1600x1200

### State Machine

```
       ┌──────────┐
       │   IDLE   │ ← camera reading frames, detection running
       └────┬─────┘
            │ person detected
            ▼
       ┌──────────┐
       │RECORDING │ ← FFmpeg capturing full-quality stereo video
       └────┬─────┘
            │ person not detected for 5 consecutive checks
            ▼
       ┌──────────┐
       │ COOLDOWN │ ← still recording, waiting for person to return
       └────┬─────┘
            │
       ┌────┴────┐
       │         │
  person      30 seconds
  returns     elapsed
       │         │
       ▼         ▼
  RECORDING    IDLE
              (stop & save)
```

### Cooldown Design
- **Duration**: 30 seconds (configurable)
- **Purpose**: Person may briefly leave frame (go to kitchen, bathroom, bend down). Don't create fragmented recordings.
- **Behavior**: Recording continues during cooldown. If person returns, seamlessly continue. If timeout expires, stop recording and finalize the segment.

### Output Structure
Each detected visit creates a session under the project's recordings directory:
```
{project}/recordings/
├── visit_20260202_120315/
│   ├── port_1.mp4           # Left camera (1600x1200, 60fps)
│   ├── port_2.mp4           # Right camera (1600x1200, 60fps)
│   ├── frame_timestamps.csv # Sync timestamps
│   └── metadata.json        # Start/end time, duration, detection log
├── visit_20260202_183042/
│   └── ...
```

### Naming Convention
- `visit_YYYYMMDD_HHMMSS` — timestamped by when the person was first detected
- Distinct from calibration recordings (`session_*`) and calibration data

### Reliability Requirements
- Must not crash or stop monitoring if detection fails on a single frame
- Must handle camera disconnection gracefully (reconnect loop)
- Must log all state transitions for debugging
- Video files must be finalized cleanly (proper MP4 container close)

---

## Phase 2: Offline Analysis Pipeline

### Step A: Person Identification

**Input**: `visit_*/port_1.mp4` (or port_2.mp4)
**Output**: `visit_*/identity.json`

- Extract faces from first N seconds of recording (when person enters, face is often visible)
- Match against enrolled subject database using face embeddings
- **Enrollment**: Pre-capture 3-5 reference photos per subject → store face embeddings
- **Library**: `face_recognition` (dlib-based) or `insightface` (ArcFace)
- **Fallback**: If face not detected, mark as "unknown" for manual annotation later

```json
{
  "subject_id": "subject_03",
  "subject_name": "Zhang Wei",
  "confidence": 0.92,
  "face_detected_frame": 45,
  "method": "arcface"
}
```

### Step B: 3D Reconstruction

**Input**: `visit_*/port_1.mp4, port_2.mp4` + `camera_array.toml`
**Output**: `visit_*/{TRACKER}/xyz_{TRACKER}.csv`

- Already implemented via caliscope integration
- Run `run_auto_reconstruction()` with selected tracker
- Tracker options: HOLISTIC (full body + hands + face), POSE (body only), HAND (hands only)
- Recommended: HOLISTIC for dining analysis (captures hand movements for eating detection)

### Step C: Dining Zone Occupancy Analysis

**Input**: `visit_*/HOLISTIC/xyz_HOLISTIC.csv` + `dining_zone.toml`
**Output**: `visit_*/zone_timeline.csv`

#### Zone Definition (One-Time Setup)
After calibration, the user defines the dining area:
1. In PipelineUI or a separate tool, user clicks 4+ points on the dining table in the camera view
2. Points are triangulated to 3D world coordinates using calibrated cameras
3. Saved as a 3D bounding region:

```toml
# dining_zone.toml
[zone]
name = "dining_table"
# 3D bounding box in world coordinates (meters)
x_min = -0.5
x_max = 0.5
y_min = -0.3
y_max = 0.3
z_min = 0.7
z_max = 1.0
```

#### Occupancy Detection
For each frame in the 3D reconstruction:
1. Check if key body landmarks (wrists, elbows, hips) are within the dining zone
2. Classify frame as: "at_table", "near_table", "away"
3. Generate temporal annotations:

```csv
sync_index,timestamp,zone_status,landmarks_in_zone
0,0.000,away,0
150,5.000,near_table,2
300,10.000,at_table,6
...
```

### Step D: Multi-Camera Alignment

**Input**: Stereo camera timestamps + food camera timestamps
**Output**: Aligned time reference

- Both cameras record with system clock timestamps
- Alignment method: match `frame_timestamps.csv` from both cameras by closest system time
- Optional: use a shared sync signal (audio clap, LED flash) for precise alignment
- Output: mapping from stereo frame indices to food camera frame indices

### Step E: Activity Summary & Export

**Input**: All analysis results from Steps A-D
**Output**: `visit_*/session_report.json`

Combine all annotations into a unified session report:

```json
{
  "visit_id": "visit_20260202_120315",
  "subject": "subject_03",
  "start_time": "2026-02-02T12:03:15",
  "end_time": "2026-02-02T12:25:42",
  "duration_seconds": 1347,
  "dining_zone_occupancy": {
    "total_at_table_seconds": 1180,
    "entry_time": "12:03:45",
    "exit_time": "12:25:10"
  },
  "reconstruction": {
    "tracker": "HOLISTIC",
    "total_frames": 80820,
    "xyz_file": "HOLISTIC/xyz_HOLISTIC.csv"
  }
}
```

---

## Project Directory Structure (Complete)

```
{project_dir}/
├── calibration/
│   ├── intrinsic/
│   │   ├── port_1.mp4
│   │   └── port_2.mp4
│   └── extrinsic/
│       ├── port_1.mp4
│       ├── port_2.mp4
│       └── CHARUCO/
│           ├── camera_array.toml
│           ├── image_points.csv
│           └── world_points.csv
│
├── recordings/
│   ├── visit_20260202_120315/        # Auto-recorded visit
│   │   ├── port_1.mp4
│   │   ├── port_2.mp4
│   │   ├── frame_timestamps.csv
│   │   ├── metadata.json             # Recording metadata
│   │   ├── identity.json             # Person ID (Step A)
│   │   ├── HOLISTIC/                 # 3D reconstruction (Step B)
│   │   │   └── xyz_HOLISTIC.csv
│   │   ├── zone_timeline.csv         # Zone analysis (Step C)
│   │   └── session_report.json       # Combined report (Step E)
│   └── visit_20260202_183042/
│       └── ...
│
├── config/
│   ├── dining_zone.toml              # 3D zone definition
│   └── subjects/                     # Face enrollment database
│       ├── subject_01/
│       │   ├── ref_001.jpg
│       │   ├── ref_002.jpg
│       │   └── embedding.npy
│       └── subject_02/
│           └── ...
│
├── camera_array.toml
├── charuco.toml
└── project_settings.toml
```

---

## Implementation Roadmap

### Stage 1: Smart Recorder (Priority: HIGH)
- [ ] Implement continuous monitoring mode with person detection
- [ ] State machine: IDLE → RECORDING → COOLDOWN → IDLE
- [ ] Integrate with existing FFmpeg recording pipeline
- [ ] Add to PipelineUI as "Panel 5: Monitor" or standalone script
- [ ] Test: reliable start/stop with real person entry/exit

### Stage 2: Dining Zone Definition Tool (Priority: HIGH)
- [ ] Add zone marking UI: user clicks table corners in camera view
- [ ] Triangulate clicked points to 3D using calibrated cameras
- [ ] Save as dining_zone.toml
- [ ] Visualize zone in Panel 4 (3D visualization) as a bounding box

### Stage 3: Person Identification (Priority: MEDIUM)
- [ ] Subject enrollment tool: capture/import reference face photos
- [ ] Face embedding extraction and storage
- [ ] Per-visit face matching: identify subject from first frames
- [ ] Output identity.json per visit

### Stage 4: Zone Occupancy Analysis (Priority: MEDIUM)
- [ ] Load 3D reconstruction + dining zone definition
- [ ] Per-frame zone occupancy classification
- [ ] Generate zone_timeline.csv with temporal annotations
- [ ] Visualize in Panel 4: highlight frames when person is at table

### Stage 5: Multi-Camera Integration (Priority: LOW)
- [ ] Food camera timestamp synchronization
- [ ] Frame index mapping between stereo and food cameras
- [ ] Combined data export

### Stage 6: Activity Summary (Priority: LOW)
- [ ] Aggregate all analysis results per visit
- [ ] Generate session_report.json
- [ ] Batch processing: run all steps on all visits at once

---

## Technical Notes

### Person Detection Choice
Recommended: **YOLOv8-nano** (ultralytics package)
- Install: `pip install ultralytics`
- Model: `yolov8n.pt` (6.3MB, ~30fps on CPU at 640x480)
- Only detect class 0 (person), ignore all other classes
- Confidence threshold: 0.5 (configurable)
- Fallback: OpenCV HOG if ultralytics not available

### Recording Strategy
- Camera reads via OpenCV (cv2.VideoCapture) for detection
- Recording via FFmpeg subprocess (same as existing CalibrationUI)
- Detection and recording use the same camera device
- Approach: OpenCV reads for detection → when triggered, release OpenCV, start FFmpeg
- Alternative: Use OpenCV VideoWriter for both (simpler but lower quality)

### Face Recognition
Recommended: `insightface` with ArcFace model
- Install: `pip install insightface onnxruntime`
- Alternative: `face_recognition` (simpler API, dlib-based)
- Enrollment: 3-5 photos per subject, frontal + slight angles
- Matching threshold: cosine similarity > 0.4

### Coordinate System
- All 3D coordinates in the calibrated world frame (set by charuco board origin)
- Units: centimeters (caliscope default)
- Dining zone defined in the same coordinate system
- Z-axis typically points up from the charuco board plane
