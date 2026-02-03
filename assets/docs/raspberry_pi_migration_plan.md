# 树莓派迁移方案 — stereo-charuco-3d-pipeline

> 编写日期: 2026-02-03
> 状态: 规划中（未实施）

---

## 1. 背景与目标

当前项目运行在 Windows + RTX 5090 平台上，但项目本身**不使用 GPU 加速**，全部计算基于 CPU。为实现低成本家庭部署，计划将项目迁移至树莓派平台。

**目标硬件配置：**

| 组件 | 规格 | 预估价格 |
|------|------|----------|
| 树莓派 5 (8GB) | BCM2712, 4核 Cortex-A76 @2.4GHz | ~¥500 |
| USB 立体相机 | 与现有相同型号 "3D USB Camera" | ~¥200-500 |
| Coral USB TPU（推荐） | Google Edge TPU, USB 3.0 | ~¥180 |
| NVMe SSD HAT + 256GB | 树莓派 5 PCIe 扩展 | ~¥150 |
| 电源 + 外壳 | 27W USB-C 官方电源 | ~¥50 |
| **总计** | | **~¥800-1200** |

**目标用户体验：**

1. 插电开机，自动运行
2. 手机打开浏览器，访问 `http://<设备IP>:8080` 查看画面
3. 站到相机前，自动开始录制
4. 录制完成后自动处理
5. 手机上查看 3D 结果 / 下载数据

---

## 2. 现状分析：哪些不用改，哪些必须改

### 2.1 不需要改动的模块（核心算法层）

以下模块使用纯 OpenCV / NumPy / SciPy，ARM 完全兼容：

| 模块 | 文件 | 说明 |
|------|------|------|
| ChArUco 检测 | `caliscope/core/charuco.py` | 纯 OpenCV ArUco 模块 |
| 内参标定 | `caliscope/core/intrinsic_calibrator.py` | `cv2.calibrateCamera()` |
| 外参标定 | `caliscope/core/bootstrap_pose/` | 立体位姿估计 |
| 三角化 | `caliscope/core/point_data.py` | Numba JIT（ARM 可用，需小改） |
| Bundle Adjustment | `caliscope/core/point_data_bundle.py` | `scipy.optimize.least_squares()` |
| 后处理滤波 | `caliscope/core/` | Butterworth 滤波（SciPy） |
| 重建管线 | `caliscope/reconstruction/reconstructor.py` | 调度逻辑 |
| 数据 I/O | `caliscope/recording/` | CSV / TRC 读写 |

### 2.2 必须改动的模块

| 模块 | 涉及文件数 | 问题 | 优先级 |
|------|-----------|------|--------|
| 采集层（DirectShow → V4L2） | 10 | Windows 专用 API | **P0 必须** |
| 检测模型（MediaPipe → MoveNet） | 6 | 树莓派上太慢 | **P0 必须** |
| 分辨率/帧率配置 | 2 | 需要树莓派专用参数 | **P0 必须** |
| UI 层（PySide6 → Web UI） | 40+ | 太重，不适合嵌入式 | **P1 建议** |
| Numba ARM fallback | 1 | 兼容性保险 | **P2 可选** |

---

## 3. 改动步骤详细说明

### 步骤 1：采集层跨平台化

**优先级：P0 必须**
**涉及文件：10 个**
**改动复杂度：低**

#### 3.1.1 相机后端切换

**文件清单及具体改动点：**

**`tools/calib_record_tool/src/recorder/smart_recorder.py`**
- 第 157-158 行：`cv2.CAP_DSHOW` → 跨平台选择

```python
# -------- 改动前 --------
# smart_recorder.py:157-158
if sys.platform == "win32":
    self._cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_DSHOW)

# -------- 改动后 --------
def _get_cv_backend():
    """根据操作系统选择相机后端"""
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    elif sys.platform == "linux":
        return cv2.CAP_V4L2
    else:
        return cv2.CAP_ANY

self._cap = cv2.VideoCapture(self.config.device_index, _get_cv_backend())
```

- 第 170 行：MJPG fourcc 设置（保持不变，V4L2 同样支持 MJPG）
- 第 396 行：VideoWriter fourcc（保持不变）

**`tools/calib_record_tool/src/recorder/calibration_ui.py`**
- 第 154 行：同上，`CAP_DSHOW` → `_get_cv_backend()`
- 第 165 行：MJPG fourcc（保持不变）
- 第 248 行：FFmpeg 命令 `-f dshow` → 跨平台

```python
# -------- 改动前 --------
# calibration_ui.py:248
"-f", "dshow",

# -------- 改动后 --------
"-f", "v4l2" if sys.platform == "linux" else "dshow",
```

- 第 252 行：FFmpeg 输入设备名

```python
# -------- 改动前 --------
"-i", f"video={self.config.device_name}"

# -------- 改动后 --------
# Linux: 使用 /dev/videoN 设备路径
# Windows: 使用 DirectShow 设备名
"-i", f"/dev/video{self.device_index}" if sys.platform == "linux"
      else f"video={self.config.device_name}"
```

**`tools/calib_record_tool/src/recorder/calibration_ui_advanced.py`**
- 第 180 行：`subprocess.CREATE_NO_WINDOW`（仅 Windows）

```python
# -------- 改动前 --------
if sys.platform == "win32":
    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

# -------- 改动后（已有 platform 判断，无需改动）--------
```

- 第 279 行：`CAP_DSHOW` → `_get_cv_backend()`
- 第 289 行：MJPG fourcc（保持不变）

**`tools/calib_record_tool/scripts/record_raw.py`**
- 第 113 行：`-f dshow` → 跨平台
- 第 116 行：`-vcodec mjpeg`（保持不变，V4L2 支持）

**`tools/calib_record_tool/scripts/record_and_split.py`**
- 第 181 行：`-f dshow` → 跨平台
- 第 184 行：`-vcodec mjpeg`（保持不变）

**`tools/calib_record_tool/scripts/probe_camera.py`**
- 第 10 行：`backend=cv2.CAP_DSHOW` → `_get_cv_backend()`
- 第 104 行：同上

#### 3.1.2 设备枚举重写

**`tools/calib_record_tool/src/recorder/camera.py`**

```python
# -------- 改动前 --------
# camera.py:9-35
def list_dshow_devices() -> List[str]:
    """FFmpeg DirectShow 设备枚举"""
    cmd = [ffmpeg, "-hide_banner", "-list_devices", "true",
           "-f", "dshow", "-i", "dummy"]
    # ... DirectShow 输出解析 ...

# -------- 改动后 --------
import sys
from pathlib import Path

def list_camera_devices() -> List[str]:
    """跨平台相机设备枚举"""
    if sys.platform == "win32":
        return _list_dshow_devices()
    elif sys.platform == "linux":
        return _list_v4l2_devices()
    else:
        return []

def _list_dshow_devices() -> List[str]:
    """Windows: FFmpeg DirectShow 枚举（原有逻辑）"""
    # ... 保持现有代码 ...

def _list_v4l2_devices() -> List[str]:
    """Linux: 扫描 /dev/video* 设备"""
    import subprocess
    devices = []
    for dev in sorted(Path("/dev").glob("video*")):
        try:
            result = subprocess.run(
                ["v4l2-ctl", "--device", str(dev), "--info"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                # 提取设备名称
                for line in result.stdout.splitlines():
                    if "Card type" in line:
                        name = line.split(":", 1)[1].strip()
                        devices.append(f"{dev} ({name})")
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            devices.append(str(dev))
    return devices
```

**需要在树莓派上安装 `v4l-utils`：**

```bash
sudo apt install v4l-utils
```

---

### 步骤 2：检测模型轻量化

**优先级：P0 必须**
**涉及文件：6 个**
**改动复杂度：中**

#### 3.2.1 性能对比

| 模型 | 关键点数 | 模型大小 | 树莓派 5 推理速度 | 树莓派 5 + Coral TPU |
|------|---------|---------|------------------|---------------------|
| MediaPipe Holistic | 593 | ~500MB | ~500-1500ms (0.7-2 fps) | 不支持 |
| MediaPipe Pose | 33 | ~200MB | ~200-500ms (2-5 fps) | 不支持 |
| **MoveNet Lightning** | **17** | **~10MB** | **~50-80ms (12-20 fps)** | **~5-10ms (100+ fps)** |
| MoveNet Thunder | 17 | ~30MB | ~150-300ms (3-7 fps) | ~15-20ms (50-65 fps) |

**推荐：MoveNet Lightning**（Google 专为边缘设备优化）

#### 3.2.2 创建 MoveNet Tracker

新建文件 `caliscope/trackers/movenet_tracker.py`：

```python
"""
MoveNet Lightning tracker — 轻量级姿态估计，适用于树莓派
17 个关键点：鼻、左右眼、左右耳、左右肩、左右肘、
            左右腕、左右髋、左右膝、左右踝
"""
import numpy as np

# 方式 A: 纯 TFLite（无额外依赖）
import tflite_runtime.interpreter as tflite

# 方式 B: 如有 Coral TPU
# from pycoral.utils.edgetpu import make_interpreter

MOVENET_KEYPOINTS = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

class MoveNetTracker:
    """MoveNet Lightning TFLite tracker"""

    def __init__(self, model_path: str, use_edgetpu: bool = False):
        if use_edgetpu:
            from pycoral.utils.edgetpu import make_interpreter
            self.interpreter = make_interpreter(model_path)
        else:
            self.interpreter = tflite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # MoveNet Lightning 输入: 192×192×3
        self.input_size = self.input_details[0]['shape'][1]

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        输入: BGR 图像 (H, W, 3)
        输出: (17, 3) — [y, x, confidence] 归一化坐标
        """
        import cv2
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.int32)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        keypoints = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        # 输出 shape: (1, 1, 17, 3) → (17, 3)
        return keypoints[0, 0, :, :]

    def get_2d_points(self, frame: np.ndarray,
                      min_confidence: float = 0.3):
        """
        返回像素坐标的关键点
        输出: dict[point_id, (x_px, y_px)] 仅包含置信度达标的点
        """
        h, w = frame.shape[:2]
        raw = self.detect(frame)  # (17, 3) → [y_norm, x_norm, conf]
        points = {}
        for i, (y, x, conf) in enumerate(raw):
            if conf >= min_confidence:
                points[i] = (x * w, y * h)
        return points
```

#### 3.2.3 需要修改的 Tracker 注册/调度文件

在 caliscope 中，tracker 的选择通过配置文件和工厂模式实现。需要：

1. 在 tracker 注册表中添加 `"movenet"` 类型
2. 将 MoveNet 的 17 个关键点映射到 caliscope 的点位 ID 体系
3. 修改 `reconstructor.py` 中的 tracker 实例化逻辑

**关键点映射（MoveNet 17 点 → caliscope 点位 ID）：**

```python
# MoveNet 关键点 → caliscope 兼容的 point_id 映射
MOVENET_TO_CALISCOPE = {
    0: "nose",
    5: "left_shoulder",   6: "right_shoulder",
    7: "left_elbow",      8: "right_elbow",
    9: "left_wrist",      10: "right_wrist",
    11: "left_hip",       12: "right_hip",
    13: "left_knee",      14: "right_knee",
    15: "left_ankle",     16: "right_ankle",
}
# 注意：MoveNet 的 17 点是 MediaPipe Pose 33 点的子集
# 缺少：中间躯干点、脚趾、脚跟等
```

#### 3.2.4 HOG 人体检测替代

**`tools/calib_record_tool/src/recorder/smart_recorder.py`**

HOG 检测器在树莓派上 0.4x 降采样可达 7-20 fps，基本可用。但如果需要更快，可以用 MoveNet 本身做检测触发：

```python
# -------- 方案 A：保留 HOG（可用但较慢）--------
# 将 detect_scale 降低到 0.25
# smart_recorder.py:43
hog_win_stride: int = 16      # 从 8 改为 16（速度翻倍，精度略降）

# -------- 方案 B：用 MoveNet 替代 HOG --------
# 如果 MoveNet 检测到任何关键点 confidence > 0.5 → 认为有人
def _detect_person_movenet(self, frame):
    points = self.movenet.get_2d_points(frame, min_confidence=0.5)
    return len(points) >= 3  # 至少检测到 3 个关键点
```

#### 3.2.5 MoveNet 模型获取

```bash
# 下载 MoveNet Lightning TFLite 模型
wget -O models/movenet_lightning.tflite \
  "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"

# 如果使用 Coral TPU，需要 Edge TPU 编译版本
wget -O models/movenet_lightning_edgetpu.tflite \
  "https://raw.githubusercontent.com/google-coral/test_data/master/movenet_single_pose_lightning_ptq_edgetpu.tflite"
```

---

### 步骤 3：树莓派专用配置文件

**优先级：P0 必须**
**涉及文件：1-2 个新建**
**改动复杂度：低**

新建 `tools/calib_record_tool/configs/raspberry_pi.yaml`：

```yaml
# 树莓派 5 专用配置
# 使用方式: python scripts/run_unified.py --config configs/raspberry_pi.yaml

platform: "raspberry_pi"

camera:
  device_name: "3D USB Camera"
  device_index: 0
  backend: "v4l2"
  video_size: "1280x480"      # 每路 640x480（树莓派 USB 带宽限制）
  fps: 30                     # 30fps 足够动作捕捉

split:
  full_w: 1280
  full_h: 480
  xsplit: 640                 # 左右各 640x480

detection:
  model: "movenet_lightning"  # 轻量模型
  model_path: "models/movenet_lightning.tflite"
  use_edgetpu: false          # 设为 true 如果有 Coral TPU
  check_interval: 10          # 每 10 帧检测一次
  detect_scale: 0.5           # 640x480 已经很小，不需要大幅降采样
  min_confidence: 0.3

recording:
  codec: "MJPG"
  absence_threshold: 5
  cooldown_seconds: 30

charuco:
  columns: 7                  # 减少角点以加速检测
  rows: 10
  aruco_scale: 0.7
  square_size_overide_cm: 5.0
  board_height_cm: 50.0
  board_width_cm: 35.0
  inverted: false
  dictionary: "DICT_4X4_50"

processing:
  tracker: "movenet"          # 使用 MoveNet 替代 MediaPipe
  batch_size: 1               # 逐帧处理，节省内存
  max_workers: 2              # 限制并行线程数

output:
  format: "csv"               # CSV 输出
  include_trc: true           # 同时输出 TRC 格式
```

**对比参考——现有 Windows 配置：**

```yaml
# configs/default.yaml (现有 Windows 配置)
camera:
  video_size: "3200x1200"     # 每路 1600x1200
  fps: 60                     # 60fps
  backend: "dshow"            # Windows DirectShow

charuco:
  columns: 10
  rows: 16
```

---

### 步骤 4：UI 层改造（Headless + Web UI）

**优先级：P1 建议**
**涉及文件：新建 5-10 个，停用 40+ 个 PySide6 文件**
**改动复杂度：高**

#### 3.4.1 架构设计

```
┌─────────────────────────────────────────────────────┐
│                    树莓派 5                           │
│                                                     │
│  systemd service (开机自启)                           │
│       │                                             │
│       ▼                                             │
│  ┌──────────────────────────────────────────────┐   │
│  │  Python 主进程                                │   │
│  │                                              │   │
│  │  ┌────────────┐  ┌────────────┐              │   │
│  │  │ 相机采集线程 │  │ 检测处理线程 │              │   │
│  │  │ (OpenCV)   │  │ (MoveNet)  │              │   │
│  │  └─────┬──────┘  └─────┬──────┘              │   │
│  │        │               │                     │   │
│  │        ▼               ▼                     │   │
│  │  ┌─────────────────────────────────────┐     │   │
│  │  │        Flask / FastAPI Web 服务       │     │   │
│  │  │                                     │     │   │
│  │  │  GET  /              → 主页          │     │   │
│  │  │  GET  /stream        → MJPEG 实时流  │     │   │
│  │  │  GET  /status        → 系统状态      │     │   │
│  │  │  POST /calibrate     → 开始标定      │     │   │
│  │  │  POST /record/start  → 手动开始录制  │     │   │
│  │  │  POST /record/stop   → 手动停止录制  │     │   │
│  │  │  GET  /results       → 查看结果列表  │     │   │
│  │  │  GET  /download/:id  → 下载数据文件  │     │   │
│  │  │  GET  /settings      → 配置页面      │     │   │
│  │  └─────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────┘   │
│                    ↕ Port 8080                       │
└─────────────────────────────────────────────────────┘
         │
    WiFi / 有线局域网
         │
         ▼
┌─────────────────────┐
│  用户手机/电脑浏览器   │
│  http://192.168.x.x  │
│       :8080          │
└─────────────────────┘
```

#### 3.4.2 Web UI 核心代码框架

新建 `tools/calib_record_tool/src/web/app.py`：

```python
"""树莓派 Web UI — 基于 Flask"""
from flask import Flask, Response, render_template, jsonify
import cv2
import threading

app = Flask(__name__)

# 全局状态
recorder = None  # SmartRecorder 实例
latest_frame = None
lock = threading.Lock()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stream")
def video_stream():
    """MJPEG 实时视频流"""
    def generate():
        while True:
            with lock:
                if latest_frame is None:
                    continue
                _, jpeg = cv2.imencode('.jpg', latest_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + jpeg.tobytes() + b'\r\n')
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def status():
    return jsonify({
        "state": recorder.state if recorder else "not_initialized",
        "recording_count": recorder.recording_count if recorder else 0,
        "disk_free_gb": get_disk_free_gb(),
    })

@app.route("/calibrate", methods=["POST"])
def calibrate():
    # 触发标定流程
    threading.Thread(target=run_calibration, daemon=True).start()
    return jsonify({"status": "calibration_started"})

@app.route("/results")
def results():
    # 列出所有处理完成的结果
    return jsonify(list_results())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

#### 3.4.3 开机自启动配置

新建 `/etc/systemd/system/stereo-pipeline.service`：

```ini
[Unit]
Description=Stereo ChArUco 3D Pipeline
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/stereo-charuco-3d-pipeline
ExecStart=/home/pi/miniconda3/envs/caliscope311/bin/python \
          tools/calib_record_tool/src/web/app.py \
          --config configs/raspberry_pi.yaml
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
# 启用自启动
sudo systemctl enable stereo-pipeline.service
sudo systemctl start stereo-pipeline.service

# 查看日志
journalctl -u stereo-pipeline -f
```

---

### 步骤 5：Numba ARM 兼容性保障

**优先级：P2 可选**
**涉及文件：1 个**
**改动复杂度：低**

**`projects/caliscope/caliscope_src/src/caliscope/core/point_data.py`**

当前代码第 55 行已有 `cache=True`（首次编译后缓存到磁盘），ARM 上通常可以工作。增加 fallback 保险：

```python
# -------- 改动后 --------
try:
    from numba import jit
    from numba.typed import Dict, List
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # fallback: 装饰器不做任何事
    def jit(**kwargs):
        def decorator(func):
            return func
        return decorator
    # 使用普通 Python dict/list
    Dict = dict
    List = list

@jit(nopython=True, cache=True)
def triangulate_sync_index(...):
    # 函数体保持不变
    # 如果 Numba 不可用，以纯 Python 速度运行（慢约 10-50x）
    # 但对于离线处理仍然可接受
    ...
```

---

## 4. 依赖变更

### 4.1 需要移除的依赖（树莓派上不需要）

```
PySide6-essentials    # 替换为 Flask Web UI
pyvista               # 3D 可视化改为 Web 端 three.js（可选）
pyvistaqt             # 同上
PyOpenGL              # 同上
pywinctl              # Windows 专用
```

### 4.2 需要新增的依赖

```
flask>=3.0            # Web UI 框架
tflite-runtime        # TFLite 推理（替代完整 TensorFlow）
# 或
pycoral               # 如果使用 Coral Edge TPU
```

### 4.3 树莓派专用 requirements 文件

新建 `requirements_raspberry_pi.txt`：

```
# 核心依赖（与现有相同）
opencv-contrib-python>=4.8.0.74
numpy>=1.24.0
scipy>=1.10.1
pandas>=1.5.0
PyYAML==6.0.2
Pillow>=10.0.0
av>=16.0.0,<16.1.0
rtoml>=0.9.0
platformdirs>=4.3.8

# 树莓派专用
flask>=3.0
tflite-runtime>=2.14.0

# 可选：Numba（ARM 上可能需要从 conda-forge 安装）
# numba>=0.57.0

# 可选：Coral Edge TPU
# pycoral>=2.0
# tflite-runtime  # pycoral 自带

# 不需要的依赖（已移除）：
# PySide6-essentials
# pyvista
# pyvistaqt
# PyOpenGL
# mediapipe        # 替换为 tflite-runtime + MoveNet 模型
```

### 4.4 系统级依赖（apt）

```bash
# 树莓派 OS 上需要安装
sudo apt update
sudo apt install -y \
    python3-dev \
    v4l-utils \              # V4L2 相机工具
    libatlas-base-dev \      # NumPy BLAS 加速
    libjasper-dev \          # OpenCV 依赖
    libhdf5-dev \            # HDF5 支持
    libqt5gui5 \             # OpenCV highgui 后端
    ffmpeg                   # 视频编解码
```

---

## 5. 树莓派环境搭建步骤

```bash
# 1. 安装 Raspberry Pi OS (64-bit, Bookworm)
#    使用 Raspberry Pi Imager 烧录到 SD 卡或 NVMe SSD

# 2. 系统配置
sudo raspi-config
#    → Interface Options → SSH → Enable
#    → Interface Options → VNC → Enable (可选)
#    → Advanced Options → Expand Filesystem

# 3. 安装 Miniconda (ARM64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh
source ~/.bashrc

# 4. 创建环境
conda create -n caliscope311 python=3.11 -y
conda activate caliscope311

# 5. 安装 Numba（推荐通过 conda-forge，pip 在 ARM 上可能失败）
conda install -c conda-forge numba -y

# 6. 安装 Python 依赖
pip install -r requirements_raspberry_pi.txt

# 7. 安装系统依赖
sudo apt install -y v4l-utils libatlas-base-dev ffmpeg

# 8. 下载 MoveNet 模型
mkdir -p models
wget -O models/movenet_lightning.tflite \
  "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"

# 9. 验证相机
v4l2-ctl --list-devices
python -c "import cv2; cap=cv2.VideoCapture(0, cv2.CAP_V4L2); print(cap.isOpened())"

# 10. 可选：安装 Coral Edge TPU 运行时
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std python3-pycoral
```

---

## 6. 精度影响评估

分辨率降低和关键点减少会影响 3D 重建精度：

| 参数 | 现有 Windows 配置 | 树莓派配置 | 精度影响 |
|------|------------------|-----------|---------|
| 分辨率 | 1600×1200 每路 | 640×480 每路 | 像素误差放大 ~2.5x |
| 帧率 | 60 fps | 30 fps | 快速运动可能模糊 |
| 关键点数 | 593 (Holistic) | 17 (MoveNet) | 仅大关节，无手指/面部 |
| ChArUco 棋盘 | 10×16 (160格) | 7×10 (70格) | 标定精度略降 |
| 三角化精度 | 亚毫米级 | 约 2-5mm 级 | 对康复评估仍可接受 |

**关键结论：** 对于基础的人体运动捕捉（步态分析、康复评估等），17 个关键点 + 640×480 分辨率 **足够满足需求**。不适合精细手指追踪或面部表情分析。

---

## 7. 实施路线图

```
阶段 1：基础跨平台化                     阶段 2：模型替换
┌──────────────────────────┐        ┌──────────────────────────┐
│ □ 抽取 _get_cv_backend() │        │ □ 实现 MoveNetTracker    │
│ □ 改 10 个文件的 CAP_DSHOW│        │ □ 下载 TFLite 模型       │
│ □ 改 camera.py 设备枚举   │        │ □ 注册到 tracker 工厂    │
│ □ 改 FFmpeg -f dshow     │        │ □ 关键点 ID 映射         │
│ □ 新建 raspberry_pi.yaml │        │ □ 集成测试               │
└──────────────────────────┘        └──────────────────────────┘
           │                                    │
           ▼                                    ▼
阶段 3：Web UI                        阶段 4：产品化
┌──────────────────────────┐        ┌──────────────────────────┐
│ □ Flask app 骨架          │        │ □ systemd 自启动          │
│ □ MJPEG 视频流            │        │ □ 自动 WiFi 配网         │
│ □ 标定/录制 API           │        │ □ OTA 远程更新           │
│ □ 结果查看/下载页面       │        │ □ 日志收集与错误上报      │
│ □ 前端 HTML 页面          │        │ □ 用户引导流程           │
└──────────────────────────┘        └──────────────────────────┘
```

---

## 8. 风险与备选方案

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|---------|
| USB 立体相机无 Linux 驱动 | 中 | 致命 | 提前测试；备选：换用两个独立 USB 摄像头 |
| MoveNet 17 点精度不够 | 低 | 高 | 改用 MoveNet Thunder（30MB，精度更高） |
| Numba ARM 编译失败 | 低 | 中 | 使用纯 NumPy fallback |
| 树莓派内存 OOM | 中 | 高 | 限制 batch_size=1；分阶段处理 |
| 树莓派 USB 带宽不足 | 中 | 高 | 降至 640×480@15fps；使用 CSI 摄像头替代 |
| Coral TPU 缺货 | 低 | 低 | MoveNet 在树莓派 5 上无 TPU 也可达 12-20fps |

### 备选硬件方案

如果树莓派方案遇到不可克服的问题：

| 方案 | 成本 | 优势 | 劣势 |
|------|------|------|------|
| **Intel N100 Mini PC** | ¥500-800 | x86 架构，代码零修改 | 体积略大 |
| **Orange Pi 5 (RK3588)** | ¥400-600 | 内置 6 TOPS NPU | 软件生态较弱 |
| **Jetson Orin Nano** | ¥1500-2000 | CUDA GPU 可用 | 成本高 |

---

## 附录 A：需要改动的完整文件清单

### P0 必须改动

| 文件路径 | 改动行 | 改动内容 |
|---------|--------|---------|
| `tools/.../recorder/smart_recorder.py` | 157-158, 170, 396 | CAP_DSHOW → 跨平台 |
| `tools/.../recorder/calibration_ui.py` | 154, 165, 248, 252 | CAP_DSHOW + FFmpeg dshow |
| `tools/.../recorder/calibration_ui_advanced.py` | 279, 289 | CAP_DSHOW → 跨平台 |
| `tools/.../recorder/camera.py` | 9-35 | 设备枚举重写 |
| `tools/.../scripts/record_raw.py` | 113, 116 | FFmpeg -f dshow |
| `tools/.../scripts/record_and_split.py` | 181, 184 | FFmpeg -f dshow |
| `tools/.../scripts/probe_camera.py` | 10, 104 | CAP_DSHOW → 跨平台 |
| `tools/.../configs/` | 新建文件 | raspberry_pi.yaml |
| `caliscope/trackers/` | 新建文件 | movenet_tracker.py |
| `requirements_raspberry_pi.txt` | 新建文件 | 树莓派依赖列表 |

### P1 建议改动

| 文件路径 | 改动内容 |
|---------|---------|
| `tools/.../src/web/app.py` | 新建 Flask Web UI |
| `tools/.../src/web/templates/` | 新建 HTML 模板 |
| `/etc/systemd/system/stereo-pipeline.service` | 新建自启动配置 |

### P2 可选改动

| 文件路径 | 改动行 | 改动内容 |
|---------|--------|---------|
| `caliscope/core/point_data.py` | 11-12 | Numba import fallback |
