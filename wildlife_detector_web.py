#!/usr/bin/env python3
"""
Raspberry Pi 5 + Hailo-8 Wildlife Detector

Inference is performed by the Hailo-8 through Picamera2's Hailo helper.
Ultralytics/PyTorch is NOT required.

Modes:
    1. PIR:
       - PIR triggers a still image.
       - If a configured target class is detected in that image, a video
         can be recorded.
       - Video can also be recorded on every PIR trigger.

    2. Continuous:
       - Detection runs continuously.
       - A configured target detection triggers either an image or a video.

All settings intended for later use by a local web UI are kept in config.json.
"""

from __future__ import annotations

import json
import shutil
import signal
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from pathlib import Path
from threading import Event, Thread

import cv2
from gpiozero import MotionSensor
from picamera2 import Picamera2
from libcamera import controls
from picamera2.devices import Hailo



# ---------------------------------------------------------------------------
# LOCAL WEB CONFIGURATION SERVER
# ---------------------------------------------------------------------------

WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
WEB_CONFIG_LOCK = __import__("threading").Lock()


WEB_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wildlife Detector - Settings</title>
<style>
:root { color-scheme: dark; }
body {
    font-family: system-ui, sans-serif;
    margin: 0; padding: 20px;
    background: #101114; color: #eee;
}
main { max-width: 1100px; margin: auto; }
h1 { margin-bottom: 6px; }
.note { color: #aaa; margin-bottom: 20px; }
section {
    background: #191b20; border: 1px solid #30333a;
    border-radius: 12px; padding: 16px; margin: 14px 0;
}
section h2 { margin-top: 0; font-size: 20px; }
.field {
    display: grid; grid-template-columns: minmax(180px, 1fr) minmax(180px, 2fr);
    gap: 12px; align-items: center; margin: 9px 0;
}
label { color: #ddd; }
input, textarea, select {
    width: 100%; box-sizing: border-box;
    background: #0f1013; color: #eee;
    border: 1px solid #444851; border-radius: 7px;
    padding: 9px;
}
textarea { min-height: 90px; font-family: monospace; }
button {
    border: 0; border-radius: 8px; padding: 11px 16px;
    cursor: pointer; margin-right: 8px; font-weight: 600;
}
#save { background: #4caf50; color: white; }
#reload { background: #444851; color: white; }
#status { margin-top: 12px; color: #8bcf8b; }
@media (max-width: 700px) {
    .field { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<main>
<h1>Raspberry Pi Wildlife Detector</h1>
<div class="note">
    Local configuration panel. Changes are saved to <code>config.json</code>.
    Restart the detector when a setting requires reinitialization.
</div>
<div id="settings"></div>
<div>
    <button id="save">Save settings</button>
    <button id="reload">Reload</button>
</div>
<div id="status"></div>
</main>

<script>
let config = {};

function isObject(v) {
    return v !== null && typeof v === "object" && !Array.isArray(v);
}

function makeInput(path, value) {
    const wrap = document.createElement("div");
    wrap.className = "field";

    const label = document.createElement("label");
    label.textContent = path.join(".");
    wrap.appendChild(label);

    let input;

    if (typeof value === "boolean") {
        input = document.createElement("select");
        ["true", "false"].forEach(x => {
            const o = document.createElement("option");
            o.value = x; o.textContent = x;
            if ((value && x === "true") || (!value && x === "false")) o.selected = true;
            input.appendChild(o);
        });
        input.dataset.type = "boolean";
    } else if (Array.isArray(value)) {
        input = document.createElement("textarea");
        input.value = value.map(x => typeof x === "object" ? JSON.stringify(x) : String(x)).join("\\n");
        input.dataset.type = "array";
    } else {
        input = document.createElement("input");
        input.value = String(value);
        input.dataset.type = typeof value;
    }

    input.dataset.path = JSON.stringify(path);
    wrap.appendChild(input);
    return wrap;
}

function renderObject(obj, path = [], parent = document.getElementById("settings")) {
    for (const [key, value] of Object.entries(obj)) {
        const p = [...path, key];

        if (isObject(value)) {
            const section = document.createElement("section");
            const h = document.createElement("h2");
            h.textContent = p.join(".");
            section.appendChild(h);
            parent.appendChild(section);
            renderObject(value, p, section);
        } else {
            parent.appendChild(makeInput(p, value));
        }
    }
}

function setAtPath(obj, path, value) {
    let cur = obj;
    for (let i = 0; i < path.length - 1; i++) cur = cur[path[i]];
    cur[path[path.length - 1]] = value;
}

function parseValue(input) {
    const t = input.dataset.type;
    const raw = input.value;

    if (t === "boolean") return raw === "true";
    if (t === "number") {
        const n = Number(raw);
        if (Number.isNaN(n)) throw new Error("Invalid number: " + raw);
        return n;
    }
    if (t === "array") {
        return raw.split("\\n").map(x => x.trim()).filter(Boolean);
    }
    return raw;
}

function collect() {
    const result = structuredClone(config);
    for (const input of document.querySelectorAll("[data-path]")) {
        setAtPath(result, JSON.parse(input.dataset.path), parseValue(input));
    }
    return result;
}

async function load() {
    const r = await fetch("/api/config");
    config = await r.json();
    document.getElementById("settings").innerHTML = "";
    renderObject(config);
    document.getElementById("status").textContent = "Settings loaded.";
}

async function save() {
    try {
        const newConfig = collect();
        const r = await fetch("/api/config", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(newConfig)
        });
        const data = await r.json();
        document.getElementById("status").textContent =
            data.ok ? "Saved. Restart detector if needed." : ("Error: " + data.error);
    } catch (e) {
        document.getElementById("status").textContent = "Error: " + e.message;
    }
}

document.getElementById("save").onclick = save;
document.getElementById("reload").onclick = load;
load();
</script>
</body>
</html>
"""


def _write_config_atomic(config):
    """Save config without leaving a partially-written JSON file."""
    temporary = CONFIG_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(config, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(CONFIG_PATH)


class ConfigWebHandler(BaseHTTPRequestHandler):
    """Small dependency-free web UI for the Raspberry Pi local network."""

    def _send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            data = WEB_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/api/config":
            try:
                with WEB_CONFIG_LOCK:
                    current = load_config()
                self._send_json(current)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/api/config":
            self._send_json({"error": "Not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("Invalid request size.")

            body = self.rfile.read(length)
            new_config = json.loads(body.decode("utf-8"))

            if not isinstance(new_config, dict):
                raise ValueError("Configuration must be a JSON object.")

            with WEB_CONFIG_LOCK:
                _write_config_atomic(new_config)

            self._send_json({"ok": True})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def log_message(self, format, *args):
        print(f"[WEB] {self.address_string()} - {format % args}")


def get_local_ip():
    """Return the Pi's LAN IP without requiring an internet connection."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


def start_web_server():
    server = ThreadingHTTPServer((WEB_HOST, WEB_PORT), ConfigWebHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    ip = get_local_ip()
    print(f"Web configuration: http://{ip}:{WEB_PORT}")
    print("Web server listens on the Raspberry Pi LAN interfaces.")

    return server


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
STOP_EVENT = Event()


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {'mode': 'pir', '_mode_comment': 'pir = PIR-triggered operation; continuous = continuous detection.', 'model': {'path': 'yolov11l.hef', '_path_comment': 'Hailo HEF model file used for inference.', 'labels': 'labels.txt', '_labels_comment': 'Text file containing one class name per line, in the same order as the HEF class IDs.', 'confidence': 0.5, '_confidence_comment': 'Minimum detection confidence. Detections below this value are ignored.'}, 'camera': {'width': 1280, '_width_comment': 'Main camera output width in pixels.', 'height': 960, '_height_comment': 'Main camera output height in pixels.', 'fps_min': 5, '_fps_min_comment': 'Minimum allowed camera/video FPS.', 'fps_max': 30, '_fps_max_comment': 'Maximum allowed camera/video FPS.', 'fps': 15, '_fps_comment': 'Requested camera FPS. It is clamped between fps_min and fps_max.'}, 'pir': {'_comment': 'Settings for PIR-triggered operation.', 'gpio_pin': 13, '_gpio_pin_comment': 'GPIO pin connected to the PIR sensor.', 'cooldown_seconds': 2.0, '_cooldown_comment': 'Minimum time between two PIR-triggered events.', 'capture': {'image': True, 'video': True}, '_capture_comment': 'Set image and/or video to true to choose what is captured.', 'video': {'trigger': 'detected', '_trigger_comment': 'detected = start video only when a trigger class is detected; always = start video on every PIR trigger.'}}, 'continuous': {'_comment': 'Settings for continuous detection operation.', 'cooldown_seconds': 10.0, '_cooldown_comment': 'Minimum time between two continuous capture events.', 'capture': {'image': True, 'video': False}, '_capture_comment': 'Set image and/or video to true to choose what is captured.'}, 'video': {'_comment': 'Global video recording settings shared by PIR and continuous modes.', 'seconds': 10, '_seconds_comment': 'Video duration in seconds.', 'fps': 15, '_fps_comment': 'Video recording FPS. The value is clamped between camera.fps_min and camera.fps_max.', 'codec': 'mp4v', '_codec_comment': 'OpenCV four-character video codec used for saved video files.'}, 'overlay': {'_comment': 'Information drawn onto the live preview and saved media.', 'show_fps': True, '_show_fps_comment': 'Show the current FPS in the top-left corner.', 'show_datetime': True, '_show_datetime_comment': 'Show the Raspberry Pi system date and time in the top-left corner.', 'show_class_name': True, '_show_class_name_comment': 'Show the detected class name next to each bounding box.', 'show_confidence': True, '_show_confidence_comment': 'Show the detection confidence next to each class name.', 'font_scale': 0.7, '_font_scale_comment': 'OpenCV text size.', 'line_thickness': 2, '_line_thickness_comment': 'Bounding-box and text line thickness.', 'background': True, '_background_comment': 'Draw a coloured background behind detection labels.'}, 'detection': {'_comment': 'Detection classes and bounding-box colours.', 'trigger_classes': ['bear', 'boar', 'wolf', 'deer'], '_trigger_classes_comment': 'Only these classes can trigger capture events. Other detected classes can still have boxes displayed.', 'box_colors': {'bear': '#FF8C00', 'boar': '#8B4513', 'wolf': '#808080', 'deer': '#D2691E'}, '_box_colors_comment': 'Per-class bounding-box colours in #RRGGBB format. Classes without a custom colour use default_box_color.', 'default_box_color': '#00FF00', '_default_box_color_comment': 'Bounding-box colour used for any class not listed in box_colors.'}, 'output': {'directory': 'outputs', '_directory_comment': 'Root directory where captured images and videos are stored.'}, 'preview': {'_comment': 'Live preview window settings.', 'enabled': True, '_enabled_comment': 'Show the live camera preview window.', 'window_name': 'Wildlife Detector', '_window_name_comment': 'Title of the live preview window.', 'width': 1280, '_width_comment': 'Preview window width in pixels.', 'height': 720, '_height_comment': 'Preview window height in pixels.'}}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            json.dumps(DEFAULT_CONFIG, indent=4),
            encoding="utf-8"
        )
        return DEFAULT_CONFIG

    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON configuration: {CONFIG_PATH}\n{exc}"
        ) from exc


def get_config(config, *keys, default=None):
    value = config

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def camera_fps(config) -> float:
    minimum = float(
        get_config(config, "camera", "fps_min", default=5)
    )
    maximum = float(
        get_config(config, "camera", "fps_max", default=30)
    )
    requested = float(
        get_config(config, "camera", "fps", default=15)
    )

    minimum = max(1.0, minimum)
    maximum = max(minimum, maximum)

    return clamp(requested, minimum, maximum)


def video_fps(config) -> float:
    requested = float(
        get_config(
            config,
            "video",
            "fps",
            default=camera_fps(config)
        )
    )
    return clamp(
        requested,
        float(get_config(config, "camera", "fps_min", default=5)),
        float(get_config(config, "camera", "fps_max", default=30))
    )


def normalise_class(name: str) -> str:
    return str(name).strip().casefold()


def target_classes(config) -> set[str]:
    return {
        normalise_class(name)
        for name in get_config(
            config,
            "detection",
            "trigger_classes",
            default=[]
        )
    }


# ---------------------------------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------------------------------

def load_labels(config) -> list[str]:
    labels_path = Path(
        get_config(
            config,
            "model",
            "labels",
            default="labels.txt"
        )
    )

    if not labels_path.is_absolute():
        labels_path = BASE_DIR / labels_path

    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}"
        )

    labels = [
        line.strip()
        for line in labels_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not labels:
        raise RuntimeError(f"No class labels found in {labels_path}")

    return labels


# ---------------------------------------------------------------------------
# COLOUR / DRAWING
# ---------------------------------------------------------------------------

def hex_to_bgr(value: str):
    if not isinstance(value, str):
        return (0, 255, 0)

    text = value.strip().lstrip("#")

    if len(text) != 6:
        return (0, 255, 0)

    try:
        red = int(text[0:2], 16)
        green = int(text[2:4], 16)
        blue = int(text[4:6], 16)
        return (blue, green, red)
    except ValueError:
        return (0, 255, 0)


def class_colour(config, class_name: str):
    colours = get_config(
        config,
        "detection",
        "box_colors",
        default={}
    ) or {}

    default_colour = get_config(
        config,
        "detection",
        "default_box_color",
        default="#00FF00"
    )

    for configured_name, colour in colours.items():
        if normalise_class(configured_name) == normalise_class(class_name):
            return hex_to_bgr(colour)

    return hex_to_bgr(default_colour)


def draw_overlay(
    frame,
    config,
    detections,
    fps_value: float,
    main_width: int,
    main_height: int
):
    """
    Draw all detections.

    Hailo returns normalized coordinates:
        y0, x0, y1, x1, confidence

    The coordinates are converted to the output frame size.
    """

    overlay = get_config(config, "overlay", default={}) or {}

    font_scale = float(overlay.get("font_scale", 0.70))
    thickness = int(overlay.get("line_thickness", 2))

    show_class = bool(overlay.get("show_class_name", True))
    show_confidence = bool(overlay.get("show_confidence", True))

    for class_name, score, bbox in detections:
        x0, y0, x1, y1 = bbox
        colour = class_colour(config, class_name)

        cv2.rectangle(
            frame,
            (x0, y0),
            (x1, y1),
            colour,
            thickness
        )

        label_parts = []

        if show_class:
            label_parts.append(class_name)

        if show_confidence:
            label_parts.append(f"{score:.2f}")

        if label_parts:
            label = " ".join(label_parts)

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )

            label_y = max(text_height + baseline + 2, y0)

            if overlay.get("background", True):
                cv2.rectangle(
                    frame,
                    (x0, label_y - text_height - baseline - 2),
                    (x0 + text_width + 4, label_y + 2),
                    colour,
                    -1
                )

            cv2.putText(
                frame,
                label,
                (x0 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

    info_y = 28

    if bool(overlay.get("show_fps", True)):
        cv2.putText(
            frame,
            f"FPS: {fps_value:.1f}",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        info_y += 30

    if bool(overlay.get("show_datetime", True)):
        cv2.putText(
            frame,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return frame


# ---------------------------------------------------------------------------
# HAILO DETECTION
# ---------------------------------------------------------------------------

def extract_detections(
    hailo_output,
    labels: list[str],
    frame_width: int,
    frame_height: int,
    confidence_threshold: float
):
    """
    Convert the Picamera2 Hailo helper output into:

        [(class_name, confidence, (x0, y0, x1, y1)), ...]

    The official Picamera2 Hailo example uses the same normalized
    [y0, x0, y1, x1, score] format for Hailo post-process output.
    """

    detections = []

    for class_id, class_detections in enumerate(hailo_output):
        if class_id >= len(labels):
            continue

        class_name = labels[class_id]

        for detection in class_detections:
            if len(detection) < 5:
                continue

            score = float(detection[4])

            if score < confidence_threshold:
                continue

            y0, x0, y1, x1 = map(float, detection[:4])

            x0 = int(clamp(x0 * frame_width, 0, frame_width - 1))
            y0 = int(clamp(y0 * frame_height, 0, frame_height - 1))
            x1 = int(clamp(x1 * frame_width, 0, frame_width - 1))
            y1 = int(clamp(y1 * frame_height, 0, frame_height - 1))

            if x1 <= x0 or y1 <= y0:
                continue

            detections.append(
                (
                    class_name,
                    score,
                    (x0, y0, x1, y1)
                )
            )

    return detections


def target_detections(detections, config):
    targets = target_classes(config)

    return [
        detection
        for detection in detections
        if normalise_class(detection[0]) in targets
    ]


# ---------------------------------------------------------------------------
# OUTPUT DIRECTORIES
# ---------------------------------------------------------------------------

def output_root(config) -> Path:
    directory = Path(
        get_config(
            config,
            "output",
            "directory",
            default="outputs"
        )
    )

    if not directory.is_absolute():
        directory = BASE_DIR / directory

    directory.mkdir(parents=True, exist_ok=True)

    for class_name in target_classes(config):
        (directory / class_name / "image").mkdir(
            parents=True,
            exist_ok=True
        )
        (directory / class_name / "video").mkdir(
            parents=True,
            exist_ok=True
        )

    return directory


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def class_names_from_detections(detections):
    return {
        normalise_class(detection[0])
        for detection in detections
    }


def destination_paths(
    root: Path,
    media_type: str,
    classes: set[str],
    extension: str
):
    filename = f"{timestamp()}.{extension}"

    return [
        root / class_name / media_type / filename
        for class_name in sorted(classes)
    ]


def save_image(frame, root: Path, classes: set[str]):
    if not classes:
        return []

    paths = destination_paths(
        root,
        "image",
        classes,
        "jpg"
    )

    temporary = root / f".temporary_{timestamp()}.jpg"

    try:
        success = cv2.imwrite(
            str(temporary),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        if not success:
            raise RuntimeError(
                f"Could not write image: {temporary}"
            )

        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temporary, path)

        return paths

    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CAMERA / FPS
# ---------------------------------------------------------------------------

class FPSCounter:
    def __init__(self):
        self.last_time = time.monotonic()
        self.frames = 0
        self.value = 0.0

    def update(self):
        self.frames += 1

        now = time.monotonic()
        elapsed = now - self.last_time

        if elapsed >= 1.0:
            self.value = self.frames / elapsed
            self.frames = 0
            self.last_time = now

        return self.value


def configure_autofocus(camera, config):
    """
    Configure autofocus for cameras that expose libcamera autofocus controls,
    such as Camera Module 3.

    Unsupported autofocus controls are handled gracefully so fixed-focus
    cameras can still run.
    """
    autofocus = get_config(config, "autofocus", default={}) or {}

    if not bool(autofocus.get("enabled", True)):
        print("Autofocus disabled.")
        return

    mode_name = str(autofocus.get("mode", "continuous")).casefold()
    range_name = str(autofocus.get("range", "normal")).casefold()

    mode_map = {
        "continuous": getattr(controls.AfModeEnum, "Continuous", None),
        "auto": getattr(controls.AfModeEnum, "Auto", None),
        "manual": getattr(controls.AfModeEnum, "Manual", None),
    }

    range_map = {
        "normal": getattr(controls.AfRangeEnum, "Normal", None),
        "macro": getattr(controls.AfRangeEnum, "Macro", None),
        "full": getattr(controls.AfRangeEnum, "Full", None),
    }

    selected_mode = mode_map.get(mode_name)
    selected_range = range_map.get(range_name)

    if selected_mode is None:
        raise ValueError(
            "autofocus.mode must be 'continuous', 'auto', or 'manual'."
        )

    if selected_range is None:
        raise ValueError(
            "autofocus.range must be 'normal', 'macro', or 'full'."
        )

    camera.set_controls({
        "AfMode": selected_mode,
        "AfRange": selected_range,
    })

    print(
        f"Autofocus: enabled, mode={mode_name}, range={range_name}"
    )


def create_camera(config):
    camera = Picamera2()

    width = int(
        get_config(config, "camera", "width", default=1280)
    )
    height = int(
        get_config(config, "camera", "height", default=960)
    )

    fps = camera_fps(config)

    main = {
        "size": (width, height),
        "format": "XRGB8888"
    }

    # The HEF requires 640x640x3 UINT8.
    # RGB888 is supplied directly to Hailo by the Picamera2 Hailo helper.
    lores = {
        "size": (640, 640),
        "format": "RGB888"
    }

    controls = {
        "FrameRate": fps
    }

    camera_config = camera.create_preview_configuration(
        main=main,
        lores=lores,
        controls=controls
    )

    camera.configure(camera_config)
    camera.start()

    configure_autofocus(camera, config)

    time.sleep(2)

    return camera


# ---------------------------------------------------------------------------
# LIVE PREVIEW
# ---------------------------------------------------------------------------

def preview_enabled(config):
    return bool(get_config(config, "preview", "enabled", default=True))


def show_preview(frame, config, detections=None, fps=0.0):
    if not preview_enabled(config):
        return False

    display = frame.copy()
    height, width = display.shape[:2]

    if detections is not None:
        display = draw_overlay(
            display, config, detections, fps, width, height
        )

    preview_width = int(get_config(config, "preview", "width", default=1280))
    preview_height = int(get_config(config, "preview", "height", default=720))
    if preview_width > 0 and preview_height > 0:
        display = cv2.resize(display, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

    window_name = str(get_config(config, "preview", "window_name", default="Wildlife Detector"))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display)

    key = cv2.waitKey(1) & 0xFF
    return key == ord("q") or key == 27


def close_preview(config):
    if preview_enabled(config):
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


# ---------------------------------------------------------------------------
# VIDEO
# ---------------------------------------------------------------------------

def create_video_writer(path, width, height, fps, config):
    codec_name = str(
        get_config(
            config,
            "video",
            "codec",
            default="mp4v"
        )
    )

    if len(codec_name) != 4:
        codec_name = "mp4v"

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec_name),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not open the video writer. "
            "The selected OpenCV codec may not be available."
        )

    return writer


def record_video(
    camera,
    hailo,
    labels,
    config,
    root: Path,
    seconds: float,
    first_target_required: bool = False
):
    width = int(
        get_config(config, "camera", "width", default=1280)
    )
    height = int(
        get_config(config, "camera", "height", default=960)
    )

    fps = video_fps(config)
    confidence = float(
        get_config(config, "model", "confidence", default=0.50)
    )

    temporary = root / f".temporary_{timestamp()}.mp4"

    writer = create_video_writer(
        temporary,
        width,
        height,
        fps,
        config
    )

    seen_target_classes = set()
    fps_counter = FPSCounter()
    start = time.monotonic()

    try:
        while (
            not STOP_EVENT.is_set()
            and time.monotonic() - start < seconds
        ):
            main_frame = camera.capture_array("main")
            inference_frame = camera.capture_array("lores")

            hailo_output = hailo.run(inference_frame)

            detections = extract_detections(
                hailo_output,
                labels,
                width,
                height,
                confidence
            )

            targets = target_detections(
                detections,
                config
            )

            seen_target_classes.update(
                class_names_from_detections(targets)
            )

            current_fps = fps_counter.update()

            annotated = draw_overlay(
                main_frame.copy(),
                config,
                detections,
                current_fps,
                width,
                height
            )

            if show_preview(annotated, config, None, current_fps):
                STOP_EVENT.set()
                break

            writer.write(annotated)

        writer.release()

        if first_target_required and not seen_target_classes:
            temporary.unlink(missing_ok=True)
            return []

        if not seen_target_classes:
            # "always" PIR recording can legitimately contain no target.
            # Keep the recording in an explicit folder rather than losing it.
            unclassified_dir = root / "_unclassified" / "video"
            unclassified_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            destination = (
                unclassified_dir
                / f"{timestamp()}.mp4"
            )
            shutil.copy2(temporary, destination)
            return [destination]

        paths = destination_paths(
            root,
            "video",
            seen_target_classes,
            "mp4"
        )

        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temporary, path)

        return paths

    finally:
        writer.release()
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# PIR MODE
# ---------------------------------------------------------------------------

def run_pir_mode(
    camera,
    hailo,
    labels,
    config,
    root
):
    gpio_pin = int(
        get_config(config, "pir", "gpio_pin", default=13)
    )

    cooldown = float(
        get_config(
            config,
            "pir",
            "cooldown_seconds",
            default=2.0
        )
    )

    capture_image = bool(
        get_config(
            config,
            "pir",
            "capture",
            "image",
            default=True
        )
    )

    capture_video = bool(
        get_config(
            config,
            "pir",
            "capture",
            "video",
            default=False
        )
    )

    video_trigger = str(
        get_config(
            config,
            "pir",
            "video",
            "trigger",
            default="detected"
        )
    ).casefold()

    if video_trigger not in {"detected", "always"}:
        raise ValueError(
            "pir.video.trigger must be 'detected' or 'always'."
        )

    video_seconds = float(
        get_config(
            config,
            "video",
            "seconds",
            default=10
        )
    )

    pir = MotionSensor(gpio_pin)
    last_trigger = 0.0

    print(f"PIR mode active. GPIO: {gpio_pin}")
    print("Waiting for PIR trigger...")

    confidence = float(
        get_config(
            config,
            "model",
            "confidence",
            default=0.50
        )
    )

    fps_counter = FPSCounter()

    while not STOP_EVENT.is_set():
        # Keep the live preview responsive while waiting for PIR.
        main_frame = camera.capture_array("main")
        current_fps = fps_counter.update()

        if show_preview(main_frame, config, None, current_fps):
            STOP_EVENT.set()
            break

        if not pir.motion_detected:
            continue

        now = time.monotonic()

        if now - last_trigger < cooldown:
            continue

        last_trigger = now

        inference_frame = camera.capture_array("lores")
        hailo_output = hailo.run(inference_frame)

        width = main_frame.shape[1]
        height = main_frame.shape[0]

        detections = extract_detections(
            hailo_output,
            labels,
            width,
            height,
            confidence
        )

        targets = target_detections(detections, config)
        target_names = class_names_from_detections(targets)

        # -------------------------------------------------------------------
        # TARGET CLASS DETECTED — FUTURE ACTIONS
        # -------------------------------------------------------------------
        if target_names:
            # At least one configured target class was detected.
            # Future actions can be added here.
            pass

        annotated = draw_overlay(
            main_frame.copy(),
            config,
            detections,
            camera_fps(config),
            width,
            height
        )

        if show_preview(annotated, config, None, camera_fps(config)):
            STOP_EVENT.set()
            break

        print(
            "PIR triggered. "
            f"Targets: {', '.join(sorted(target_names)) or 'none'}"
        )

        # A PIR trigger always captures a frame internally for detection.
        # capture.image only decides whether that frame is saved as a photo.
        if capture_image and target_names:
            paths = save_image(
                annotated,
                root,
                target_names
            )

            print(
                "Image saved:",
                ", ".join(str(path) for path in paths)
            )

        should_record = (
            capture_video
            and (
                video_trigger == "always"
                or (
                    video_trigger == "detected"
                    and bool(target_names)
                )
            )
        )

        if should_record:
            print(
                f"Starting {video_seconds:.1f}s video..."
            )

            paths = record_video(
                camera,
                hailo,
                labels,
                config,
                root,
                video_seconds,
                first_target_required=(
                    video_trigger == "detected"
                )
            )

            if paths:
                print(
                    "Video saved:",
                    ", ".join(str(path) for path in paths)
                )


# ---------------------------------------------------------------------------
# CONTINUOUS MODE
# ---------------------------------------------------------------------------

def run_continuous_mode(
    camera,
    hailo,
    labels,
    config,
    root
):
    capture_image = bool(
        get_config(
            config,
            "continuous",
            "capture",
            "image",
            default=True
        )
    )

    capture_video = bool(
        get_config(
            config,
            "continuous",
            "capture",
            "video",
            default=False
        )
    )

    cooldown = float(
        get_config(
            config,
            "continuous",
            "cooldown_seconds",
            default=10
        )
    )

    video_seconds = float(
        get_config(
            config,
            "video",
            "seconds",
            default=10
        )
    )

    confidence = float(
        get_config(
            config,
            "model",
            "confidence",
            default=0.50
        )
    )

    last_capture = 0.0
    fps_counter = FPSCounter()

    print("Continuous mode active.")

    while not STOP_EVENT.is_set():
        main_frame = camera.capture_array("main")
        inference_frame = camera.capture_array("lores")

        hailo_output = hailo.run(inference_frame)

        width = main_frame.shape[1]
        height = main_frame.shape[0]

        detections = extract_detections(
            hailo_output,
            labels,
            width,
            height,
            confidence
        )

        targets = target_detections(
            detections,
            config
        )

        current_fps = fps_counter.update()

        if show_preview(main_frame, config, detections, current_fps):
            STOP_EVENT.set()
            break

        if not targets:
            continue

        now = time.monotonic()

        if now - last_capture < cooldown:
            continue

        last_capture = now

        target_names = class_names_from_detections(targets)

        print(
            "Target detected:",
            ", ".join(sorted(target_names))
        )

        # -------------------------------------------------------------------
        # TARGET CLASS DETECTED — FUTURE ACTIONS
        # -------------------------------------------------------------------
        if target_names:
            # At least one configured target class was detected.
            # Future actions can be added here.
            pass

        # Image and video are independent. Both can be true.
        if capture_image:
            annotated = draw_overlay(
                main_frame.copy(),
                config,
                detections,
                current_fps,
                width,
                height
            )

            paths = save_image(
                annotated,
                root,
                target_names
            )

            print(
                "Image saved:",
                ", ".join(str(path) for path in paths)
            )

        if capture_video:
            paths = record_video(
                camera,
                hailo,
                labels,
                config,
                root,
                video_seconds,
                first_target_required=True
            )

            if paths:
                print(
                    "Video saved:",
                    ", ".join(str(path) for path in paths)
                )

        if not capture_image and not capture_video:
            print(
                "Detection found, but both continuous capture options "
                "are disabled."
            )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def install_signal_handlers():
    def stop_handler(signum, frame):
        STOP_EVENT.set()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)


def main():
    install_signal_handlers()

    config = load_config()

    # Start the local configuration web page.
    # Connect to the Raspberry Pi's Wi-Fi/LAN and open the printed URL.
    web_server = start_web_server()

    model_path = Path(
        get_config(
            config,
            "model",
            "path",
            default="yolov11l.hef"
        )
    )

    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    if not model_path.is_file():
        raise FileNotFoundError(
            f"HEF model not found: {model_path}"
        )

    labels = load_labels(config)

    if len(labels) != 7:
        raise RuntimeError(
            f"Expected 7 labels for this HEF, "
            f"but labels.txt contains {len(labels)} labels."
        )

    root = output_root(config)

    mode = str(
        get_config(
            config,
            "mode",
            default="pir"
        )
    ).casefold()

    print("=" * 60)
    print("Raspberry Pi 5 + Hailo-8 Wildlife Detector")
    print("=" * 60)
    print(f"Mode:       {mode}")
    print(f"HEF:        {model_path}")
    print(f"Labels:     {', '.join(labels)}")
    print(f"Confidence: {get_config(config, 'model', 'confidence', default=0.50)}")
    print(f"Camera FPS: {camera_fps(config):.1f}")
    print(f"Output:     {root}")
    print("=" * 60)

    camera = create_camera(config)

    try:
        # Picamera2's Hailo helper uses the HEF's input shape and
        # Hailo post-processing output.
        with Hailo(str(model_path)) as hailo:
            model_height, model_width, model_channels = (
                hailo.get_input_shape()
            )

            print(
                "Hailo input:",
                f"{model_width}x{model_height}x{model_channels}"
            )

            if (
                model_width != 640
                or model_height != 640
                or model_channels != 3
            ):
                raise RuntimeError(
                    "This detector expects a 640x640x3 Hailo input, "
                    f"but the HEF reports "
                    f"{model_width}x{model_height}x{model_channels}."
                )

            if mode == "pir":
                run_pir_mode(
                    camera,
                    hailo,
                    labels,
                    config,
                    root
                )

            elif mode == "continuous":
                run_continuous_mode(
                    camera,
                    hailo,
                    labels,
                    config,
                    root
                )

            else:
                raise ValueError(
                    "mode must be 'pir' or 'continuous'."
                )

    finally:
        close_preview(config)
        camera.stop()
        web_server.shutdown()
        web_server.server_close()

    print("Detector stopped.")


if __name__ == "__main__":
    main()
