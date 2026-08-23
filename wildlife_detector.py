#!/usr/bin/env python3
"""
Raspberry Pi Wildlife Detector

Two operating modes:
1. PIR mode:
   - PIR triggers a still image.
   - The still image is analysed for the configured trigger classes.
   - If a trigger class is detected, an optional video can be recorded.
   - Alternatively, video can be recorded on every PIR trigger.

2. Continuous mode:
   - The camera runs continuously.
   - When a configured trigger class is detected, an image or video is captured.

Configuration is kept in config.json so a local web interface can read it later.
"""

from __future__ import annotations

import json
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path
from threading import Event

import cv2
from gpiozero import MotionSensor
from picamera2 import Picamera2
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# FILES / PATHS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
OUTPUTS_DIR = BASE_DIR / "outputs"

# The JSON file is intentionally plain and human-readable.
# A local web page can read this file later.


# ---------------------------------------------------------------------------
# DEFAULT CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "mode": "pir",
    "model": {
        "path": "model.pt",
        "confidence": 0.50,
        "iou": 0.45,
        "image_size": 640
    },
    "camera": {
        "width": 1280,
        "height": 720,

        # Requested FPS range.
        # The selected FPS is clamped between these two values.
        "fps_min": 5,
        "fps_max": 30,
        "fps": 15
    },
    "pir": {
        "gpio_pin": 13,

        # PIR trigger cooldown prevents repeated triggers from creating
        # many files during one motion event.
        "cooldown_seconds": 2.0,

        # Take one still image whenever PIR fires.
        "capture_trigger_image": True,

        # "detected" = record only when a target class is found in the PIR image.
        # "always"   = record after every PIR trigger.
        "video_trigger": "detected"
    },
    "continuous": {
        # "image" or "video"
        "capture_type": "image",

        # Minimum time between continuous detections.
        "cooldown_seconds": 10.0,

        # Video length when capture_type is "video".
        "video_seconds": 10
    },
    "video": {
        "enabled": True,
        "seconds": 10,

        # Video FPS is also clamped to camera.fps_min / camera.fps_max.
        "fps": 15
    },
    "image": {
        "enabled": True
    },
    "overlay": {
        "show_fps": True,
        "show_datetime": True,
        "show_class_name": True,
        "show_confidence": True,
        "font_scale": 0.65,
        "line_thickness": 2
    },
    "detection": {
        # Only these classes can trigger an event.
        "trigger_classes": [
            "bear",
            "boar",
            "wolf",
            "deer"
        ],

        # Per-class box colours in BGR format.
        # You can also use "#RRGGBB" strings; they are converted automatically.
        # Classes not listed here use the default green colour.
        "box_colors": {
            "bear": "#FF8C00",
            "boar": "#8B4513",
            "wolf": "#808080",
            "deer": "#D2691E"
        },

        "default_box_color": "#00FF00"
    },
    "output": {
        "directory": "outputs"
    }
}


# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------

STOP_EVENT = Event()


def load_config() -> dict:
    """Create config.json if necessary and return its contents."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            json.dumps(DEFAULT_CONFIG, indent=4, ensure_ascii=False),
            encoding="utf-8"
        )
        return DEFAULT_CONFIG.copy()

    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON configuration: {CONFIG_PATH}\n{exc}"
        ) from exc


def get_nested(config: dict, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def configured_fps(config: dict, requested=None) -> float:
    fps_min = float(get_nested(config, "camera", "fps_min", default=5))
    fps_max = float(get_nested(config, "camera", "fps_max", default=30))

    if fps_min <= 0:
        fps_min = 1.0
    if fps_max < fps_min:
        fps_max = fps_min

    if requested is None:
        requested = get_nested(config, "camera", "fps", default=fps_min)

    return clamp(float(requested), fps_min, fps_max)


def video_fps(config: dict) -> float:
    requested = get_nested(
        config, "video", "fps",
        default=get_nested(config, "camera", "fps", default=15)
    )
    return configured_fps(config, requested)


def ensure_output_directories(config: dict):
    output_name = get_nested(
        config, "output", "directory", default="outputs"
    )
    output_root = (BASE_DIR / output_name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for class_name in get_nested(
        config, "detection", "trigger_classes", default=[]
    ):
        class_dir = output_root / str(class_name).strip().lower()
        (class_dir / "image").mkdir(parents=True, exist_ok=True)
        (class_dir / "video").mkdir(parents=True, exist_ok=True)

    return output_root


def class_name_normalized(name: str) -> str:
    return str(name).strip().casefold()


def hex_to_bgr(value: str):
    """Convert #RRGGBB to OpenCV BGR."""
    if not isinstance(value, str):
        return 0, 255, 0

    text = value.strip().lstrip("#")
    if len(text) != 6:
        return 0, 255, 0

    try:
        red = int(text[0:2], 16)
        green = int(text[2:4], 16)
        blue = int(text[4:6], 16)
        return blue, green, red
    except ValueError:
        return 0, 255, 0


def box_colour(config: dict, class_name: str):
    colours = get_nested(
        config, "detection", "box_colors", default={}
    ) or {}

    default_colour = get_nested(
        config, "detection", "default_box_color", default="#00FF00"
    )

    wanted = None
    for configured_name, colour in colours.items():
        if class_name_normalized(configured_name) == class_name_normalized(class_name):
            wanted = colour
            break

    return hex_to_bgr(wanted if wanted is not None else default_colour)


def trigger_class_set(config: dict):
    return {
        class_name_normalized(name)
        for name in get_nested(
            config, "detection", "trigger_classes", default=[]
        )
    }


def timestamp_string():
    # Raspberry Pi system date/time is used directly.
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def display_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def install_signal_handlers():
    def stop_handler(signum, frame):
        STOP_EVENT.set()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)


# ---------------------------------------------------------------------------
# DETECTION
# ---------------------------------------------------------------------------

class Detection:
    def __init__(self, class_name, confidence, xyxy):
        self.class_name = class_name
        self.confidence = float(confidence)
        self.xyxy = tuple(int(v) for v in xyxy)


def detect_frame(model, frame, config) -> list[Detection]:
    results = model.predict(
        source=frame,
        conf=float(get_nested(config, "model", "confidence", default=0.50)),
        iou=float(get_nested(config, "model", "iou", default=0.45)),
        imgsz=int(get_nested(config, "model", "image_size", default=640)),
        verbose=False
    )

    detections = []

    for result in results:
        names = result.names

        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            if isinstance(names, dict):
                name = names.get(class_id, str(class_id))
            else:
                name = names[class_id]

            detections.append(
                Detection(
                    class_name=str(name),
                    confidence=confidence,
                    xyxy=xyxy
                )
            )

    return detections


def target_detections(detections, config):
    targets = trigger_class_set(config)
    return [
        detection
        for detection in detections
        if class_name_normalized(detection.class_name) in targets
    ]


# ---------------------------------------------------------------------------
# OVERLAY
# ---------------------------------------------------------------------------

def draw_overlay(frame, config, fps_value, detections):
    overlay_config = get_nested(config, "overlay", default={}) or {}

    show_fps = bool(overlay_config.get("show_fps", True))
    show_datetime = bool(overlay_config.get("show_datetime", True))
    show_class_name = bool(overlay_config.get("show_class_name", True))
    show_confidence = bool(overlay_config.get("show_confidence", True))

    font_scale = float(overlay_config.get("font_scale", 0.65))
    thickness = int(overlay_config.get("line_thickness", 2))

    # Draw every detected box, not only trigger classes.
    # This means the animal's bounding box is visible in saved media.
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        colour = box_colour(config, detection.class_name)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            colour,
            thickness
        )

        label_parts = []

        if show_class_name:
            label_parts.append(detection.class_name)

        if show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")

        if label_parts:
            label = " ".join(label_parts)

            text_y = max(20, y1 - 8)
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )

            cv2.rectangle(
                frame,
                (x1, max(0, text_y - text_height - baseline)),
                (x1 + text_width, text_y + baseline),
                colour,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

    # Top-left information.
    info_y = 28
    if show_fps:
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

    if show_datetime:
        cv2.putText(
            frame,
            display_datetime(),
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return frame


# ---------------------------------------------------------------------------
# OUTPUT HANDLING
# ---------------------------------------------------------------------------

def output_file_paths(output_root, media_type, detected_classes, extension):
    """
    Return one destination path per detected target class.

    A single captured file is copied into every detected class directory.
    Example:
      outputs/bear/image/20260822_220000_123456.jpg
      outputs/wolf/image/20260822_220000_123456.jpg
    """
    filename = f"{timestamp_string()}.{extension}"

    paths = []
    for class_name in sorted(
        {class_name_normalized(name) for name in detected_classes}
    ):
        destination = output_root / class_name / media_type / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        paths.append(destination)

    return paths


def save_image(frame, output_root, detected_classes):
    paths = output_file_paths(
        output_root,
        "image",
        detected_classes,
        "jpg"
    )

    if not paths:
        return []

    temporary = output_root / f".capture_{timestamp_string()}.jpg"

    try:
        if not cv2.imwrite(
            str(temporary),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        ):
            raise RuntimeError(f"Failed to write image: {temporary}")

        for destination in paths:
            shutil.copy2(temporary, destination)

        return paths
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# VIDEO RECORDING
# ---------------------------------------------------------------------------

def create_video_writer(path, width, height, fps):
    # mp4v is widely available with OpenCV.
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not open the MP4 video writer. "
            "Check the OpenCV codecs installed on the Raspberry Pi."
        )

    return writer


def record_video(camera, model, config, output_root, seconds):
    fps = video_fps(config)
    width = int(get_nested(config, "camera", "width", default=1280))
    height = int(get_nested(config, "camera", "height", default=720))

    # First frame is analysed before creating the final class-specific paths.
    first_frame = camera.capture_array()
    first_detections = detect_frame(model, first_frame, config)
    first_targets = target_detections(first_detections, config)

    # If the caller requests a detection-based video, no target means no file.
    if not first_targets:
        return []

    detected_classes = {
        class_name_normalized(d.class_name)
        for d in first_targets
    }

    paths = output_file_paths(
        output_root,
        "video",
        detected_classes,
        "mp4"
    )

    if not paths:
        return []

    temporary = output_root / f".recording_{timestamp_string()}.mp4"
    writer = create_video_writer(temporary, width, height, fps)

    start = time.monotonic()

    try:
        while (
            not STOP_EVENT.is_set()
            and time.monotonic() - start < seconds
        ):
            frame = camera.capture_array()
            detections = detect_frame(model, frame, config)

            # Include newly detected target classes as they appear.
            current_targets = target_detections(detections, config)
            detected_classes.update(
                class_name_normalized(d.class_name)
                for d in current_targets
            )

            annotated = draw_overlay(
                frame.copy(),
                config,
                fps,
                detections
            )
            writer.write(annotated)

        writer.release()

        # A video can contain multiple classes. The same file is copied
        # into every corresponding class directory.
        final_paths = output_file_paths(
            output_root,
            "video",
            detected_classes,
            "mp4"
        )

        for destination in final_paths:
            shutil.copy2(temporary, destination)

        # Remove any duplicate first set generated before newly detected
        # classes were discovered.
        for old_path in paths:
            if old_path not in final_paths:
                old_path.unlink(missing_ok=True)

        return final_paths

    finally:
        writer.release()
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CAMERA
# ---------------------------------------------------------------------------

def create_camera(config):
    camera = Picamera2()

    width = int(get_nested(config, "camera", "width", default=1280))
    height = int(get_nested(config, "camera", "height", default=720))
    fps = configured_fps(config)

    video_config = camera.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameRate": fps}
    )

    camera.configure(video_config)
    camera.start()

    # Allow auto exposure / white balance to settle.
    time.sleep(2)

    return camera


# ---------------------------------------------------------------------------
# PIR MODE
# ---------------------------------------------------------------------------

def run_pir_mode(camera, model, config, output_root):
    pin = int(get_nested(config, "pir", "gpio_pin", default=13))
    cooldown = float(
        get_nested(config, "pir", "cooldown_seconds", default=2.0)
    )

    capture_trigger_image = bool(
        get_nested(config, "pir", "capture_trigger_image", default=True)
    )

    video_enabled = bool(
        get_nested(config, "video", "enabled", default=True)
    )

    video_trigger = str(
        get_nested(config, "pir", "video_trigger", default="detected")
    ).casefold()

    video_seconds = float(
        get_nested(config, "video", "seconds", default=10)
    )

    pir = MotionSensor(pin)
    last_trigger = 0.0

    print(f"PIR mode active. GPIO: {pin}")
    print("Waiting for PIR trigger...")

    while not STOP_EVENT.is_set():
        pir.wait_for_motion()

        if STOP_EVENT.is_set():
            break

        now = time.monotonic()
        if now - last_trigger < cooldown:
            pir.wait_for_no_motion()
            continue

        last_trigger = now

        frame = camera.capture_array()
        detections = detect_frame(model, frame, config)
        targets = target_detections(detections, config)

        annotated = draw_overlay(
            frame.copy(),
            config,
            configured_fps(config),
            detections
        )

        target_names = sorted(
            {
                class_name_normalized(d.class_name)
                for d in targets
            }
        )

        if capture_trigger_image and get_nested(
            config, "image", "enabled", default=True
        ):
            # The PIR trigger image is saved only if at least one target
            # class was detected.
            if target_names:
                paths = save_image(
                    annotated,
                    output_root,
                    target_names
                )
                print(
                    "Target detected: "
                    + ", ".join(target_names)
                )
                print("Image:", ", ".join(map(str, paths)))

        should_record = False

        if video_enabled:
            if video_trigger == "always":
                should_record = True
            elif video_trigger == "detected" and target_names:
                should_record = True

        if should_record:
            print(
                "Video trigger: "
                + (
                    ", ".join(target_names)
                    if target_names
                    else "PIR trigger"
                )
            )

            if video_trigger == "detected":
                # The initial trigger image already proved that a target
                # exists, so record_video can safely record.
                paths = record_video(
                    camera,
                    model,
                    config,
                    output_root,
                    video_seconds
                )
            else:
                # Always mode records regardless of the first image.
                paths = record_video_always(
                    camera,
                    model,
                    config,
                    output_root,
                    video_seconds
                )

            if paths:
                print("Video:", ", ".join(map(str, paths)))

        pir.wait_for_no_motion()


def record_video_always(camera, model, config, output_root, seconds):
    """
    Record a video after every PIR trigger.

    The final class directories are decided from all target detections
    encountered during the recording.
    """
    fps = video_fps(config)
    width = int(get_nested(config, "camera", "width", default=1280))
    height = int(get_nested(config, "camera", "height", default=720))

    temporary = output_root / f".recording_{timestamp_string()}.mp4"
    writer = create_video_writer(temporary, width, height, fps)

    detected_classes = set()
    start = time.monotonic()

    try:
        while (
            not STOP_EVENT.is_set()
            and time.monotonic() - start < seconds
        ):
            frame = camera.capture_array()
            detections = detect_frame(model, frame, config)
            targets = target_detections(detections, config)

            detected_classes.update(
                class_name_normalized(d.class_name)
                for d in targets
            )

            annotated = draw_overlay(
                frame.copy(),
                config,
                fps,
                detections
            )
            writer.write(annotated)

        writer.release()

        if not detected_classes:
            print("PIR video contained no configured target classes.")
            temporary.unlink(missing_ok=True)
            return []

        paths = output_file_paths(
            output_root,
            "video",
            detected_classes,
            "mp4"
        )

        for destination in paths:
            shutil.copy2(temporary, destination)

        return paths

    finally:
        writer.release()
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CONTINUOUS MODE
# ---------------------------------------------------------------------------

def run_continuous_mode(camera, model, config, output_root):
    capture_type = str(
        get_nested(config, "continuous", "capture_type", default="image")
    ).casefold()

    cooldown = float(
        get_nested(config, "continuous", "cooldown_seconds", default=10)
    )

    video_seconds = float(
        get_nested(config, "continuous", "video_seconds", default=10)
    )

    last_capture = 0.0
    print("Continuous detection mode active.")

    while not STOP_EVENT.is_set():
        frame = camera.capture_array()

        detections = detect_frame(model, frame, config)
        targets = target_detections(detections, config)

        if not targets:
            continue

        now = time.monotonic()
        if now - last_capture < cooldown:
            continue

        last_capture = now

        target_names = sorted(
            {
                class_name_normalized(d.class_name)
                for d in targets
            }
        )

        print(
            "Target detected: "
            + ", ".join(target_names)
        )

        if capture_type == "image":
            annotated = draw_overlay(
                frame.copy(),
                config,
                configured_fps(config),
                detections
            )

            paths = save_image(
                annotated,
                output_root,
                target_names
            )

            print("Image:", ", ".join(map(str, paths)))

        elif capture_type == "video":
            paths = record_video(
                camera,
                model,
                config,
                output_root,
                video_seconds
            )

            print("Video:", ", ".join(map(str, paths)))

        else:
            raise ValueError(
                "continuous.capture_type must be 'image' or 'video'."
            )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    install_signal_handlers()

    config = load_config()
    output_root = ensure_output_directories(config)

    model_path = Path(
        get_nested(config, "model", "path", default="model.pt")
    )
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}\n"
            "Set model.path in config.json."
        )

    mode = str(
        get_nested(config, "mode", default="pir")
    ).casefold()

    print("Raspberry Pi Wildlife Detector")
    print(f"Mode: {mode}")
    print(f"Configuration: {CONFIG_PATH}")
    print(f"Output directory: {output_root}")
    print(f"YOLO model: {model_path}")

    model = YOLO(str(model_path))
    camera = create_camera(config)

    try:
        if mode == "pir":
            run_pir_mode(
                camera,
                model,
                config,
                output_root
            )
        elif mode == "continuous":
            run_continuous_mode(
                camera,
                model,
                config,
                output_root
            )
        else:
            raise ValueError(
                "mode must be either 'pir' or 'continuous'."
            )
    finally:
        camera.stop()

    print("Detector stopped.")


if __name__ == "__main__":
    main()
