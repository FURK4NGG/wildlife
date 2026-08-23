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
from datetime import datetime
from pathlib import Path
from threading import Event

import cv2
from gpiozero import MotionSensor
from picamera2 import Picamera2
from picamera2.devices import Hailo


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
STOP_EVENT = Event()


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "mode": "pir",

    "model": {
        "path": "yolov11l.hef",
        "labels": "labels.txt",
        "confidence": 0.50
    },

    "camera": {
        "width": 1280,
        "height": 960,
        "fps_min": 5,
        "fps_max": 30,
        "fps": 15
    },

    "pir": {
        "gpio_pin": 13,
        "cooldown_seconds": 2.0,
        "capture_image": True,

        "video": {
            "enabled": True,

            # "detected" = record only if the trigger image contains a
            # configured target class.
            # "always" = record after every PIR trigger.
            "trigger": "detected",
            "seconds": 10
        }
    },

    "continuous": {
        # "image" or "video"
        "capture_type": "image",
        "cooldown_seconds": 10.0,
        "video_seconds": 10
    },

    "video": {
        "fps": 15,
        "codec": "mp4v"
    },

    "overlay": {
        "show_fps": True,
        "show_datetime": True,
        "show_class_name": True,
        "show_confidence": True,
        "font_scale": 0.70,
        "line_thickness": 2,
        "background": True
    },

    "detection": {
        "trigger_classes": [
            "bear",
            "boar",
            "wolf",
            "deer"
        ],

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
        # Keep the live camera window updating while PIR is idle.
        main_frame = camera.capture_array("main")
        current_fps = fps_counter.update()

        if show_preview(main_frame, config, None, current_fps):
            STOP_EVENT.set()
            break

        # Do not block on wait_for_motion(); blocking would freeze the preview.
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

        targets = target_detections(
            detections,
            config
        )

        target_names = class_names_from_detections(
            targets
        )

        fps = camera_fps(config)

        annotated = draw_overlay(
            main_frame.copy(),
            config,
            detections,
            fps,
            width,
            height
        )

        if show_preview(annotated, config, None, fps):
            STOP_EVENT.set()
            break

        print(
            "PIR triggered. "
            f"Targets: {', '.join(sorted(target_names)) or 'none'}"
        )

        # The trigger image is saved when a configured target is found.
        if (
            target_names
            and bool(
                get_config(
                    config,
                    "pir",
                    "capture_image",
                    default=True
                )
            )
        ):
            paths = save_image(
                annotated,
                root,
                target_names
            )

            print(
                "Image saved:",
                ", ".join(str(path) for path in paths)
            )

        video_enabled = bool(
            get_config(
                config,
                "pir",
                "video",
                "enabled",
                default=True
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

        video_seconds = float(
            get_config(
                config,
                "pir",
                "video",
                "seconds",
                default=10
            )
        )

        should_record = (
            video_enabled
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
    capture_type = str(
        get_config(
            config,
            "continuous",
            "capture_type",
            default="image"
        )
    ).casefold()

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
            "continuous",
            "video_seconds",
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

        target_names = class_names_from_detections(
            targets
        )

        print(
            "Target detected:",
            ", ".join(sorted(target_names))
        )

        if capture_type == "image":
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

        elif capture_type == "video":
            paths = record_video(
                camera,
                hailo,
                labels,
                config,
                root,
                video_seconds,
                first_target_required=True
            )

            print(
                "Video saved:",
                ", ".join(str(path) for path in paths)
            )

        else:
            raise ValueError(
                "continuous.capture_type must be "
                "'image' or 'video'."
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

    print("Detector stopped.")


if __name__ == "__main__":
    main()
