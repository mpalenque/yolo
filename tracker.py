"""
tracker.py - Motor de inferencia con pipeline optimizado para RTX 3090.

Arquitectura de 3 hilos:
  1. Hilo de CAPTURA+INFERENCIA - lee frames y corre YOLO
  2. Hilo de ENCODE             - comprime a JPEG para el stream MJPEG
  
Los frames viajan por deque(maxlen=1) entre hilos, nunca se bloquean
mutuamente. El cuello de botella es solo la GPU (o la fuente de video).
"""

import collections
import json
import math
import os
import platform
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# -- Constantes --
MODEL_PATH  = "yolo11n.pt"
ZONA_FILE   = "zona_coords.json"
MAX_PUNTOS  = 40
RTSP_URL    = "rtsp://mpalenque:madariaga1@192.168.0.96:554/stream1"
DEFAULT_CONF    = 0.35
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_IMGSZ   = 256
DEFAULT_HALF    = True

COL_ZONA   = (255, 120,   0)
COL_BOX    = (  0, 220,   0)
COL_TRAIL  = (  0, 220, 220)
COL_POINT  = (  0,  60, 255)
COL_PLANE  = (220, 120, 255)
COL_TXT_BG = (  0,   0,   0)
COL_TXT_FG = (255, 255, 255)


# -- FrameBuffer thread-safe --

class FrameBuffer:
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame: Optional[bytes] = None
        self._event = threading.Event()

    def put(self, jpeg_bytes: bytes):
        with self._lock:
            self._frame = jpeg_bytes
        self._event.set()

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def wait_for_frame(self, timeout=2.0) -> Optional[bytes]:
        if self._event.wait(timeout):
            self._event.clear()
        return self.get()


# -- Helpers de dibujo --

def _put_text_bg(frame, text, org, scale=0.55, thick=1,
                 fg=COL_TXT_FG, bg=COL_TXT_BG):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, thick, cv2.LINE_AA)


# -- Tracker --

class Tracker:
    def __init__(self):
        self.buffer = FrameBuffer()

        self._source     = RTSP_URL
        self._source_label = "Tapo RTSP"

        self._mode       = "MENU"
        self._zona_pts   : list                  = []
        self._zona_valida: Optional[np.ndarray] = None
        self._plane_pts  : list                  = []
        self._plane_valida: Optional[np.ndarray] = None
        self._homography: Optional[np.ndarray] = None
        self._mirror_x = False
        self._aruco_marker_id = 0
        self._aruco_marker_size_mm = 120.0
        self._camera_height_m = 0.94
        self._screen_config = {
            "enabled": True, "x_m": 0.0, "z_m": 1.2,
            "width_m": 3.5, "height_m": 2.0, "yaw_deg": 0.0,
        }
        self._view_config = {
            "content_yaw_deg": 45.0,
            "content_pitch_deg": 37.2,
            "content_distance_m": 4.6,
            "content_target_x_m": 0.0,
            "content_target_y_m": 0.25,
            "content_target_z_m": 0.0,
        }
        self._visual_config = {
            "room_width_m": 8.0,
            "min_depth_m": 1.2,
            "max_depth_m": 12.0,
            "tracking_max_depth_rel": 1.0,
            "person_height_m": 1.7,
            "person_width_m": 0.55,
            "height_scale": 1.0,
            "width_scale": 1.0,
            "media_scale": 1.0,
            "offset_x_px": 0.0,
            "offset_y_px": 0.0,
            "bottom_offset_px": 0.0,
            "fov_deg": 84.0,
            "flip_depth": False,
            "particles_enabled": True,
            "particles_count": 260,
            "particles_size_px": 3.2,
            "particles_opacity": 0.3,
            "particles_speed": 1.0,
            "particles_life": 2.8,
            "particles_preset": "classic",
        }
        self._aruco_auto_active = False
        self._aruco_detected = False
        self._aruco_last_message = "Aruco inactivo"
        self._aruco_quality = 0
        self._aruco_quality_label = "bad"
        self._historial  : dict = {}
        self._people_plane: list = []
        self._simulation_enabled = False
        self._simulation_count = 3
        self._simulation_started_at = time.monotonic()
        self._model      = None
        self._device     = "cpu"
        self._device_label = "CPU"
        self._conf       = DEFAULT_CONF
        self._tracker    = DEFAULT_TRACKER
        self._imgsz      = DEFAULT_IMGSZ
        self._half       = DEFAULT_HALF
        self._model_name = MODEL_PATH
        self._fps        = 0.0
        self._display_max_side = 640
        self._jpeg_quality = 50
        self._draw_trails = False

        # Geo-cache: evita copiar zona/plane/H en cada frame
        self._geo_version = 0
        self._cached_geo_version = -1
        self._cached_zona = None
        self._cached_plane = None
        self._cached_H = None

        # Stats snapshot (lock-free via GIL)
        self._stats_snapshot: dict = {
            "personas": 0, "mode": "MENU", "source": "---",
            "model": MODEL_PATH, "conf": DEFAULT_CONF,
            "fps": 0.0, "imgsz": DEFAULT_IMGSZ,
            "source_fps": 0.0, "perf_profile": "turbo",
            "device": self._device_label,
            "plane_ready": False,
            "mirror_x": self._mirror_x,
            "aruco_auto_active": self._aruco_auto_active,
            "aruco_detected": self._aruco_detected,
            "aruco_marker_id": self._aruco_marker_id,
            "aruco_marker_size_mm": self._aruco_marker_size_mm,
            "camera_height_m": self._camera_height_m,
            "screen_config": dict(self._screen_config),
            "view_config": dict(self._view_config),
            "visual_config": dict(self._visual_config),
            "aruco_quality": self._aruco_quality,
            "aruco_quality_label": self._aruco_quality_label,
            "simulation_enabled": self._simulation_enabled,
        }

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._encode_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._run_token = 0

        # Deque de un solo slot para pasar frames numpy al hilo de encode
        self._encode_deque: collections.deque = collections.deque(maxlen=1)
        self._encode_event = threading.Event()

        # Captura persistente: evita re-abrir al cambiar perfil/zona
        self._persistent_cap = None
        self._persistent_cap_source = None

        self._aruco_available = hasattr(cv2, "aruco")
        self._aruco_dict = None
        self._aruco_params = None
        if self._aruco_available:
            self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self._aruco_params = cv2.aruco.DetectorParameters()

        self._load_zona()
        # Update stats snapshot with loaded config
        self._flush_stats_snapshot()

    # -- Helpers de source --

    def _normalize_source(self, source):
        if isinstance(source, int):
            return source
        if isinstance(source, str):
            s = source.strip()
            if s.isdigit():
                return int(s)
        return source

    def _open_capture(self, source):
        normalized = self._normalize_source(source)
        is_windows = platform.system().lower().startswith("win")

        def _quick_has_frames(cap_obj, attempts=5):
            if cap_obj is None or not cap_obj.isOpened():
                return False
            for _ in range(attempts):
                ok, fr = cap_obj.read()
                if ok and fr is not None and fr.size > 0:
                    return True
                time.sleep(0.01)
            return False

        def _configure_webcam(cap_obj):
            if cap_obj is None or not cap_obj.isOpened():
                return
            try:
                cap_obj.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            try:
                cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap_obj.set(cv2.CAP_PROP_FPS, 30)
                cap_obj.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        def _measure_fps(cap_obj, n=10):
            if cap_obj is None or not cap_obj.isOpened():
                return 0.0
            t0 = time.time()
            ok_n = 0
            for _ in range(n):
                ok, _ = cap_obj.read()
                if ok:
                    ok_n += 1
                if (time.time() - t0) > 0.6:
                    break
            return ok_n / max(1e-6, time.time() - t0)

        if isinstance(normalized, int):
            if is_windows:
                best_cap = None
                best_fps = 0.0
                for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
                    cap = cv2.VideoCapture(normalized, backend)
                    if not cap.isOpened():
                        cap.release()
                        continue
                    _configure_webcam(cap)
                    if not _quick_has_frames(cap):
                        cap.release()
                        continue
                    fps = _measure_fps(cap)
                    if fps > best_fps:
                        if best_cap is not None:
                            best_cap.release()
                        best_cap = cap
                        best_fps = fps
                    else:
                        cap.release()
                if best_cap is not None:
                    return best_cap

            cap = cv2.VideoCapture(normalized)
            _configure_webcam(cap)
            if _quick_has_frames(cap):
                return cap
            cap.release()
            return cv2.VideoCapture(-1)

        return cv2.VideoCapture(normalized)

    def scan_webcams(self, max_idx=8):
        found = []
        for idx in range(max_idx):
            backend = cv2.CAP_DSHOW if platform.system().lower().startswith("win") else -1
            cap = cv2.VideoCapture(idx, backend)
            if cap and cap.isOpened():
                ok, _ = cap.read()
                cap.release()
                if ok:
                    found.append({"index": idx, "label": f"Webcam {idx}"})
            else:
                if cap:
                    cap.release()
        return found

    # -- API publica --

    def set_source(self, source, label="") -> Tuple[bool, str]:
        normalized = self._normalize_source(source)
        should_load_model = False

        new_cap = self._open_capture(normalized)
        if new_cap is None or not new_cap.isOpened():
            if new_cap:
                new_cap.release()
            return False, f"No se pudo abrir: {normalized}"

        with self._lock:
            self._source = normalized
            self._source_label = label or str(normalized)
            self._historial = {}
            if self._mode == "MENU":
                if self._zona_valida is not None:
                    self._mode = "TRACKING"
                    should_load_model = True
                else:
                    self._mode = "CALIBRANDO"

        if should_load_model:
            self._ensure_model()

        # Pasar la captura validada al hilo (evita doble apertura)
        self._persistent_cap = new_cap
        self._persistent_cap_source = normalized
        self.restart()
        return True, f"Fuente activa: {label or str(normalized)}"

    def add_calibration_point(self, x, y, frame_w, frame_h, display_w, display_h):
        if self._mode != "CALIBRANDO":
            return
        rx = int(x * frame_w / display_w) if display_w else x
        ry = int(y * frame_h / display_h) if display_h else y
        with self._lock:
            self._zona_pts.append((rx, ry))

    def add_plane_point(self, x, y, frame_w, frame_h, display_w, display_h):
        rx = int(x * frame_w / display_w) if display_w else x
        ry = int(y * frame_h / display_h) if display_h else y
        with self._lock:
            if len(self._plane_pts) < 4:
                self._plane_pts.append((rx, ry))
            else:
                self._plane_pts[-1] = (rx, ry)

    def undo_last_point(self):
        with self._lock:
            if self._zona_pts:
                self._zona_pts.pop()

    def undo_last_plane_point(self):
        with self._lock:
            if self._plane_pts:
                self._plane_pts.pop()

    def confirm_zone(self):
        with self._lock:
            if len(self._zona_pts) < 3:
                return False
            self._zona_valida = np.array(self._zona_pts, dtype=np.int32)
            self._geo_version += 1
            self._save_zona()
            self._historial = {}
            self._mode = "TRACKING"
        self._ensure_model()
        self.restart()
        return True

    def confirm_plane(self):
        with self._lock:
            if len(self._plane_pts) != 4:
                return False
            self._plane_valida = np.array(self._plane_pts, dtype=np.int32)
            src = np.array(self._plane_pts, dtype=np.float32)
            dst = np.array(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                dtype=np.float32,
            )
            self._homography = cv2.getPerspectiveTransform(src, dst)
            self._geo_version += 1
            self._save_zona()
            return True

    def reset_zone(self):
        with self._lock:
            self._zona_pts = []
            self._zona_valida = None
            self._geo_version += 1
            self._historial = {}
            self._mode = "CALIBRANDO"
            self._people_plane = []
        if self._thread is None or not self._thread.is_alive():
            self.restart()

    def reset_plane(self):
        with self._lock:
            self._plane_pts = []
            self._plane_valida = None
            self._homography = None
            self._geo_version += 1
            self._save_zona()

    def set_mirror(self, mirror_x):
        with self._lock:
            self._mirror_x = bool(mirror_x)
            self._save_zona()

    def set_marker_spec(self, marker_id, marker_size_mm):
        with self._lock:
            self._aruco_marker_id = max(0, int(marker_id))
            self._aruco_marker_size_mm = max(20.0, float(marker_size_mm))
            self._save_zona()

    def set_camera_height(self, camera_height_m):
        with self._lock:
            self._camera_height_m = max(0.2, min(5.0, float(camera_height_m)))
            self._save_zona()

    def set_screen_config(self, config):
        with self._lock:
            screen = dict(self._screen_config)
            if "enabled" in config:
                screen["enabled"] = bool(config.get("enabled"))
            if "x_m" in config:
                screen["x_m"] = float(config.get("x_m", screen["x_m"]))
            if "z_m" in config:
                screen["z_m"] = float(config.get("z_m", screen["z_m"]))
            if "width_m" in config:
                screen["width_m"] = max(0.2, float(config.get("width_m", screen["width_m"])))
            if "height_m" in config:
                screen["height_m"] = max(0.2, float(config.get("height_m", screen["height_m"])))
            if "yaw_deg" in config:
                screen["yaw_deg"] = float(config.get("yaw_deg", screen["yaw_deg"]))
            self._screen_config = screen
            self._save_zona()

    def set_view_config(self, config):
        with self._lock:
            view = dict(self._view_config)
            if "content_yaw_deg" in config:
                view["content_yaw_deg"] = float(config.get("content_yaw_deg", view["content_yaw_deg"]))
            if "content_pitch_deg" in config:
                view["content_pitch_deg"] = float(config.get("content_pitch_deg", view["content_pitch_deg"]))
            if "content_distance_m" in config:
                view["content_distance_m"] = max(0.2, float(config.get("content_distance_m", view["content_distance_m"])))
            if "content_target_x_m" in config:
                view["content_target_x_m"] = float(config.get("content_target_x_m", view["content_target_x_m"]))
            if "content_target_y_m" in config:
                view["content_target_y_m"] = float(config.get("content_target_y_m", view["content_target_y_m"]))
            if "content_target_z_m" in config:
                view["content_target_z_m"] = float(config.get("content_target_z_m", view["content_target_z_m"]))
            self._view_config = view
            self._save_zona()
        self._flush_stats_snapshot()

    def get_view_config(self):
        with self._lock:
            return dict(self._view_config)

    def set_visual_config(self, config):
        with self._lock:
            visual = dict(self._visual_config)
            if "room_width_m" in config:
                visual["room_width_m"] = max(1.0, float(config.get("room_width_m", visual["room_width_m"])))
            if "min_depth_m" in config:
                visual["min_depth_m"] = max(0.15, float(config.get("min_depth_m", visual["min_depth_m"])))
            if "max_depth_m" in config:
                visual["max_depth_m"] = max(0.3, float(config.get("max_depth_m", visual["max_depth_m"])))
            if "tracking_max_depth_rel" in config:
                visual["tracking_max_depth_rel"] = max(0.0, min(1.0, float(config.get("tracking_max_depth_rel", visual.get("tracking_max_depth_rel", 1.0)))))
            if "person_height_m" in config:
                visual["person_height_m"] = max(0.4, float(config.get("person_height_m", visual["person_height_m"])))
            if "person_width_m" in config:
                visual["person_width_m"] = max(0.1, float(config.get("person_width_m", visual["person_width_m"])))
            if "height_scale" in config:
                visual["height_scale"] = max(0.2, float(config.get("height_scale", visual["height_scale"])))
            if "width_scale" in config:
                visual["width_scale"] = max(0.2, float(config.get("width_scale", visual["width_scale"])))
            if "media_scale" in config:
                visual["media_scale"] = max(0.1, min(6.0, float(config.get("media_scale", visual["media_scale"]))))
            if "offset_x_px" in config:
                visual["offset_x_px"] = float(config.get("offset_x_px", visual["offset_x_px"]))
            if "offset_y_px" in config:
                visual["offset_y_px"] = float(config.get("offset_y_px", visual["offset_y_px"]))
            if "bottom_offset_px" in config:
                visual["bottom_offset_px"] = float(config.get("bottom_offset_px", visual["bottom_offset_px"]))
            if "fov_deg" in config:
                visual["fov_deg"] = max(35.0, min(130.0, float(config.get("fov_deg", visual["fov_deg"]))))
            if "flip_depth" in config:
                visual["flip_depth"] = bool(config.get("flip_depth", visual["flip_depth"]))
            if "particles_enabled" in config:
                visual["particles_enabled"] = bool(config.get("particles_enabled", visual["particles_enabled"]))
            if "particles_count" in config:
                visual["particles_count"] = int(max(0, min(200000, int(config.get("particles_count", visual["particles_count"])))) )
            if "particles_size_px" in config:
                visual["particles_size_px"] = max(0.2, min(30.0, float(config.get("particles_size_px", visual["particles_size_px"]))))
            if "particles_opacity" in config:
                visual["particles_opacity"] = max(0.0, min(1.0, float(config.get("particles_opacity", visual["particles_opacity"]))))
            if "particles_speed" in config:
                visual["particles_speed"] = max(0.0, min(10.0, float(config.get("particles_speed", visual["particles_speed"]))))
            if "particles_life" in config:
                visual["particles_life"] = max(0.1, min(12.0, float(config.get("particles_life", visual.get("particles_life", 2.8)))))
            if "particles_preset" in config:
                preset = str(config.get("particles_preset", visual.get("particles_preset", "classic"))).strip().lower()
                visual["particles_preset"] = preset if preset in ("classic", "flow") else "classic"
            visual["max_depth_m"] = max(visual["min_depth_m"] + 0.1, visual["max_depth_m"])
            self._visual_config = visual
            self._save_zona()
        self._flush_stats_snapshot()

    def get_visual_config(self):
        with self._lock:
            return dict(self._visual_config)

    def get_screen_config(self):
        with self._lock:
            return dict(self._screen_config)

    def set_simulation(self, enabled):
        with self._lock:
            self._simulation_enabled = bool(enabled)
            if self._simulation_enabled:
                self._simulation_started_at = time.monotonic()
        self._flush_stats_snapshot()
        return True, (
            "Simulación de posiciones activada"
            if enabled else
            "Simulación de posiciones desactivada"
        )

    def start_auto_plane(self):
        with self._lock:
            self._mode = "CALIBRANDO"
            self._aruco_auto_active = True
            self._aruco_detected = False
            self._aruco_quality = 0
            self._aruco_quality_label = "bad"
            self._aruco_last_message = "Buscando marcador ArUco..."
        if self._thread is None or not self._thread.is_alive():
            self.restart()

    def stop_auto_plane(self):
        with self._lock:
            self._aruco_auto_active = False
            if not self._aruco_detected:
                self._aruco_last_message = "Aruco inactivo"

    def _calc_aruco_quality(self, corners, frame_shape):
        h, w = frame_shape[:2]
        pts = corners.astype(np.float32)
        area = float(abs(cv2.contourArea(pts)))
        frame_area = float(max(1, w * h))
        area_ratio = area / frame_area
        edges = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            edges.append(float(np.linalg.norm(p1 - p0)))
        min_edge = max(1e-6, min(edges))
        max_edge = max(edges)
        edge_uniformity = min_edge / max_edge
        margin_px = min(
            float(np.min(pts[:, 0])), float(np.min(pts[:, 1])),
            float(w - np.max(pts[:, 0])), float(h - np.max(pts[:, 1])),
        )
        area_score = np.clip((area_ratio - 0.005) / 0.05, 0.0, 1.0)
        uniformity_score = np.clip((edge_uniformity - 0.35) / 0.65, 0.0, 1.0)
        margin_score = np.clip(margin_px / 40.0, 0.0, 1.0)
        score = int(round(100 * (0.5 * area_score + 0.35 * uniformity_score + 0.15 * margin_score)))
        if score >= 75:
            label = "good"
        elif score >= 45:
            label = "warn"
        else:
            label = "bad"
        return score, label

    def confirm_auto_plane(self):
        with self._lock:
            if len(self._plane_pts) != 4:
                return False
        ok = self.confirm_plane()
        if not ok:
            return False
        with self._lock:
            if self._zona_valida is None:
                self._zona_pts = list(self._plane_pts)
                self._zona_valida = np.array(self._zona_pts, dtype=np.int32)
                self._geo_version += 1
            self._mode = "TRACKING"
            self._aruco_auto_active = False
            self._aruco_last_message = "Plano auto-calibrado y tracking activo"
            self._save_zona()
        self._ensure_model()
        self.restart()
        return True

    def get_aruco_status(self):
        with self._lock:
            return {
                "available": self._aruco_available,
                "auto_active": self._aruco_auto_active,
                "detected": self._aruco_detected,
                "marker_id": self._aruco_marker_id,
                "marker_size_mm": self._aruco_marker_size_mm,
                "camera_height_m": self._camera_height_m,
                "screen_config": dict(self._screen_config),
                "quality": self._aruco_quality,
                "quality_label": self._aruco_quality_label,
                "message": self._aruco_last_message,
            }

    def start_stream(self):
        if self._mode == "MENU":
            if self._zona_valida is not None:
                self._mode = "TRACKING"
                self._ensure_model()
            else:
                self._mode = "CALIBRANDO"
        self.restart()

    def set_conf(self, conf):
        self._conf = max(0.05, min(0.95, conf))

    def set_model(self, model_name, imgsz=640, half=True):
        self._model_name = model_name
        self._imgsz      = imgsz
        self._half       = half
        self._model      = None
        self._ensure_model()
        self.restart()

    def set_perf_profile(self, profile):
        profile = (profile or "balanced").lower()

        if profile == "turbo":
            new_model = "yolo11n.pt"
            self._imgsz = 256
            self._conf = 0.35
            self._half = True
            self._display_max_side = 480
            self._jpeg_quality = 40
            self._draw_trails = False
            selected = "turbo"
        else:
            new_model = "yolo11n.pt"
            self._imgsz = 384
            self._conf = 0.35
            self._half = True
            self._display_max_side = 640
            self._jpeg_quality = 55
            self._draw_trails = False
            selected = "balanced"

        if new_model != self._model_name or self._model is None:
            self._model_name = new_model
            self._model = None
            self._ensure_model()
        else:
            self._model_name = new_model

        self._flush_stats_snapshot(perf_profile=selected)
        self.restart()

    def get_stats(self):
        return dict(self._stats_snapshot)

    def get_zone_points(self):
        with self._lock:
            return list(self._zona_pts)

    def get_plane_points(self):
        with self._lock:
            return list(self._plane_pts)

    def get_people_plane(self):
        if self._simulation_enabled:
            return self._build_simulated_people()
        with self._lock:
            return list(self._people_plane)

    def get_config(self):
        with self._lock:
            return {
                "zone_points": list(self._zona_pts),
                "plane_points": list(self._plane_pts),
                "mirror_x": self._mirror_x,
                "aruco_marker_id": self._aruco_marker_id,
                "aruco_marker_size_mm": self._aruco_marker_size_mm,
                "camera_height_m": self._camera_height_m,
                "screen_config": dict(self._screen_config),
                "view_config": dict(self._view_config),
                "visual_config": dict(self._visual_config),
                "aruco_auto_active": self._aruco_auto_active,
            }

    def get_frame_size(self):
        return self._last_w, self._last_h

    # -- Control del hilo --

    def restart(self):
        self.stop()
        with self._lock:
            self._run_token += 1
            run_token = self._run_token
        self._stop_flag.clear()
        self._encode_deque.clear()
        self._thread = threading.Thread(target=self._run, args=(run_token,), daemon=True)
        self._thread.start()
        self._encode_thread = threading.Thread(target=self._encode_loop, args=(run_token,), daemon=True)
        self._encode_thread.start()

    def stop(self):
        with self._lock:
            self._run_token += 1
        self._stop_flag.set()
        self._encode_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4)
        if self._encode_thread and self._encode_thread.is_alive():
            self._encode_thread.join(timeout=2)

    # -- Hilo de ENCODE (JPEG) --

    def _encode_loop(self, run_token):
        while not self._stop_flag.is_set() and run_token == self._run_token:
            self._encode_event.wait(timeout=0.1)
            self._encode_event.clear()
            if self._encode_deque:
                if run_token != self._run_token:
                    break
                frame = self._encode_deque[-1]
                _, buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                self.buffer.put(buf.tobytes())

    def _submit_for_encode(self, frame):
        self._encode_deque.append(frame)
        self._encode_event.set()

    # -- Hilo de inferencia --

    _last_w: int = 1280
    _last_h: int = 720

    def _flush_stats_snapshot(self, **overrides):
        with self._lock:
            simulation_enabled = self._simulation_enabled
            snap = {
                "personas": (self._simulation_count if simulation_enabled else len(self._people_plane)),
                "mode": self._mode,
                "source": (f"{self._source_label} · SIM" if simulation_enabled else self._source_label),
                "model": self._model_name,
                "conf": self._conf,
                "fps": self._fps,
                "imgsz": self._imgsz,
                "source_fps": 0.0,
                "perf_profile": overrides.get("perf_profile",
                    self._stats_snapshot.get("perf_profile", "turbo")),
                "device": self._device_label,
                "plane_ready": (self._homography is not None) or simulation_enabled,
                "mirror_x": self._mirror_x,
                "aruco_auto_active": self._aruco_auto_active,
                "aruco_detected": self._aruco_detected,
                "aruco_marker_id": self._aruco_marker_id,
                "aruco_marker_size_mm": self._aruco_marker_size_mm,
                "camera_height_m": self._camera_height_m,
                "screen_config": dict(self._screen_config),
                "view_config": dict(self._view_config),
                "visual_config": dict(self._visual_config),
                "aruco_quality": self._aruco_quality,
                "aruco_quality_label": self._aruco_quality_label,
                "simulation_enabled": simulation_enabled,
            }
        snap.update(overrides)
        self._stats_snapshot = snap  # atomic reference swap

    def _build_simulated_people(self):
        with self._lock:
            width = max(1, int(self._last_w))
            height = max(1, int(self._last_h))
            mirror_x = bool(self._mirror_x)
            elapsed = time.monotonic() - self._simulation_started_at
            screen_z_m = max(0.8, float(self._screen_config.get("z_m", 1.2)))
            screen_width_m = max(1.5, float(self._screen_config.get("width_m", 3.5)))
            tracking_max_depth_rel = max(0.0, min(1.0, float(self._visual_config.get("tracking_max_depth_rel", 1.0))))

        people = []
        specs = (
            {"id": 101, "base_u": 0.28, "amp_u": 0.13, "base_v": 0.28, "amp_v": 0.08, "speed": 0.55, "phase": 0.0},
            {"id": 102, "base_u": 0.52, "amp_u": 0.19, "base_v": 0.58, "amp_v": 0.12, "speed": 0.37, "phase": 1.4},
            {"id": 103, "base_u": 0.76, "amp_u": 0.10, "base_v": 0.40, "amp_v": 0.09, "speed": 0.61, "phase": 2.3},
        )
        now_ts = time.time()

        for spec in specs[:self._simulation_count]:
            t = elapsed * spec["speed"] + spec["phase"]
            u = spec["base_u"] + math.sin(t) * spec["amp_u"]
            v = spec["base_v"] + math.cos(t * 0.83) * spec["amp_v"]
            u = min(0.96, max(0.04, u))
            v = min(0.95, max(0.08, v))
            if v > tracking_max_depth_rel:
                continue
            if mirror_x:
                u = 1.0 - u

            world_x_m = (u - 0.5) * (screen_width_m * 0.78)
            world_z_m = 0.22 + v * max(0.25, screen_z_m - 0.34)
            screen_x = int(round(u * width))
            screen_y = int(round(height * (0.52 + v * 0.36)))

            people.append({
                "id": spec["id"],
                "u": round(u, 4),
                "v": round(v, 4),
                "depth_rel": round(v, 4),
                "x": round(world_x_m, 4),
                "y": 0.0,
                "z": round(world_z_m, 4),
                "screen_x": screen_x,
                "screen_y": screen_y,
                "ts": now_ts,
                "simulated": True,
            })

        return people

    def _run(self, run_token):
        source = self._source

        # Reusar captura persistente si coincide con la fuente
        cap = None
        if self._persistent_cap is not None and self._persistent_cap_source == source:
            cap = self._persistent_cap
            self._persistent_cap = None
            self._persistent_cap_source = None
        if cap is None or not cap.isOpened():
            if cap:
                cap.release()
            cap = self._open_capture(source)

        if not cap.isOpened():
            self._push_error_frame(f"No se pudo abrir: {source}")
            return

        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            source_fps = 0.0

        old_snap = self._stats_snapshot
        self._stats_snapshot = {**old_snap, "source_fps": round(source_fps, 2)}

        fps_t0    = time.time()
        fps_count = 0
        read_failures = 0

        while not self._stop_flag.is_set() and run_token == self._run_token:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and os.path.isfile(source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                read_failures += 1

                # RTSP puede perder frames puntuales; evitar frame negro (titileo)
                if read_failures <= 15:
                    time.sleep(0.01)
                    continue

                # Si persiste, intentar reabrir captura en caliente
                try:
                    cap.release()
                except Exception:
                    pass

                cap = self._open_capture(source)
                if cap is not None and cap.isOpened():
                    read_failures = 0
                    continue

                # Solo mostrar error si el corte ya es sostenido
                if read_failures % 30 == 0:
                    self._push_error_frame("Sin senal de la camara (reconectando)")

                time.sleep(0.2)
                continue

            read_failures = 0

            h, w = frame.shape[:2]
            self._last_w = w
            self._last_h = h
            mode = self._mode

            if mode == "CALIBRANDO":
                out = self._draw_calibration(frame)
            elif mode == "TRACKING":
                out = self._process_tracking(frame)
            else:
                out = self._draw_menu(frame)

            # Resize solo para display/encode
            dh, dw = out.shape[:2]
            max_side = self._display_max_side
            if max(dh, dw) > max_side:
                s = max_side / max(dh, dw)
                out = cv2.resize(out, (int(dw * s), int(dh * s)),
                                 interpolation=cv2.INTER_LINEAR)

            # Enviar a encode thread (no bloquea)
            if run_token != self._run_token:
                break
            self._submit_for_encode(out)

            fps_count += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 1.0:
                self._fps = round(fps_count / elapsed, 1)
                fps_count = 0
                fps_t0 = time.time()
                self._flush_stats_snapshot(source_fps=round(source_fps, 2))

        cap.release()

    # -- Dibujo calibracion --

    def _draw_calibration(self, frame):
        self._update_auto_plane_from_aruco(frame)
        with self._lock:
            pts = list(self._zona_pts)
            plane_pts = list(self._plane_pts)
            aruco_msg = self._aruco_last_message
            aruco_active = self._aruco_auto_active
            aruco_detected = self._aruco_detected

        for idx, p in enumerate(pts):
            cv2.circle(frame, p, 7, COL_POINT, -1)
            cv2.circle(frame, p, 9, COL_TXT_FG, 1)
            _put_text_bg(frame, str(idx + 1), (p[0] + 10, p[1] - 6))

        if len(pts) >= 2:
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(frame, [arr],
                          isClosed=(len(pts) >= 3), color=COL_ZONA, thickness=2)

        for idx, p in enumerate(plane_pts):
            cv2.circle(frame, p, 6, COL_PLANE, -1)
            cv2.circle(frame, p, 8, COL_TXT_FG, 1)
            _put_text_bg(frame, f"P{idx + 1}", (p[0] + 10, p[1] + 16), scale=0.45, fg=COL_PLANE)

        if len(plane_pts) >= 2:
            arr_plane = np.array(plane_pts, dtype=np.int32)
            cv2.polylines(frame, [arr_plane],
                          isClosed=(len(plane_pts) >= 4), color=COL_PLANE, thickness=2)

        _put_text_bg(frame, f"CALIBRANDO --- zona:{len(pts)} plano:{len(plane_pts)}/4", (10, 28),
                     scale=0.65, fg=(0, 255, 150))
        if aruco_active:
            _put_text_bg(
                frame, f"AUTO ARUCO: {aruco_msg}", (10, 56),
                scale=0.55,
                fg=(0, 255, 170) if aruco_detected else (0, 220, 255),
            )
        return frame

    def _update_auto_plane_from_aruco(self, frame):
        with self._lock:
            enabled = self._aruco_auto_active
            marker_id = self._aruco_marker_id

        if not enabled:
            return
        if not self._aruco_available or self._aruco_dict is None:
            with self._lock:
                self._aruco_detected = False
                self._aruco_last_message = "cv2.aruco no disponible"
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = None
        ids = None

        try:
            if hasattr(cv2.aruco, "ArucoDetector"):
                detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self._aruco_dict, parameters=self._aruco_params,
                )
        except Exception as e:
            with self._lock:
                self._aruco_detected = False
                self._aruco_last_message = f"Error ArUco: {e}"
            return

        if ids is None or len(ids) == 0:
            with self._lock:
                self._aruco_detected = False
                self._aruco_quality = 0
                self._aruco_quality_label = "bad"
                self._aruco_last_message = f"No se detecta marker {marker_id}"
            return

        flat_ids = ids.flatten().tolist()
        if marker_id not in flat_ids:
            with self._lock:
                self._aruco_detected = False
                self._aruco_quality = 0
                self._aruco_quality_label = "bad"
                self._aruco_last_message = f"Marker detectado, falta ID {marker_id}"
            return

        idx = flat_ids.index(marker_id)
        c = np.array(corners[idx]).reshape(4, 2)
        ordered = [(int(p[0]), int(p[1])) for p in c]
        quality, quality_label = self._calc_aruco_quality(c, frame.shape)

        with self._lock:
            self._plane_pts = ordered
            self._aruco_detected = True
            self._aruco_quality = quality
            self._aruco_quality_label = quality_label
            qt = "verde" if quality_label == "good" else "amarillo" if quality_label == "warn" else "rojo"
            self._aruco_last_message = f"ID {marker_id} detectado - calidad {quality}/100 ({qt})"

    def _draw_menu(self, frame):
        _put_text_bg(frame, "Selecciona una fuente en el panel ->",
                     (10, 28), scale=0.7, fg=(0, 255, 150))
        return frame

    # -- Tracking --

    def _get_cached_geo(self):
        with self._lock:
            ver = self._geo_version
            mirror_x = self._mirror_x
            source_label = self._source_label

        if ver != self._cached_geo_version:
            with self._lock:
                self._cached_zona = self._zona_valida.copy() if self._zona_valida is not None else None
                self._cached_plane = self._plane_valida.copy() if self._plane_valida is not None else None
                self._cached_H = self._homography.copy() if self._homography is not None else None
                self._cached_geo_version = ver

        return self._cached_zona, self._cached_plane, self._cached_H, mirror_x, source_label

    def _fallback_zone_projection(self, zona, cx, by, mirror_x):
        if zona is None or len(zona) == 0:
            return None, None

        zx, zy, zw, zh = cv2.boundingRect(zona)
        if zw <= 1 or zh <= 1:
            return None, None

        u = (float(cx) - float(zx)) / float(zw)
        v = (float(by) - float(zy)) / float(zh)
        u = float(np.clip(u, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        if mirror_x:
            u = 1.0 - u
        return u, v

    def _project_person_point(self, H, zona, cx, by, mirror_x):
        if H is not None:
            try:
                src_pt = np.array([[[float(cx), float(by)]]], dtype=np.float32)
                dst_pt = cv2.perspectiveTransform(src_pt, H)[0][0]
                u = float(dst_pt[0])
                v = float(dst_pt[1])
                if np.isfinite(u) and np.isfinite(v):
                    if -0.25 <= u <= 1.25 and -0.25 <= v <= 1.25:
                        if mirror_x:
                            u = 1.0 - u
                        return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0)), True
            except Exception:
                pass

        u, v = self._fallback_zone_projection(zona, cx, by, mirror_x)
        if u is None or v is None:
            return None, None, False
        return u, v, False

    def _process_tracking(self, frame):
        n_en_zona = 0
        zona, plane, H, mirror_x, source_label = self._get_cached_geo()
        with self._lock:
            tracking_max_depth_rel = max(0.0, min(1.0, float(self._visual_config.get("tracking_max_depth_rel", 1.0))))

        if zona is None and plane is not None:
            zona = plane

        people_frame = []

        if zona is not None:
            cv2.polylines(frame, [zona], isClosed=True, color=COL_ZONA, thickness=2)
        if plane is not None:
            cv2.polylines(frame, [plane], isClosed=True, color=COL_PLANE, thickness=2)

        if not ULTRALYTICS_AVAILABLE or self._model is None or zona is None:
            _put_text_bg(frame, "YOLO no disponible o zona no configurada",
                         (10, 28), fg=(0, 0, 255))
            with self._lock:
                self._people_plane = []
            return frame

        try:
            results = self._model.track(
                frame, persist=True, classes=[0],
                conf=self._conf, tracker=self._tracker,
                imgsz=self._imgsz, half=self._half,
                device=self._device, verbose=False,
            )
            r = results[0]
            boxes = getattr(r, "boxes", None)
            draw_trails = self._draw_trails

            if boxes is not None:
                for box in boxes:
                    try:
                        coords = np.array(box.xyxy).reshape(-1)
                        x1, y1, x2, y2 = map(int, coords[:4])
                    except Exception:
                        continue

                    cx = (x1 + x2) // 2
                    by = int(y2)

                    inside = cv2.pointPolygonTest(zona, (float(cx), float(by)), False)
                    if inside < 0:
                        continue

                    try:
                        pid = int(np.array(box.id).flatten()[0])
                    except Exception:
                        pid = None

                    u = None
                    v = None
                    depth_rel = None
                    world_x_m = None
                    world_y_m = 0.0
                    world_z_m = None
                    u, v, used_plane_projection = self._project_person_point(H, zona, cx, by, mirror_x)
                    if u is not None and v is not None:
                        depth_rel = v
                        if used_plane_projection and depth_rel > tracking_max_depth_rel:
                            continue
                        if used_plane_projection:
                            marker_size_m = max(0.02, self._aruco_marker_size_mm / 1000.0)
                            world_x_m = (u - 0.5) * marker_size_m
                            world_z_m = (v - 0.5) * marker_size_m

                    n_en_zona += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_BOX, 2)
                    _put_text_bg(frame, f"ID {pid}" if pid is not None else "persona",
                                 (x1, y1 - 6), fg=COL_BOX)

                    if pid is not None:
                        trail = None
                        if draw_trails:
                            with self._lock:
                                self._historial.setdefault(pid, []).append((cx, by))
                                if len(self._historial[pid]) > MAX_PUNTOS:
                                    self._historial[pid].pop(0)
                                trail = self._historial[pid]

                        people_frame.append({
                            "id": pid, "u": u, "v": v,
                            "depth_rel": depth_rel,
                            "x": world_x_m, "y": world_y_m, "z": world_z_m,
                            "screen_x": cx, "screen_y": by, "ts": time.time(),
                        })

                        if draw_trails and trail and len(trail) >= 2:
                            cv2.polylines(frame, [np.array(trail, dtype=np.int32)],
                                          isClosed=False, color=COL_TRAIL, thickness=2)
                            cv2.circle(frame, trail[-1], 5, COL_TRAIL, -1)

        except Exception as e:
            _put_text_bg(frame, f"Error tracking: {e}", (10, 60), fg=(0, 0, 255))

        with self._lock:
            self._people_plane = people_frame

        _put_text_bg(frame, f"Personas en zona: {n_en_zona}",
                     (10, 28), scale=0.65, fg=(0, 255, 150))
        return frame

    def _push_error_frame(self, msg):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _put_text_bg(frame, msg, (20, 240), scale=0.7, fg=(0, 0, 255))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.buffer.put(buf.tobytes())

    # -- Persistencia zona --

    def _save_zona(self):
        try:
            with open(ZONA_FILE, "w") as f:
                payload = {
                    "version": 3,
                    "zone_points": self._zona_pts,
                    "plane_points": self._plane_pts,
                    "mirror_x": self._mirror_x,
                    "aruco": {
                        "marker_id": self._aruco_marker_id,
                        "marker_size_mm": self._aruco_marker_size_mm,
                        "camera_height_m": self._camera_height_m,
                    },
                    "screen": dict(self._screen_config),
                    "view": dict(self._view_config),
                    "visual": dict(self._visual_config),
                }
                json.dump(payload, f)
        except Exception as e:
            print(f"[tracker] No se pudo guardar zona: {e}")

    def _load_zona(self):
        if not os.path.exists(ZONA_FILE):
            return
        try:
            with open(ZONA_FILE) as f:
                data = json.load(f)

            if isinstance(data, list):
                pts = data
                if len(pts) >= 3:
                    self._zona_pts = [tuple(p) for p in pts]
                    self._zona_valida = np.array(pts, dtype=np.int32)
                    self._geo_version += 1
                    print(f"[tracker] Zona cargada ({len(pts)} puntos).")
                return

            pts = data.get("zone_points", [])
            plane_pts = data.get("plane_points", [])
            self._mirror_x = bool(data.get("mirror_x", False))
            aruco_cfg = data.get("aruco", {}) if isinstance(data, dict) else {}
            self._aruco_marker_id = int(aruco_cfg.get("marker_id", self._aruco_marker_id))
            self._aruco_marker_size_mm = float(aruco_cfg.get("marker_size_mm", self._aruco_marker_size_mm))
            self._camera_height_m = float(aruco_cfg.get("camera_height_m", self._camera_height_m))
            screen_cfg = data.get("screen", {}) if isinstance(data, dict) else {}
            if isinstance(screen_cfg, dict):
                self._screen_config.update(screen_cfg)
            view_cfg = data.get("view", {}) if isinstance(data, dict) else {}
            if isinstance(view_cfg, dict):
                self._view_config.update(view_cfg)
            visual_cfg = data.get("visual", {}) if isinstance(data, dict) else {}
            if isinstance(visual_cfg, dict):
                self._visual_config.update(visual_cfg)
                self._visual_config["tracking_max_depth_rel"] = max(
                    0.0,
                    min(1.0, float(self._visual_config.get("tracking_max_depth_rel", 1.0)))
                )
                self._visual_config["particles_life"] = max(
                    0.1,
                    min(12.0, float(self._visual_config.get("particles_life", 2.8)))
                )
                preset = str(self._visual_config.get("particles_preset", "classic")).strip().lower()
                self._visual_config["particles_preset"] = preset if preset in ("classic", "flow") else "classic"

            if len(pts) >= 3:
                self._zona_pts = [tuple(p) for p in pts]
                self._zona_valida = np.array(pts, dtype=np.int32)
                self._geo_version += 1
                print(f"[tracker] Zona cargada ({len(pts)} puntos).")

            if len(plane_pts) == 4:
                self._plane_pts = [tuple(p) for p in plane_pts]
                self._plane_valida = np.array(plane_pts, dtype=np.int32)
                src = np.array(plane_pts, dtype=np.float32)
                dst = np.array(
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    dtype=np.float32,
                )
                self._homography = cv2.getPerspectiveTransform(src, dst)
                self._geo_version += 1
        except Exception as e:
            print(f"[tracker] Error cargando zona: {e}")

    def _ensure_model(self):
        if self._model is None and ULTRALYTICS_AVAILABLE:
            device = "cpu"
            device_label = "CPU"
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = "cuda:0"
                device_label = torch.cuda.get_device_name(0)
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.benchmark = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            elif TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                device_label = "Apple MPS"
            self._device = device
            self._device_label = device_label
            if self._device == "cpu" and self._half:
                self._half = False
            print(
                f"[tracker] Cargando {self._model_name} en {self._device} "
                f"({self._device_label}) (imgsz={self._imgsz}, half={self._half})..."
            )
            self._model = YOLO(self._model_name)
            print("[tracker] Modelo listo.")
