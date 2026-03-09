"""
tracker.py — Motor de inferencia con thread dedicado.

La clase Tracker corre YOLO en su propio hilo. El resultado de cada frame
(JPEG codificado) se guarda en un FrameBuffer thread-safe. El servidor
web lee ese buffer sin bloquear nunca el hilo de inferencia.
"""

import json
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

# ── Constantes ──────────────────────────────────────────────────────────────
MODEL_PATH  = "yolo11n.pt"   # YOLO11 nano — prioriza FPS en tiempo real
ZONA_FILE   = "zona_coords.json"
MAX_PUNTOS  = 40
RTSP_URL    = "rtsp://mpalenque:madariaga1@192.168.0.96:554/stream1"
DEFAULT_CONF    = 0.30
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_IMGSZ   = 416          # resolución de inferencia (menor = más rápido)
DEFAULT_HALF    = True         # fp16 — MPS del M4 Pro lo soporta

# Colores BGR
COL_ZONA   = (255, 120,   0)
COL_BOX    = (  0, 220,   0)
COL_TRAIL  = (  0, 220, 220)
COL_POINT  = (  0,  60, 255)
COL_PLANE  = (220, 120, 255)
COL_TXT_BG = (  0,   0,   0)
COL_TXT_FG = (255, 255, 255)


# ── FrameBuffer thread-safe ─────────────────────────────────────────────────

class FrameBuffer:
    """Guarda siempre el último JPEG disponible. Sin colas: nunca bloquea YOLO."""

    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None          # bytes JPEG
        self._event = threading.Event()

    def put(self, jpeg_bytes: bytes):
        with self._lock:
            self._frame = jpeg_bytes
            self._event.set()

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def wait_for_frame(self, timeout=2.0) -> Optional[bytes]:
        """Bloquea hasta que haya un frame nuevo, útil para el generador MJPEG."""
        self._event.wait(timeout)
        self._event.clear()
        return self.get()


# ── Helpers de dibujo ───────────────────────────────────────────────────────

def _put_text_bg(frame, text, org, scale=0.55, thick=1,
                 fg=COL_TXT_FG, bg=COL_TXT_BG):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, thick, cv2.LINE_AA)


# ── Tracker ─────────────────────────────────────────────────────────────────

class Tracker:
    def __init__(self):
        self.buffer = FrameBuffer()

        # Estado de fuente
        self._source     = RTSP_URL
        self._source_label = "Tapo RTSP"

        # Estado de calibración / tracking
        self._mode       = "MENU"       # "MENU" | "CALIBRANDO" | "TRACKING"
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
            "enabled": True,
            "x_m": 0.0,
            "z_m": 1.2,
            "width_m": 3.5,
            "height_m": 2.0,
            "yaw_deg": 0.0,
        }
        self._aruco_auto_active = False
        self._aruco_detected = False
        self._aruco_last_message = "Aruco inactivo"
        self._aruco_quality = 0
        self._aruco_quality_label = "bad"
        self._historial  : dict         = {}
        self._people_plane: list = []
        self._model                     = None
        self._device                    = "cpu"
        self._device_label              = "CPU"
        self._conf      = DEFAULT_CONF
        self._tracker   = DEFAULT_TRACKER
        self._imgsz     = DEFAULT_IMGSZ
        self._half      = DEFAULT_HALF
        self._model_name = MODEL_PATH
        self._fps       = 0.0
        self._max_side = 720
        self._jpeg_quality = 65
        self._draw_trails = False
        self._capture_width = 960
        self._capture_height = 540
        self._capture_fps = 30
        self._stats      : dict         = {"personas": 0, "mode": "MENU", "source": "—",
                                           "model": MODEL_PATH, "conf": DEFAULT_CONF,
                           "fps": 0.0, "imgsz": DEFAULT_IMGSZ,
                                           "source_fps": 0.0,
                   "perf_profile": "turbo",
                                           "device": self._device_label,
                                           "plane_ready": False,
                                           "mirror_x": self._mirror_x,
                                           "aruco_auto_active": self._aruco_auto_active,
                                           "aruco_detected": self._aruco_detected,
                                           "aruco_marker_id": self._aruco_marker_id,
                                           "aruco_marker_size_mm": self._aruco_marker_size_mm,
                                           "camera_height_m": self._camera_height_m,
                                           "screen_config": dict(self._screen_config),
                                           "aruco_quality": self._aruco_quality,
                                           "aruco_quality_label": self._aruco_quality_label}

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        self._aruco_available = hasattr(cv2, "aruco")
        self._aruco_dict = None
        self._aruco_params = None
        if self._aruco_available:
            self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self._aruco_params = cv2.aruco.DetectorParameters()

        # Intentar cargar zona guardada
        self._load_zona()

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

        def _capture_has_frames(cap_obj, attempts: int = 12) -> bool:
            if cap_obj is None or not cap_obj.isOpened():
                return False
            for _ in range(attempts):
                ok, frame = cap_obj.read()
                if ok and frame is not None and frame.size > 0:
                    return True
                time.sleep(0.03)
            return False

        def _configure_webcam_capture(cap_obj):
            if cap_obj is None or not cap_obj.isOpened():
                return
            try:
                cap_obj.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            try:
                cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_width)
                cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_height)
                cap_obj.set(cv2.CAP_PROP_FPS, self._capture_fps)
                cap_obj.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        def _measure_capture_score(cap_obj, sample_frames: int = 20) -> float:
            if cap_obj is None or not cap_obj.isOpened():
                return 0.0
            start = time.time()
            ok_count = 0
            for _ in range(sample_frames):
                ok, frame = cap_obj.read()
                if ok and frame is not None and frame.size > 0:
                    ok_count += 1
                if (time.time() - start) > 1.4:
                    break
            elapsed = max(1e-6, time.time() - start)
            return ok_count / elapsed

        if isinstance(normalized, int):
            if is_windows:
                best_cap = None
                best_score = 0.0
                for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
                    cap = cv2.VideoCapture(normalized, backend)
                    if not cap.isOpened():
                        cap.release()
                        continue
                    _configure_webcam_capture(cap)
                    if not _capture_has_frames(cap):
                        cap.release()
                        continue
                    score = _measure_capture_score(cap)
                    if score > best_score:
                        if best_cap is not None:
                            best_cap.release()
                        best_cap = cap
                        best_score = score
                    else:
                        cap.release()
                if best_cap is not None:
                    return best_cap

            cap = cv2.VideoCapture(normalized)
            _configure_webcam_capture(cap)
            if _capture_has_frames(cap):
                return cap
            cap.release()
            return cap

        return cv2.VideoCapture(normalized)

    def scan_webcams(self, max_idx: int = 8) -> list:
        found = []
        for idx in range(max_idx):
            cap = self._open_capture(idx)
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                continue
            ok, _ = cap.read()
            cap.release()
            if ok:
                found.append({"index": idx, "label": f"Webcam {idx}"})
        return found

    # ── API pública (llamada desde el servidor web) ──────────────────────────

    def set_source(self, source: str, label: str = "") -> Tuple[bool, str]:
        """Cambia la fuente de video. Reinicia el hilo."""
        normalized = self._normalize_source(source)
        should_load_model = False

        if isinstance(normalized, int):
            test_cap = self._open_capture(normalized)
            if test_cap is None or not test_cap.isOpened():
                return False, f"No se pudo abrir Webcam {normalized}"
            test_cap.release()

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
                self._stats["mode"] = self._mode

        if should_load_model:
            self._ensure_model()
        self.restart()
        return True, f"Fuente activa: {self._source_label}"

    def add_calibration_point(self, x: int, y: int,
                               frame_w: int, frame_h: int,
                               display_w: int, display_h: int):
        """
        Agrega un punto de calibración. El browser envía coordenadas relativas
        al <canvas> de display; las escalamos al tamaño real del frame.
        """
        if self._mode != "CALIBRANDO":
            return
        # Escalar del espacio display al espacio frame real
        rx = int(x * frame_w / display_w) if display_w else x
        ry = int(y * frame_h / display_h) if display_h else y
        with self._lock:
            self._zona_pts.append((rx, ry))

    def add_plane_point(self, x: int, y: int,
                        frame_w: int, frame_h: int,
                        display_w: int, display_h: int):
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

    def confirm_zone(self) -> bool:
        with self._lock:
            if len(self._zona_pts) < 3:
                return False
            self._zona_valida = np.array(self._zona_pts, dtype=np.int32)
            self._save_zona()
            self._historial = {}
            self._mode = "TRACKING"
            self._stats["mode"] = "TRACKING"
        self._ensure_model()
        # Reiniciar el thread para que YOLO resetee los IDs de tracking
        self.restart()
        return True

    def confirm_plane(self) -> bool:
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
            self._stats["plane_ready"] = True
            self._save_zona()
            return True

    def reset_zone(self):
        with self._lock:
            self._zona_pts = []
            self._zona_valida = None
            self._historial = {}
            self._mode = "CALIBRANDO"
            self._stats["mode"] = "CALIBRANDO"
            self._people_plane = []
        # Asegurarse de que el thread esté corriendo (para ver el video al calibrar)
        if self._thread is None or not self._thread.is_alive():
            self.restart()

    def reset_plane(self):
        with self._lock:
            self._plane_pts = []
            self._plane_valida = None
            self._homography = None
            self._stats["plane_ready"] = False
            self._save_zona()

    def set_mirror(self, mirror_x: bool):
        with self._lock:
            self._mirror_x = bool(mirror_x)
            self._stats["mirror_x"] = self._mirror_x
            self._save_zona()

    def set_marker_spec(self, marker_id: int, marker_size_mm: float):
        with self._lock:
            self._aruco_marker_id = max(0, int(marker_id))
            self._aruco_marker_size_mm = max(20.0, float(marker_size_mm))
            self._stats["aruco_marker_id"] = self._aruco_marker_id
            self._stats["aruco_marker_size_mm"] = self._aruco_marker_size_mm
            self._save_zona()

    def set_camera_height(self, camera_height_m: float):
        with self._lock:
            self._camera_height_m = max(0.2, min(5.0, float(camera_height_m)))
            self._stats["camera_height_m"] = self._camera_height_m
            self._save_zona()

    def set_screen_config(self, config: dict):
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
            self._stats["screen_config"] = dict(self._screen_config)
            self._save_zona()

    def get_screen_config(self) -> dict:
        with self._lock:
            return dict(self._screen_config)

    def start_auto_plane(self):
        with self._lock:
            self._mode = "CALIBRANDO"
            self._stats["mode"] = "CALIBRANDO"
            self._aruco_auto_active = True
            self._aruco_detected = False
            self._aruco_quality = 0
            self._aruco_quality_label = "bad"
            self._aruco_last_message = "Buscando marcador ArUco…"
            self._stats["aruco_auto_active"] = True
            self._stats["aruco_detected"] = False
            self._stats["aruco_quality"] = self._aruco_quality
            self._stats["aruco_quality_label"] = self._aruco_quality_label
        if self._thread is None or not self._thread.is_alive():
            self.restart()

    def stop_auto_plane(self):
        with self._lock:
            self._aruco_auto_active = False
            self._stats["aruco_auto_active"] = False
            if not self._aruco_detected:
                self._aruco_last_message = "Aruco inactivo"

    def _calc_aruco_quality(self, corners: np.ndarray, frame_shape) -> tuple[int, str]:
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
            float(np.min(pts[:, 0])),
            float(np.min(pts[:, 1])),
            float(w - np.max(pts[:, 0])),
            float(h - np.max(pts[:, 1])),
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

    def confirm_auto_plane(self) -> bool:
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

            self._mode = "TRACKING"
            self._stats["mode"] = "TRACKING"
            self._aruco_auto_active = False
            self._stats["aruco_auto_active"] = False
            self._aruco_last_message = "Plano auto-calibrado y tracking activo"
            self._save_zona()

        self._ensure_model()
        self.restart()
        return True

    def get_aruco_status(self) -> dict:
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
        """Activa el stream sin cambiar la zona ni el modo."""
        if self._mode == "MENU":
            if self._zona_valida is not None:
                self._mode = "TRACKING"
                self._ensure_model()
            else:
                self._mode = "CALIBRANDO"
            self._stats["mode"] = self._mode
        self.restart()

    def set_conf(self, conf: float):
        """Cambia el umbral de confianza (0.0–1.0)."""
        self._conf = max(0.05, min(0.95, conf))
        with self._lock:
            self._stats["conf"] = self._conf

    def set_model(self, model_name: str, imgsz: int = 640, half: bool = True):
        """Cambia el modelo YOLO en caliente. Reinicia el hilo."""
        self._model_name = model_name
        self._imgsz      = imgsz
        self._half       = half
        self._model      = None   # forzar recarga
        with self._lock:
            self._stats["model"] = model_name
            self._stats["imgsz"] = imgsz
        self._ensure_model()
        self.restart()

    def set_perf_profile(self, profile: str):
        profile = (profile or "balanced").lower()

        if profile == "turbo":
            self._model_name = "yolo11n.pt"
            self._imgsz = 256
            self._conf = 0.40
            self._half = True
            self._max_side = 512
            self._jpeg_quality = 55
            self._draw_trails = False
            self._capture_width = 640
            self._capture_height = 480
            self._capture_fps = 30
            selected = "turbo"
        else:
            self._model_name = "yolo11n.pt"
            self._imgsz = 416
            self._conf = 0.35
            self._half = True
            self._max_side = 800
            self._jpeg_quality = 65
            self._draw_trails = False
            self._capture_width = 960
            self._capture_height = 540
            self._capture_fps = 30
            selected = "balanced"

        self._model = None
        with self._lock:
            self._stats["model"] = self._model_name
            self._stats["imgsz"] = self._imgsz
            self._stats["conf"] = self._conf
            self._stats["perf_profile"] = selected

        self._ensure_model()
        self.restart()

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def get_zone_points(self) -> list:
        """Devuelve los puntos de calibración actuales (en coords de frame)."""
        with self._lock:
            return list(self._zona_pts)

    def get_plane_points(self) -> list:
        with self._lock:
            return list(self._plane_pts)

    def get_people_plane(self) -> list:
        with self._lock:
            return list(self._people_plane)

    def get_config(self) -> dict:
        with self._lock:
            return {
                "zone_points": list(self._zona_pts),
                "plane_points": list(self._plane_pts),
                "mirror_x": self._mirror_x,
                "aruco_marker_id": self._aruco_marker_id,
                "aruco_marker_size_mm": self._aruco_marker_size_mm,
                "camera_height_m": self._camera_height_m,
                "screen_config": dict(self._screen_config),
                "aruco_auto_active": self._aruco_auto_active,
            }

    def get_frame_size(self) -> Tuple[int, int]:
        """Devuelve (w, h) del último frame conocido."""
        with self._lock:
            return self._last_w, self._last_h

    # ── Control del hilo ────────────────────────────────────────────────────

    def restart(self):
        self.stop()
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4)

    # ── Hilo de inferencia ───────────────────────────────────────────────────

    _last_w: int = 1280
    _last_h: int = 720

    def _run(self):
        source = self._source
        cap = self._open_capture(source)
        if not cap.isOpened():
            self._push_error_frame(f"No se pudo abrir: {source}")
            return

        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            source_fps = 0.0
        with self._lock:
            self._stats["source_fps"] = round(source_fps, 2)

        fps_t0    = time.time()
        fps_count = 0

        while not self._stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and os.path.isfile(source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._push_error_frame("Sin señal de la cámara")
                time.sleep(0.5)
                continue

            h, w = frame.shape[:2]
            with self._lock:
                self._last_w = w
                self._last_h = h
                mode = self._mode

            # ── Resize para inferencia eficiente ────────────────────────
            # Reducir frames grandes (ej: 2560x1440) antes de enviar a YOLO
            max_side = self._max_side
            if max(h, w) > max_side:
                scale  = max_side / max(h, w)
                frame  = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                    interpolation=cv2.INTER_LINEAR)

            if mode == "CALIBRANDO":
                out = self._draw_calibration(frame)
            elif mode == "TRACKING":
                out = self._process_tracking(frame)
            else:
                out = self._draw_menu(frame)

            # ── Medir FPS ────────────────────────────────────────────────
            fps_count += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 1.0:
                self._fps = round(fps_count / elapsed, 1)
                with self._lock:
                    self._stats["fps"] = self._fps
                fps_count = 0
                fps_t0    = time.time()

            _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            self.buffer.put(buf.tobytes())

        cap.release()

    # ── Dibujo calibración ───────────────────────────────────────────────────

    def _draw_calibration(self, frame):
        frame = frame.copy()
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

        _put_text_bg(frame, f"CALIBRANDO — zona:{len(pts)} plano:{len(plane_pts)}/4", (10, 28),
                     scale=0.65, fg=(0, 255, 150))
        if aruco_active:
            _put_text_bg(
                frame,
                f"AUTO ARUCO: {aruco_msg}",
                (10, 56),
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
                self._stats["aruco_detected"] = False
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
                    gray,
                    self._aruco_dict,
                    parameters=self._aruco_params,
                )
        except Exception as e:
            with self._lock:
                self._aruco_detected = False
                self._aruco_last_message = f"Error ArUco: {e}"
                self._stats["aruco_detected"] = False
            return

        if ids is None or len(ids) == 0:
            with self._lock:
                self._aruco_detected = False
                self._aruco_quality = 0
                self._aruco_quality_label = "bad"
                self._aruco_last_message = f"No se detecta marker {marker_id}"
                self._stats["aruco_detected"] = False
                self._stats["aruco_quality"] = self._aruco_quality
                self._stats["aruco_quality_label"] = self._aruco_quality_label
            return

        flat_ids = ids.flatten().tolist()
        if marker_id not in flat_ids:
            with self._lock:
                self._aruco_detected = False
                self._aruco_quality = 0
                self._aruco_quality_label = "bad"
                self._aruco_last_message = f"Marker detectado, falta ID {marker_id}"
                self._stats["aruco_detected"] = False
                self._stats["aruco_quality"] = self._aruco_quality
                self._stats["aruco_quality_label"] = self._aruco_quality_label
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
            quality_text = "verde" if quality_label == "good" else "amarillo" if quality_label == "warn" else "rojo"
            self._aruco_last_message = f"ID {marker_id} detectado • calidad {quality}/100 ({quality_text})"
            self._stats["aruco_detected"] = True
            self._stats["aruco_quality"] = self._aruco_quality
            self._stats["aruco_quality_label"] = self._aruco_quality_label

    def _draw_menu(self, frame):
        frame = frame.copy()
        _put_text_bg(frame, "Seleccioná una fuente en el panel →",
                     (10, 28), scale=0.7, fg=(0, 255, 150))
        return frame

    # ── Tracking ─────────────────────────────────────────────────────────────

    def _process_tracking(self, frame):
        frame = frame.copy()
        n_en_zona = 0

        with self._lock:
            zona = self._zona_valida.copy() if self._zona_valida is not None else None
            plane = self._plane_valida.copy() if self._plane_valida is not None else None
            H = self._homography.copy() if self._homography is not None else None
            mirror_x = self._mirror_x
            source_label = self._source_label

        if zona is None and plane is not None:
            zona = plane.copy()

        people_frame = []

        if zona is not None:
            cv2.polylines(frame, [zona], isClosed=True, color=COL_ZONA, thickness=2)
        if plane is not None:
            cv2.polylines(frame, [plane], isClosed=True, color=COL_PLANE, thickness=2)

        if not ULTRALYTICS_AVAILABLE or self._model is None or zona is None:
            _put_text_bg(frame, "YOLO no disponible o zona no configurada",
                         (10, 28), fg=(0, 0, 255))
            with self._lock:
                self._stats["personas"] = 0
            return frame

        try:
            results = self._model.track(
                frame,
                persist=True,
                classes=[0],
                conf=self._conf,
                tracker=self._tracker,
                imgsz=self._imgsz,
                half=self._half,
                device=self._device,
                verbose=False
            )
            r = results[0]
            boxes = getattr(r, "boxes", None)

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

                    n_en_zona += 1

                    try:
                        pid = int(np.array(box.id).flatten()[0])
                    except Exception:
                        pid = None

                    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_BOX, 2)
                    _put_text_bg(frame, f"ID {pid}" if pid is not None else "persona",
                                 (x1, y1 - 6), fg=COL_BOX)

                    if pid is not None:
                        with self._lock:
                            self._historial.setdefault(pid, []).append((cx, by))
                            if len(self._historial[pid]) > MAX_PUNTOS:
                                self._historial[pid].pop(0)
                            trail = list(self._historial[pid])

                        u = None
                        v = None
                        depth_rel = None
                        world_x_m = None
                        world_y_m = 0.0
                        world_z_m = None
                        if H is not None:
                            src_pt = np.array([[[float(cx), float(by)]]], dtype=np.float32)
                            dst_pt = cv2.perspectiveTransform(src_pt, H)[0][0]
                            u = float(dst_pt[0])
                            v = float(dst_pt[1])
                            if mirror_x:
                                u = 1.0 - u
                            depth_rel = v

                            marker_size_m = max(0.02, self._aruco_marker_size_mm / 1000.0)
                            world_x_m = (u - 0.5) * marker_size_m
                            world_z_m = (v - 0.5) * marker_size_m

                        people_frame.append({
                            "id": pid,
                            "u": u,
                            "v": v,
                            "depth_rel": depth_rel,
                            "x": world_x_m,
                            "y": world_y_m,
                            "z": world_z_m,
                            "screen_x": cx,
                            "screen_y": by,
                            "ts": time.time(),
                        })

                        if self._draw_trails:
                            if len(trail) >= 2:
                                cv2.polylines(frame, [np.array(trail, dtype=np.int32)],
                                              isClosed=False, color=COL_TRAIL, thickness=2)
                            cv2.circle(frame, trail[-1], 5, COL_TRAIL, -1)

        except Exception as e:
            _put_text_bg(frame, f"Error tracking: {e}", (10, 60), fg=(0, 0, 255))

        with self._lock:
            self._stats["personas"] = n_en_zona
            self._stats["mode"]     = "TRACKING"
            self._stats["source"]   = source_label
            self._stats["plane_ready"] = H is not None
            self._stats["mirror_x"] = mirror_x
            self._people_plane = people_frame

        _put_text_bg(frame, f"Personas en zona: {n_en_zona}",
                     (10, 28), scale=0.65, fg=(0, 255, 150))
        return frame

    def _push_error_frame(self, msg: str):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _put_text_bg(frame, msg, (20, 240), scale=0.7, fg=(0, 0, 255))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.buffer.put(buf.tobytes())

    # ── Persistencia zona ────────────────────────────────────────────────────

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
                    print(f"[tracker] Zona cargada desde '{ZONA_FILE}' ({len(pts)} puntos).")
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

            if len(pts) >= 3:
                self._zona_pts = [tuple(p) for p in pts]
                self._zona_valida = np.array(pts, dtype=np.int32)
                print(f"[tracker] Zona cargada desde '{ZONA_FILE}' ({len(pts)} puntos).")

            if len(plane_pts) == 4:
                self._plane_pts = [tuple(p) for p in plane_pts]
                self._plane_valida = np.array(plane_pts, dtype=np.int32)
                src = np.array(plane_pts, dtype=np.float32)
                dst = np.array(
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    dtype=np.float32,
                )
                self._homography = cv2.getPerspectiveTransform(src, dst)
                self._stats["plane_ready"] = True

            self._stats["mirror_x"] = self._mirror_x
            self._stats["aruco_marker_id"] = self._aruco_marker_id
            self._stats["aruco_marker_size_mm"] = self._aruco_marker_size_mm
            self._stats["camera_height_m"] = self._camera_height_m
            self._stats["screen_config"] = dict(self._screen_config)
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
                f"({self._device_label}) (imgsz={self._imgsz}, half={self._half})…"
            )
            self._model = YOLO(self._model_name)
            with self._lock:
                self._stats["device"] = self._device_label
            print("[tracker] Modelo listo.")
