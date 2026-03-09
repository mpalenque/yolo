"""
app.py  —  Tracking de audiencia: calibración + YOLO en un solo script.

Controles
---------
MODO CALIBRACIÓN (al inicio):
  Clic izquierdo        → agregar punto a la zona
  Backspace             → borrar último punto
  Enter / Space         → confirmar zona y arrancar tracking (mínimo 3 puntos)
  q                     → salir

MODO TRACKING:
  r                     → volver a calibrar
  q                     → salir
"""

import os
import json
import sys
import glob

import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Dependencia opcional: ultralytics ─────────────────────────────────────────
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# ── Constantes ─────────────────────────────────────────────────────────────────
MODEL_PATH   = "yolov8n.pt"
ZONA_FILE    = "zona_coords.json"
MAX_PUNTOS   = 40           # máximo de puntos de rastro por persona
WINDOW       = "Tracking Audiencia"
RTSP_URL     = "rtsp://mpalenque:madariaga1@192.168.0.96:554/stream1"

# ── Estado global de UI ────────────────────────────────────────────────────────
MODE         = "CALIBRANDO"   # "CALIBRANDO" | "TRACKING"
zona_pts     = []             # lista de tuplas (x, y) durante calibración
zona_valida  = None           # np.array cuando se confirma
historial    = {}             # id → [(x,y), ...]
model        = None           # instancia YOLO cargada en el primer frame
DEVICE       = "cpu"
DEVICE_LABEL = "CPU"
USE_HALF     = False


# ── Colores BGR ────────────────────────────────────────────────────────────────
COL_ZONA     = (255, 120,  0)   # naranja
COL_BOX      = (  0, 220,  0)   # verde
COL_TRAIL    = (  0, 220, 220)  # amarillo-cian
COL_POINT    = (  0,  60, 255)  # rojo-cian
COL_TEXT_BG  = (  0,   0,   0)
COL_TEXT_FG  = (255, 255, 255)


# ── Helpers ────────────────────────────────────────────────────────────────────

def put_text_bg(frame, text, org, scale=0.6, thickness=1, fg=COL_TEXT_FG, bg=COL_TEXT_BG):
    """Texto con fondo negro para legibilidad."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def draw_instructions(frame):
    lines = [
        "CALIBRACIÓN",
        "Clic izq: agregar punto  |  Backspace: borrar",
        "Enter/Space: confirmar zona  |  q: salir",
        f"Puntos: {len(zona_pts)} (mínimo 3)",
    ]
    for i, line in enumerate(lines):
        color = (0, 255, 150) if i == 0 else COL_TEXT_FG
        put_text_bg(frame, line, (10, 28 + i * 26), scale=0.65, thickness=1, fg=color)


def draw_calibration(frame):
    """Dibuja puntos y líneas de la zona en modo calibración."""
    draw_instructions(frame)
    for idx, p in enumerate(zona_pts):
        cv2.circle(frame, p, 6, COL_POINT, -1)
        cv2.circle(frame, p, 8, COL_TEXT_FG, 1)
        put_text_bg(frame, str(idx + 1), (p[0] + 10, p[1] - 6), scale=0.5)
    if len(zona_pts) >= 2:
        pts_arr = np.array(zona_pts, dtype=np.int32)
        cv2.polylines(frame, [pts_arr], isClosed=(len(zona_pts) >= 3), color=COL_ZONA, thickness=2)


def draw_tracking_hud(frame, n):
    """HUD mínima durante tracking."""
    h, w = frame.shape[:2]
    put_text_bg(frame, f"Personas en zona: {n}", (10, 28), scale=0.65, fg=(0, 255, 150))
    put_text_bg(frame, "r: recalibrar  |  q: salir", (10, h - 12), scale=0.5)


def draw_trails(frame):
    for pid, puntos in historial.items():
        if len(puntos) >= 2:
            pts = np.array(puntos, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=COL_TRAIL, thickness=2)
            # punto actual más grande
            cv2.circle(frame, puntos[-1], 5, COL_TRAIL, -1)


def ensure_model():
    """Carga YOLO la primera vez (descarga automática si no existe)."""
    global model, DEVICE, DEVICE_LABEL, USE_HALF
    if model is None and ULTRALYTICS_AVAILABLE:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            DEVICE = "cuda:0"
            DEVICE_LABEL = torch.cuda.get_device_name(0)
            USE_HALF = True
        elif TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
            DEVICE_LABEL = "Apple MPS"
            USE_HALF = True
        else:
            DEVICE = "cpu"
            DEVICE_LABEL = "CPU"
            USE_HALF = False
        print(f"Cargando modelo YOLO en {DEVICE} ({DEVICE_LABEL})…")
        model = YOLO(MODEL_PATH)
        print("Modelo listo.")


def process_tracking(frame):
    """Inferencia YOLO + filtro ROI + rastros. Devuelve frame anotado."""
    global historial
    n_en_zona = 0

    # Dibujar zona
    cv2.polylines(frame, [zona_valida], isClosed=True, color=COL_ZONA, thickness=2)

    if not ULTRALYTICS_AVAILABLE or model is None:
        put_text_bg(frame, "ultralytics no disponible", (10, 60), fg=(0, 0, 255))
        draw_tracking_hud(frame, 0)
        return frame

    try:
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            device=DEVICE,
            half=USE_HALF,
            verbose=False,
        )
        r = results[0]
        boxes = getattr(r, "boxes", None)

        if boxes is not None:
            for box in boxes:
                # Coordenadas
                try:
                    coords = np.array(box.xyxy).reshape(-1)
                    x1, y1, x2, y2 = map(int, coords[:4])
                except Exception:
                    continue

                cx = (x1 + x2) // 2
                by = int(y2)

                # Filtro ROI
                inside = cv2.pointPolygonTest(zona_valida, (float(cx), float(by)), False)
                if inside < 0:
                    continue

                n_en_zona += 1

                # ID de tracking
                try:
                    pid = int(np.array(box.id).flatten()[0])
                except Exception:
                    pid = None

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), COL_BOX, 2)
                label = f"ID {pid}" if pid is not None else "persona"
                put_text_bg(frame, label, (x1, y1 - 6), scale=0.55, fg=COL_BOX)

                # Historial de rastro
                if pid is not None:
                    historial.setdefault(pid, []).append((cx, by))
                    if len(historial[pid]) > MAX_PUNTOS:
                        historial[pid].pop(0)

    except Exception as e:
        put_text_bg(frame, f"Error: {e}", (10, 60), fg=(0, 0, 255))

    draw_trails(frame)
    draw_tracking_hud(frame, n_en_zona)
    return frame


# ── Mouse callback ─────────────────────────────────────────────────────────────

def mouse_cb(event, x, y, flags, param):
    if MODE == "CALIBRANDO" and event == cv2.EVENT_LBUTTONDOWN:
        zona_pts.append((x, y))


# ── Menú de selección de fuente ────────────────────────────────────────────────

def detectar_webcams(max_idx=5):
    """Detecta webcams disponibles probando índices 0..max_idx-1."""
    disponibles = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                disponibles.append(i)
        cap.release()
    return disponibles


def menu_fuente():
    """Muestra un menú en consola y devuelve el argumento para VideoCapture."""
    print("\n" + "═" * 50)
    print("   TRACKING DE AUDIENCIA — Selección de fuente")
    print("═" * 50)

    opciones = []

    # Opción Tapo siempre presente
    opciones.append(("Cámara Tapo (RTSP)", RTSP_URL))
    print(f"  [1] Cámara Tapo  →  {RTSP_URL}")

    # Webcams disponibles
    print("      Buscando webcams...", end="\r")
    webcams = detectar_webcams()
    for idx in webcams:
        label = f"Webcam (índice {idx})"
        opciones.append((label, idx))
        print(f"  [{len(opciones)}] {label}")

    # Archivo de video
    opciones.append(("Archivo de video", "__FILE__"))
    print(f"  [{len(opciones)}] Cargar archivo de video")

    print("═" * 50)

    while True:
        try:
            eleccion = input(f"Elegí una opción [1-{len(opciones)}]: ").strip()
            n = int(eleccion)
            if 1 <= n <= len(opciones):
                label, fuente = opciones[n - 1]
                if fuente == "__FILE__":
                    path = input("Ruta del archivo de video: ").strip().strip('"').strip("'")
                    if not os.path.exists(path):
                        print(f"  ✗ No se encontró el archivo: {path}")
                        continue
                    fuente = path
                    label = f"Video: {os.path.basename(path)}"
                print(f"\n  → Usando: {label}\n")
                return fuente
            else:
                print(f"  Ingresá un número entre 1 y {len(opciones)}.")
        except (ValueError, KeyboardInterrupt):
            print("\nSaliendo.")
            sys.exit(0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global MODE, zona_pts, zona_valida, historial

    fuente = menu_fuente()
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la fuente seleccionada: {fuente}")
        sys.exit(1)

    # Si ya existe zona calibrada, arrancar directo en tracking
    if os.path.exists(ZONA_FILE):
        try:
            with open(ZONA_FILE) as f:
                pts = json.load(f)
            if len(pts) >= 3:
                zona_valida = np.array(pts, dtype=np.int32)
                MODE = "TRACKING"
                ensure_model()
                print(f"Zona cargada desde '{ZONA_FILE}'. Arrancando en modo tracking.")
        except Exception as e:
            print(f"No se pudo leer '{ZONA_FILE}': {e}. Iniciando calibración.")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, mouse_cb)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: No se pudo leer frame de la cámara.")
            break

        if MODE == "CALIBRANDO":
            draw_calibration(frame)
        else:
            frame = process_tracking(frame)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        # ── Teclas globales ──────────────────────────────────────
        if key == ord("q"):
            break

        # ── Teclas modo calibración ──────────────────────────────
        if MODE == "CALIBRANDO":
            if key == 8:  # Backspace
                if zona_pts:
                    zona_pts.pop()
            elif key in (13, 32):  # Enter o Space → confirmar
                if len(zona_pts) < 3:
                    print("Necesitás al menos 3 puntos. Seguí haciendo clic.")
                else:
                    zona_valida = np.array(zona_pts, dtype=np.int32)
                    # Guardar zona
                    try:
                        with open(ZONA_FILE, "w") as f:
                            json.dump(zona_pts, f)
                        print(f"Zona guardada en '{ZONA_FILE}': {zona_pts}")
                    except Exception as e:
                        print(f"No se pudo guardar zona: {e}")
                    historial = {}
                    MODE = "TRACKING"
                    ensure_model()
                    print("Modo TRACKING activo.")

        # ── Teclas modo tracking ─────────────────────────────────
        elif MODE == "TRACKING":
            if key == ord("r"):
                zona_pts = []
                zona_valida = None
                historial = {}
                MODE = "CALIBRANDO"
                print("Volviendo a CALIBRACIÓN.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
