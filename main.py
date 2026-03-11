"""
main.py

Esqueleto para el tracking con YOLO + filtrado por polígono y visualización de rastros.

Notas:
- Instalar dependencias con `pip install -r requirements.txt`.
- Ajustar `zona_valida` con las coordenadas obtenidas por `calibrador.py`.
"""
import os
import json
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False


MODEL_PATH = 'yolo11s.pt'  # Ajustar si es necesario

# Carga automática de coordenadas desde calibrador.py, o usa el ejemplo por defecto
_ZONA_FILE = 'zona_coords.json'
if os.path.exists(_ZONA_FILE):
    with open(_ZONA_FILE) as _f:
        _pts = json.load(_f)
    zona_valida = np.array(_pts, dtype=np.int32)
    print(f'Zona cargada desde {_ZONA_FILE}: {_pts}')
else:
    zona_valida = np.array([[100, 500], [1180, 500], [1000, 700], [280, 700]], dtype=np.int32)
    print(f'No se encontró {_ZONA_FILE}. Usando zona de ejemplo. Ejecuta calibrador.py primero.')

historial_rutas = {}  # id -> list de (x,y)
MAX_PUNTOS = 30


def draw_zone(frame):
    cv2.polylines(frame, [zona_valida], isClosed=True, color=(255, 0, 0), thickness=2)


def main():
    if ULTRALYTICS_AVAILABLE:
        model = YOLO(MODEL_PATH)
        print('Modelo YOLO cargado')
    else:
        model = None
        print('ultralytics no disponible. El script seguirá mostrando la cámara sin detección.')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('No se pudo abrir la cámara 0')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_zone(frame)

        personas_en_zona = []

        if model:
            try:
                results = model.track(frame, persist=True, classes=[0])
                r = results[0]
                boxes = getattr(r, 'boxes', None)
                if boxes is not None:
                    for box in boxes:
                        # Manejo robusto de las coordenadas
                        xy = None
                        if hasattr(box, 'xyxy'):
                            xy = box.xyxy
                        elif hasattr(box, 'xyxyn'):
                            xy = box.xyxyn

                        if xy is None:
                            continue

                        try:
                            coords = np.array(xy).reshape(-1)
                            x1, y1, x2, y2 = list(map(int, coords[:4]))
                        except Exception:
                            continue

                        centro_x = int((x1 + x2) / 2)
                        base_y = int(y2)

                        inside = cv2.pointPolygonTest(zona_valida, (centro_x, base_y), False)
                        if inside >= 0:
                            # intentar obtener id si existe
                            obj_id = getattr(box, 'id', None)
                            if obj_id is None:
                                try:
                                    obj_id = int(getattr(box, 'tracker_id'))
                                except Exception:
                                    obj_id = None

                            # dibujar bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"ID:{obj_id}" if obj_id is not None else "persona"
                            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                            personas_en_zona.append({"id": obj_id, "x": centro_x, "y": base_y})

                            # historial
                            if obj_id is not None:
                                historial_rutas.setdefault(obj_id, []).append((centro_x, base_y))
                                if len(historial_rutas[obj_id]) > MAX_PUNTOS:
                                    historial_rutas[obj_id].pop(0)

            except Exception as e:
                cv2.putText(frame, f"Tracking error: {e}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Dibujar rastros
        for pid, puntos in historial_rutas.items():
            if len(puntos) >= 2:
                pts = np.array(puntos, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=False, color=(0,255,255), thickness=2)

        # Mostrar lista simple en consola
        if personas_en_zona:
            print('personas_en_zona =', personas_en_zona)

        cv2.imshow('Tracking', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
