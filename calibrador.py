import cv2
import json

WINDOW_NAME = "Calibrador - Haz clic para definir puntos (q para salir)"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara 0")
        return

    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
            print(f"Clic guardado: ({x}, {y})")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for p in points:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{p}", (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Coordenadas finales:")
    print(points)

    try:
        with open('zona_coords.json', 'w') as f:
            json.dump(points, f)
        print("Guardadas en zona_coords.json")
    except Exception as e:
        print("No se pudo guardar las coordenadas:", e)

if __name__ == '__main__':
    main()
