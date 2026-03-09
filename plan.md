🗺️ Plan Maestro y Prompts: Tracking de Audiencia con YOLO y OpenCV

🎯 Objetivo del Proyecto

Detectar y trackear personas frente a una pantalla LED usando una cámara a 4 metros de altura, filtrando la detección a una zona específica (Polígono/ROI), exportando las coordenadas de las personas válidas y visualizando su recorrido en tiempo real sobre la cámara.

📂 Arquitectura del Proyecto

Crea una carpeta tracking-pantalla con estos archivos vacíos para empezar:

requirements.txt

calibrador.py

main.py

🛠️ Fase 1: Entorno (Hazlo tú mismo en la terminal)

Copia y pega esto en la terminal de VS Code:

python -m venv venv
.\venv\Scripts\activate
pip install ultralytics opencv-python numpy python-osc


📐 Fase 2: Script Auxiliar (calibrador.py)

Abre el archivo calibrador.py, ve al chat de tu IA (Copilot/GPT) y pégale exactamente este prompt:

Prompt para la IA:
"Actúa como un experto en Python y OpenCV. Escribe el código para calibrador.py. El objetivo es que lea la cámara web (índice 0), muestre el video en una ventana y use cv2.setMouseCallback. Cada vez que el usuario haga clic izquierdo en el video, debe guardar las coordenadas (X, Y) en una lista, dibujar un pequeño círculo rojo en ese punto y mostrar la coordenada en la consola. Cuando el usuario presione la tecla 'q', el script se cierra e imprime la lista final de coordenadas."

Nota para vos: Ejecutá ese código, hacé 4 clics en el piso formando tu zona válida, anotá los 4 números que te tira al final y cerralo.

🧠 Fase 3: El Script Principal (main.py)

Abre main.py. Como este script es más complejo, se lo vamos a pedir a la IA en tres partes para que no se equivoque y quede perfecto.

Paso 1: Estructura base y tracking

Pégale este prompt a la IA:

Prompt para la IA (Parte 1):
"Actúa como un experto en Computer Vision. Escribe la estructura base para main.py. Importa cv2, numpy y YOLO de ultralytics.

Carga el modelo 'yolov8n.pt'.

Inicia la captura de video con cv2.VideoCapture(0).

Crea un bucle while True para leer los frames.

En cada frame, haz inferencia usando resultados = model.track(frame, persist=True, classes=[0]).

Muestra el frame resultante en una ventana llamada 'Tracking'. Cierra con la tecla 'q'."

Paso 2: El Filtro del Polígono

Una vez que el código anterior funcione, pídele a la IA que agregue la lógica del área. (Asegúrate de cambiar las coordenadas del prompt por las tuyas).

Prompt para la IA (Parte 2):
"Excelente. Ahora modifica el código que acabas de hacer para agregar un filtro de zona de interés (ROI).

Antes del bucle while, define un array de numpy llamado zona_valida con estas coordenadas: 

$$\[100, 500$$

, 

$$1180, 500$$

, 

$$1000, 700$$

, 

$$280, 700$$

] de tipo np.int32.

En el bucle while, dibuja este polígono en el frame usando cv2.polylines (color azul, grosor 2).

Crea una lista vacía personas_en_zona = [] al inicio de cada frame.

Itera sobre los boxes detectados por YOLO. Obtén las coordenadas x1, y1, x2, y2 de cada persona y su ID.

Calcula el punto de los pies: centro_x = int((x1 + x2) / 2) y base_y = int(y2).

Usa cv2.pointPolygonTest para evaluar si el punto (centro_x, base_y) está dentro de zona_valida.

Solo si está dentro: dibuja el bounding box en verde y el ID sobre la cabeza, y agrega el diccionario {"id": id, "x": centro_x, "y": base_y} a personas_en_zona.

Imprime la lista personas_en_zona en la consola."

Paso 3: Visualización de Rastros (Trails) sobre la cámara

Para que quede súper visual y veas por dónde se mueve cada persona, le pedimos esto:

Prompt para la IA (Parte 3):
"Perfecto. Ahora vamos a hacer que la visualización sobre la webcam sea más avanzada mostrando el recorrido de cada persona.

Antes del bucle while, crea un diccionario global llamado historial_rutas = {} y una variable MAX_PUNTOS = 30.

Dentro de la lógica de 'Solo si está dentro', además de lo que ya hiciste, agrega la tupla (centro_x, base_y) a la lista de puntos de ese ID en historial_rutas. Si la lista supera MAX_PUNTOS, elimina el punto más antiguo.

Usa un bucle para recorrer historial_rutas y utiliza cv2.polylines (o múltiples cv2.line) para dibujar una línea de color llamativo (ej. amarillo o magenta) que conecte el historial de puntos de cada ID, creando un efecto de 'estela' o 'rastro' del movimiento de la persona sobre el frame de video."

🚀 Siguientes pasos

Con esos prompts, la IA te va a generar un código modular, limpio y documentado. Si llega a fallar algo (por ejemplo, YOLO a veces devuelve los tensores de diferente forma en nuevas versiones), simplemente copiás el error que te tira la terminal y se lo pegás a la IA; al tener el contexto dividido en bloques, lo va a arreglar en segundos.