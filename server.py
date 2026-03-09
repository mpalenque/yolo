"""
server.py — FastAPI + MJPEG stream + WebSocket + upload de video.

Arrancar:
    python server.py
Luego abrir:
    http://localhost:8000
"""

import os
import shutil
import asyncio
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles

from tracker import Tracker, RTSP_URL

# ── Contador de clientes activos ─────────────────────────────────
active_clients = 0

def client_connected():
    global active_clients
    active_clients += 1
    if active_clients == 1:
        tracker.start_stream()

def client_disconnected():
    global active_clients
    active_clients = max(0, active_clients - 1)
    if active_clients == 0:
        tracker.stop()
        print("[server] Sin clientes — tracker pausado.")


@asynccontextmanager
async def lifespan(app):
    yield  # El tracker arranca solo cuando el primer cliente se conecta
    tracker.stop()


# ── App y tracker global ────────────────────────────────────────────
tracker = Tracker()
app     = FastAPI(title="Tracking Audiencia", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


# ── Página principal ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text("utf-8"))
    return HTMLResponse("<h1>index.html no encontrado en static/</h1>", status_code=500)


@app.get("/content.html", response_class=HTMLResponse)
async def content_page():
    html_file = STATIC_DIR / "content.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text("utf-8"))
    return HTMLResponse("<h1>content.html no encontrado en static/</h1>", status_code=500)


@app.get("/aruco.html", response_class=HTMLResponse)
async def aruco_page():
    html_file = STATIC_DIR / "aruco.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text("utf-8"))
    return HTMLResponse("<h1>aruco.html no encontrado en static/</h1>", status_code=500)


@app.get("/visual.html", response_class=HTMLResponse)
async def visual_page():
    html_file = STATIC_DIR / "visual.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text("utf-8"))
    return HTMLResponse("<h1>visual.html no encontrado en static/</h1>", status_code=500)


@app.get("/fullscreen", response_class=HTMLResponse)
async def fullscreen_page():
    html_file = STATIC_DIR / "fullscreen.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text("utf-8"))
    return HTMLResponse("<h1>fullscreen.html no encontrado en static/</h1>", status_code=500)


@app.get("/fullscreen2", response_class=HTMLResponse)
async def fullscreen2_page():
    html_file = STATIC_DIR / "fullscreen2.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text("utf-8"))
    return HTMLResponse("<h1>fullscreen2.html no encontrado en static/</h1>", status_code=500)


@app.get("/media/visual-loop.mp4")
async def visual_loop_video():
    video_file = Path(__file__).parent / "160723_00001_int_loop (1).mp4"
    if not video_file.exists():
        return Response(content=b"video no encontrado", media_type="text/plain", status_code=404)
    return FileResponse(str(video_file), media_type="video/mp4", filename=video_file.name)


@app.get("/api/aruco/marker.png")
async def aruco_marker_png(marker_id: int = 0, px: int = 700):
    try:
        import cv2
        if not hasattr(cv2, "aruco"):
            return Response(content=b"Aruco unavailable", media_type="text/plain", status_code=500)
        marker_id = max(0, int(marker_id))
        px = max(256, min(2400, int(px)))
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, px)
        ok, encoded = cv2.imencode(".png", img)
        if not ok:
            return Response(content=b"encode error", media_type="text/plain", status_code=500)
        return Response(content=encoded.tobytes(), media_type="image/png")
    except Exception as e:
        return Response(content=str(e).encode("utf-8"), media_type="text/plain", status_code=500)


@app.get("/api/view-config")
async def get_view_config():
    return JSONResponse({"ok": True, "view_config": tracker.get_view_config()})


@app.get("/api/visual-config")
async def get_visual_config():
    return JSONResponse({"ok": True, "visual_config": tracker.get_visual_config()})


@app.post("/api/visual-config")
async def set_visual_config(request: Request):
    data = await request.json()
    tracker.set_visual_config(data if isinstance(data, dict) else {})
    return JSONResponse({"ok": True, "visual_config": tracker.get_visual_config()})


@app.post("/api/camera-height")
async def set_camera_height_api(request: Request):
    data = await request.json()
    camera_height_m = float((data or {}).get("camera_height_m", 0.94))
    tracker.set_camera_height(camera_height_m)
    return JSONResponse({"ok": True, "camera_height_m": tracker.get_stats().get("camera_height_m", 0.94)})


@app.post("/api/view-config")
async def set_view_config(request: Request):
    data = await request.json()
    tracker.set_view_config(data if isinstance(data, dict) else {})
    return JSONResponse({"ok": True, "view_config": tracker.get_view_config()})


# ── MJPEG stream ─────────────────────────────────────────────────────────────
def _mjpeg_generator():
    client_connected()
    boundary = b"--frame\r\n"
    header   = b"Content-Type: image/jpeg\r\n\r\n"
    try:
        while True:
            jpeg = tracker.buffer.wait_for_frame(timeout=2.0)
            if jpeg is None:
                continue
            yield boundary + header + jpeg + b"\r\n"
    finally:
        client_disconnected()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Stats (polling fallback) ─────────────────────────────────────────────────
@app.get("/stats")
def stats():
    return JSONResponse(tracker.get_stats())


@app.get("/sources/webcams")
async def sources_webcams():
    loop = asyncio.get_running_loop()
    webcams = await loop.run_in_executor(None, tracker.scan_webcams)
    return JSONResponse({"ok": True, "webcams": webcams})


# ── Upload de video ──────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    ext  = Path(file.filename).suffix or ".mp4"
    dest = UPLOADS_DIR / f"video{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    tracker.set_source(str(dest), f"Video: {file.filename}")
    return JSONResponse({"ok": True, "path": str(dest), "name": file.filename})


# ── WebSocket de control ─────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Manda stats periódicas (10 Hz) mientras el cliente esté conectado
    async def push_stats():
        while True:
            try:
                await ws.send_json({"type": "stats", **tracker.get_stats()})
                await asyncio.sleep(0.2)
            except Exception:
                break

    asyncio.create_task(push_stats())

    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")

            if action == "set_source":
                source = data.get("source")
                if isinstance(source, str) and source.strip().isdigit():
                    source = int(source.strip())
                label  = data.get("label", source)
                # Run in thread to avoid blocking asyncio event loop
                loop = asyncio.get_running_loop()
                ok, msg = await loop.run_in_executor(None, tracker.set_source, source, label)
                await ws.send_json({"type": "ack", "action": action, "ok": ok, "msg": msg})

            elif action == "start_stream":
                tracker.start_stream()
                await ws.send_json({"type": "ack", "action": action, "ok": True})

            elif action == "add_point":
                tracker.add_calibration_point(
                    x=data["x"], y=data["y"],
                    frame_w=data.get("fw", 1280), frame_h=data.get("fh", 720),
                    display_w=data.get("dw", 1280), display_h=data.get("dh", 720),
                )
                await ws.send_json({"type": "ack", "action": action, "ok": True,
                                    "points": tracker.get_zone_points()})

            elif action == "add_plane_point":
                tracker.add_plane_point(
                    x=data["x"], y=data["y"],
                    frame_w=data.get("fw", 1280), frame_h=data.get("fh", 720),
                    display_w=data.get("dw", 1280), display_h=data.get("dh", 720),
                )
                await ws.send_json({"type": "ack", "action": action, "ok": True,
                                    "plane_points": tracker.get_plane_points()})

            elif action == "undo_point":
                tracker.undo_last_point()
                await ws.send_json({"type": "ack", "action": action, "ok": True,
                                    "points": tracker.get_zone_points()})

            elif action == "undo_plane_point":
                tracker.undo_last_plane_point()
                await ws.send_json({"type": "ack", "action": action, "ok": True,
                                    "plane_points": tracker.get_plane_points()})

            elif action == "confirm_zone":
                ok = tracker.confirm_zone()
                await ws.send_json({"type": "ack", "action": action, "ok": ok,
                                    "msg": "Zona confirmada" if ok else "Necesitás al menos 3 puntos"})

            elif action == "confirm_plane":
                ok = tracker.confirm_plane()
                await ws.send_json({"type": "ack", "action": action, "ok": ok,
                                    "msg": "Plano confirmado" if ok else "Necesitás 4 puntos para el plano"})

            elif action == "set_conf":
                tracker.set_conf(float(data.get("conf", 0.35)))
                await ws.send_json({"type": "ack", "action": action, "ok": True})

            elif action == "set_model":
                tracker.set_model(
                    model_name=data.get("model", "yolo11m.pt"),
                    imgsz=int(data.get("imgsz", 640)),
                    half=bool(data.get("half", True)),
                )
                await ws.send_json({"type": "ack", "action": action, "ok": True})

            elif action == "set_perf_profile":
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, tracker.set_perf_profile, str(data.get("profile", "balanced")))
                await ws.send_json({"type": "ack", "action": action, "ok": True,
                                    "msg": f"Perfil {data.get('profile', 'balanced')} aplicado"})

            elif action == "reset_zone":
                tracker.reset_zone()
                await ws.send_json({"type": "ack", "action": action, "ok": True})

            elif action == "reset_plane":
                tracker.reset_plane()
                await ws.send_json({"type": "ack", "action": action, "ok": True})

            elif action == "set_mirror":
                tracker.set_mirror(bool(data.get("mirror_x", False)))
                await ws.send_json({"type": "ack", "action": action, "ok": True})

            elif action == "set_marker_spec":
                tracker.set_marker_spec(
                    marker_id=int(data.get("marker_id", 0)),
                    marker_size_mm=float(data.get("marker_size_mm", 120.0)),
                )
                await ws.send_json({"type": "ack", "action": action, "ok": True,
                                    "msg": "Parámetros ArUco guardados"})

            elif action == "set_camera_height":
                tracker.set_camera_height(float(data.get("camera_height_m", 0.94)))
                await ws.send_json({"type": "ack", "action": action, "ok": True,
                                    "msg": "Altura de cámara guardada"})

            elif action == "set_screen_config":
                tracker.set_screen_config({
                    "enabled": data.get("enabled", True),
                    "x_m": data.get("x_m", 0.0),
                    "z_m": data.get("z_m", 1.2),
                    "width_m": data.get("width_m", 3.5),
                    "height_m": data.get("height_m", 2.0),
                    "yaw_deg": data.get("yaw_deg", 0.0),
                })
                await ws.send_json({"type": "ack", "action": action, "ok": True,
                                    "msg": "Configuración de pantalla guardada"})

            elif action == "set_simulation":
                ok, msg = tracker.set_simulation(bool(data.get("enabled", False)))
                await ws.send_json({"type": "ack", "action": action, "ok": ok, "msg": msg})

            elif action == "start_auto_plane":
                tracker.start_auto_plane()
                await ws.send_json({"type": "ack", "action": action, "ok": True,
                                    "msg": "Auto-calibración ArUco iniciada"})

            elif action == "stop_auto_plane":
                tracker.stop_auto_plane()
                await ws.send_json({"type": "ack", "action": action, "ok": True,
                                    "msg": "Auto-calibración ArUco detenida"})

            elif action == "confirm_auto_plane":
                ok = tracker.confirm_auto_plane()
                await ws.send_json({"type": "ack", "action": action, "ok": ok,
                                    "msg": "Plano auto confirmado" if ok else "No hay detección ArUco válida"})

            elif action == "get_aruco":
                await ws.send_json({"type": "aruco", **tracker.get_aruco_status()})

            elif action == "get_zone":
                w, h = tracker.get_frame_size()
                config = tracker.get_config()
                await ws.send_json({"type": "zone",
                                    "points": config["zone_points"],
                                    "plane_points": config["plane_points"],
                                    "mirror_x": config["mirror_x"],
                                    "aruco_marker_id": config["aruco_marker_id"],
                                    "aruco_marker_size_mm": config["aruco_marker_size_mm"],
                                    "camera_height_m": config["camera_height_m"],
                                    "screen_config": config["screen_config"],
                                    "view_config": config["view_config"],
                                        "visual_config": config["visual_config"],
                                    "aruco_auto_active": config["aruco_auto_active"],
                                    "fw": w, "fh": h})

    except WebSocketDisconnect:
        pass


@app.websocket("/ws/positions")
async def websocket_positions(ws: WebSocket):
    await ws.accept()
    client_connected()

    try:
        while True:
            stats = tracker.get_stats()
            await ws.send_json({
                "type": "positions",
                "people": tracker.get_people_plane(),
                "mirror_x": stats.get("mirror_x", False),
                "plane_ready": stats.get("plane_ready", False),
                "camera_name": stats.get("source", "Camera"),
                "aruco_marker_id": stats.get("aruco_marker_id", 0),
                "aruco_marker_size_mm": stats.get("aruco_marker_size_mm", 120.0),
                "camera_height_m": stats.get("camera_height_m", 0.94),
                "screen_config": stats.get("screen_config", {}),
                "view_config": stats.get("view_config", {}),
                    "visual_config": stats.get("visual_config", {}),
                "mode": stats.get("mode", "MENU"),
                "ts": asyncio.get_running_loop().time(),
            })
            await asyncio.sleep(0.067)
    except WebSocketDisconnect:
        pass
    finally:
        client_disconnected()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("  Tracking Audiencia — Web UI")
    print("  http://localhost:8000")
    print("=" * 52)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")
